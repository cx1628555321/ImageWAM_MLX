"""Fused Qwen Q/K/V projection; weights concatenated without changing values."""
import argparse,json,sys,time,shutil
from pathlib import Path
import numpy as np
import mlx.core as mx
sys.path.insert(0,str(Path(__file__).resolve().parent))
from compile_experiments import Candidates,stats,error
from imagewam_mlx import ImageWAM
from imagewam_mlx.qwen import Qwen
from imagewam_mlx.ops import rms,silu,attention

class FusedQwen(Qwen):
 def __init__(self,original):
  super().__init__(original.w)
  self.qkv={f'layers.{i}':mx.concatenate([self.w[f'layers.{i}.self_attn.{k}_proj.weight'] for k in ('q','k','v')],axis=0) for i in range(27)}
  mx.eval(self.qkv)
 def __call__(self,ids):
  x=self.w['embed_tokens.weight'][ids];n=ids.shape[1];hidden=[]
  freq=1/(1000000**(mx.arange(0,128,2,dtype=mx.float32)/128));a=mx.arange(n,dtype=mx.float32)[:,None]*freq[None]
  c=mx.concatenate([mx.cos(a),mx.cos(a)],axis=-1).astype(x.dtype)[None,None];s=mx.concatenate([mx.sin(a),mx.sin(a)],axis=-1).astype(x.dtype)[None,None]
  def rope(z):
   first,last=mx.split(z,2,axis=-1);return z*c+mx.concatenate([-last,first],axis=-1)*s
  for i in range(27):
   p=f'layers.{i}';y=rms(x,self.w[p+'.input_layernorm.weight']);q,k,v=mx.split(y@self.qkv[p].T,[4096,5120],axis=-1)
   q=q.reshape(1,n,32,128).transpose(0,2,1,3);k=k.reshape(1,n,8,128).transpose(0,2,1,3);v=v.reshape(1,n,8,128).transpose(0,2,1,3)
   q=rope(rms(q,self.w[p+'.self_attn.q_norm.weight']));k=rope(rms(k,self.w[p+'.self_attn.k_norm.weight']))
   x=x+self.linear(attention(q,k,v,'causal'),p+'.self_attn.o_proj');y=rms(x,self.w[p+'.post_attention_layernorm.weight'])
   x=x+self.linear(silu(self.linear(y,p+'.mlp.gate_proj'))*self.linear(y,p+'.mlp.up_proj'),p+'.mlp.down_proj')
   if i+1 in (9,18,27):hidden.append(x)
  return mx.concatenate(hidden,axis=-1)

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);shutil.copyfile(__file__,a.output/'qkv_experiment.py')
 model=ImageWAM('weights/bf16','assets');c=Candidates(model);c.set('both_compile');original=model.qwen;fused=FusedQwen(original);compiled=mx.compile(fused.__call__)
 ex=Path('examples/robotwin_ep0_frame0');raw=np.load(ex/'inputs.npz');im={k:raw[k] for k in ('cam_high','cam_left_wrist','cam_right_wrist')};pr=(ex/'prompt.txt').read_text();noise=np.load(ex/'initial_noise_fp32.npy')
 def once(mode,prompt=pr,profile=False):
  model.qwen={'baseline':original,'fused':fused,'fused_compile':compiled}[mode];mx.synchronize();t=time.perf_counter();x,n=model.predict(im,raw['state'],prompt,noise,profile=profile);mx.synchronize();return x,(time.perf_counter()-t)*1000
 r={'results':{},'status':'running','rows':[],'extra_qkv_bytes':sum(w.nbytes for w in fused.qkv.values())}
 for _ in range(8):once('baseline')
 for candidate in ('fused','fused_compile'):
  _,first=once(candidate)
  for _ in range(6):once(candidate)
  for _ in range(6):once('baseline')
  times={'baseline':[],candidate:[]};outputs={}
  for i in range(40):
   for mode in (('baseline',candidate) if i%2==0 else (candidate,'baseline')):
    x,ms=once(mode);outputs[mode]=x;times[mode].append(ms);r['rows'].append({'candidate':candidate,'mode':mode,'index':i,'ms':ms});np.save(a.output/f'{candidate}_{mode}_{i:03d}.npy',x)
  modules=[]
  for _ in range(5):once(candidate,profile=True);modules.append(dict(model.profile))
  result={'first_ms':first,'times':{k:stats(v) for k,v in times.items()},'error':error(outputs[candidate],outputs['baseline']),'modules':{k:float(np.mean([m[k] for m in modules])) for k in modules[0]},'dynamic':[]}
  inv=json.loads(Path('../ImageWAM/development_records/m5pro_robotwin/20261103_dynamic_optimization/text_lengths.json').read_text())['examples_by_length']
  for length in (31,79,48,31):
   prompt=inv[str(length)]['prompt'];ref,_=once('baseline',prompt);out,ms=once(candidate,prompt);result['dynamic'].append({'tokens':length,'error':error(out,ref),'ms':ms})
  r['results'][candidate]=result;(a.output/'results.json').write_text(json.dumps(r,indent=2));print('RESULT',candidate,result,flush=True)
 r['status']='completed';(a.output/'results.json').write_text(json.dumps(r,indent=2))
if __name__=='__main__':main()
