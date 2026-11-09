"""Encoder optimizations on top of the established both_compile baseline."""
import argparse,copy,json,sys,time,types,shutil
from pathlib import Path
import numpy as np
import mlx.core as mx
sys.path.insert(0,str(Path(__file__).resolve().parent))
from compile_experiments import Candidates,stats,error
import imagewam_mlx.model as mm
import imagewam_mlx.qwen as qm

class EncoderCandidates:
 def __init__(self,model):
  self.model=model;self.base=Candidates(model);self.base.set('both_compile')
  self.qwen=model.qwen;self.vae=model.vae
  self.qwen_compile=mx.compile(self.qwen.__call__)
  self.vae_compile=mx.compile(self.vae.__call__)
  self.contiguous=copy.copy(self.vae)
  self.contiguous.w={k:mx.contiguous(v) if v.ndim==4 else v for k,v in self.vae.w.items()};mx.eval(self.contiguous.w)
  def fast_rms(x,w,eps=1e-6):
   # Preserve BF16 normalization then weight multiply as separate operations.
   return mx.fast.rms_norm(x,None,eps)*w
  env=dict(qm.Qwen.__call__.__globals__);env['rms']=fast_rms
  fast_function=types.FunctionType(qm.Qwen.__call__.__code__,env)
  self.fast_qwen=mx.compile(types.MethodType(fast_function,self.qwen))
 def set(self,mode):
  self.base.set('both_compile')
  self.model.qwen=self.qwen_compile if mode in ('qwen_compile','encoders_compile') else (self.fast_qwen if mode=='qwen_fast_rms' else self.qwen)
  self.model.vae=self.vae_compile if mode=='encoders_compile' else (self.contiguous if mode=='vae_contiguous' else self.vae)

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--iterations',type=int,default=30);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 ex=Path('examples/robotwin_ep0_frame0');raw=np.load(ex/'inputs.npz');images={k:raw[k] for k in ('cam_high','cam_left_wrist','cam_right_wrist')};pr=(ex/'prompt.txt').read_text();noise=np.load(ex/'initial_noise_fp32.npy');model=mm.ImageWAM('weights/bf16','assets');c=EncoderCandidates(model)
 r={'status':'running','rows':[],'results':{},'baseline':'both_compile'}
 shutil.copyfile(__file__,a.output/'encoder_experiments.py')
 def persist():(a.output/'results.json').write_text(json.dumps(r,indent=2))
 def once(mode,prompt=pr,im=images,st=raw['state'],n=noise,profile=False):
  c.set(mode);mx.synchronize();t=time.perf_counter();out,norm=model.predict(im,st,prompt,n,profile=profile);mx.synchronize();ms=(time.perf_counter()-t)*1000;assert np.isfinite(out).all();return out,ms
 try:
  for _ in range(8):once('baseline')
  for mode in ('vae_contiguous','qwen_compile','encoders_compile','qwen_fast_rms'):
   print('START',mode,flush=True);_,first=once(mode)
   for _ in range(6):once(mode)
   for _ in range(6):once('baseline')
   timings={'baseline':[],mode:[]};outputs={}
   for i in range(a.iterations):
    for k in (('baseline',mode) if i%2==0 else (mode,'baseline')):
     x,ms=once(k);timings[k].append(ms);outputs[k]=x;r['rows'].append({'candidate':mode,'mode':k,'index':i,'ms':ms});np.save(a.output/f'{mode}_{k}_{i:03d}.npy',x)
   modules=[]
   for _ in range(5):once(mode,profile=True);modules.append(dict(model.profile))
   row={'first_ms':first,'times':{k:stats(v) for k,v in timings.items()},'error_vs_baseline':error(outputs[mode],outputs['baseline']),'modules':{k:float(np.mean([m[k] for m in modules])) for k in modules[0]},'dynamic':[]}
   inventory=json.loads(Path('../ImageWAM/development_records/m5pro_robotwin/20261103_dynamic_optimization/text_lengths.json').read_text())['examples_by_length']
   for length in (31,79,48,31):
    prompt=inventory[str(length)]['prompt'];ref,_=once('baseline',prompt);x,ms=once(mode,prompt);row['dynamic'].append({'tokens':length,'first_shape_ms':ms,'error':error(x,ref)})
   r['results'][mode]=row;persist();print('RESULT',mode,row,flush=True)
  c.set('baseline');r['status']='completed';persist()
 except BaseException:
  import traceback
  r['status']='failed';r['traceback']=traceback.format_exc();persist();raise
if __name__=='__main__':main()
