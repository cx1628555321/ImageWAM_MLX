"""Cache only fixed-schedule time modulation; frozen-model inference only."""
import argparse,inspect,textwrap,types,json,time,sys,shutil
from pathlib import Path
from collections import OrderedDict
import numpy as np
import mlx.core as mx
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from compile_experiments import stats,error
from graph_experiments import GraphCandidates
import imagewam_mlx.model as mm
from imagewam_mlx.ops import D,flux_rope,silu

class ConditioningCandidates:
 def __init__(self,model):
  self.model=model;self.base=GraphCandidates(model);self.base.set('lazy_prefix');self.schedules=OrderedDict()
  src=textwrap.dedent(inspect.getsource(mm.ImageWAM.denoise));start=src.index("        t=(times[step:step+1]/1000).astype(D); vec=a.embed_time(t)");end=src.index('        # Bound the lazy graph;');body=textwrap.dedent(src[start:end])
  lines=body.splitlines();lines=[line for line in lines if not line.startswith(('t=(times[step:step+1]','md=a.modulation','shift,scale=mx.split'))];body='\n'.join(lines).replace('deltas[step]','delta')+'\n'
  ns=dict(vars(mm));ns['a']=model.action
  exec('def step(cache,x,md,ms,shift,scale,delta,pe):\n'+textwrap.indent(body,'    ')+'    return x\n',ns);self.step=mx.compile(ns['step'])
  a=model.action
  def condition(t):
   vec=a.embed_time(t);md=a.modulation(vec,'double_stream_modulation_img',True);ms=a.modulation(vec,'single_stream_modulation');shift,scale=mx.split(a.linear(silu(vec),'head.adaLN_modulation.1'),2,axis=-1)
   return md,ms,shift,scale
  self.condition=mx.compile(condition)
 def schedule(self,steps):
  if steps in self.schedules:self.schedules.move_to_end(steps);return self.schedules[steps]
  u=mx.linspace(1,0,steps+1,dtype=mx.float32);sig=5*u/(1+4*u);times=(sig[:-1]*1000).astype(D);deltas=(sig[1:]-sig[:-1]).astype(D)
  values=[self.condition((times[i:i+1]/1000).astype(D)) for i in range(steps)];mx.eval(values,deltas)
  self.schedules[steps]=(values,deltas)
  if len(self.schedules)>4:self.schedules.popitem(last=False)
  return values,deltas
 def denoise(self,cache,noise,steps=10,trace=False):
  if noise.ndim!=3 or noise.shape[0]!=1 or noise.shape[2]!=14 or not 1<=noise.shape[1]<=64 or steps<1:raise ValueError('invalid noise/steps')
  values,deltas=self.schedule(steps);x=mx.array(noise).astype(D)
  ids=np.zeros((1,x.shape[1],4),np.float32);ids[...,0]=2;ids[0,:,1]=np.arange(x.shape[1]);pe=flux_rope(mx.array(ids).astype(D))
  for i in range(steps):
   md,ms,shift,scale=values[i];x=self.step(cache,x,md,ms,shift,scale,deltas[i],pe);mx.async_eval(x)
   if trace:self.model.trace[f'action_step_{i}']=np.array(x.astype(mx.float32))
  return x
 def set(self,mode):
  self.base.set('lazy_prefix')
  if mode=='conditioning':self.model.denoise=self.denoise

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--iterations',type=int,default=30);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 shutil.copyfile(__file__,a.output/'conditioning_experiments.py')
 model=mm.ImageWAM('weights/bf16','assets');c=ConditioningCandidates(model)
 ex=Path('examples/robotwin_ep0_frame0');raw=np.load(ex/'inputs.npz');images={k:raw[k] for k in ('cam_high','cam_left_wrist','cam_right_wrist')};prompt=(ex/'prompt.txt').read_text();noise=np.load(ex/'initial_noise_fp32.npy')
 r={'status':'running','rows':[],'results':{}}
 def persist():(a.output/'results.json').write_text(json.dumps(r,indent=2))
 def once(mode,pr=prompt,n=noise,steps=10):
  c.set(mode);mx.synchronize();t=time.perf_counter();out,_=model.predict(images,raw['state'],pr,n,steps=steps);mx.synchronize();assert np.isfinite(out).all();return out,(time.perf_counter()-t)*1000
 try:
  for _ in range(8):once('baseline')
  for mode in ('conditioning',):
   print('START',mode,flush=True);_,first=once(mode)
   for _ in range(6):once(mode)
   times={'baseline':[],mode:[]};outputs={}
   for i in range(a.iterations):
    for k in (('baseline',mode) if i%2==0 else (mode,'baseline')):
     x,ms=once(k);outputs[k]=x;times[k].append(ms);r['rows'].append({'candidate':mode,'mode':k,'index':i,'ms':ms});np.save(a.output/f'{mode}_{k}_{i:03d}.npy',x)
   row={'first_ms':first,'times':{k:stats(v) for k,v in times.items()},'error':error(outputs[mode],outputs['baseline']),'dynamic':[]}
   inv=json.loads(Path('../ImageWAM/development_records/m5pro_robotwin/20261103_dynamic_optimization/text_lengths.json').read_text())['examples_by_length']
   for length in (31,79,48,31):
    pr=inv[str(length)]['prompt'];ref,_=once('baseline',pr);x,ms=once(mode,pr);row['dynamic'].append({'tokens':length,'error':error(x,ref)})
   for horizon,steps in ((8,3),(16,7),(32,10)):
    n=np.random.default_rng(7).standard_normal((1,horizon,14)).astype(np.float32);ref,_=once('baseline',n=n,steps=steps);x,_=once(mode,n=n,steps=steps);row['dynamic'].append({'horizon':horizon,'steps':steps,'error':error(x,ref)})
   r['results'][mode]=row;persist();print('RESULT',mode,row,flush=True)
  r['status']='completed';persist()
 except BaseException:
  import traceback
  r['status']='failed';r['traceback']=traceback.format_exc();persist();raise
if __name__=='__main__':main()
