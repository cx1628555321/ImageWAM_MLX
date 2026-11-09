"""Opt-in action-step / visual-prefix compilation. No persistent feature caches.
The compiled step receives noise/latents, time, delta, position encoding and all
prefix K/V as tensor inputs. Shape-specific compilation never pads/truncates text.
"""
import inspect,textwrap,types,time,json,argparse,sys,shutil
from pathlib import Path
import numpy as np
import mlx.core as mx
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import imagewam_mlx.model as mm
from imagewam_mlx.ops import D,flux_rope

class Candidates:
 def __init__(self,model):
  self.model=model;self.original_denoise=model.denoise;self.original_prefill=model.prefill
  src=textwrap.dedent(inspect.getsource(mm.ImageWAM.denoise));namespace=dict(vars(mm))
  assert src.count('mx.eval(x)')==1
  exec(src.replace('mx.eval(x)','mx.async_eval(x)'),namespace)
  self.async_denoise=types.MethodType(namespace['denoise'],model)
  # Extract original per-step arithmetic without its scheduling/debug side effects.
  start=src.index('        t=(times[step:step+1]/1000).astype(D); vec=a.embed_time(t)')
  end=src.index('        # Bound the lazy graph;')
  body=textwrap.dedent(src[start:end])
  body=body.replace('t=(times[step:step+1]/1000).astype(D); vec=a.embed_time(t)','vec=a.embed_time(t)').replace('deltas[step]','delta')
  namespace=dict(vars(mm));namespace['a']=model.action
  exec('def step_fn(cache,x,t,delta,pe):\n'+textwrap.indent(body,'    ')+'    return x\n',namespace)
  self.compiled_step=mx.compile(namespace['step_fn'])
  self.compiled_prefill=mx.compile(self.original_prefill)
 def denoise_compiled(self,cache,noise,steps=10,trace=False):
  if noise.ndim!=3 or noise.shape[0]!=1 or noise.shape[2]!=14 or not 1<=noise.shape[1]<=64 or steps<1:raise ValueError('invalid noise/steps')
  x=mx.array(noise).astype(D);u=mx.linspace(1,0,steps+1,dtype=mx.float32);sig=5*u/(1+4*u)
  times=(sig[:-1]*1000).astype(D);deltas=(sig[1:]-sig[:-1]).astype(D)
  ids=np.zeros((1,x.shape[1],4),np.float32);ids[...,0]=2;ids[0,:,1]=np.arange(x.shape[1]);pe=flux_rope(mx.array(ids).astype(D))
  for i in range(steps):
   t=(times[i:i+1]/1000).astype(D);x=self.compiled_step(cache,x,t,deltas[i],pe);mx.async_eval(x)
   if trace:self.model.trace[f'action_step_{i}']=np.array(x.astype(mx.float32))
  return x
 def set(self,mode):
  self.model.denoise=self.original_denoise if mode=='original' else (self.denoise_compiled if mode in ('action_compile','both_compile') else self.async_denoise)
  self.model.prefill=self.compiled_prefill if mode in ('visual_compile','both_compile') else self.original_prefill

def stats(values):
 x=np.array(values);return {'count':len(x),'mean':float(x.mean()),'median':float(np.median(x)),'p95':float(np.percentile(x,95)),'min':float(x.min()),'max':float(x.max())}
def error(x,y):
 d=x.astype(np.float64)-y.astype(np.float64);return {'max_abs':float(abs(d).max()),'mae':float(abs(d).mean()),'rmse':float(np.sqrt(np.mean(d*d)))}

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--iterations',type=int,default=30);args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
 root=Path.cwd();example=root/'examples/robotwin_ep0_frame0';raw=np.load(example/'inputs.npz');images={k:raw[k] for k in ('cam_high','cam_left_wrist','cam_right_wrist')};state=raw['state'];prompt=(example/'prompt.txt').read_text();noise=np.load(example/'initial_noise_fp32.npy')
 model=mm.ImageWAM('weights/bf16','assets');c=Candidates(model);rows=[];report={'status':'running','results':{},'rows':rows,'method':'alternating original async baseline vs candidate, unchanged raw observation and text; no quantization or token padding'}
 shutil.copyfile(__file__,args.output/'compile_experiments.py');shutil.copytree(root/'imagewam_mlx',args.output/'source_snapshot',ignore=shutil.ignore_patterns('__pycache__'))
 def persist():(args.output/'results.json').write_text(json.dumps(report,indent=2))
 def once(mode,pr=prompt,im=images,st=state,n=noise,profile=False):
  c.set(mode);mx.synchronize();start=time.perf_counter();x,y=model.predict(im,st,pr,n,profile=profile);mx.synchronize();ms=(time.perf_counter()-start)*1000
  assert np.isfinite(x).all();return x,y,ms
 try:
  for _ in range(8):once('async')
  for mode in ('action_compile','visual_compile','both_compile'):
   print('START',mode,flush=True);x,_,first=once(mode)
   for _ in range(6):once(mode)
   for _ in range(6):once('async')
   times={'async':[],mode:[]};out={}
   for i in range(args.iterations):
    for k in (('async',mode) if i%2==0 else (mode,'async')):
     x,y,ms=once(k);times[k].append(ms);out[k]=x;rows.append({'candidate':mode,'mode':k,'index':i,'ms':ms})
     np.save(args.output/f'{mode}_{k}_{i:03d}.npy',x)
   mods=[]
   for _ in range(5):once(mode,profile=True);mods.append(dict(model.profile))
   report['results'][mode]={'first_ms':first,'times':{k:stats(v) for k,v in times.items()},'error_vs_async':error(out[mode],out['async']),'error_vs_mps':error(out[mode],np.load(example/'reference_action.npy')),'module_means':{k:float(np.mean([r[k] for r in mods])) for k in mods[0]},'dynamic':[]}
   inventory=json.loads((root/'../ImageWAM/development_records/m5pro_robotwin/20261103_dynamic_optimization/text_lengths.json').read_text())['examples_by_length']
   for length in (31,79,48,31):
    pr=inventory[str(length)]['prompt'];ref,_,_=once('async',pr);x,_,ms=once(mode,pr)
    report['results'][mode]['dynamic'].append({'tokens':length,'shape_first_ms':ms,'error':error(x,ref)})
   # Detect accidental capture of noise/state/text content by the compiler.
   changed=noise.copy();changed[:]=np.flip(changed,axis=1)
   ref,_,_=once('async',n=changed,st=state+.1);x,_,_=once(mode,n=changed,st=state+.1)
   report['results'][mode]['changed_noise_and_state']=error(x,ref)
   persist();print('RESULT',mode,report['results'][mode],flush=True)
  c.set('original');report['status']='completed';report['memory']={'active':mx.get_active_memory(),'peak':mx.get_peak_memory(),'cache':mx.get_cache_memory()};persist()
 except BaseException:
  import traceback
  report['status']='failed';report['traceback']=traceback.format_exc();persist();raise
if __name__=='__main__':main()
