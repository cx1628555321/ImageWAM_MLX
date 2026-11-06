import os
os.environ['MLX_MAX_OPS_PER_BUFFER']='256'
import sys,time,json,types,traceback
from pathlib import Path
sys.path.insert(0,str(Path.cwd()));sys.path.insert(0,str(Path('experiments/roadmap_20261105').resolve()))
import numpy as np
import mlx.core as mx
from common import inputs
from imagewam_mlx import ImageWAM
from imagewam_mlx.model import ImageWAM as Base
from imagewam_mlx.workspace import workspace_step
from candidates import install,action_int8
mode=sys.argv[1];r=Path(sys.argv[2]);r.mkdir(parents=True,exist_ok=False);refroot=Path(os.environ.get('BASE_RUN',str(r)))
def save(d): (r/'results.json').write_text(json.dumps(d,indent=2))
def error(a,b):
 diff=a.astype(np.float64)-b.astype(np.float64);return dict(max_abs=float(np.max(np.abs(diff))),mae=float(np.mean(np.abs(diff))),rmse=float(np.sqrt(np.mean(diff**2))),exact=bool(np.array_equal(a,b)))
report=dict(mode=mode,status='running');save(report)
try:
 t=time.perf_counter();m=ImageWAM('weights/bf16','assets',profile='speed',text_cache_size=1);mx.synchronize();report['base_load_s']=time.perf_counter()-t
 scales=None;raw=None;t=time.perf_counter()
 if mode.startswith('action_') or mode=='combined_w4':
  raw=mx.load(str(Path(m.weights_path)/'action.safetensors'));mx.eval(raw)
 if mode=='fixed_scale':
  # Calibrate on 3 observations/prompts, and use a safety margin; validation is separate.
  scales={};old=m.visual.linear
  def collect(self,x,p):
   if p in self.packed and p.endswith('.linear1'):
    value=mx.max(mx.abs(x.astype(mx.float32)));mx.eval(value);scales[p]=max(scales.get(p,0),float(value.item())/127*1.25)
   return old(x,p)
  m.visual.linear=types.MethodType(collect,m.visual);m.prefill=types.MethodType(Base.prefill,m)
  for idx in range(3):
   args=inputs()
   if idx:args['prompt']+=' Please move carefully.'*idx;args['images']['cam_high']=np.roll(args['images']['cam_high'],idx*7,axis=1)
   m.predict(**args)
  del m.visual.linear
  (r/'calibration.json').write_text(json.dumps(scales,indent=2))
 if mode=='action_int8':
  m.action=action_int8(m.action,raw);m._compiled_workspace_step=mx.compile(lambda c,x,t,d,p:workspace_step(m,c,x,t,d,p));inventory=[]
 else:inventory=install(m,mode,raw,scales)
 del raw;mx.synchronize();report['adapt_s']=time.perf_counter()-t;report['inventory']=inventory
 args=inputs();report['text_prepare']=m.prepare_text(args['prompt'])
 t=time.perf_counter();a,n=m.predict(**args);mx.synchronize();report['first_ms']=(time.perf_counter()-t)*1000
 if not np.isfinite(a).all():raise ValueError('Nonfinite first output')
 np.savez(r/'standard.npz',action=a,normalized=n)
 report['supplied_reference_error']=error(a,np.load('examples/robotwin_ep0_frame0/reference_action.npy'))
 if mode!='baseline':
  ref=np.load(refroot/'standard.npz');report['standard_error']=error(a,ref['action'])
 # Preserve full 50 measurements unless candidate is catastrophically slow.
 count=50 if report['first_ms']<30000 else 10;report['steady_count']=count
 for _ in range(10):m.predict(**args)
 times=[]
 for i in range(count):
  mx.synchronize();t=time.perf_counter();a,n=m.predict(**args);mx.synchronize();times.append((time.perf_counter()-t)*1000)
 report['steady']=dict(mean_ms=float(np.mean(times)),p95_ms=float(np.percentile(times,95)),samples_ms=times);save(report);print('PERF',mode,report['steady']['mean_ms'],flush=True)
 profiles=[]
 for i in range(count):m.predict(**args,profile=True);profiles.append(dict(m.profile))
 report['modules']={k:float(np.mean([v[k] for v in profiles])) for k in profiles[0]};report['profile_rows']=profiles
 source=Path('../ImageWAM/evaluate_results/m5pro_robotwin/20261103_104832_dynamic_full');cases=json.loads((source/'comparison.json').read_text())['cases'];validation=[]
 for i,c in enumerate(cases):
  rawcase=np.load(source/('base_inputs.npz' if c['label'].startswith('length_') else c['label']+'_inputs.npz'));args=dict(images={k:rawcase[k] for k in ('cam_high','cam_left_wrist','cam_right_wrist')},state=rawcase['state'],prompt=c['prompt'],noise=inputs()['noise'])
  a,n=m.predict(**args);assert np.isfinite(a).all();np.savez(r/f'case_{i:03d}.npz',action=a,normalized=n)
  if mode=='baseline':e=dict(exact=True,max_abs=0,rmse=0,mae=0)
  else:e=error(a,np.load(refroot/f'case_{i:03d}.npz')['action'])
  validation.append(dict(label=c['label'],**e))
  if i%10==0:print('CASE',mode,i,flush=True)
 report['validation']=validation
 if mode=='fixed_scale':
  clipped=[];original=m.visual.linear;compiled=m.prefill
  def audit(self,x,p):
   if p in self.scales:
    clipped.append(dict(layer=p,elements=x.size,clipped=int(mx.sum(mx.abs(x.astype(mx.float32))>127*self.scales[p]).item())))
   return original(x,p)
  m.visual.linear=types.MethodType(audit,m.visual);m.prefill=types.MethodType(Base.prefill,m)
  for j in range(3):
   sample=inputs();sample['prompt']+=' Please carefully follow the instruction.'*(j*10)
   if j:sample['images']['cam_high']=255-sample['images']['cam_high']
   m.predict(**sample)
  del m.visual.linear;m.prefill=compiled;report['clip_audit']=clipped
 dynamic=[]
 for i,(h,long) in enumerate([(1,False),(64,False),(16,True)]):
  args=inputs();args['noise']=np.resize(args['noise'],(1,h,14))
  if long:args['prompt']+=' Please carefully follow the instruction.'*25
  a,n=m.predict(**args,trace=True);assert np.isfinite(a).all();np.savez(r/f'dynamic_{i}.npz',action=a,normalized=n,**m.trace)
  dynamic.append(dict(h=h,long=long,error=error(a,np.load(refroot/f'dynamic_{i}.npz')['action']) if mode!='baseline' else {}))
 report['dynamic']=dynamic;report['cache']=m.text_cache_stats();report['peak_bytes']=mx.get_peak_memory();report['status']='completed';save(report);print('DONE',mode,flush=True)
except Exception as e:
 report['status']='failed';report['error']=repr(e);report['traceback']=traceback.format_exc();save(report);print(report['traceback'],flush=True);sys.exit(1)
