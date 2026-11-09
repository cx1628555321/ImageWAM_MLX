"""Isolated detailed profiling and reversible optimization experiments.
All steady rounds execute online encoders. Diagnostic operator passes synchronize
inputs before timing, and are not used as end-to-end performance measurements.
"""
import argparse,collections,contextlib,hashlib,inspect,json,sys,time,types,shutil,traceback
from pathlib import Path
import numpy as np
import mlx.core as mx
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import imagewam_mlx.model as mm
import imagewam_mlx.ops as ops
import imagewam_mlx.qwen as qm
import imagewam_mlx.vae as vm
from imagewam_mlx.model import ImageWAM
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--rounds',type=int,default=50);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
report={'status':'running','load':[],'runs':[],'method':'steady: sync at E2E boundaries, online inputs; operator diagnostics: sync inputs and outputs, intrusive wall-time attribution, not GPU kernel trace'}
def persist(): (a.output/'analysis.json').write_text(json.dumps(report,indent=2))
def stats(x):
 x=np.array(x);return dict(count=len(x),mean=float(x.mean()),median=float(np.median(x)),p90=float(np.percentile(x,90)),p95=float(np.percentile(x,95)),p99=float(np.percentile(x,99)),std=float(x.std()),min=float(x.min()),max=float(x.max()),cv=float(x.std()/x.mean()))
def error(x,y):
 d=x.astype(np.float64)-y.astype(np.float64);return dict(max_abs=float(abs(d).max()),mae=float(abs(d).mean()),rmse=float(np.sqrt(np.mean(d*d))))
root=Path.cwd();example=root/'examples/robotwin_ep0_frame0';raw=np.load(example/'inputs.npz');images={k:raw[k] for k in ('cam_high','cam_left_wrist','cam_right_wrist')};state=raw['state'];prompt=(example/'prompt.txt').read_text();noise=np.load(example/'initial_noise_fp32.npy')
shutil.copytree(root/'imagewam_mlx',a.output/'source_snapshot/imagewam_mlx',ignore=shutil.ignore_patterns('__pycache__'))
shutil.copyfile(__file__,a.output/'source_snapshot/detailed_analysis.py')
old_load=mx.load
# Timed reads include array evaluation; OS page cache is not flushed.
def load(path,*args,**kwargs):
 t=time.perf_counter();w=old_load(path,*args,**kwargs);mx.eval(w);mx.synchronize();report['load'].append({'file':Path(path).name,'bytes':Path(path).stat().st_size,'ms':(time.perf_counter()-t)*1000});return w
mx.load=load
t=time.perf_counter();model=ImageWAM('weights/bf16','assets');mx.synchronize();report['load_total_ms']=(time.perf_counter()-t)*1000;mx.load=old_load
report['load_non_file_ms']=report['load_total_ms']-sum(x['ms'] for x in report['load'])
print('LOAD',report['load_total_ms'],flush=True);persist()
def once(label,prompt=prompt,im=images,st=state,profile=False):
 mx.synchronize();t=time.perf_counter();out,n=model.predict(im,st,prompt,noise,profile=profile);mx.synchronize();ms=(time.perf_counter()-t)*1000
 assert np.isfinite(out).all();report['runs'].append({'label':label,'ms':ms,'modules':dict(model.profile)});return out,n,ms
try:
 first,_,ms=once('first');report['first_ms']=ms
 for _ in range(8):once('warmup')
 blocks=[]
 for b in range(3):
  times=[]
  for i in range(a.rounds):
   out,n,ms=once(f'steady_{b}');times.append(ms)
   np.save(a.output/f'action_steady_{b}_{i:03d}.npy',out)
  blocks.append(stats(times));print('STEADY',b,blocks[-1],flush=True);persist()
 report['steady_blocks']=blocks;report['steady_all']=stats([r['ms'] for r in report['runs'] if r['label'].startswith('steady_')])
 report['repeatability_max_abs']=max(float(np.max(np.abs(np.load(f)-first))) for f in a.output.glob('action_steady*.npy'))
 for i in range(10):once('modules',profile=True)
 report['modules']={k:stats([r['modules'][k] for r in report['runs'] if r['label']=='modules']) for k in model.profile}
 # Input length sweeps, independently warmed; no change to the text policy.
 inventory=json.loads((root/'../ImageWAM/development_records/m5pro_robotwin/20261103_dynamic_optimization/text_lengths.json').read_text())['examples_by_length']
 report['length_sweep']={}
 for length in (31,48,64,79):
  pr=inventory[str(length)]['prompt']
  for _ in range(6):once(f'length_{length}_warm',pr)
  times=[once(f'length_{length}',pr)[2] for _ in range(20)];report['length_sweep'][str(length)]=stats(times)
  print('LENGTH',length,report['length_sweep'][str(length)]['mean'],flush=True)
 # Fine operation attribution. Parent/module timing was measured separately.
 counters=collections.defaultdict(list);current=['other'];patches=[]
 def arrays(obj):
  if isinstance(obj,mx.array):return [obj]
  if isinstance(obj,(tuple,list)):return [y for x in obj for y in arrays(x)]
  if isinstance(obj,dict):return [y for x in obj.values() for y in arrays(x)]
  return []
 def patch(obj,name,fn):patches.append((obj,name,getattr(obj,name)));setattr(obj,name,fn)
 def wrap_role(fn,role):
  def wrapped(*args,**kwargs):
   old=current[0];current[0]=role
   try:return fn(*args,**kwargs)
   finally:current[0]=old
  return wrapped
 patch(qm.Qwen,'__call__',wrap_role(qm.Qwen.__call__,'text'))
 patch(vm.VAE,'__call__',wrap_role(vm.VAE.__call__,'vae'))
 patch(model,'prefill',wrap_role(model.prefill,'visual'))
 patch(model,'denoise',wrap_role(model.denoise,'action'))
 def timed(fn,label):
  def wrapped(*args,**kwargs):
   mx.eval(arrays(args),arrays(kwargs));mx.synchronize();t=time.perf_counter();y=fn(*args,**kwargs);mx.eval(y);mx.synchronize()
   counters[label(args)].append((time.perf_counter()-t)*1000);return y
  return wrapped
 patch(ops.Weights,'linear',timed(ops.Weights.linear,lambda ar:current[0]+'.linear.'+ar[2]))
 patch(vm.VAE,'conv',timed(vm.VAE.conv,lambda ar:'vae.conv.'+ar[2]))
 patch(vm.VAE,'gn',timed(vm.VAE.gn,lambda ar:'vae.gn.'+ar[2]))
 for module in (mm,qm,vm):patch(module,'attention',timed(module.attention,lambda ar:current[0]+'.attention'))
 for i in range(3):once('operator_profile')
 for obj,name,fn in reversed(patches):setattr(obj,name,fn)
 report['operators']={k:{'calls_per_inference':len(v)/3,'mean_ms_per_inference':sum(v)/3,'call_ms':stats(v)} for k,v in counters.items()}
 print('OPERATORS_DONE',flush=True);persist()
 # Reversible candidates; do not change the production model files/defaults.
 old_gn=model.vae.gn;old_denoise=model.denoise
 def fast_gn(x,p):
  b,h,w,c=x.shape;z=x.astype(mx.float32).reshape(b,h*w,32,c//32).transpose(0,2,1,3).reshape(b,32,-1)
  z=mx.fast.layer_norm(z,None,None,eps=1e-6).reshape(b,32,h*w,c//32).transpose(0,2,1,3).reshape(x.shape)
  return (z*model.vae.w[p+'.weight'].astype(mx.float32)+model.vae.w[p+'.bias'].astype(mx.float32)).astype(x.dtype)
 def gn_kernel(x,weight,bias):
  b,h,w,c=x.shape;z=x.astype(mx.float32).reshape(b,h,w,32,c//32);mean=mx.mean(z,axis=(1,2,4),keepdims=True);var=mx.mean((z-mean)**2,axis=(1,2,4),keepdims=True)
  z=((z-mean)*mx.rsqrt(var+1e-6)).reshape(x.shape)
  return (z*weight.astype(mx.float32)+bias.astype(mx.float32)).astype(x.dtype)
 compiled_gn=mx.compile(gn_kernel)
 def compile_gn(x,p):return compiled_gn(x,model.vae.w[p+'.weight'],model.vae.w[p+'.bias'])
 # Same function body and input-dependent shapes, only scheduling differs.
 import textwrap
 src=textwrap.dedent(inspect.getsource(mm.ImageWAM.denoise)).replace('mx.eval(x)','mx.async_eval(x)')
 namespace=dict(vars(mm));exec(src,namespace);async_fn=types.MethodType(namespace['denoise'],model)
 def setmode(mode):
  model.vae.gn={'baseline':old_gn,'gn_fast':fast_gn,'gn_compile':compile_gn,'async':old_gn,'combined':compile_gn}[mode]
  model.denoise=async_fn if mode in ('async','combined') else old_denoise
 report['experiments']={}
 for candidate in ('gn_fast','gn_compile','async','combined'):
  setmode(candidate);t=time.perf_counter();cand,_,_=once(candidate+'_first');first_cost=(time.perf_counter()-t)*1000
  for _ in range(6):once(candidate+'_warm')
  setmode('baseline')
  for _ in range(6):once('comparison_warm')
  times={'baseline':[],candidate:[]};outputs={}
  for i in range(30):
   for mode in (('baseline',candidate) if i%2==0 else (candidate,'baseline')):
    setmode(mode);out,n,ms=once('experiment_'+candidate+'_'+mode);times[mode].append(ms);outputs[mode]=out
  np.save(a.output/(candidate+'_action.npy'),outputs[candidate])
  # Verify changing shapes and changed content; compare to unmodified MLX each time.
  cases=[]
  for length in (31,79,48,31):
   pr=inventory[str(length)]['prompt'];setmode('baseline');ref,_,_=once('candidate_reference',pr);setmode(candidate);out,_,_=once('candidate_dynamic',pr);cases.append({'tokens':length,'error_vs_mlx':error(out,ref)})
  report['experiments'][candidate]={'first_candidate_ms':first_cost,'timings':{k:stats(v) for k,v in times.items()},'error_vs_mlx':error(outputs[candidate],outputs['baseline']),'error_vs_mps':error(outputs[candidate],np.load(example/'reference_action.npy')),'dynamic_cases':cases}
  print('EXPERIMENT',candidate,report['experiments'][candidate],flush=True);persist()
 setmode('baseline')
 report['memory']={'active':mx.get_active_memory(),'peak':mx.get_peak_memory(),'cache':mx.get_cache_memory()};report['status']='completed';persist()
except BaseException:
 report['status']='failed';report['error']=traceback.format_exc();persist();raise
