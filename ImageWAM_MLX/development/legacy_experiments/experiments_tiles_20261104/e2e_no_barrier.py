import sys,json,time,argparse,traceback
from pathlib import Path
import numpy as np
import mlx.core as mx
sys.path.insert(0,str(Path.cwd()))
sys.path.insert(0,str(Path.cwd()/'development/legacy_experiments/experiments_lowbit_20261104'))
from imagewam_mlx import ImageWAM
from adapters import install
p=argparse.ArgumentParser();p.add_argument('--case',required=True);p.add_argument('--iterations',type=int,default=12);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
configs={'tile32k128':dict(components=['visual'],fused=True),'tile32k512':dict(components=['visual'],fused=True),'tile64n64':dict(components=['visual'],fused=True),'accuracy':{},'current_speed':{},'speed_native_visual':dict(components=['visual'],fused=True),'fast_vae':{},'native_action':dict(components=['action']),'fused_action':dict(components=['action'],fused=True),'native_visual':dict(components=['visual']),'fused_visual':dict(components=['visual'],fused=True),'native_all':dict(components=['visual','action','qwen']),'fused_all':dict(components=['visual','action','qwen'],fused=True),'int8_kv':dict(kv=True),'int8_conv':dict(conv=True),'all_lowbit':dict(components=['visual','action','qwen'],fused=True,kv=True,conv=True)}
r=dict(case=a.case,status='running',config=configs[a.case],iterations=[])
def save():(a.output/'results.json').write_text(json.dumps(r,indent=2))
save()
try:
 t=time.perf_counter()
 if a.case=='accuracy':m=ImageWAM('weights/bf16','assets')
 elif a.case in ('current_speed','speed_native_visual','tile32k128','tile32k512','tile64n64'):
  m=ImageWAM('weights/bf16','assets',profile='speed')
  if a.case!='current_speed':
   if a.case.startswith('tile'):
    import native_int8,adapters
    from tune_no_barrier import gemm
    tile={'tile32k128':(32,128,128),'tile32k512':(32,128,512),'tile64n64':(64,64,128)}[a.case]
    def patched(q,w,sx,sw,bias):return gemm(q,w,sx,sw,bias,*tile)
    native_int8.int8_gemm=patched;adapters.int8_gemm=patched
   install(m,**configs[a.case])
 else:m=ImageWAM('weights/bf16','assets',profile='speed',quantization={},speed_options={'vae':'fast_bf16'});install(m,**configs[a.case])
 mx.synchronize();r['load_s']=time.perf_counter()-t;r['active_bytes']=mx.get_active_memory();mx.clear_cache()
 ex=Path('examples/robotwin_ep0_frame0');raw=np.load(ex/'inputs.npz');ims={k:raw[k] for k in ('cam_high','cam_left_wrist','cam_right_wrist')};pr=(ex/'prompt.txt').read_text();noise=np.load(ex/'initial_noise_fp32.npy')
 def call(pr=pr,noise=noise,steps=10,profile=False):
  mx.synchronize();t=time.perf_counter();act,n=m.predict(ims,raw['state'],pr,noise,steps=steps,profile=profile);mx.synchronize();ms=(time.perf_counter()-t)*1000
  assert act.shape==(noise.shape[1],14) and np.isfinite(act).all() and np.isfinite(n).all();return act,n,ms
 act,n,r['first_ms']=call();save()
 for _ in range(3):call()
 for _ in range(a.iterations):
  act,n,ms=call();r['iterations'].append(ms)
 np.savez(a.output/'output.npz',action=act,normalized=n)
 ref=np.load('records/final_default_20261103/reference_action.npy');d=act-ref;r['difference']=dict(max_abs=float(np.max(np.abs(d))),rmse=float(np.sqrt(np.mean(d*d))))
 r['mean_ms']=float(np.mean(r['iterations']));r['p95_ms']=float(np.percentile(r['iterations'],95));r['modules']=[]
 for _ in range(3):call(profile=True);r['modules'].append(dict(m.profile))
 r['dynamic']=[]
 for i,(prompt,h,steps) in enumerate([(pr,8,3),('Move carefully. '*60,16,10),(pr,32,3)]):
  z=np.resize(noise,(1,h,14));act,n,ms=call(prompt,z,steps);np.savez(a.output/f'dynamic_{i}.npz',action=act,normalized=n,noise=z);r['dynamic'].append(dict(tokens=int(m.tokenize(prompt).shape[1]),horizon=h,steps=steps,finite=True))
 r['status']='completed';save();print({k:v for k,v in r.items() if k not in ('iterations','modules')},flush=True)
except Exception:r.update(status='failed',error=traceback.format_exc());save();print(r['error'],flush=True);raise
