import os
os.environ.setdefault('MLX_MAX_OPS_PER_BUFFER','256')
import sys,time,json,resource,subprocess,hashlib
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
import numpy as np
import mlx.core as mx
from imagewam_mlx import ImageWAM
from common import install,predict
mode,label=sys.argv[1:];r=Path('records/remaining_20261105/independent');r.mkdir(exist_ok=True)
assert not (r/(label+'.json')).exists()
t=time.perf_counter();base=ImageWAM('weights/bf16','assets',profile='speed');m,steps=install(base,mode);mx.synchronize();load=(time.perf_counter()-t)*1000
p=Path('examples/robotwin_ep0_frame0');raw=np.load(p/'inputs.npz');args=dict(images={k:raw[k] for k in ('cam_high','cam_left_wrist','cam_right_wrist')},state=raw['state'],prompt=(p/'prompt.txt').read_text(),noise=np.load(p/'initial_noise_fp32.npy'))
def run():
 mx.synchronize();t=time.perf_counter();a,n=predict(m,mode,steps,args);mx.synchronize();return a,n,(time.perf_counter()-t)*1000
a,n,first=run()
for _ in range(8):a,n,_=run()
Path('records/remaining_20261105/'+label+'.ready.json').write_text(json.dumps(dict(pid=os.getpid())))
rows=[];cpu=time.process_time();usage=resource.getrusage(resource.RUSAGE_SELF)
for _ in range(60):
 a,n,dt=run();rows.append(dt)
 if not np.isfinite(a).all():raise ValueError('nonfinite action')
cpu=(time.process_time()-cpu)*1000/60;fault=resource.getrusage(resource.RUSAGE_SELF).ru_majflt-usage.ru_majflt
np.savez(r/(label+'.npz'),action=a,normalized=n)
profiles=[]
for _ in range(3):
 if mode=='cache_miss':m.qwen.clear()
 m.predict(**args,steps=steps,profile=True);profiles.append(dict(m.profile))
res=dict(diagnostic_only=True,mode=mode,steps=steps,load_ms=load,first_ms=first,mean_ms=float(np.mean(rows)),p95_ms=float(np.percentile(rows,95)),samples_ms=rows,process_cpu_ms=cpu,hard_fault_delta=fault,profiles=profiles,active_bytes=mx.get_active_memory(),peak_bytes=mx.get_peak_memory(),cache_stats={k:getattr(m.qwen,k) for k in ('hits','misses','bytes')} if mode.startswith('cache') else None)
(r/(label+'.json')).write_text(json.dumps(res,indent=2));print(label,res['mean_ms'],flush=True)
