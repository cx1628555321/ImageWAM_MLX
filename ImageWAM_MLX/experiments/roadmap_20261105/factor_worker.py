import os
os.environ['MLX_MAX_OPS_PER_BUFFER']='256'
import sys,time,json,resource,hashlib
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
from common import *
mode,label=sys.argv[1:];r=ROOT/'factorial';r.mkdir(exist_ok=True);assert not (r/(label+'.json')).exists()
t=time.perf_counter();m=ImageWAM('weights/bf16','assets',profile='speed',text_cache_size=0,visual_shape_dispatch=mode.startswith('shape'));
if mode.endswith('lambda'):m.prefill=mx.compile(lambda h,l,s:Base.prefill(m,h,l,s))
mx.synchronize();load=(time.perf_counter()-t)*1000;args=inputs()
def run():
 mx.synchronize();t=time.perf_counter();a,n=m.predict(**args,steps=10);mx.synchronize();return a,n,(time.perf_counter()-t)*1000
a,n,first=run()
for _ in range(30):a,n,_=run()
rows=[];outs=[];cpu=time.process_time();usage=resource.getrusage(resource.RUSAGE_SELF)
for _ in range(100):
 a,n,t=run();rows.append(t);outs.append(a.copy());assert np.isfinite(a).all()
cpu=(time.process_time()-cpu)*10;fault=resource.getrusage(resource.RUSAGE_SELF).ru_majflt-usage.ru_majflt
np.savez(r/(label+'.npz'),actions=np.stack(outs),normalized=n)
profiles=[]
for _ in range(3):m.predict(**args,steps=10,profile=True);profiles.append(dict(m.profile))
res=dict(mode=mode,label=label,load_ms=load,first_ms=first,mean_ms=float(np.mean(rows)),p95_ms=float(np.percentile(rows,95)),samples_ms=rows,process_cpu_ms=cpu,hard_fault_delta=fault,profiles=profiles,peak_bytes=mx.get_peak_memory())
(r/(label+'.json')).write_text(json.dumps(res,indent=2));print(label,res['mean_ms'],flush=True)
