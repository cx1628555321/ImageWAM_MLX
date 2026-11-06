import os
os.environ['MLX_MAX_OPS_PER_BUFFER']='256'
import sys,time,json,functools
from pathlib import Path
sys.path.insert(0,str(Path.cwd()));sys.path.insert(0,str(Path('experiments/roadmap_20261105').resolve()))
from common import inputs
import numpy as np
import mlx.core as mx
from imagewam_mlx import ImageWAM
r=Path('records/audit4_20261105');m=ImageWAM('weights/bf16','assets',profile='speed',text_cache_size=0);args=inputs()
for _ in range(10):a,n=m.predict(**args)
events=[];request=-1
oldstep=m._compiled_workspace_step;oldasync=mx.async_eval;oldarray=np.array

def wrap(fn,name,condition=lambda a:True):
 @functools.wraps(fn)
 def call(*a,**kw):
  if not condition(a):return fn(*a,**kw)
  wall=time.perf_counter();cpu=time.thread_time();out=fn(*a,**kw);events.append(dict(request=request,kind=name,wall_ms=(time.perf_counter()-wall)*1000,caller_cpu_ms=(time.thread_time()-cpu)*1000));return out
 return call
m._compiled_workspace_step=wrap(oldstep,'compiled_step');mx.async_eval=wrap(oldasync,'async_eval');np.array=wrap(oldarray,'mlx_to_numpy',lambda a:bool(a) and isinstance(a[0],mx.array))
rows=[]
try:
 for request in range(30):
  wall=time.perf_counter();cpu=time.thread_time();out,z=m.predict(**args);mx.synchronize();rows.append(dict(wall_ms=(time.perf_counter()-wall)*1000,caller_cpu_ms=(time.thread_time()-cpu)*1000));assert np.array_equal(a,out) and np.array_equal(n,z)
finally:m._compiled_workspace_step=oldstep;mx.async_eval=oldasync;np.array=oldarray
(r/'boundary.json').write_text(json.dumps(dict(requests=rows,events=events,exact=True,note='wall time includes native evaluation and waiting; thread CPU is only Python calling thread, not total backend CPU'),indent=2));print('boundary complete',len(events),flush=True)
