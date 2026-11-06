import os,sys,time,json,types
from pathlib import Path
os.environ['MLX_MAX_OPS_PER_BUFFER']='256'
root=Path.cwd();sys.path.insert(0,str(root));sys.path.insert(0,str(root/'experiments/background_20261105'));sys.path.insert(0,str(root/'experiments/roadmap_20261105'));sys.path.insert(0,str(root/'experiments/prefix_tiles_20261106'))
import mlx.core as mx
import numpy as np
import imagewam_mlx.runtime as rt
source=Path(rt.__file__).read_text();assert "version('mlx')=='0.32.4'" in source
exec(compile(source.replace("version('mlx')=='0.32.4'",'True'),rt.__file__,'exec'),rt.__dict__)
from common import inputs
from imagewam_mlx.model import ImageWAM as Base
from candidates import install
from candidate import make_kernel,wide,shape
r=root/'records/prefix_tiles_20261106/diagnostic';r.mkdir(parents=True,exist_ok=False)
m=rt.ImageWAM('weights/bf16','assets',profile='speed',prefix_precision='all_int8',text_cache_size=1);install(m,'native_w1')
old=m.visual.mm;captured={};counts={}
def capture(**kw):
 key=shape(kw);counts[key]=counts.get(key,0)+1
 if key not in captured:captured[key]=kw
 return old(**kw)
m.visual.mm=capture;m.prefill=types.MethodType(Base.prefill,m)
sample=inputs();m.prepare_text(sample['prompt']);m.predict(**sample);mx.synchronize()
m.visual.mm=old
kernels={wn:make_kernel(wn) for wn in (2,4)};rows=[];selection=[]
def bench(fn):
 for _ in range(5):mx.eval(fn())
 mx.synchronize();times=[]
 for _ in range(20):
  t=time.perf_counter();mx.eval(fn());mx.synchronize();times.append((time.perf_counter()-t)*1000)
 return float(np.median(times))
for key,kw in captured.items():
 mx.eval(kw['inputs']);reference=old(**kw)[0];mx.eval(reference)
 baseline=bench(lambda:old(**kw));entry=dict(shape=key,count=counts[key],baseline_ms=baseline,candidates=[])
 if key[1]%128==0:
  for wn,kernel in kernels.items():
   try:
    out=wide(kernel,wn,kw)[0];mx.eval(out)
    exact=bool(mx.array_equal(out,reference).item());delta=float(mx.max(mx.abs(out.astype(mx.float32)-reference.astype(mx.float32))).item())
    elapsed=bench(lambda:wide(kernel,wn,kw));entry['candidates'].append(dict(wn=wn,ms=elapsed,exact=exact,max_abs=delta))
   except Exception as exc:entry['candidates'].append(dict(wn=wn,error=str(exc)))
 good=[x for x in entry['candidates'] if x.get('exact') and x['ms']<baseline*.95]
 if good:
  best=min(good,key=lambda x:x['ms']);selection.append(dict(shape=key,wn=best['wn'],micro_saved_ms=baseline-best['ms']))
 rows.append(entry);(r/'results.json').write_text(json.dumps(rows,indent=2));print(json.dumps(entry),flush=True)
(r/'selection.json').write_text(json.dumps(selection,indent=2));print('selected',selection,flush=True)
