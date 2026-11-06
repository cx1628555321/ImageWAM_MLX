import os
os.environ['MLX_MAX_OPS_PER_BUFFER']='256'
import sys,time,json
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
import mlx.core as mx
import numpy as np
from imagewam_mlx.staged_visual import quantize
from imagewam_mlx.ops import norm
from candidate import q as modq
from candidate_norm import q as normq
mx.random.seed(513)
x=mx.random.normal((328,3072)).astype(mx.bfloat16);scale=mx.random.normal((3072,)).astype(mx.bfloat16);shift=mx.random.normal((3072,)).astype(mx.bfloat16)
mx.eval(x,scale,shift)
funcs={'baseline':mx.compile(lambda x,s,b:quantize((1+s)*norm(x)+b))}
for mode,kernel in [('mod',modq),('norm',normq)]:
 for threads in [128,256,512,1024]:
  def f(x,s,b,kernel=kernel,threads=threads,mode=mode):
   a=norm(x) if mode=='mod' else x;rows,k=a.shape;mp=(rows+63)//64*64
   return kernel(inputs=[a,s,b],template=[('T',a.dtype),('XK',k),('THREADS',threads)],grid=(threads,mp,1),threadgroup=(threads,1,1),output_shapes=[(mp,k),(rows,),(mp,k//64+1)],output_dtypes=[mx.int8,mx.float32,mx.uint8])
  funcs[f'{mode}{threads}']=mx.compile(f)
rows={k:[] for k in funcs};errors={};ref=None
for key,f in funcs.items():
 for _ in range(10):out=f(x,scale,shift);mx.eval(out)
 arr=[np.array(t) for t in out]
 if ref is None:ref=arr
 errors[key]=[dict(different=int(np.count_nonzero(a!=b)),max_abs=float(np.max(np.abs(a.astype(float)-b.astype(float))))) for a,b in zip(arr,ref)]
rng=np.random.default_rng(935)
for i in range(200):
 for key in rng.permutation(list(funcs)):
  mx.synchronize();t=time.perf_counter();mx.eval(funcs[key](x,scale,shift));mx.synchronize();rows[key].append((time.perf_counter()-t)*1000)
res={k:dict(mean_ms=float(np.mean(v)),median_ms=float(np.median(v)),errors=errors[k],samples_ms=v) for k,v in rows.items()}
Path('records/mod_quant_20261104/micro.json').write_text(json.dumps(res,indent=2))
print({k:{a:b for a,b in v.items() if a!='samples_ms'} for k,v in res.items()},flush=True)
