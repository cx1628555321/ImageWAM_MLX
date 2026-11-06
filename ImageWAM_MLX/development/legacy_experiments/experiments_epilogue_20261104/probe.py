import sys,json,time
from pathlib import Path
import mlx.core as mx
import numpy as np
from fused import fused
mx.random.seed(4);root=Path('records/epilogue_20261104');root.mkdir(exist_ok=True)
rows=[]
for m in (1,8,16,32,64):
 k,n=6144,3072
 x=mx.random.normal((1,m,k)).astype(mx.bfloat16);w=mx.random.normal((n,k)).astype(mx.bfloat16);parts=mx.quantize(w,bits=1,group_size=64);r=mx.random.normal((1,m,n)).astype(mx.bfloat16);g=mx.random.normal((1,1,n)).astype(mx.bfloat16);b=mx.random.normal((n,)).astype(mx.bfloat16)
 base=mx.compile(lambda x,r: r+g*(mx.quantized_matmul(x,*parts,transpose=True,bits=1,group_size=64)+b));cand=mx.compile(lambda x,r:fused(x,parts,r,g,b))
 a=base(x,r);z=cand(x,r);mx.eval(a,z);diff=np.array(a.astype(mx.float32))-np.array(z.astype(mx.float32));assert np.isfinite(np.array(z.astype(mx.float32))).all()
 times={}
 for name,fn in [('base',base),('fused',cand)]:
  for _ in range(3):mx.eval(fn(x,r))
  ts=[]
  for _ in range(30):
   mx.synchronize();t=time.perf_counter();mx.eval(fn(x,r));mx.synchronize();ts.append((time.perf_counter()-t)*1000)
  times[name]=float(np.median(ts))
 row=dict(m=m,max_abs=float(abs(diff).max()),rmse=float(np.sqrt((diff*diff).mean())),median_ms=times);rows.append(row);print(row,flush=True)
(root/'probe.json').write_text(json.dumps(rows,indent=2))
