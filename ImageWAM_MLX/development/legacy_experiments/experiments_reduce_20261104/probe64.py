import json
from pathlib import Path
import mlx.core as mx
import numpy as np
from reduceopt import reduce
from tiled import reduce as original
root=Path('records/reduce_20261104/probe64');root.mkdir(exist_ok=False);rows=[];mx.random.seed(47)
for m in (1,8,14,16,32,64):
 for s in (1,2,4,8,16):
  n=3072;p=mx.random.normal((s,m,n));r=mx.random.normal((1,m,n)).astype(mx.bfloat16);g=mx.random.normal((1,1,n)).astype(mx.bfloat16);b=mx.random.normal((n,)).astype(mx.bfloat16);inputs=[p,r,g,b]
  ref=original(inputs=inputs,template=[('T',r.dtype),('SPLITS',s)],grid=(r.size,1,1),threadgroup=(256,1,1),output_shapes=[r.shape],output_dtypes=[r.dtype])[0]
  out=reduce(inputs=inputs,template=[('T',r.dtype),('SPLITS',s),('VEC',4)],grid=(n//4,m,1),threadgroup=(64,1,1),output_shapes=[r.shape],output_dtypes=[r.dtype])[0]
  mx.eval(ref,out);a=np.array(ref.astype(mx.float32));z=np.array(out.astype(mx.float32));assert np.array_equal(a,z),(m,s,float(abs(a-z).max()));rows.append(dict(horizon=m,splits=s,exact=True))
(root/'probe.json').write_text(json.dumps(rows,indent=2));print('30 probes exact',flush=True)
