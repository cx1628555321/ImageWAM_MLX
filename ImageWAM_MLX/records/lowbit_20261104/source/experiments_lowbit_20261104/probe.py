import json,traceback,time
from pathlib import Path
import numpy as np
import mlx.core as mx
from native_int8 import int8_gemm,pack_weight,linear
out=Path('records/lowbit_20261104');out.mkdir(parents=True,exist_ok=True)
rows=[]
try:
 for m,k,n in [(1,64,64),(16,1024,128),(39,256,512),(79,512,256),(328,1024,256)]:
  a=mx.random.randint(-127,128,(m,k)).astype(mx.int8);b=mx.random.randint(-127,128,(n,k)).astype(mx.int8);y=int8_gemm(a,b,raw=True);mx.eval(y)
  ref=np.array(a).astype(np.int64)@np.array(b).astype(np.int64).T
  err=int(np.max(np.abs(np.array(y)-ref)));assert err==0,(m,k,n,err)
  x=mx.random.normal((m,k)).astype(mx.bfloat16);w=mx.random.normal((n,k)).astype(mx.bfloat16);z=linear(x,pack_weight(w));mx.eval(z)
  rows.append(dict(shape=[m,k,n],integer_max_error=err,finite=bool(np.isfinite(np.array(z.astype(mx.float32))).all())));print(rows[-1],flush=True)
 result=dict(status='passed',cases=rows)
except Exception:
 result=dict(status='failed',cases=rows,error=traceback.format_exc());print(result['error'],flush=True)
(out/'native_probe.json').write_text(json.dumps(result,indent=2))
if result['status']!='passed':raise SystemExit(1)
