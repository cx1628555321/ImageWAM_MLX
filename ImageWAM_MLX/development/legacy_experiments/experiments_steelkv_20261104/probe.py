import sys,json,time
from pathlib import Path
import numpy as np
import mlx.core as mx
sys.path.insert(0,str(Path.cwd()))
from imagewam_mlx.ops import attention
from segmented import segmented
root=Path('records/steelkv_20261104');rows=[]
for n,p in [(1,320),(8,328),(16,328),(32,368),(64,482)]:
 q=mx.random.normal((1,24,n,128)).astype(mx.bfloat16);pk=mx.random.normal((1,24,p,128)).astype(mx.bfloat16);pv=mx.random.normal(pk.shape).astype(mx.bfloat16);ak=mx.random.normal(q.shape).astype(mx.bfloat16);av=mx.random.normal(q.shape).astype(mx.bfloat16)
 y=segmented(q,pk,pv,ak,av);ref=attention(q,mx.concatenate([pk,ak],2),mx.concatenate([pv,av],2));mx.eval(y,ref);diff=np.array((y.astype(mx.float32)-ref.astype(mx.float32)))
 assert np.isfinite(np.array(y.astype(mx.float32))).all();assert float(np.max(np.abs(diff)))<.02
 rows.append(dict(query=n,prefix=p,max_abs=float(np.max(np.abs(diff))),rmse=float(np.sqrt(np.mean(diff**2)))))
(root/'probe.json').write_text(json.dumps(rows,indent=2));print(rows,flush=True)
a=np.load('records/lowbit_20261104/profile/attention.npz');q,k,v=[mx.array(a[z]).astype(mx.bfloat16) for z in ('q','k','v')];n=q.shape[2];pk,pv=k[:,:,:-n],v[:,:,:-n];ak,av=k[:,:,-n:],v[:,:,-n:]
results={}
for name in ('sdpa_concat','segmented'):
 def one(q):return segmented(q,pk,pv,ak,av) if name=='segmented' else attention(q,mx.concatenate([pk,ak],2),mx.concatenate([pv,av],2))
 def chain(q):
  for _ in range(8):q=one(q).reshape(1,n,24,128).transpose(0,2,1,3)
  return q
 fn=mx.compile(chain)
 for _ in range(4):mx.eval(fn(q))
 ts=[]
 for _ in range(25):
  mx.synchronize();t=time.perf_counter();mx.eval(fn(q));mx.synchronize();ts.append((time.perf_counter()-t)*1000/8)
 results[name]=dict(mean_ms=float(np.mean(ts)),median_ms=float(np.median(ts)),chain=8,measurements=25)
(root/'chain.json').write_text(json.dumps(results,indent=2));print(results,flush=True)
