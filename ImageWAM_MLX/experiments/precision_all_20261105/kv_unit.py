import sys,json
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
import mlx.core as mx
import numpy as np
from kv_fused import quant,attention
rng=np.random.default_rng(123);rows=[]
for h,m,p in [(2,1,39),(2,16,83),(2,64,189)]:
 q,k,v,ak,av=[mx.array(rng.normal(size=(1,h,n,128)).astype(np.float32)).astype(mx.bfloat16) for n in (m,p,p,m,m)]
 cache=quant(k,v);out=np.array(attention(q,cache,ak,av).astype(mx.float32)).reshape(1,m,h,128).transpose(0,2,1,3)
 pk,ks,pv,vs=cache;kk=np.concatenate([np.array(pk).astype(np.float64)*np.array(ks)[...,None],np.array(ak.astype(mx.float32))],axis=2);vv=np.concatenate([np.array(pv).astype(np.float64)*np.array(vs)[...,None],np.array(av.astype(mx.float32))],axis=2)
 logits=np.array(q.astype(mx.float32)).astype(np.float64)@kk.swapaxes(-1,-2)/np.sqrt(128);logits-=logits.max(axis=-1,keepdims=True);soft=np.exp(logits);soft/=soft.sum(axis=-1,keepdims=True);ref=soft@vv;diff=out-ref
 row=dict(h=h,m=m,p=p,max_abs=float(abs(diff).max()),rmse=float(np.sqrt((diff**2).mean())));rows.append(row);assert row['max_abs']<0.01,row
Path(sys.argv[1]).write_text(json.dumps(rows,indent=2));print(rows)
