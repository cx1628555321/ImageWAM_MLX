import os
os.environ['MLX_MAX_OPS_PER_BUFFER']='256'
import sys,json,time
from pathlib import Path
import mlx.core as mx
import numpy as np
sys.path.insert(0,str(Path.cwd()))
from fusion_candidates import qx,sv,gated
r=Path('records/fusion_confirm_20261104/micro');r.mkdir(exist_ok=False)
M=328;KA=3072;KB=9216;K=KA+KB;mp=(M+63)//64*64;nb=K//64;threads=(nb+31)//32*32
mx.random.seed(29);a=mx.random.normal((M,KA)).astype(mx.bfloat16);b=mx.random.normal((M,KB*2)).astype(mx.bfloat16);mx.eval(a,b)
old=mx.compile(lambda a,b:sv.quantize(mx.concatenate([a,gated(b)],axis=-1)))
new=mx.compile(lambda a,b:qx(inputs=[a,gated(b)],template=[('T',a.dtype)],grid=(threads,mp,1),threadgroup=(threads,1,1),output_shapes=[(mp,K),(M,),(mp,nb+1)],output_dtypes=[mx.int8,mx.float32,mx.uint8]))
fns={'baseline':old,'split':new};ref=old(a,b);mx.eval(ref)
for f in fns.values():
 out=f(a,b);mx.eval(out);assert all(np.array_equal(np.array(x),np.array(y)) for x,y in zip(ref,out))
 for i in range(20):mx.eval(f(a,b))
rows={k:[] for k in fns};rng=np.random.default_rng(91)
for i in range(500):
 for k in rng.permutation(list(fns)):
  mx.synchronize();start=time.perf_counter();out=fns[k](a,b);mx.eval(out);mx.synchronize();rows[k].append((time.perf_counter()-start)*1000)
d=np.array(rows['baseline'])-rows['split'];ci=np.percentile(np.mean(d[rng.integers(0,len(d),(10000,len(d)))],1),[2.5,97.5]);result=dict(shape=[M,KA,KB*2],measurement='synthetic inputs, actual projection dimensions, synchronized wall time; not GPU counters',exact=True,mean_ms={k:float(np.mean(v)) for k,v in rows.items()},saving_ms=float(d.mean()),ci95=ci.tolist(),samples_ms=rows)
(r/'results.json').write_text(json.dumps(result,indent=2));print({k:v for k,v in result.items() if k!='samples_ms'})
