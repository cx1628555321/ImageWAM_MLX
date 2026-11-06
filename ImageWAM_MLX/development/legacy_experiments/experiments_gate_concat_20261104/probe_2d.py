import sys,json
from pathlib import Path
import numpy as np
import mlx.core as mx
sys.path.insert(0,str(Path.cwd()))
from imagewam_mlx.ops import gated
from gate_2d import gate_concat
root=Path('records/gate_concat_20261104/probe_2d');root.mkdir(exist_ok=False)
mx.random.seed(9);results=[]
for m in (1,8,14,16,32,64):
 a=mx.random.normal((1,m,3072)).astype(mx.bfloat16);full=(mx.random.normal((1,m,9216+12288))*4).astype(mx.bfloat16);z=full[...,9216:]
 ref=mx.compile(lambda a,z:mx.concatenate([a,gated(z)],-1))(a,z);out=gate_concat(a,z);mx.eval(ref,out)
 r=np.array(ref.astype(mx.float32));o=np.array(out.astype(mx.float32));d=r-o
 result=dict(horizon=m,max_abs=float(abs(d).max()),rmse=float(np.sqrt(np.mean(d*d))),different=int(np.count_nonzero(d)))
 assert np.isfinite(o).all();assert result['max_abs']==0;results.append(result);print(result,flush=True)
(root/'probe.json').write_text(json.dumps(results,indent=2))
