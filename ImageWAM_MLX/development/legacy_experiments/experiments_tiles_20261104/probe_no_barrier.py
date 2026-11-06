import sys,json
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'development/legacy_experiments/experiments_lowbit_20261104'))
import mlx.core as mx
import numpy as np
import native_int8 as native
from tune_no_barrier import gemm
rows=[]
for m,k,n in [(1,64,64),(16,1024,128),(39,256,512),(79,512,256),(328,1024,256),(193,3072,128)]:
 x=mx.random.normal((m,k)).astype(mx.bfloat16);w=mx.random.normal((n,k)).astype(mx.bfloat16)
 q,s=native.quantize_rows(x);w,sw=native.pack_weight(w);bias=mx.zeros((n,),dtype=mx.bfloat16)
 expected=native.int8_gemm(q,w,s,sw,bias);y=gemm(q,w,s,sw,bias,32,128,512)
 mx.eval(y,expected);err=float(mx.max(mx.abs(y.astype(mx.float32)-expected.astype(mx.float32))).item());assert err==0
 rows.append(dict(shape=[m,k,n],max_abs=err))
Path('records/tiles_retest_20261104/no_barrier_probe.json').write_text(json.dumps(rows,indent=2));print(rows)
