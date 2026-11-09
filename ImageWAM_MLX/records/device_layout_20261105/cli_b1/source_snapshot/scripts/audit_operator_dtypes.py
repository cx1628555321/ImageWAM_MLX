"""Probe public operators; integer success alone does not imply quantized semantics."""
import json
from pathlib import Path
import mlx.core as mx
out=Path('records/speed_operator_audit_20261103');out.mkdir(exist_ok=False)
rows=[]
for dtype in (mx.bfloat16,mx.float16,mx.float32,mx.int8,mx.uint32):
 x=mx.ones((1,16,64),dtype=dtype);w=mx.ones((64,64),dtype=dtype)
 funcs={'matmul':lambda:x@w,'layer_norm':lambda:mx.fast.layer_norm(x,None,None,eps=1e-6),'softmax':lambda:mx.softmax(x),
 'silu':lambda:x*mx.sigmoid(x),'sdpa':lambda:mx.fast.scaled_dot_product_attention(x[:,None],x[:,None],x[:,None],scale=.125),
 'conv2d':lambda:mx.conv2d(mx.ones((1,8,8,32),dtype=dtype),mx.ones((64,3,3,32),dtype=dtype),padding=1),
 'mean':lambda:mx.mean(x,axis=-1),'rsqrt':lambda:mx.rsqrt(x),'add':lambda:x+x,'concat':lambda:mx.concatenate([x,x],axis=1)}
 for name,fn in funcs.items():
  r={'operator':name,'input_dtype':str(dtype)}
  try:y=fn();mx.eval(y);r.update(supported=True,output_dtype=str(y.dtype))
  except Exception as e:r.update(supported=False,error=str(e))
  rows.append(r)
(out/'results.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))
