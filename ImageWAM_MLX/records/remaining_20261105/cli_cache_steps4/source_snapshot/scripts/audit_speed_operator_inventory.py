"""Arithmetic/dtype audit, not a claim that integer tensors have quantized semantics."""
import sys,json
from pathlib import Path
import mlx.core as mx
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from imagewam_mlx.ops import rms,gated
out=Path('records/speed_operator_inventory_20261103');out.mkdir(exist_ok=False);rows=[]
for dt in (mx.bfloat16,mx.float16,mx.float32,mx.int8,mx.uint32):
 x=mx.ones((1,16,64),dtype=dt);w=mx.ones((64,),dtype=dt)
 fs={'rms_expression':lambda:rms(x,w),'gated_silu':lambda:gated(x),'sigmoid':lambda:mx.sigmoid(x),'exp':lambda:mx.exp(x),'cos':lambda:mx.cos(x),'sin':lambda:mx.sin(x),'power':lambda:x**2,'multiply':lambda:x*x,'divide':lambda:x/x,'subtract':lambda:x-x,'clip':lambda:mx.clip(x,0,1),'sum':lambda:mx.sum(x,axis=-1),'reshape':lambda:x.reshape(16,64),'transpose':lambda:x.transpose(0,2,1),'split':lambda:mx.split(x,2,-1)[0],'gather_embedding':lambda:x[:,mx.array([0,2],dtype=mx.int32)],'pad':lambda:mx.pad(x,[(0,0),(0,1),(0,0)])}
 for op,fn in fs.items():
  r=dict(operator=op,input_dtype=str(dt))
  try:
   y=fn();mx.eval(y);r.update(supported=True,output_dtype=str(y.dtype))
  except Exception as e:r.update(supported=False,error=str(e))
  rows.append(r)
(out/'results.json').write_text(json.dumps({'operators':rows,'note':'Combine with speed_operator_audit and quantization_support. Raw integer success is not scale-aware low-bit inference. Custom RoPE production tensors remain BF16/FP32; token IDs and shape indexing remain integer.'},indent=2))
