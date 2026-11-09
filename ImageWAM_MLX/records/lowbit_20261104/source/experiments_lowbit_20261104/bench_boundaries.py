import sys,time,json,traceback
from pathlib import Path
import numpy as np
import mlx.core as mx
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from imagewam_mlx.ops import attention,silu,gated,rms
from native_int8 import quantize_rows,int8_gemm,pack_weight,linear
from fused_quant import quantize_fused
from quant_attention import quant_kv,attend_quantized
from quant_conv import QuantConv
out=Path('records/lowbit_20261104');prof=out/'profile';rows=[]
def bench(name,fn,args,ref=None):
 try:
  fn=mx.compile(fn)
  for _ in range(4):mx.eval(fn(*args))
  ts=[]
  for _ in range(20):
   mx.synchronize();t=time.perf_counter();y=fn(*args);mx.eval(y);mx.synchronize();ts.append((time.perf_counter()-t)*1000)
  arr=np.array(y.astype(mx.float32)) if isinstance(y,mx.array) else np.array((y[0].astype(mx.float32)*y[1][:,None]).astype(mx.float32))
  r=dict(name=name,mean_ms=float(np.mean(ts)),median_ms=float(np.median(ts)),finite=bool(np.isfinite(arr).all()))
  if ref is not None:
   d=arr-ref;r.update(max_abs=float(np.max(np.abs(d))),rmse=float(np.sqrt(np.mean(d*d))))
 except Exception:r=dict(name=name,error=traceback.format_exc())
 rows.append(r);(out/'boundaries.json').write_text(json.dumps(rows,indent=2));print(r,flush=True)
# Eager reference quantities and real activations.
x=mx.array(np.load(prof/'action__single_blocks.0.linear1.npy')).astype(mx.bfloat16);k=x.shape[-1];g=mx.ones((k,),dtype=x.dtype)
for kind,fn in [('layernorm',lambda x:mx.fast.layer_norm(x,None,None,eps=1e-6)),('rms',lambda x:rms(x,g)),('silu',silu),('gated',gated)]:
 ref=np.array(fn(x).astype(mx.float32))
 bench(kind+'_unfused_quant',lambda x:quantize_rows(fn(x)),[x],ref)
 bench(kind+'_fused_quant',lambda x:quantize_fused(x,kind),[x],ref)
# Actual attention dimensions, including all dynamic text lengths tested via prefixes below.
a=np.load(prof/'attention.npz');q,k,v=[mx.array(a[n]).astype(mx.bfloat16) for n in ('q','k','v')];cache=quant_kv(k,v);mx.eval(cache);ref=np.array(attention(q,k,v).astype(mx.float32))
bench('attention_bf16',attention,[q,k,v],ref)
bench('attention_int8_online_kv',lambda q,k,v:attend_quantized(q,quant_kv(k,v)),[q,k,v],ref)
bench('attention_int8_reuse_kv',attend_quantized,[q,cache],ref)
for p in sorted(prof.glob('vae__*.npy')):
 key=p.stem.split('__')[1];ww=mx.load('weights/bf16/action_only/vae.safetensors');w=ww[key+'.weight'].transpose(0,2,3,1);bias=ww[key+'.bias'];del ww;mx.eval(w,bias)
 x=mx.array(np.load(p)).astype(mx.bfloat16);pad=1 if w.shape[1]==3 else 0
 qc=QuantConv(w,bias);ref=np.array((mx.conv2d(x,w,padding=pad)+bias).astype(mx.float32))
 bench(key+'_conv_bf16',lambda x:mx.conv2d(x,w,padding=pad)+bias,[x],ref)
 bench(key+'_conv_int8',lambda x:qc(x,pad=pad),[x],ref)
 del qc,w,bias,x;mx.clear_cache()
