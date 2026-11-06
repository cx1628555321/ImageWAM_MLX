import sys,json,time,gc,traceback
from pathlib import Path
import numpy as np
import mlx.core as mx
from native_int8 import pack_weight,linear
out=Path('records/lowbit_20261104');prof=out/'profile';weights=Path('weights/bf16/action_only');man=json.loads((weights/'manifest.json').read_text());rows=[]
def bench(fn,x):
 for _ in range(5):mx.eval(fn(x))
 ts=[]
 for _ in range(25):
  mx.synchronize();t=time.perf_counter();y=fn(x);mx.eval(y);mx.synchronize();ts.append((time.perf_counter()-t)*1000)
 return y,dict(mean_ms=float(np.mean(ts)),median_ms=float(np.median(ts)),p95_ms=float(np.percentile(ts,95)))
for path in sorted(prof.glob('*.npy')):
 component,key=path.stem.split('__',1)
 if component=='vae':continue
 file=next(n for n,d in man['files'].items() if (n.startswith('qwen') if component=='qwen' else n==component+'.safetensors') and key+'.weight' in d['tensors'])
 ww=mx.load(str(weights/file));w=ww[key+'.weight'];bias=ww.get(key+'.bias');del ww
 mx.eval(w);x=mx.array(np.load(path)).astype(mx.bfloat16);mx.eval(x)
 modes={'bf16':lambda a:a@w.T+(bias if bias is not None else 0)}
 packed=pack_weight(w)
 modes['native_w8a8']=lambda a:linear(a,packed,bias)
 parts={}
 for bits in (1,4,8):
  parts[bits]=mx.quantize(w,group_size=64,bits=bits);mx.eval(parts[bits])
  modes[f'affine_w{bits}a16']=lambda a,bits=bits:mx.quantized_matmul(a,*parts[bits],group_size=64,bits=bits)+(bias if bias is not None else 0)
 ref=np.array(modes['bf16'](x).astype(mx.float32))
 for name,fn in modes.items():
  try:
   y,t=bench(mx.compile(fn),x);arr=np.array(y.astype(mx.float32));diff=arr-ref
   row=dict(component=component,key=key,shape=[x.size//x.shape[-1],w.shape[1],w.shape[0]],mode=name,**t,finite=bool(np.isfinite(arr).all()),max_abs=float(np.max(np.abs(diff))),rmse=float(np.sqrt(np.mean(diff**2))))
  except Exception:row=dict(component=component,key=key,mode=name,error=traceback.format_exc())
  rows.append(row);(out/'gemm.json').write_text(json.dumps(rows,indent=2));print(row,flush=True)
 del modes,parts,packed,w,x,ref;gc.collect();mx.clear_cache()
