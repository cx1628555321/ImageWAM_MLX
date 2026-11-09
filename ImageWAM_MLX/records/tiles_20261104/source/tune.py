import sys,json,time
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'experiments_lowbit_20261104'))
import mlx.core as mx
import numpy as np
import native_int8 as native
OUT=Path('records/tiles_20261104');OUT.mkdir(exist_ok=True)
KERNELS={}
def gemm(q,w,sx,sw,bias,bm,bn,bk):
 m,k=q.shape;n=w.shape[0];kp=(k+bk-1)//bk*bk;mp=(m+bm-1)//bm*bm;npad=(n+bn-1)//bn*bn
 q=native.pad2(q,mp,kp);w=native.pad2(w,npad,kp);threads=(bm//32)*(bn//32)*32
 sx=mx.pad(sx,[(0,max(0,16-sx.size))]);sw=mx.pad(sw,[(0,max(0,16-sw.size))]);bias=mx.pad(bias,[(0,max(0,16-bias.size))])
 key=bm,bn,bk
 if key not in KERNELS:
  src=f'w8a8_gemm_impl<{bm},{bn},{bk},32,{bm//32},{bn//32}>(q,w,out,M,N,K,sx,sw,bias,0,(M+{bm}-1)/{bm},(N+{bn}-1)/{bn},threadgroup_position_in_grid.xy,thread_position_in_threadgroup.x/32,thread_position_in_threadgroup.x%32);'
  KERNELS[key]=mx.fast.metal_kernel(name=f'tile_i8_{bm}_{bn}_{bk}',input_names=['q','w','sx','sw','bias'],output_names=['out'],header=native.HEADER,source=src)
 return KERNELS[key](inputs=[q,w,sx,sw,bias],template=[('M',m),('N',n),('K',kp)],grid=(((n+bn-1)//bn)*threads,(m+bm-1)//bm,1),threadgroup=(threads,1,1),output_shapes=[(m,n)],output_dtypes=[mx.bfloat16])[0]
def bench(fn,x):
 fn=mx.compile(fn)
 for _ in range(4):mx.eval(fn(x))
 ts=[]
 for _ in range(15):
  mx.synchronize();t=time.perf_counter();y=fn(x);mx.eval(y);ts.append((time.perf_counter()-t)*1000)
 return y,float(np.median(ts))
def run():
 rows=[];man=json.loads(Path('weights/bf16/action_only/manifest.json').read_text())
 for path in sorted(Path('records/lowbit_20261104/profile').glob('*.npy')):
  component,key=path.stem.split('__',1)
  if component not in ('visual','action'):continue
  ww=mx.load('weights/bf16/action_only/'+component+'.safetensors');w=ww[key+'.weight'];b=ww.get(key+'.bias',mx.zeros((w.shape[0],),dtype=mx.bfloat16));del ww
  packed,sw=native.pack_weight(w);x=mx.array(np.load(path)).astype(mx.bfloat16);mx.eval(x,b);q,s=native.quantize_rows(x);ref=native.int8_gemm(q,packed,s,sw,b);mx.eval(ref)
  for bm,bn,bk in [(32,64,128),(32,128,128),(64,64,128),(64,128,128),(128,64,128),(128,128,128),(32,128,512),(128,128,512)]:
   def fn(x):
    q,s=native.quantize_rows(x);return gemm(q,packed,s,sw,b,bm,bn,bk)
   y,ms=bench(fn,x);err=float(mx.max(mx.abs(y.astype(mx.float32)-ref.astype(mx.float32))).item());assert err==0,err
   rows.append(dict(component=component,key=key,shape=[q.shape[0],w.shape[1],w.shape[0]],tile=[bm,bn,bk],median_ms=ms,max_error=err));print(rows[-1],flush=True)
  (OUT/'tuning.json').write_text(json.dumps(rows,indent=2));del w,packed,x,ref;mx.clear_cache()
if __name__=='__main__':run()
