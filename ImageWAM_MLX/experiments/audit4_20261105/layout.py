import os
os.environ['MLX_MAX_OPS_PER_BUFFER']='256'
import sys,json,collections
from pathlib import Path
sys.path.insert(0,str(Path.cwd()));sys.path.insert(0,str(Path('experiments/roadmap_20261105').resolve()))
import mlx.core as mx
original=mx.fast.metal_kernel
probe=original(name='wam_stride_metadata',input_names=['x'],output_names=['out'],ensure_row_contiguous=False,source='out[0]=x_ndim; for(int i=0;i<x_ndim;i++)out[i+1]=x_strides[i];')
rows=[];counts=collections.Counter();seen=set();enabled=False
def factory(*a,**kw):
 f=original(*a,**kw);name=kw.get('name',a[0] if a else '?');ensure=kw.get('ensure_row_contiguous',True)
 def call(*args,**kwargs):
  if enabled:
   inputs=kwargs.get('inputs',args[0] if args else []);counts[name]+=1;sig=(name,tuple((tuple(x.shape),str(x.dtype)) for x in inputs))
   if sig not in seen:
    seen.add(sig);data=[]
    for x in inputs:
     meta=probe(inputs=[x],grid=(1,1,1),threadgroup=(1,1,1),output_shapes=[(x.ndim+1,)],output_dtypes=[mx.int32])[0];mx.eval(meta);strides=meta.tolist()[1:];expected=1;contiguous=True
     for size,stride in zip(reversed(x.shape),reversed(strides)):
      if size>1 and stride!=expected:contiguous=False
      expected*=size
     data.append(dict(shape=x.shape,dtype=str(x.dtype),strides=strides,row_contiguous=contiguous,logical_bytes=x.nbytes))
    rows.append(dict(name=name,ensure_row_contiguous=ensure,inputs=data))
  return f(*args,**kwargs)
 return call
mx.fast.metal_kernel=factory
from common import inputs
from imagewam_mlx import ImageWAM
from imagewam_mlx.model import ImageWAM as Base
from imagewam_mlx.workspace import workspace_step
from imagewam_mlx.ops import D,flux_rope
import numpy as np
r=Path('records/audit4_20261105');mx.disable_compile();m=ImageWAM('weights/bf16','assets',profile='speed',text_cache_size=0);args=inputs();enabled=True;a,n=m.predict(**args,steps=10);enabled=False
np.savez(r/'layout_output.npz',action=a,normalized=n)
(r/'layout.json').write_text(json.dumps(dict(kind='eager diagnostic with compilation disabled; first occurrence per kernel/shape/dtype; not proof of compiled physical copies',calls=counts,signatures=rows),indent=2));print('layout',len(rows),sum(counts.values()),flush=True)
# Export logical step dependencies separately, with probes disabled and inputs materialized.
x,st=m.preprocess(args['images'],args['state']);h=m.qwen(mx.array(m.tokenize(args['prompt'])));lat=m.vae(x);cache=Base.prefill(m,h,lat,st);mx.eval(cache)
noise=mx.array(args['noise']).astype(D);H=noise.shape[1];ids=np.zeros((1,H,4),np.float32);ids[...,0]=2;ids[0,:,1]=np.arange(H);pe=flux_rope(mx.array(ids).astype(D));cache=[(mx.pad(k,[(0,0),(0,0),(0,H),(0,0)]),mx.pad(v,[(0,0),(0,0),(0,H),(0,0)])) for k,v in cache];t=mx.array([1.]).astype(D);delta=mx.array(-.02).astype(D);mx.eval(cache,noise,t,delta,pe)
y,updated=workspace_step(m,cache,noise,t,delta,pe);mx.export_to_dot(str(r/'logical_step.dot'),y,*[v for kv in updated for v in kv]);print('graph saved',flush=True)
