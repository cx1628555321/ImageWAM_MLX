import os
os.environ['MLX_MAX_OPS_PER_BUFFER']='256'
import sys,time,json,collections,copy
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
import numpy as np
import mlx.core as mx
from common import *
from imagewam_mlx.ops import D,flux_rope
import imagewam_mlx.action_epilogue as ae
import imagewam_mlx.action_kernel as ak
ROOT.mkdir(exist_ok=False,parents=True)
m=ImageWAM('weights/bf16','assets',profile='speed',text_cache_size=0);args=inputs()
def bench(fn,repeats=20):
 for _ in range(3):mx.eval(fn())
 rows=[]
 for _ in range(repeats):
  mx.synchronize();t=time.perf_counter();mx.eval(fn());mx.synchronize();rows.append((time.perf_counter()-t)*1000)
 return float(np.median(rows))
x,st=m.preprocess(args['images'],args['state']);hidden=m.qwen(mx.array(m.tokenize(args['prompt'])));latent=m.vae(x);mx.eval(hidden,latent,st)
visual=[];original=m.visual.mm;g=GroupVisual(m.visual,64)
def mm(**kw):
 mx.eval(kw['inputs']);t=dict(kw['template']);a=original(**kw);b=g.mm(**dict(kw));mx.eval(a,b)
 err=float(mx.max(mx.abs(a[0].astype(mx.float32)-b[0].astype(mx.float32))).item())
 row=dict(shape=[t['ROWS'],kw['inputs'][0].shape[0],t['XK']],base_ms=bench(lambda:original(**kw)),group_ms=bench(lambda:g.mm(**dict(kw))),exact=bool(mx.array_equal(a[0],b[0]).item()),max_abs=err)
 visual.append(row);print('VISUAL',row,flush=True)
 return a
m.visual.mm=mm
cache=Base.prefill(m,hidden,latent,st);mx.eval(cache);m.visual.mm=original
# Audit one actual denoising step; only first occurrence of each shape is microbenchmarked.
action=[];origf=ae.fused;seen={}
def fused(x,parts,residual,gate,bias,target=512,tile=(32,64,2,2)):
 mx.eval(x,parts,residual,gate,bias);shape=(x.shape[1],residual.shape[-1],x.shape[-1]);key=str(shape)
 if key not in seen:
  refs=origf(x,parts,residual,gate,bias,target=512,tile=tile);mx.eval(refs);rows=[]
  for target in (128,256,512,1024):
   bm,bn,wm,wn=tile;nt=(shape[1]+bn-1)//bn;mt=(shape[0]+bm-1)//bm;s=min(max(1,target//(nt*mt)),shape[2]//64)
   while s>1 and shape[2]%(s*64):s-=1
   fn=lambda:origf(x,parts,residual,gate,bias,target=target,tile=tile)
   out=fn();mx.eval(out);rows.append(dict(target=target,splits=s,ms=bench(fn),partial_logical_rw_bytes=8*s*shape[0]*shape[1],exact=bool(mx.array_equal(out,refs).item()),max_abs=float(mx.max(mx.abs(out.astype(mx.float32)-refs.astype(mx.float32))).item())))
  seen[key]=dict(shape=shape,results=rows);print('ACTION',seen[key],flush=True)
 action.append(key)
 return origf(x,parts,residual,gate,bias,target=512,tile=tile)
ae.fused=fused
noise=mx.array(args['noise']).astype(D);h=noise.shape[1];ids=np.zeros((1,h,4),np.float32);ids[...,0]=2;ids[0,:,1]=np.arange(h);pe=flux_rope(mx.array(ids).astype(D));workspace=[(mx.pad(k,[(0,0),(0,0),(0,h),(0,0)]),mx.pad(v,[(0,0),(0,0),(0,h),(0,0)])) for k,v in cache];mx.eval(workspace)
result=workspace_step(m,workspace,noise,mx.array([1.0]).astype(D),mx.array(-0.021728515625).astype(D),pe);mx.eval(result);ae.fused=origf
# Functional KV ownership experiment on real prefix/action shapes. This is allocation evidence, not a physical-copy trace.
k=workspace[0][0];tail=k[:,:,-h:,:];mx.eval(tail);before=mx.get_active_memory();updated=mx.slice_update(k,tail,mx.array([k.shape[2]-h]),axes=(2,));mx.eval(updated);retained=mx.get_active_memory()-before
shapes={}
for row in visual:
 key=tuple(row['shape']);shapes.setdefault(key,[]).append(row)
selected=[list(k) for k,v in shapes.items() if all(a['exact'] for a in v) and np.mean([a['group_ms'] for a in v])<.95*np.mean([a['base_ms'] for a in v])]
res=dict(visual=visual,selected_visual_shapes=selected,action_shapes=seen,action_calls_per_step=dict(collections.Counter(action)),kv=dict(shape=k.shape,bytes=k.nbytes,retained_allocation_delta=retained,note='retained references; not full-request physical-copy proof'),profile_kind='isolated synchronized wall times; not GPU timestamps',device=mx.device_info())
(ROOT/'diagnostic.json').write_text(json.dumps(res,indent=2));print('DONE',selected,flush=True)
