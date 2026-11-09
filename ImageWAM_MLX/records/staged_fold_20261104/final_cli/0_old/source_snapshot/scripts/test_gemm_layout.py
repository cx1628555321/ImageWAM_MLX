"""Real action activations; compare BF16 transposed-view and packed GEMM."""
import sys,time,json,argparse
from pathlib import Path
import numpy as np
import mlx.core as mx
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from imagewam_mlx import ImageWAM
from imagewam_mlx.ops import D,flux_rope
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--variant',choices=['packed','flatten'],default='packed');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
m=ImageWAM('weights/bf16','assets')
ex=Path('examples/robotwin_ep0_frame0');raw=np.load(ex/'inputs.npz');images={k:raw[k] for k in ('cam_high','cam_left_wrist','cam_right_wrist')};prompt=(ex/'prompt.txt').read_text();noise=np.load(ex/'initial_noise_fp32.npy')
image,state=m.preprocess(images,raw['state']);hidden=m.qwen(mx.array(m.tokenize(prompt)));latent=m.vae(image);cache=m.prefill(hidden,latent,state);mx.eval(cache)
names=['double_blocks.0.img_attn.qkv','single_blocks.0.linear1','single_blocks.0.linear2'];captured={};original=m.action.linear
def capture(x,p):
 if p in names:captured[p]=x
 return original(x,p)
m.action.linear=capture
ids=np.zeros((1,16,4),np.float32);ids[...,0]=2;ids[0,:,1]=np.arange(16)
y=m._step(cache,mx.array(noise).astype(D),mx.ones((1,),dtype=D),mx.array(-.02,dtype=D),flux_rope(mx.array(ids).astype(D)))
mx.eval(y,*captured.values());m.action.linear=original
r={'variant':a.variant,'rows':[],'note':'M>16 tiles real first-step activations; isolated GEMMs, not end-to-end speedup. packed_ms labels candidate timing also for flatten variant.'}
for name in names:
 w=m.action.w[name+'.weight'];b=m.action.w.get(name+'.bias')
 mx.synchronize();t=time.perf_counter();wt=mx.contiguous(w.T) if a.variant=='packed' else w.T;mx.eval(wt);pack_ms=(time.perf_counter()-t)*1000
 for rows in (1,8,16,32,64):
  x=mx.tile(captured[name],(1,4,1))[:,:rows];mx.eval(x)
  def old(x):
   y=x@w.T
   return y if b is None else y+b
  def new(x):
   y=x@wt if a.variant=='packed' else (x.reshape(-1,x.shape[-1])@wt).reshape(*x.shape[:-1],w.shape[0])
   return y if b is None else y+b
  fs=[mx.compile(old),mx.compile(new)]
  for f in fs:
   for _ in range(5):mx.eval(f(x))
  times=[[],[]]
  for i in range(40):
   for k in ((0,1) if i%2==0 else (1,0)):
    mx.synchronize();t=time.perf_counter();z=fs[k](x);mx.eval(z);mx.synchronize();times[k].append((time.perf_counter()-t)*1000)
  u,v=[np.array(f(x).astype(mx.float32)) for f in fs]
  row={'name':name,'M':rows,'weight_shape':list(w.shape),'pack_ms':pack_ms,'packed_bytes':w.nbytes if a.variant=='packed' else 0,'base_ms':float(np.mean(times[0])),'packed_ms':float(np.mean(times[1])),'equal':bool(np.array_equal(u,v)),'max_abs':float(np.max(np.abs(u-v))),'times_ms':times}
  r['rows'].append(row);(a.output/'results.json').write_text(json.dumps(r,indent=2));print({k:v for k,v in row.items() if k!='times_ms'},flush=True)
 del wt,fs
