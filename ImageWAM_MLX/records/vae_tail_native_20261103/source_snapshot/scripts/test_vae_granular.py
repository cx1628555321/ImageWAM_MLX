"""Fine-grained compilation / pointwise-convolution candidates; shared weights."""
import sys,time,json,copy,argparse
from pathlib import Path
import numpy as np
import mlx.core as mx
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from imagewam_mlx import ImageWAM
from imagewam_mlx.fused_vae import FusedVAE

def build(base,mode):
 v=FusedVAE.from_vae(base)
 if mode=='res_safe':
  cache={}
  def res(x,p):
   y=v.conv(v.gn_silu(x,p+'.norm1'),p+'.conv1',pad=1)
   y=v.gn_silu(y,p+'.norm2')
   if p not in cache:
    def finish(x,y):
     z=v.conv(y,p+'.conv2',pad=1)
     if p+'.nin_shortcut.weight' in v.w:x=v.conv(x,p+'.nin_shortcut')
     return x+z
    cache[p]=mx.compile(finish)
   return cache[p](x,y)
  v.res=res
  return v
 if mode in ('conv','res','gn'):
  name={'conv':'conv','res':'res','gn':'gn_silu'}[mode];original=getattr(v,name);cache={}
  def call(x,p,*args,**kwargs):
   key=(p,args,tuple(sorted(kwargs.items())))
   if key not in cache:cache[key]=mx.compile(lambda z:original(z,p,*args,**kwargs))
   return cache[key](x)
  setattr(v,name,call)
 elif mode=='pointwise':
  original=v.conv
  def conv(x,p,stride=1,pad=0):
   w=v.w[p+'.weight']
   if w.shape[1:3]==(1,1) and stride==1 and pad==0:return x@w[:,0,0,:].T+v.w[p+'.bias']
   return original(x,p,stride,pad)
  v.conv=conv
 return v

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--mode',choices=['conv','res','gn','pointwise','res_safe','all'],default='all');p.add_argument('--pairs',type=int,default=40);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 m=ImageWAM('weights/bf16','assets');ex=Path('examples/robotwin_ep0_frame0');raw=np.load(ex/'inputs.npz');images={k:raw[k] for k in ('cam_high','cam_left_wrist','cam_right_wrist')};x,_=m.preprocess(images,raw['state']);mx.eval(x)
 r=[]
 def once(v):
  mx.synchronize();t=time.perf_counter();z=v(x);mx.eval(z);mx.synchronize();return z,(time.perf_counter()-t)*1000
 for mode in (['conv','res','gn','pointwise'] if a.mode=='all' else [a.mode]):
  v=build(m.vae,mode)
  for _ in range(5):once(m.vae);once(v)
  ts=[[],[]]
  for i in range(a.pairs):
   for j in ((0,1) if i%2==0 else (1,0)):
    z,ms=once([m.vae,v][j]);ts[j].append(ms)
  u,z=[np.array(once(o)[0].astype(mx.float32)) for o in (m.vae,v)]
  d=np.array(ts[0])-ts[1];ci=np.percentile(np.random.default_rng(42).choice(d,(10000,len(d))).mean(1),[2.5,97.5])
  row={'mode':mode,'baseline_ms':float(np.mean(ts[0])),'candidate_ms':float(np.mean(ts[1])),'saving_ci95':ci.tolist(),'equal':u.tobytes()==z.tobytes(),'max_abs':float(np.max(np.abs(u-z))),'times':ts}
  np.savez(a.output/(mode+'_latent.npz'),baseline=u,candidate=z);r.append(row);(a.output/'results.json').write_text(json.dumps(r,indent=2));print({k:v for k,v in row.items() if k!='times'},flush=True)
