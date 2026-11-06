import sys,time,json,copy
from pathlib import Path
import mlx.core as mx
import numpy as np
sys.path.insert(0,str(Path.cwd()))
from imagewam_mlx import ImageWAM
from imagewam_mlx.runtime import RopeWeights
from imagewam_mlx.model import ImageWAM as Base
class PadVisual(RopeWeights):
 def linear(self,x,p):
  if x.ndim==3 and x.shape[0]==1 and x.shape[1]>=128:
   m=x.shape[1];pad=(-m)%self.align
   if pad:return super().linear(mx.pad(x,[(0,0),(0,pad),(0,0)]),p)[:,:m]
  return super().linear(x,p)
root=Path('records/best_native_20261104/next_visual');root.mkdir(exist_ok=False)
base=ImageWAM('weights/bf16','assets',profile='speed');models={0:base}
for a in (32,64):
 c=copy.copy(base);c.visual=PadVisual(base.visual.w);c.visual.align=a
 def bind(model):return mx.compile(lambda h,l,s:Base.prefill(model,h,l,s))
 c.prefill=bind(c);models[a]=c
ex=Path('examples/robotwin_ep0_frame0');raw=np.load(ex/'inputs.npz');images={k:raw[k] for k in ('cam_high','cam_left_wrist','cam_right_wrist')};prompt=(ex/'prompt.txt').read_text();noise=np.load(ex/'initial_noise_fp32.npy')
def run(a):
 mx.synchronize();t=time.perf_counter();out,n=models[a].predict(images,raw['state'],prompt,noise);mx.synchronize();assert np.isfinite(out).all();return out,n,(time.perf_counter()-t)*1000
for a in models:
 for _ in range(4):run(a)
rows={a:[] for a in models};outs={};rng=np.random.default_rng(24)
for i in range(40):
 for a in rng.permutation(list(models)):
  out,n,ms=run(int(a));rows[int(a)].append(ms);outs[int(a)]=(out,n)
 if i%10==0:print(i,{a:float(np.mean(v)) for a,v in rows.items()},flush=True)
results={}
for a in models:
 out,n=outs[a];np.savez(root/f'align{a}.npz',action=out,normalized=n)
 diff=out-outs[0][0];d=np.array(rows[0])-rows[a];ci=np.percentile(np.mean(d[rng.integers(0,len(d),(10000,len(d)))],1),[2.5,97.5])
 results[a]=dict(ms=rows[a],mean=float(np.mean(rows[a])),saving=float(d.mean()),ci95=ci.tolist(),max_abs=float(abs(diff).max()),rmse=float(np.sqrt(np.mean(diff*diff))))
(root/'results.json').write_text(json.dumps(results,indent=2));print({a:{k:v for k,v in r.items() if k!='ms'} for a,r in results.items()},flush=True)
