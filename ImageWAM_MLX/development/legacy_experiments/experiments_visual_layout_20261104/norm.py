import sys,time,json,copy
from pathlib import Path
import mlx.core as mx
import numpy as np
sys.path.insert(0,str(Path.cwd()))
from imagewam_mlx import ImageWAM
from imagewam_mlx.runtime import RopeWeights
from imagewam_mlx.model import ImageWAM as Base
from imagewam_mlx.ops import gated
class Flat(RopeWeights):
 def modnorm(self,x,shift,scale):
  return mx.fast.layer_norm(x,(1+scale).reshape(-1),shift.reshape(-1),eps=1e-6)
 def double_prepare(self,x,p,m,pe,stream='img'):
  if self.mode=='single_norm':return super().double_prepare(x,p,m,pe,stream)
  z=self.modnorm(x,m[0],m[1]);return self.qkv(self.linear(z,p+'.'+stream+'_attn.qkv'),p+'.'+stream+'_attn.norm',pe)
 def double_post(self,x,att,p,m,stream='img'):
  if self.mode=='single_norm':return super().double_post(x,att,p,m,stream)
  x=x+m[2]*self.linear(att,p+'.'+stream+'_attn.proj');z=self.modnorm(x,m[3],m[4])
  return x+m[5]*self.linear(gated(self.linear(z,p+'.'+stream+'_mlp.0')),p+'.'+stream+'_mlp.2')
 def single_prepare(self,x,p,m,pe):
  z=self.linear(self.modnorm(x,m[0],m[1]),p+'.linear1');q,k,v=self.qkv(z[...,:9216],p+'.norm',pe)
  return q,k,v,z[...,9216:]
root=Path('records/visual_layout_20261104/norm');root.mkdir(exist_ok=False)
base=ImageWAM('weights/bf16','assets',profile='speed');models={'baseline':base}
for mode in ('single_norm','all_norm'):
 c=copy.copy(base);c.visual=Flat(base.visual.w);c.visual.mode=mode
 def bind(model):return mx.compile(lambda h,l,s:Base.prefill(model,h,l,s))
 c.prefill=bind(c);models[mode]=c
ex=Path('examples/robotwin_ep0_frame0');raw=np.load(ex/'inputs.npz');images={k:raw[k] for k in ('cam_high','cam_left_wrist','cam_right_wrist')};prompt=(ex/'prompt.txt').read_text();noise=np.load(ex/'initial_noise_fp32.npy')
def run(a,profile=False):
 mx.synchronize();t=time.perf_counter();out,n=models[a].predict(images,raw['state'],prompt,noise,profile=profile);mx.synchronize();assert np.isfinite(out).all();return out,n,(time.perf_counter()-t)*1000
for a in models:
 for _ in range(5):run(a)
rows={a:[] for a in models};outs={};rng=np.random.default_rng(73)
for i in range(100):
 for a in rng.permutation(list(models)):
  out,n,ms=run(a);rows[a].append(ms);outs[a]=(out,n)
 if i%20==0:print(i,{a:float(np.mean(v)) for a,v in rows.items()},flush=True)
results={}
for a in models:
 out,n=outs[a];np.savez(root/f'{a}.npz',action=out,normalized=n)
 diff=out-outs['baseline'][0];d=np.array(rows['baseline'])-rows[a];ci=np.percentile(np.mean(d[rng.integers(0,len(d),(10000,len(d)))],1),[2.5,97.5]);prof=[]
 for _ in range(5):run(a,True);prof.append(dict(models[a].profile))
 results[a]=dict(ms=rows[a],mean=float(np.mean(rows[a])),saving=float(d.mean()),ci95=ci.tolist(),max_abs=float(abs(diff).max()),rmse=float(np.sqrt(np.mean(diff*diff))),profiles=prof)
(root/'results.json').write_text(json.dumps(results,indent=2));print({a:{k:v for k,v in r.items() if k not in ('ms','profiles')} for a,r in results.items()},flush=True)
