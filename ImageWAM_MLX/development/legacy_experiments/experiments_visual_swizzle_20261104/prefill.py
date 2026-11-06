import sys,time,json,copy
from pathlib import Path
import mlx.core as mx
import numpy as np
sys.path.insert(0,str(Path.cwd()))
from imagewam_mlx import ImageWAM
from imagewam_mlx.runtime import RopeWeights
from imagewam_mlx.model import ImageWAM as Base
from candidate import InputVisual as Visual
root=Path('records/visual_swizzle_20261104/prefill');root.mkdir(exist_ok=False)
base=ImageWAM('weights/bf16','assets',profile='speed');models={'baseline':base}
for mode,sw in [('sw1',1),('sw2',2)]:
 tile=(64,64,2,2,64)
 c=copy.copy(base);c.visual=Visual(base.visual.w);c.visual.tile=tile;c.visual.swizzle=sw
 def bind(model):return mx.compile(lambda h,l,s:Base.prefill(model,h,l,s))
 c.prefill=bind(c);models[mode]=c
ex=Path('examples/robotwin_ep0_frame0');raw=np.load(ex/'inputs.npz');images={k:raw[k] for k in ('cam_high','cam_left_wrist','cam_right_wrist')};prompt=(ex/'prompt.txt').read_text();noise=np.load(ex/'initial_noise_fp32.npy')
x,st=base.preprocess(images,raw['state']);hidden=base.qwen(mx.array(base.tokenize(prompt)));latent=base.vae(x);mx.eval(hidden,latent,st)
def run(a,profile=False):
 mx.synchronize();t=time.perf_counter();cache=models[a].prefill(hidden,latent,st);mx.eval(cache);mx.synchronize();dt=(time.perf_counter()-t)*1000
 return cache,dt
for a in models:
 for _ in range(5):run(a)
rows={a:[] for a in models};outs={};rng=np.random.default_rng(73)
for i in range(150):
 for a in rng.permutation(list(models)):
  cache,ms=run(a);rows[a].append(ms);outs[a]=cache
 if i%20==0:print(i,{a:float(np.mean(v)) for a,v in rows.items()},flush=True)
results={}
for a in models:
 cache=outs[a];basecache=outs['baseline'];maxdiff=max(float(mx.max(mx.abs(x.astype(mx.float32)-y.astype(mx.float32))).item()) for pair,ref in zip(cache,basecache) for x,y in zip(pair,ref));d=np.array(rows['baseline'])-rows[a];ci=np.percentile(np.mean(d[rng.integers(0,len(d),(10000,len(d)))],1),[2.5,97.5])
 results[a]=dict(ms=rows[a],mean=float(np.mean(rows[a])),saving=float(d.mean()),ci95=ci.tolist(),cache_max_abs=maxdiff)
(root/'results.json').write_text(json.dumps(results,indent=2));print({a:{k:v for k,v in r.items() if k!='ms'} for a,r in results.items()},flush=True)
