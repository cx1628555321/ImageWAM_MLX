import sys,time,json,copy
from pathlib import Path
import mlx.core as mx
import numpy as np
sys.path.insert(0,str(Path.cwd()))
from imagewam_mlx import ImageWAM
from imagewam_mlx.runtime import RopeWeights
from imagewam_mlx.model import ImageWAM as Base
from candidate import InputVisual as Visual
root=Path('records/visual_swizzle_20261104/paired');root.mkdir(exist_ok=False)
base=ImageWAM('weights/bf16','assets',profile='speed');models={'baseline':base}
for mode,sw in [('sw2',2)]:
 tile=(64,64,2,2,64)
 c=copy.copy(base);c.visual=Visual(base.visual.w);c.visual.tile=tile;c.visual.swizzle=sw
 def bind(model):return mx.compile(lambda h,l,s:Base.prefill(model,h,l,s))
 c.prefill=bind(c);models[mode]=c
ex=Path('examples/robotwin_ep0_frame0');raw=np.load(ex/'inputs.npz');images={k:raw[k] for k in ('cam_high','cam_left_wrist','cam_right_wrist')};prompt=(ex/'prompt.txt').read_text();noise=np.load(ex/'initial_noise_fp32.npy')
def run(a,profile=False):
 mx.synchronize();t=time.perf_counter();out,n=models[a].predict(images,raw['state'],prompt,noise,profile=profile);mx.synchronize();assert np.isfinite(out).all();return out,n,(time.perf_counter()-t)*1000
for a in models:
 for _ in range(5):run(a)
rows={a:[] for a in models};outs={};rng=np.random.default_rng(73)
for i in range(150):
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
