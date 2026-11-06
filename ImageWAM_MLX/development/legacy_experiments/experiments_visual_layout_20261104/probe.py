import sys,time,json,collections
from pathlib import Path
import numpy as np
import mlx.core as mx
sys.path.insert(0,str(Path.cwd()))
from imagewam_mlx import ImageWAM
from imagewam_mlx.runtime import RopeWeights
from imagewam_mlx.model import ImageWAM as Base
root=Path('records/visual_layout_20261104');root.mkdir(exist_ok=False)
m=ImageWAM('weights/bf16','assets',profile='speed')
ex=Path('examples/robotwin_ep0_frame0');raw=np.load(ex/'inputs.npz');images={k:raw[k] for k in ('cam_high','cam_left_wrist','cam_right_wrist')};prompt=(ex/'prompt.txt').read_text()
x,st=m.preprocess(images,raw['state']);h=m.qwen(mx.array(m.tokenize(prompt)));l=m.vae(x);mx.eval(h,l,st)
class Capture(RopeWeights):
 def linear(self,x,p):
  mx.eval(x);key=str((tuple(x.shape),tuple(self.w[p+'.weight'].shape)))
  self.count[key]+=1
  if key not in self.samples:self.samples[key]=(x,p)
  return super().linear(x,p)
v=Capture(m.visual.w);v.count=collections.Counter();v.samples={};m.visual=v
mx.eval(Base.prefill(m,h,l,st));rows=[];rng=np.random.default_rng(52)
for key,(x,p) in v.samples.items():
 w=v.w[p+'.weight'];b=v.w.get(p+'.bias');wt=mx.contiguous(w.T);mx.eval(wt)
 def original(x):return x@w.T
 def flat(x):return (x.reshape(-1,x.shape[-1])@w.T).reshape(*x.shape[:-1],w.shape[0])
 def packed(x):return x@wt
 fs={n:mx.compile(f) for n,f in [('original',original),('flat',flat),('packed',packed)]}
 vals={};times={n:[] for n in fs}
 for n,f in fs.items():
  for _ in range(3):mx.eval(f(x))
 for i in range(20):
  for n in rng.permutation(list(fs)):
   mx.synchronize();t=time.perf_counter();y=fs[n](x);mx.eval(y);mx.synchronize();times[n].append((time.perf_counter()-t)*1000);vals[n]=y
 ref=np.array(vals['original'].astype(mx.float32))
 row=dict(shape=key,path=p,count=v.count[key],ms={n:float(np.median(t)) for n,t in times.items()},samples=times,max_abs={n:float(abs(np.array(y.astype(mx.float32))-ref).max()) for n,y in vals.items()})
 rows.append(row);print({k:z for k,z in row.items() if k!='samples'},flush=True)
 del fs,wt
(root/'probe.json').write_text(json.dumps(rows,indent=2));print('DONE',flush=True)
