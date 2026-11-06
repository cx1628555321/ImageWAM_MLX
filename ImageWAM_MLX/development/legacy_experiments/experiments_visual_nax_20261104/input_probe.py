import sys,time,json
from pathlib import Path
import numpy as np
import mlx.core as mx
sys.path.insert(0,str(Path.cwd()))
from candidate import Visual,project
from imagewam_mlx import ImageWAM
from imagewam_mlx.runtime import RopeWeights
from imagewam_mlx.model import ImageWAM as Base
from imagewam_mlx.ops import gated
root=Path('records/visual_nax_20261104/input_probe');root.mkdir(exist_ok=False)
m=ImageWAM('weights/bf16','assets',profile='speed');ex=Path('examples/robotwin_ep0_frame0');raw=np.load(ex/'inputs.npz');images={k:raw[k] for k in ('cam_high','cam_left_wrist','cam_right_wrist')};prompt=(ex/'prompt.txt').read_text()
class Capture(RopeWeights):
 def linear(self,x,p):
  if p=='single_blocks.0.linear1':self.sample=(x,p)
  return super().linear(x,p)
v=Capture(m.visual.w);m.visual=v
x,st=m.preprocess(images,raw['state']);h=m.qwen(mx.array(m.tokenize(prompt)));l=m.vae(x);mx.eval(Base.prefill(m,h,l,st));y,p=v.sample;w=v.w[p+'.weight'];b=v.w.get(p+'.bias',mx.zeros((w.shape[0],),dtype=w.dtype));res=mx.zeros((1,),dtype=y.dtype);gate=res;mx.eval(y,res,gate)
ref=mx.compile(lambda x,r,g:x@w.T+b);fs={'baseline':ref};tiles={}
for tile in [(32,64,2,2,32),(64,64,2,2,32),(64,128,2,2,32),(128,64,2,2,32),(64,64,2,2,64),(64,128,2,2,64),(128,128,2,2,32)]:
 name='_'.join(map(str,tile));tiles[name]=tile
 def bind(t):return mx.compile(lambda x,r,g:project(x,w,b,r,g,t,fused=False))
 fs[name]=bind(tile)
rows={};outs={};times={};rng=np.random.default_rng(22)
for name,fn in fs.items():
 try:
  for _ in range(3):o=fn(y,res,gate);mx.eval(o)
  outs[name]=np.array(o.astype(mx.float32));times[name]=[]
 except Exception as e:rows[name]={'error':str(e)};print(name,str(e)[:1200],flush=True)
for i in range(40):
 for name in rng.permutation(list(times)):
  mx.synchronize();t=time.perf_counter();o=fs[name](y,res,gate);mx.eval(o);mx.synchronize();times[name].append((time.perf_counter()-t)*1000)
for name,t in times.items():
 diff=outs[name]-outs['baseline'];rows[name]=dict(tile=tiles.get(name),mean_ms=float(np.mean(t)),median_ms=float(np.median(t)),max_abs=float(abs(diff).max()),rmse=float(np.sqrt(np.mean(diff*diff))),times=t,finite=bool(np.isfinite(outs[name]).all()));print(name,{k:v for k,v in rows[name].items() if k!='times'},flush=True)
(root/'probe.json').write_text(json.dumps(rows,indent=2));np.savez(root/'probe_inputs.npz',x=np.array(y.astype(mx.float32)),residual=np.array(res.astype(mx.float32)),gate=np.array(gate.astype(mx.float32)));print('DONE',flush=True)
