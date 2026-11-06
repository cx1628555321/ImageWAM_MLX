import os
os.environ['MLX_MAX_OPS_PER_BUFFER']='256'
import sys,time,json
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
import numpy as np
import mlx.core as mx
from imagewam_mlx import ImageWAM
from imagewam_mlx.model import ImageWAM as Base
from imagewam_mlx import prepacked_visual as pv
from candidate import Candidate
m=ImageWAM('weights/bf16','assets',profile='speed');orig=pv.kernel;captured={}
def capture(**kw):
 n=kw['inputs'][0].shape[0]
 if n not in captured:captured[n]=dict(kw)
 return orig(**kw)
pv.kernel=capture;m.prefill=lambda h,l,s:Base.prefill(m,h,l,s)
p=Path('examples/robotwin_ep0_frame0');raw=np.load(p/'inputs.npz')
try:
 m.predict({k:raw[k] for k in ('cam_high','cam_left_wrist','cam_right_wrist')},raw['state'],(p/'prompt.txt').read_text(),np.load(p/'initial_noise_fp32.npy'))
finally:pv.kernel=orig
rng=np.random.default_rng(572);result={}
for n,kw in captured.items():
 inputs=kw['inputs'];mx.eval(inputs)
 funcs={}
 for name,fn in [('baseline',orig),('wn2',Candidate(m.visual,2).mm),('wn4',Candidate(m.visual,4).mm)]:
  def f(x,s,e,fn=fn,kw=kw,inputs=inputs):
   d=dict(kw);d['inputs']=[*inputs[:5],x,s,e];return fn(**d)[0]
  funcs[name]=mx.compile(f)
 args=inputs[5:];ref=None;rows={k:[] for k in funcs};errors={}
 for name,f in funcs.items():
  for _ in range(10):y=f(*args);mx.eval(y)
  a=np.array(y.astype(mx.float32))
  if ref is None:ref=a
  errors[name]=dict(exact=bool(np.array_equal(a,ref)),max_abs=float(np.max(np.abs(a.astype(float)-ref.astype(float)))))
  assert errors[name]['exact']
 for i in range(100):
  for name in rng.permutation(list(funcs)):
   mx.synchronize();t=time.perf_counter();y=funcs[name](*args);mx.eval(y);mx.synchronize();rows[name].append((time.perf_counter()-t)*1000)
 stats={}
 for name,v in rows.items():
  d=np.array(rows['baseline'])-v;ci=np.percentile(np.mean(d[rng.integers(0,len(d),(10000,len(d)))],1),[2.5,97.5])
  stats[name]=dict(mean_ms=float(np.mean(v)),saving_ms=float(d.mean()),ci95=ci.tolist(),samples_ms=v,**errors[name])
 result[str(n)]=dict(template=dict(kw['template']),input_shapes=[list(x.shape) for x in inputs],stats=stats)
 print(n,{k:{a:b for a,b in v.items() if a!='samples_ms'} for k,v in stats.items()},flush=True)
Path('records/int8_wide_20261104/probe.json').write_text(json.dumps(result,indent=2,default=str))
