"""Synchronized operator attribution and GN compilation numerical localization."""
import sys,time,json
from pathlib import Path
import numpy as np
import mlx.core as mx
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from imagewam_mlx import ImageWAM
out=Path('records/vae_deep_diagnostic_20261103');out.mkdir(exist_ok=False)
m=ImageWAM('weights/bf16','assets');v=m.vae
raw=np.load('examples/robotwin_ep0_frame0/inputs.npz');im={k:raw[k] for k in ('cam_high','cam_left_wrist','cam_right_wrist')};x,_=m.preprocess(im,raw['state']);mx.eval(x)
for _ in range(5):mx.eval(v(x))
captured={};rows=[];originals={k:getattr(v,k) for k in ('conv','gn_silu','gn')}
def wrap(kind,fn):
 def call(z,p,*args,**kwargs):
  mx.eval(z);mx.synchronize();start=time.perf_counter();y=fn(z,p,*args,**kwargs);mx.eval(y);mx.synchronize();ms=(time.perf_counter()-start)*1000
  rows.append({'kind':kind,'layer':p,'shape':list(z.shape),'ms':ms})
  if kind=='gn_silu' and p not in captured:captured[p]=z
  return y
 return call
for k,f in originals.items():setattr(v,k,wrap(k,f))
for _ in range(3):mx.eval(v(x))
for k,f in originals.items():setattr(v,k,f)
def stats(z):
 b,h,w,c=z.shape;t=z.astype(mx.float32).reshape(b,h,w,32,c//32);mean=mx.mean(t,axis=(1,2,4),keepdims=True);center=t-mean;sq=center**2;var=mx.mean(sq,axis=(1,2,4),keepdims=True)
 return mean,center,sq,var
compiled=mx.compile(stats);checks=[]
for p,z in captured.items():
 aa=stats(z);bb=compiled(z);mx.eval(aa,bb)
 row={'layer':p,'shape':list(z.shape),'stages':{}}
 for name,a,b in zip(('mean','centered','square','var'),aa,bb):
  a,b=np.array(a),np.array(b);row['stages'][name]={'equal':a.tobytes()==b.tobytes(),'max_abs':float(np.max(np.abs(a-b))),'different':int(np.sum(a!=b))}
 a=originals['gn_silu'](z,p);b=mx.compile(lambda x:originals['gn_silu'](x,p))(z);mx.eval(a,b);a,b=np.array(a.astype(mx.float32)),np.array(b.astype(mx.float32));row['output']={'equal':a.tobytes()==b.tobytes(),'max_abs':float(np.max(np.abs(a-b))),'different':int(np.sum(a!=b))};checks.append(row)
summary={k:sum(r['ms'] for r in rows if r['kind']==k)/3 for k in originals}
r={'operator_rows':rows,'operator_sums_ms':summary,'gn_checks':checks,'note':'Per-operator synchronization perturbs execution: these sums are diagnostic, not an end-to-end GPU profile. Returning intermediates can alter compilation.'}
(out/'results.json').write_text(json.dumps(r,indent=2));print(summary,flush=True)
for c in checks:print(c,flush=True)
