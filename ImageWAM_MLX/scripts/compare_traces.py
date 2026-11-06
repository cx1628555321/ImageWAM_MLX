"""Compare separately generated native PyTorch and native MLX intermediate outputs."""
import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--mlx',type=Path,required=True);p.add_argument('--mps',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();m=np.load(a.mlx);r=np.load(a.mps);rows={}
for key in r.files:
    x,y=m[key].astype(np.float64),r[key].astype(np.float64)
    assert x.shape==y.shape and np.isfinite(x).all()
    d=x-y;rows[key]={'shape':list(x.shape),'max_abs':float(np.abs(d).max()),'mae':float(np.abs(d).mean()),'rmse':float(np.sqrt(np.mean(d*d))),'cosine':float(np.dot(x.ravel(),y.ravel())/(np.linalg.norm(x)*np.linalg.norm(y))) if np.linalg.norm(x)*np.linalg.norm(y)>0 else None}
for key in ('image','state'):assert rows[key]['max_abs']==0, f'Input changed: {key}'
with a.output.open('x') as f:json.dump({'input_preservation_passed':True,'components':rows,'note':'Intermediate numeric differences are reported; no unapproved accuracy threshold or closed-loop equivalence claim.'},f,indent=2)
print(json.dumps(rows,indent=2))
