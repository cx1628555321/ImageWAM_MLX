import os
os.environ['MLX_MAX_OPS_PER_BUFFER']='256'
import sys,time,json,copy,functools
from pathlib import Path
sys.path.insert(0,str(Path.cwd()));sys.path.insert(0,str(Path('experiments/roadmap_20261105').resolve()))
import numpy as np
import mlx.core as mx
from common import inputs
from candidate import fused
from imagewam_mlx import ImageWAM
from imagewam_mlx import action_kernel as ak
from imagewam_mlx.workspace import workspace_step
r=Path('records/split_reduce_20261105');r.mkdir(exist_ok=True)
p=Path('records/roadmap_20261105/capture');d=mx.load(str(p/'projection_inputs.npz'));kw=json.loads((p/'projection_parameters.json').read_text());a=[d['x'],[d[f'part_{i}'] for i in range(3)],d['residual'],d['gate'],d['bias']]
modes={'baseline':ak.fused,'u1_256':functools.partial(fused,vec=1,tg=256),'u1_128':functools.partial(fused,vec=1,tg=128),'u4_128':functools.partial(fused,vec=4,tg=128)}
def make(fn):
 c=mx.compile(lambda x,w,s,q,r,g,b:fn(x,[w,s,q],r,g,b,**kw))
 return lambda:c(a[0],*a[1],*a[2:])
fns={k:make(fn) for k,fn in modes.items()};ref=fns['baseline']();mx.eval(ref);micro={}
for k,fn in fns.items():
 for _ in range(15):mx.eval(fn())
 micro[k]={'exact':bool(mx.array_equal(fn(),ref).item()),'ms':[]}
rng=np.random.default_rng(5118)
for _ in range(200):
 for k in rng.permutation(list(fns)):
  t=time.perf_counter();mx.eval(fns[k]());micro[k]['ms'].append((time.perf_counter()-t)*1000)
(r/'micro.json').write_text(json.dumps(micro,indent=2));print('micro',{k:(float(np.mean(v['ms'])),v['exact']) for k,v in micro.items()},flush=True)
