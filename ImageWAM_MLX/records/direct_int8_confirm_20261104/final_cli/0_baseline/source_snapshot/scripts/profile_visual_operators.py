"""Synchronized eager operator attribution; NOT GPU kernel or compiled timings."""
import sys,time,json,collections,functools
from pathlib import Path
import numpy as np
import mlx.core as mx
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import imagewam_mlx.ops as ops
import imagewam_mlx.model as modelmod
import imagewam_mlx.runtime as runtime
out=Path('records/next_optimizations_20261103/visual_operator_profile');out.mkdir(exist_ok=False)
m=runtime.ImageWAM('weights/bf16','assets');raw=np.load('examples/robotwin_ep0_frame0/inputs.npz');images={k:raw[k] for k in ('cam_high','cam_left_wrist','cam_right_wrist')};prompt=Path('examples/robotwin_ep0_frame0/prompt.txt').read_text()
x,st=m.preprocess(images,raw['state']);hidden=m.qwen(mx.array(m.tokenize(prompt)));latent=m.vae(x);mx.eval(hidden,latent,st)
ref=m.prefill(hidden,latent,st);mx.eval(ref)
rows=[]
def record(name,fn):
 @functools.wraps(fn)
 def f(*args,**kw):
  # Flush previous work before measuring this operator. Deliberately serialized.
  mx.synchronize();start=time.perf_counter();y=fn(*args,**kw);mx.eval(y);mx.synchronize()
  rows.append({'name':name,'ms':(time.perf_counter()-start)*1000});return y
 return f
m.visual.linear=record('linear',m.visual.linear)
ops.norm=record('norm',ops.norm);modelmod.norm=ops.norm
ops.rms=record('rms',ops.rms);runtime.rms=ops.rms
runtime.fused_rotate=record('rope',runtime.fused_rotate)
modelmod.attention=record('attention',modelmod.attention)
ops.gated=record('gated',ops.gated)
for _ in range(3):
 cache=modelmod.ImageWAM.prefill(m,hidden,latent,st);mx.eval(cache)
 assert all(np.array_equal(np.array(a.astype(mx.float32)),np.array(b.astype(mx.float32))) for pair1,pair2 in zip(ref,cache) for a,b in zip(pair1,pair2))
sums=collections.defaultdict(float);counts=collections.Counter()
for r in rows:sums[r['name']]+=r['ms'];counts[r['name']]+=1
result={'warning':'Eager graph with forced per-op synchronization; not production compiled latency or GPU kernel timeline. Residual and concat work may be charged to subsequent consumer.','iterations':3,'rows':rows,'category_mean_total_ms':{k:v/3 for k,v in sums.items()},'calls_per_iteration':{k:v/3 for k,v in counts.items()},'cache_equal':True}
(out/'results.json').write_text(json.dumps(result,indent=2));print(result['category_mean_total_ms'])
