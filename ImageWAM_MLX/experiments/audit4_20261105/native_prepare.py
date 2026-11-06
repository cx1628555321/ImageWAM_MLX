import json,time
from pathlib import Path
import mlx.core as mx
from imagewam_mlx.action_kernel import fused
p=Path('records/roadmap_20261105/capture');r=Path('records/audit4_20261105/export')
d=mx.load(str(p/'projection_inputs.npz'));mx.save_safetensors(str(r/'inputs.safetensors'),d)
a=[d[k] for k in ('x','part_0','part_1','part_2','residual','gate','bias')];mx.eval(*a)
kw=json.loads((p/'projection_parameters.json').read_text());imp=mx.import_function(str(r/'real_projection.mlxfn'))
def original():return fused(a[0],a[1:4],*a[4:],**kw)
compiled=mx.compile(lambda *args:fused(args[0],args[1:4],*args[4:],**kw))
result={}
for name,fn in [('original',original),('compiled',lambda:compiled(*a)),('imported',lambda:imp(*a)[0])]:
 for _ in range(20):mx.eval(fn())
 times=[]
 for _ in range(200):
  t=time.perf_counter();y=fn();mx.eval(y);times.append((time.perf_counter()-t)*1000)
 result[name]=times
mx.save(str(r/'python_output.npy'),original());(r/'python_timings.json').write_text(json.dumps(result))

import random
paired=[];rng=random.Random(5117)
for i in range(300):
 order=['compiled','imported'];rng.shuffle(order);row={}
 for mode in order:
  t=time.perf_counter();y=compiled(*a) if mode=='compiled' else imp(*a)[0];mx.eval(y);row[mode]=(time.perf_counter()-t)*1000
 paired.append(row)
(r/'paired_import.json').write_text(json.dumps(paired))
