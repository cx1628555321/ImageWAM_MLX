import os
os.environ['MLX_MAX_OPS_PER_BUFFER']='256'
import sys,time,json,copy
from pathlib import Path
sys.path.insert(0,str(Path.cwd()));sys.path.insert(0,str(Path('experiments/roadmap_20261105').resolve()))
import numpy as np
import mlx.core as mx
from common import inputs
from imagewam_mlx import ImageWAM
r=Path('records/text_prepare_20261105/full');r.mkdir(exist_ok=False)
t=time.perf_counter();m=ImageWAM('weights/bf16','assets',profile='speed');mx.synchronize();load=time.perf_counter()-t
b=copy.copy(m);b.qwen=m.qwen.encoder;b.text_cache_size=0
args=inputs();a=args['prompt'];other=a+' Please move carefully.';long=a+' Please carefully follow the instruction.'*25
initial=m.prepare_text(a);checks=[]
for i,(prompt,alter) in enumerate([(a,False),(a,True),(other,False),(other,True),(a,False),(long,False),(long,True)]):
 args=inputs();args['prompt']=prompt
 if alter:
  args['state']=args['state'].copy();args['state'][0]+=.1
  args['images']['cam_high']=255-args['images']['cam_high']
 before=m.text_cache_stats();t=time.perf_counter();out,z=m.predict(**args,steps=10);mx.synchronize();ms=(time.perf_counter()-t)*1000;after=m.text_cache_stats()
 ref,n=b.predict(**args,steps=10)
 exact=bool(np.array_equal(out,ref) and np.array_equal(z,n));assert exact
 np.savez(r/f'case_{i}.npz',action=out,normalized=z,reference=ref,reference_normalized=n)
 checks.append(dict(index=i,tokens=int(m.tokenize(prompt).shape[1]),changed_observation=alter,exact=exact,before=before,after=after,ms=ms))
 print('case',i,exact,flush=True)
args=inputs();prep=m.prepare_text(a)
for _ in range(5):m.predict(**args);b.predict(**args)
rows={'cached':[],'uncached':[]};rng=np.random.default_rng(5119)
for i in range(30):
 for key in rng.permutation(list(rows)):
  model=m if key=='cached' else b;mx.synchronize();t=time.perf_counter();out,z=model.predict(**args,steps=10);mx.synchronize();rows[key].append((time.perf_counter()-t)*1000)
result=dict(load_s=load,initial_prepare=initial,checks=checks,prepare_before_timing=prep,performance={k:dict(mean_ms=float(np.mean(v)),p95_ms=float(np.percentile(v,95)),samples_ms=v) for k,v in rows.items()},cache=m.text_cache_stats())
(r/'results.json').write_text(json.dumps(result,indent=2));print('DONE',{k:v['mean_ms'] for k,v in result['performance'].items()},flush=True)
