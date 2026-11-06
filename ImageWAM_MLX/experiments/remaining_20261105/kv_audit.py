import os
os.environ.setdefault('MLX_MAX_OPS_PER_BUFFER','256')
import sys,time,json,gc
from pathlib import Path
import mlx.core as mx
import numpy as np
r=Path('records/remaining_20261105');results=[]
for p in (320,328,368,478):
 for retain in (False,True):
  gc.collect();mx.clear_cache();a=mx.ones((1,24,p+16,128),dtype=mx.bfloat16);u=mx.full((1,24,16,128),2,dtype=mx.bfloat16);idx=mx.array([p]);mx.eval(a,u,idx)
  owned_bytes=a.nbytes;before=mx.get_active_memory();keep=a if retain else None
  b=mx.slice_update(a,u,idx,axes=(2,));del a;mx.eval(b);mx.synchronize();after=mx.get_active_memory()
  assert np.all(np.array(b[:,:,p:,:].astype(mx.float32))==2)
  if retain:assert np.all(np.array(keep.astype(mx.float32))==1)
  results.append(dict(prefix=p,retain=retain,buffer_bytes=owned_bytes,active_delta=after-before,expected_output=True,retained_input_unchanged=True if retain else None))
  del b,u,idx,keep
# Isolated update-chain timing, not end-to-end or a physical copy counter.
rows={};rng=np.random.default_rng(91)
for mode in ('reusable','retained'):
 samples=[]
 for rep in range(25):
  a=mx.ones((1,24,344,128),dtype=mx.bfloat16);u=mx.ones((1,24,16,128),dtype=mx.bfloat16);idx=mx.array([328]);mx.eval(a,u,idx);keep=[];t=time.perf_counter()
  for step in range(10):
   if mode=='retained':keep.append(a)
   a=mx.slice_update(a,u,idx,axes=(2,));mx.eval(a)
  mx.synchronize();samples.append((time.perf_counter()-t)*1000)
 rows[mode]=dict(mean_ms=float(np.mean(samples[5:])),samples_ms=samples)
(r/'kv_audit.json').write_text(json.dumps(dict(memory_cases=results,ten_updates_diagnostic=rows,note='Active allocation deltas support ownership-dependent reuse; not a GPU copy count for the full model.'),indent=2));print(results,rows,flush=True)
