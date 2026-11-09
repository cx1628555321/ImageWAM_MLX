import sys,time,json,types
from pathlib import Path
import numpy as np
import mlx.core as mx
sys.path.insert(0,str(Path.cwd()))
from imagewam_mlx import ImageWAM
from candidates import Workspace,CompiledQwen
root=Path('records/next_20261104/native_paired');root.mkdir(exist_ok=False)
m=ImageWAM('weights/bf16','assets',profile='speed');
def select(mode):m.kv_workspace=mode!='baseline'
ex=Path('examples/robotwin_ep0_frame0');raw=np.load(ex/'inputs.npz');images={k:raw[k] for k in ('cam_high','cam_left_wrist','cam_right_wrist')};prompt=(ex/'prompt.txt').read_text();noise=np.load(ex/'initial_noise_fp32.npy')
def run(mode):
 select(mode);mx.synchronize();t=time.perf_counter();a,n=m.predict(images,raw['state'],prompt,noise);mx.synchronize();return a,n,(time.perf_counter()-t)*1000
results={}
for mode in ('workspace',):
 for _ in range(4):run('baseline');run(mode)
 rows=[];maxerr=0
 for i in range(100):
  pair={}
  for v in (('baseline',mode) if i%2==0 else (mode,'baseline')):
   a,n,ms=run(v);pair[v]=(a,n,ms)
  maxerr=max(maxerr,float(np.max(np.abs(pair['baseline'][0]-pair[mode][0]))));rows.append([pair['baseline'][2],pair[mode][2]])
  assert np.array_equal(pair['baseline'][0],pair[mode][0]) and np.array_equal(pair['baseline'][1],pair[mode][1]),(mode,maxerr)
 np.savez(root/(mode+'_output.npz'),action=a,normalized=n)
 x=np.array(rows);diff=x[:,0]-x[:,1];rng=np.random.default_rng(42);ci=np.percentile(np.mean(diff[rng.integers(0,len(diff),(10000,len(diff)))],axis=1),[2.5,97.5])
 results[mode]=dict(pairs=rows,baseline_mean=float(x[:,0].mean()),candidate_mean=float(x[:,1].mean()),saving_ms=float(diff.mean()),ci95_ms=ci.tolist(),exact=True,max_abs=maxerr)
 (root/'results.json').write_text(json.dumps(results,indent=2));print(mode,{k:v for k,v in results[mode].items() if k!='pairs'},flush=True)
