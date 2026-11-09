import sys,time,json,types
from pathlib import Path
import numpy as np
import mlx.core as mx
sys.path.insert(0,str(Path.cwd()))
from imagewam_mlx import ImageWAM
from candidates import Workspace,CompiledQwen
root=Path('records/next_20261104/native_validation');root.mkdir(exist_ok=False)
m=ImageWAM('weights/bf16','assets',profile='speed');
def select(mode):m.kv_workspace=mode!='baseline'
ex=Path('examples/robotwin_ep0_frame0');raw=np.load(ex/'inputs.npz');images={k:raw[k] for k in ('cam_high','cam_left_wrist','cam_right_wrist')};prompt=(ex/'prompt.txt').read_text();noise=np.load(ex/'initial_noise_fp32.npy')
def run(mode):
 select(mode);mx.synchronize();t=time.perf_counter();a,n=m.predict(images,raw['state'],prompt,noise);mx.synchronize();return a,n,(time.perf_counter()-t)*1000

ref=Path('../ImageWAM/evaluate_results/m5pro_robotwin/20261103_104832_dynamic_full');data=json.loads((ref/'comparison.json').read_text());results=[]
for case in data['cases']:
 label=case['label'];raw=np.load(ref/('base_inputs.npz' if label.startswith('length_') else label+'_inputs.npz'));images={k:raw[k] for k in ('cam_high','cam_left_wrist','cam_right_wrist')};prompt=case['prompt']
 a,n,_=run('baseline');b,c,_=run('workspace');err=float(np.max(np.abs(a-b)));exact=np.array_equal(a,b) and np.array_equal(n,c)
 np.savez(root/(label+'.npz'),baseline=a,workspace=b,baseline_normalized=n,workspace_normalized=c)
 results.append(dict(label=label,tokens=int(m.tokenize(prompt).shape[1]),exact=bool(exact),max_abs=err));(root/'results.json').write_text(json.dumps(results,indent=2));assert exact,(label,err)
 print(label,'exact',flush=True)

# Native trace and horizon boundaries, including default accuracy fallback.
extra=[]
for horizon in (1,64):
 z=np.resize(noise,(1,horizon,14));select('baseline');a,n=m.predict(images,raw['state'],prompt,z,steps=3,trace=True);ta={k:v.copy() for k,v in m.trace.items()}
 select('workspace');b,c=m.predict(images,raw['state'],prompt,z,steps=3,trace=True)
 assert np.array_equal(a,b) and np.array_equal(n,c)
 assert all(np.array_equal(v,m.trace[k]) for k,v in ta.items())
 np.savez(root/f'horizon_{horizon}.npz',baseline=a,workspace=b)
 extra.append(dict(horizon=horizon,trace_exact=True))
(root/'extra.json').write_text(json.dumps(extra,indent=2))
import gc
del m;gc.collect();mx.clear_cache()
m=ImageWAM('weights/bf16','assets',profile='accuracy');assert not m.kv_workspace
raw=np.load(ex/'inputs.npz');images={k:raw[k] for k in ('cam_high','cam_left_wrist','cam_right_wrist')};prompt=(ex/'prompt.txt').read_text()
a,n=m.predict(images,raw['state'],prompt,noise);reference=np.load('records/final_default_20261103/reference_action.npy');assert np.array_equal(a,reference)
np.savez(root/'accuracy.npz',action=a,normalized=n)
(root/'completed.json').write_text(json.dumps(dict(cases=len(results),accuracy_exact=True,workspace_default_speed=True),indent=2))
