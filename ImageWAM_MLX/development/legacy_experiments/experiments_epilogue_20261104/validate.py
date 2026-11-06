import sys,time,json,types
from pathlib import Path
import numpy as np
import mlx.core as mx
sys.path.insert(0,str(Path.cwd()))
from imagewam_mlx import ImageWAM
import candidates
from split import fused
candidates.fused=fused
root=Path('records/epilogue_20261104/validation');root.mkdir(exist_ok=False)
m=ImageWAM('weights/bf16','assets',profile='speed');
candidate=candidates.install(m)
def select(mode):return m if mode=='baseline' else candidate
ex=Path('examples/robotwin_ep0_frame0');raw=np.load(ex/'inputs.npz');images={k:raw[k] for k in ('cam_high','cam_left_wrist','cam_right_wrist')};prompt=(ex/'prompt.txt').read_text();noise=np.load(ex/'initial_noise_fp32.npy')
def run(mode):
 model=select(mode);mx.synchronize();t=time.perf_counter();a,n=model.predict(images,raw['state'],prompt,noise);mx.synchronize();return a,n,(time.perf_counter()-t)*1000

ref=Path('../ImageWAM/evaluate_results/m5pro_robotwin/20261103_104832_dynamic_full');data=json.loads((ref/'comparison.json').read_text());results=[]
for case in data['cases']:
 label=case['label'];raw=np.load(ref/('base_inputs.npz' if label.startswith('length_') else label+'_inputs.npz'));images={k:raw[k] for k in ('cam_high','cam_left_wrist','cam_right_wrist')};prompt=case['prompt']
 a,n,_=run('baseline');b,c,_=run('workspace');err=float(np.max(np.abs(a-b)));exact=np.array_equal(a,b) and np.array_equal(n,c)
 np.savez(root/(label+'.npz'),baseline=a,workspace=b,baseline_normalized=n,workspace_normalized=c)
 results.append(dict(label=label,tokens=int(m.tokenize(prompt).shape[1]),exact=bool(exact),max_abs=err));(root/'results.json').write_text(json.dumps(results,indent=2));assert np.isfinite(a).all() and np.isfinite(b).all(),label
 print(label,err,flush=True)

(root/'completed.json').write_text(json.dumps(dict(cases=len(results),max_abs=max(v['max_abs'] for v in results),exact_count=sum(v['exact'] for v in results)),indent=2))
