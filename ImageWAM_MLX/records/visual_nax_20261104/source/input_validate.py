import sys,time,json,copy
from pathlib import Path
import mlx.core as mx
import numpy as np
sys.path.insert(0,str(Path.cwd()))
from imagewam_mlx import ImageWAM
from imagewam_mlx.runtime import RopeWeights
from imagewam_mlx.model import ImageWAM as Base
from candidate import InputVisual as Visual
root=Path('records/visual_nax_20261104/input_validation');root.mkdir(exist_ok=False)
base=ImageWAM('weights/bf16','assets',profile='speed');models={'baseline':base}
for mode,tile in [('input64',(64,64,2,2,64))]:
 c=copy.copy(base);c.visual=Visual(base.visual.w);c.visual.tile=tile
 def bind(model):return mx.compile(lambda h,l,s:Base.prefill(model,h,l,s))
 c.prefill=bind(c);models[mode]=c
ex=Path('examples/robotwin_ep0_frame0');raw=np.load(ex/'inputs.npz');images={k:raw[k] for k in ('cam_high','cam_left_wrist','cam_right_wrist')};prompt=(ex/'prompt.txt').read_text();noise=np.load(ex/'initial_noise_fp32.npy')
def run(a,profile=False):
 mx.synchronize();t=time.perf_counter();out,n=models[a].predict(images,raw['state'],prompt,noise,profile=profile);mx.synchronize();assert np.isfinite(out).all();return out,n,(time.perf_counter()-t)*1000

ref=Path('../ImageWAM/evaluate_results/m5pro_robotwin/20261103_104832_dynamic_full');cases=json.loads((ref/'comparison.json').read_text())['cases']
selected=[dict(label="standard",prompt=prompt),min(cases,key=lambda c:base.tokenize(c['prompt']).shape[1]),max(cases,key=lambda c:base.tokenize(c['prompt']).shape[1])]
selected.append(dict(label='long',prompt=selected[-1]['prompt']+' Please carefully follow the instruction.'*20))
results=[]
for ci,case in enumerate(selected):
 prompt=case['prompt']
 for horizon in (1,16,64):
  noise=np.resize(np.load(ex/'initial_noise_fp32.npy'),(1,horizon,14));outs={}
  for name in models:
   a,n,_=run(name);outs[name]=a;np.savez(root/f'case{ci}_h{horizon}_{name}.npz',action=a,normalized=n,token_ids=base.tokenize(prompt))
  diffs={name:dict(max_abs=float(abs(a-outs['baseline']).max()),rmse=float(np.sqrt(np.mean((a-outs['baseline'])**2)))) for name,a in outs.items()}
  results.append(dict(case=case['label'],prompt=prompt,tokens=int(base.tokenize(prompt).shape[1]),horizon=horizon,differences=diffs));print(results[-1],flush=True)
(root/'results.json').write_text(json.dumps(results,indent=2))
