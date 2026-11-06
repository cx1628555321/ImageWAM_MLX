import os
os.environ.setdefault('MLX_MAX_OPS_PER_BUFFER','256')
import sys,copy,json
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
import numpy as np
import mlx.core as mx
from imagewam_mlx import ImageWAM
from imagewam_mlx.model import ImageWAM as Base
from imagewam_mlx.gate_quant_visual import GateQuantVisual
root=Path('records/int8_tail_20261105/default_validation');root.mkdir(exist_ok=False)
new=ImageWAM('weights/bf16','assets',profile='speed');assert type(new.visual).__name__=='TailVisual'
old=copy.copy(new);old.visual=GateQuantVisual.__new__(GateQuantVisual);old.visual.__dict__.update(new.visual.__dict__);old.prefill=mx.compile(Base.prefill.__get__(old,type(old)))
ref=Path('../ImageWAM/evaluate_results/m5pro_robotwin/20261103_104832_dynamic_full');cases=json.loads((ref/'comparison.json').read_text())['cases'];noise=np.load('examples/robotwin_ep0_frame0/initial_noise_fp32.npy');rows=[]
def check(label,raw,prompt,noise,steps):
 args=dict(images={k:raw[k] for k in ('cam_high','cam_left_wrist','cam_right_wrist')},state=raw['state'],prompt=prompt,noise=noise,steps=steps)
 a,n=old.predict(**args);b,z=new.predict(**args)
 assert np.array_equal(a,b) and np.array_equal(n,z),label
 np.savez(root/(label+'.npz'),baseline=a,action=b,normalized=z,noise=noise)
 rows.append(dict(label=label,tokens=int(new.tokenize(prompt).shape[1]),horizon=int(noise.shape[1]),steps=steps,exact=True))
for c in cases:
 label=c['label'];raw=np.load(ref/('base_inputs.npz' if label.startswith('length_') else label+'_inputs.npz'));check(label,raw,c['prompt'],noise,10)
ex=Path('examples/robotwin_ep0_frame0');raw=np.load(ex/'inputs.npz');prompt=(ex/'prompt.txt').read_text()
for i,(p,h,s) in enumerate([(prompt,1,1),(prompt,8,3),(prompt+' Please carefully follow the instruction.'*25,16,10),(prompt,64,11)]):check('dynamic'+str(i),raw,p,np.resize(noise,(1,h,14)),s)
(root/'results.json').write_text(json.dumps(rows,indent=2));print('DEFAULT ALL EXACT',len(rows),flush=True)
