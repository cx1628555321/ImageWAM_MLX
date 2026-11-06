import os
os.environ['MLX_MAX_OPS_PER_BUFFER']='256'
import sys,json,types
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
import numpy as np
from imagewam_mlx import ImageWAM
from submit import denoise
root=Path('records/graph_dispatch_20261105/submit_dynamic');root.mkdir(exist_ok=False)
m=ImageWAM('weights/bf16','assets',profile='speed');original=m.denoise
p=Path('examples/robotwin_ep0_frame0');raw=np.load(p/'inputs.npz');images={k:raw[k] for k in ('cam_high','cam_left_wrist','cam_right_wrist')};prompt=(p/'prompt.txt').read_text();seed=np.load(p/'initial_noise_fp32.npy');rows=[]
for i,(text,h,steps) in enumerate([(prompt,1,1),(prompt,8,3),(prompt+' Please carefully follow the instruction.'*25,16,10),(prompt,64,11)]):
 noise=np.resize(seed,(1,h,14));m.denoise=original;a,n=m.predict(images,raw['state'],text,noise,steps=steps)
 for name,interval in [('xonly',1),('every5',5)]:
  m.submit_interval=interval;m.denoise=types.MethodType(denoise,m);b,z=m.predict(images,raw['state'],text,noise,steps=steps)
  assert np.array_equal(a,b) and np.array_equal(n,z),(i,name)
  np.savez(root/f'{i}_{name}.npz',baseline=a,candidate=b,normalized=z,noise=noise)
  rows.append(dict(tokens=int(m.tokenize(text).shape[1]),horizon=h,steps=steps,mode=name,exact=True))
(root/'results.json').write_text(json.dumps(rows,indent=2));print(rows,flush=True)
