import os
os.environ.setdefault('MLX_MAX_OPS_PER_BUFFER','256')
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
import mlx.core as mx
import numpy as np
from imagewam_mlx import ImageWAM
from common import install,predict
r=Path('records/remaining_20261105/dynamic');r.mkdir(exist_ok=False)
base=ImageWAM('weights/bf16','assets',profile='speed');models={k:install(base,k)[0] for k in ('static','packed','cache_hit')}
p=Path('examples/robotwin_ep0_frame0');raw=np.load(p/'inputs.npz');prompt=(p/'prompt.txt').read_text();noise=np.load(p/'initial_noise_fp32.npy');rows=[]
cases=[(prompt,1,1,False),(prompt,8,3,False),(prompt+' Please carefully follow the instruction.'*25,16,10,False),(prompt,64,11,False),(prompt,16,10,True)]
for i,(text,h,steps,changed) in enumerate(cases):
 images={k:raw[k].copy() for k in ('cam_high','cam_left_wrist','cam_right_wrist')};state=raw['state'].copy()
 if changed:
  images['cam_high']=np.flip(images['cam_high'],axis=1).copy();state[0]+=0.1
 args=dict(images=images,state=state,prompt=text,noise=np.resize(noise,(1,h,14)),steps=steps)
 a,n=base.predict(**args)
 for mode,m in models.items():
  for repeat in range(2 if mode=='cache_hit' else 1):
   b,z=m.predict(**args);assert np.array_equal(a,b) and np.array_equal(n,z),(i,mode)
  np.savez(r/f'{i}_{mode}.npz',action=b,normalized=z,baseline=a)
 rows.append(dict(tokens=int(base.tokenize(text).shape[1]),horizon=h,steps=steps,changed_image_state=changed,exact=True))
(r/'results.json').write_text(json.dumps(rows,indent=2));print(rows,flush=True)
