import os
os.environ['MLX_MAX_OPS_PER_BUFFER']='256'
import sys,json,copy
import mlx.core as mx
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path.cwd()))
from imagewam_mlx import ImageWAM
r=Path('records/int8_pipeline_20261105/dynamic');r.mkdir(exist_ok=True)
a=ImageWAM('weights/bf16','assets',profile='speed')
from candidate import Candidate
from imagewam_mlx.model import ImageWAM as Base
b=copy.copy(a);b.visual=Candidate(a.visual);b.prefill=mx.compile(lambda h,l,s:Base.prefill(b,h,l,s))
ex=Path('examples/robotwin_ep0_frame0');raw=np.load(ex/'inputs.npz');prompt=(ex/'prompt.txt').read_text();rows=[]
for i,(p,h,steps) in enumerate([(prompt,1,1),(prompt,8,3),(prompt+' Please carefully follow the instruction.'*25,16,10),(prompt,64,11)]):
 noise=np.resize(np.load(ex/'initial_noise_fp32.npy'),(1,h,14));args=dict(images={k:raw[k] for k in ('cam_high','cam_left_wrist','cam_right_wrist')},state=raw['state'],prompt=p,noise=noise,steps=steps)
 x,n=a.predict(**args);y,z=b.predict(**args);assert np.array_equal(x,y) and np.array_equal(n,z)
 np.savez(r/f'case{i}.npz',baseline=x,candidate=y,normalized=z,noise=noise);rows.append(dict(tokens=int(a.tokenize(p).shape[1]),horizon=h,steps=steps,exact=True))
(r/'results.json').write_text(json.dumps(rows,indent=2));print(rows,flush=True)
