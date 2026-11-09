"""Test noise/seed and horizon variability, beyond the benchmark's fixed shape."""
import argparse,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from compile_experiments import Candidates,error
from imagewam_mlx import ImageWAM
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
ex=Path('examples/robotwin_ep0_frame0');raw=np.load(ex/'inputs.npz');images={k:raw[k] for k in ('cam_high','cam_left_wrist','cam_right_wrist')};pr=(ex/'prompt.txt').read_text();model=ImageWAM('weights/bf16','assets');c=Candidates(model);rows=[]
for horizon,seed in ((8,0),(16,7),(32,42),(16,123)):
 noise=np.random.default_rng(seed).standard_normal((1,horizon,14)).astype(np.float32);out={}
 for mode in ('original','both_compile'):
  c.set(mode);x,n=model.predict(images,raw['state'],pr,noise);out[mode]=x;np.save(a.output/f'h{horizon}_s{seed}_{mode}.npy',x)
 assert out['both_compile'].shape==(horizon,14)
 rows.append({'horizon':horizon,'numpy_seed':seed,'bitwise_equal':bool(np.array_equal(out['original'],out['both_compile'])),'error':error(out['both_compile'],out['original'])})
(a.output/'variability.json').write_text(json.dumps({'cases':rows},indent=2));print(json.dumps(rows),flush=True)
