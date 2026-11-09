"""Finite outputs and shape changes for the selected speed profile."""
import sys,json
from pathlib import Path
import numpy as np
import mlx.core as mx
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from imagewam_mlx import ImageWAM
out=Path('records/speed_final_20261103/dynamic_shapes');out.mkdir(parents=True,exist_ok=False)
m=ImageWAM('weights/bf16','assets',profile='speed');ref=Path('examples/robotwin_ep0_frame0');raw=np.load(ref/'inputs.npz');images={k:raw[k] for k in ('cam_high','cam_left_wrist','cam_right_wrist')};prompt=(ref/'prompt.txt').read_text();rng=np.random.default_rng(42);rows=[]
for horizon,steps in [(1,1),(8,5),(16,10),(32,3),(64,2),(16,10)]:
 noise=rng.standard_normal((1,horizon,14)).astype(np.float32)
 action,norm=m.predict(images,raw['state'],prompt,noise,steps=steps)
 assert action.shape==(horizon,14) and np.isfinite(action).all() and np.isfinite(norm).all()
 np.savez(out/f'case_{len(rows)}.npz',action=action,normalized=norm,noise=noise)
 rows.append(dict(horizon=horizon,steps=steps,finite=True))
(out/'results.json').write_text(json.dumps(dict(status='completed',cases=rows),indent=2))
