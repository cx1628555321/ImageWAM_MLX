import sys,json,time,os,argparse
from pathlib import Path
import numpy as np
import mlx.core as mx
sys.path.insert(0,str(Path.cwd()))
from imagewam_mlx import ImageWAM
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--iterations',type=int,default=20);a=p.parse_args();root=Path(a.output);root.mkdir(parents=True,exist_ok=False)
t=time.perf_counter();m=ImageWAM('weights/bf16','assets',profile='speed',staged_visual=False);mx.synchronize();load=time.perf_counter()-t
ex=Path('examples/robotwin_ep0_frame0');r=np.load(ex/'inputs.npz');images={k:r[k] for k in ('cam_high','cam_left_wrist','cam_right_wrist')};prompt=(ex/'prompt.txt').read_text();noise=np.load(ex/'initial_noise_fp32.npy')
def run():
 mx.synchronize();t=time.perf_counter();out,n=m.predict(images,r['state'],prompt,noise);mx.synchronize();assert np.isfinite(out).all();return out,n,(time.perf_counter()-t)*1000
_,_,first=run()
for _ in range(4):run()
rows=[]
for _ in range(a.iterations):out,n,dt=run();rows.append(dt)
np.savez(root/'output.npz',action=out,normalized=n)
result=dict(load_s=load,first_ms=first,mean_ms=float(np.mean(rows)),median_ms=float(np.median(rows)),p95_ms=float(np.percentile(rows,95)),samples_ms=rows,env={k:v for k,v in os.environ.items() if k.startswith('MLX_')},peak_bytes=mx.get_peak_memory(),device=mx.device_info());(root/'results.json').write_text(json.dumps(result,indent=2));print(result['mean_ms'],flush=True)
