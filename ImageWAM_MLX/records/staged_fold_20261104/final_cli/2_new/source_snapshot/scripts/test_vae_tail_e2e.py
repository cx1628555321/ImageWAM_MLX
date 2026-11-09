import sys,time,json,copy,argparse
from pathlib import Path
import numpy as np
import mlx.core as mx
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from imagewam_mlx import ImageWAM
from imagewam_mlx.compiled_vae import CompiledResidualVAE
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--pairs',type=int,default=100);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
b=ImageWAM('weights/bf16','assets');c=copy.copy(b);c.vae=CompiledResidualVAE.from_vae(b.vae)
ex=Path('examples/robotwin_ep0_frame0');raw=np.load(ex/'inputs.npz');im={k:raw[k] for k in ('cam_high','cam_left_wrist','cam_right_wrist')};pr=(ex/'prompt.txt').read_text();n=np.load(ex/'initial_noise_fp32.npy')
def once(m):
 mx.synchronize();t=time.perf_counter();out,norm=m.predict(im,raw['state'],pr,n);mx.synchronize();return out,(time.perf_counter()-t)*1000
for _ in range(8):once(b);once(c)
ts=[[],[]];equal=[]
for i in range(a.pairs):
 outs={}
 for j in ((0,1) if i%2==0 else (1,0)):
  out,ms=once((b,c)[j]);outs[j]=out;ts[j].append(ms)
 equal.append(outs[0].tobytes()==outs[1].tobytes())
 if i%20==0:print(i,ts[0][-1],ts[1][-1],flush=True)
d=np.array(ts[0])-ts[1];ci=np.percentile(np.random.default_rng(42).choice(d,(10000,len(d))).mean(1),[2.5,97.5])
r={'baseline_ms':float(np.mean(ts[0])),'candidate_ms':float(np.mean(ts[1])),'saving_ci95':ci.tolist(),'all_equal':all(equal),'times_ms':ts}
np.savez(a.output/'actions.npz',baseline=outs[0],candidate=outs[1]);(a.output/'results.json').write_text(json.dumps(r,indent=2));print({k:v for k,v in r.items() if k!='times_ms'},flush=True)
