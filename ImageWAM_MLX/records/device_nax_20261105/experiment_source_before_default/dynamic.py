import os
os.environ.setdefault('MLX_MAX_OPS_PER_BUFFER','256')
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
import numpy as np
import mlx.core as mx
from imagewam_mlx import ImageWAM
from common import install
MODES=['baseline','direct','combined','reduce4','combo']
r=Path('records/device_nax_20261105/dynamic');r.mkdir(exist_ok=False)
base=ImageWAM('weights/bf16','assets',profile='speed',text_cache_size=0);models={k:install(base,k) for k in MODES};p=Path('examples/robotwin_ep0_frame0');raw=np.load(p/'inputs.npz');prompt=(p/'prompt.txt').read_text();noise=np.load(p/'initial_noise_fp32.npy');rows=[]
for idx,(text,h,alter) in enumerate([(prompt,1,False),(prompt+' Please carefully follow the instruction.'*25,16,False),(prompt,64,False),(prompt,16,True)]):
 images={k:raw[k].copy() for k in ('cam_high','cam_left_wrist','cam_right_wrist')};state=raw['state'].copy()
 if alter:
  images['cam_high']=images['cam_high'][::2,::2].copy();images['cam_left_wrist']=np.flip(images['cam_left_wrist'],axis=1).copy();state[0]+=0.1
 args=dict(images=images,state=state,prompt=text,noise=np.resize(noise,(1,h,14)),steps=10)
 a,n=base.predict(**args);x,st=base.preprocess(images,state);latent=base.vae(x);mx.eval(x,latent);pre=np.array(x.astype(mx.float32));ref=np.array(latent.astype(mx.float32));errs={}
 for mode in MODES[1:]:
  m=models[mode];b,z=m.predict(**args);y,_=m.preprocess(images,state);la=m.vae(y);mx.eval(y,la);yp=np.array(y.astype(mx.float32));lp=np.array(la.astype(mx.float32));errs[mode]=dict(action_exact=bool(np.array_equal(a,b) and np.array_equal(n,z)),preprocess_exact=bool(np.array_equal(pre,yp)),latent_exact=bool(np.array_equal(ref,lp)),latent_max_abs=float(np.abs(ref-lp).max()))
  np.savez(r/f'{idx}_{mode}.npz',action=b,normalized=z,latent=lp)
 rows.append(dict(tokens=int(base.tokenize(text).shape[1]),horizon=h,altered_observation=alter,errors=errs))
(r/'results.json').write_text(json.dumps(rows,indent=2));print(rows,flush=True)
