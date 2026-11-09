"""Check saved action arrays and every residual on four real observations."""
import sys,json
from pathlib import Path
import numpy as np
import mlx.core as mx
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from imagewam_mlx import ImageWAM
from imagewam_mlx.pow_vae import LookupPowVAE
out=Path('records/vae_lookup_checks_20261103');out.mkdir(exist_ok=False)
old=Path('records/final_validation_20261103');new=Path('records/vae_lookup_validation_20261103')
checks=[]
for f in sorted(old.glob('*.npy')):
 a,b=np.load(f),np.load(new/f.name)
 checks.append({'file':f.name,'equal':a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()})
assert len(checks)==162
m=ImageWAM('weights/bf16','assets');v=LookupPowVAE.from_vae(m.vae);res=v.res;rows=[]
def check(x,p):
 a=res(x,p);b=m.vae.res(x,p);mx.eval(a,b)
 aa,bb=np.array(a.astype(mx.float32)),np.array(b.astype(mx.float32))
 rows.append({'layer':p,'equal':aa.tobytes()==bb.tobytes(),'max_abs':float(np.max(np.abs(aa-bb)))})
 return a
v.res=check
ref=Path('../ImageWAM/evaluate_results/m5pro_robotwin/20261103_104832_dynamic_full');latents=[]
for i,f in enumerate(sorted(ref.glob('*_inputs.npz'))):
 raw=np.load(f);images={k:raw[k] for k in ('cam_high','cam_left_wrist','cam_right_wrist')};x,_=m.preprocess(images,raw['state'])
 a,b=m.vae(x),v(x);mx.eval(a,b);aa,bb=np.array(a.astype(mx.float32)),np.array(b.astype(mx.float32))
 np.savez(out/f'latent_{i}.npz',baseline=aa,candidate=bb)
 latents.append({'input':str(f),'equal':aa.tobytes()==bb.tobytes(),'max_abs':float(np.max(np.abs(aa-bb)))})
r={'actions':checks,'residuals':rows,'latents':latents,'all_equal':all(v['equal'] for v in checks+rows+latents)}
(out/'results.json').write_text(json.dumps(r,indent=2));print('equal',r['all_equal'],'actions',len(checks),'residuals',len(rows),'latents',len(latents));assert r['all_equal']
