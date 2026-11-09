import sys,json,types,argparse
from pathlib import Path
import numpy as np
import mlx.core as mx
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from imagewam_mlx import ImageWAM
from imagewam_mlx.ops import silu
from next_optimization_candidates import vae_model,FusedVAE
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('records/next_optimizations_20261103/vae_intermediates'));args=p.parse_args();root=args.output;root.mkdir(exist_ok=False)
m=ImageWAM('weights/bf16','assets',fused_vae=False);c=vae_model(m);rows=[]
def checked(self,x,p):
 y=FusedVAE.gn_silu(self,x,p);ref=silu(self.gn(x,p));mx.eval(y,ref)
 a,b=np.array(y.astype(mx.float32)),np.array(ref.astype(mx.float32));rows.append({'layer':p,'equal':bool(np.array_equal(a,b)),'max_abs':float(np.max(np.abs(a-b)))})
 return y
c.vae.gn_silu=types.MethodType(checked,c.vae)
ref=Path('../ImageWAM/evaluate_results/m5pro_robotwin/20261103_104832_dynamic_full')
paths=[ref/'base_inputs.npz']+sorted(p for p in ref.glob('*_inputs.npz') if p.name!='base_inputs.npz')
latents=[]
for i,p in enumerate(paths):
 raw=np.load(p);images={k:raw[k] for k in ('cam_high','cam_left_wrist','cam_right_wrist')};x,st=m.preprocess(images,raw['state'])
 a=m.vae(x);b=c.vae(x);mx.eval(a,b);aa,bb=np.array(a.astype(mx.float32)),np.array(b.astype(mx.float32))
 np.savez(root/f'latent_{i}.npz',baseline=aa,candidate=bb);latents.append({'input':str(p),'equal':bool(np.array_equal(aa,bb)),'max_abs':float(np.max(np.abs(aa-bb)))})
r={'rows':rows,'latents':latents,'all_equal':all(x['equal'] for x in rows+latents)}
(root/'results.json').write_text(json.dumps(r,indent=2));print('ALL_EQUAL',r['all_equal'],'GN',len(rows),'LATENTS',len(latents));assert r['all_equal']
