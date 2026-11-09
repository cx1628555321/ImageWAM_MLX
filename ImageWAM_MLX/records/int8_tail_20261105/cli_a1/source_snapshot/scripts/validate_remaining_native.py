"""Validate packaged opt-ins and async trace against pre-round default."""
import sys,copy,json
from pathlib import Path
import numpy as np
import mlx.core as mx
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from imagewam_mlx import ImageWAM
from imagewam_mlx.fused_qwen_rope import FusedRopeQwen
from imagewam_mlx.fused_vae_inv import GroupInverseVAE
root=Path('records/remaining_20261103/native_options');root.mkdir(exist_ok=False)
base=ImageWAM('weights/bf16','assets',overlap=False,qwen_rope=False,vae_group_inv=False)
models={'baseline':base}
for mode in ('default','qwen','group','all'):
 m=copy.copy(base);m.overlap=True
 if mode in ('qwen','all'):m.qwen=FusedRopeQwen.from_qwen(base.qwen);m.qwen_rope=True
 if mode in ('group','all'):m.vae=GroupInverseVAE.from_vae(base.vae);m.vae_group_inv=True
 models[mode]=m
ex=Path('examples/robotwin_ep0_frame0');raw=np.load(ex/'inputs.npz');images={k:raw[k] for k in ('cam_high','cam_left_wrist','cam_right_wrist')}
inv=json.loads(Path('../ImageWAM/development_records/m5pro_robotwin/20261103_dynamic_optimization/text_lengths.json').read_text())['examples_by_length']
cases=[(inv['31']['prompt'],16,10),(inv['79']['prompt'],16,10),('Move carefully. '*60,16,10)]+[(inv['31']['prompt'],h,s) for h,s in ((1,1),(8,3),(32,10),(64,2))]
rows=[]
for i,(prompt,h,steps) in enumerate(cases):
 noise=np.random.default_rng(h).standard_normal((1,h,14)).astype(np.float32);outputs={}
 for name,m in models.items():
  a,n=m.predict(images,raw['state'],prompt,noise,steps=steps,trace=True)
  arrays=dict(action=a,normalized=n,**m.trace);outputs[name]=arrays
  np.savez(root/f'{i}_{name}.npz',**arrays)
  if name!='baseline':
   for key,x in arrays.items():
    y=outputs['baseline'][key];rows.append({'case':i,'mode':name,'tensor':key,'equal':bool(np.array_equal(x,y)),'bytes_equal':x.dtype==y.dtype and x.tobytes()==y.tobytes()})
r={'all_equal':all(x['equal'] for x in rows),'all_bytes_equal':all(x['bytes_equal'] for x in rows),'checks':len(rows),'rows':rows};(root/'results.json').write_text(json.dumps(r,indent=2));print('ALL_EQUAL',r['all_equal'],'BYTES',r['all_bytes_equal'],'CHECKS',len(rows));assert r['all_equal']
