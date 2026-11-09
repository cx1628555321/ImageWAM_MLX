"""Current baseline versus combined candidate: encoders, all prefix K/V, step traces."""
import sys,json
from pathlib import Path
import numpy as np
import mlx.core as mx
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from imagewam_mlx import ImageWAM
from remaining_candidates import combined,build
root=Path('records/remaining_20261103/intermediates');root.mkdir(exist_ok=False)
base=ImageWAM('weights/bf16','assets',overlap=False,qwen_rope=False,vae_group_inv=False);candidate=combined(base);group=build(base,'group_inv')
ex=Path('examples/robotwin_ep0_frame0');raw=np.load(ex/'inputs.npz');images={k:raw[k] for k in ('cam_high','cam_left_wrist','cam_right_wrist')};noise=np.load(ex/'initial_noise_fp32.npy')
inv=json.loads(Path('../ImageWAM/development_records/m5pro_robotwin/20261103_dynamic_optimization/text_lengths.json').read_text())['examples_by_length']
rows=[]
def compare(label,a,b):
 mx.eval(a,b);x,y=np.array(a.astype(mx.float32)),np.array(b.astype(mx.float32))
 rows.append({'tensor':label,'equal':bool(np.array_equal(x,y)),'max_abs':float(np.max(np.abs(x-y)))})
for idx,prompt in enumerate([inv['31']['prompt'],inv['79']['prompt'],'Move carefully. '*60]):
 x,st=base.preprocess(images,raw['state']);ids=base.tokenize(prompt)
 h0,h1=base.qwen(mx.array(ids)),candidate.qwen(mx.array(ids));compare(f'text_{idx}',h0,h1)
 v0,v1=base.vae(x),candidate.vae(x);compare(f'latent_{idx}',v0,v1);compare(f'group_inverse_latent_{idx}',v0,group.vae(x))
 k0,k1=base.prefill(h0,v0,st),candidate.prefill(h1,v1,st)
 for i,(a,b) in enumerate(zip(k0,k1)):
  for kind,va,vb in zip(('k','v'),a,b):compare(f'prefix_{idx}_{i}_{kind}',va,vb)
 a,an=base.predict(images,raw['state'],prompt,noise,trace=True);b,bn=candidate.predict(images,raw['state'],prompt,noise,trace=True)
 np.savez(root/f'outputs_{idx}.npz',baseline=a,candidate=b,baseline_normalized=an,candidate_normalized=bn)
 for key in base.trace:
  x,y=base.trace[key],candidate.trace[key];rows.append({'tensor':f'trace_{idx}_{key}','equal':bool(np.array_equal(x,y)),'max_abs':float(np.max(np.abs(x.astype(float)-y.astype(float))))})
 rows.append({'tensor':f'action_{idx}','equal':bool(np.array_equal(a,b)),'max_abs':float(np.max(np.abs(a-b)))})
r={'all_equal':all(x['equal'] for x in rows),'rows':rows};(root/'results.json').write_text(json.dumps(r,indent=2));print('ALL_EQUAL',r['all_equal'],'CHECKS',len(rows));assert r['all_equal']
