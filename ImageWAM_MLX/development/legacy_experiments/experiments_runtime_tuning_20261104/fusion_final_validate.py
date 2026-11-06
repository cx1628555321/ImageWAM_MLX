import os
os.environ['MLX_MAX_OPS_PER_BUFFER']='256'
import sys,json,gc
from pathlib import Path
import numpy as np
import mlx.core as mx
sys.path.insert(0,str(Path.cwd()))
from imagewam_mlx import ImageWAM
root=Path('records/fusion_confirm_20261104/final_validation');root.mkdir(exist_ok=False)
m=ImageWAM('weights/bf16','assets',profile='speed');assert m.staged_visual and m.action_epilogue and m.quantization_info['visual']['concat_quant_fused'];assert m.quantization_info['visual']['linear_count']==38
ref=Path('../ImageWAM/evaluate_results/m5pro_robotwin/20261103_104832_dynamic_full');cases=json.loads((ref/'comparison.json').read_text())['cases'];ex=Path('examples/robotwin_ep0_frame0');noise=np.load(ex/'initial_noise_fp32.npy');rows=[]
for case in cases:
 label=case['label'];raw=np.load(ref/('base_inputs.npz' if label.startswith('length_') else label+'_inputs.npz'));images={k:raw[k] for k in ('cam_high','cam_left_wrist','cam_right_wrist')};a,n=m.predict(images,raw['state'],case['prompt'],noise);r=np.load(Path('records/runtime_tuning_20261104/native_confirm')/(label+'.npz'));assert np.array_equal(a,r['staged']) and np.array_equal(n,r['normalized']),label;rows.append(dict(label=label,exact=True));np.savez(root/(label+'.npz'),action=a,normalized=n)
(root/'cases.json').write_text(json.dumps(rows,indent=2));del m;gc.collect();mx.clear_cache()
raw=np.load(ex/'inputs.npz');images={k:raw[k] for k in ('cam_high','cam_left_wrist','cam_right_wrist')};prompt=(ex/'prompt.txt').read_text()
m=ImageWAM('weights/bf16','assets',profile='accuracy');assert not m.staged_visual and not m.action_epilogue;a,n=m.predict(images,raw['state'],prompt,noise);assert np.array_equal(a,np.load('records/final_default_20261103/reference_action.npy'));np.savez(root/'accuracy.npz',action=a,normalized=n);del m;gc.collect();mx.clear_cache()
m=ImageWAM('weights/bf16','assets',profile='speed',staged_visual=False);assert not m.staged_visual;a,n=m.predict(images,raw['state'],prompt,noise);assert np.array_equal(a,np.load('records/runtime_tuning_20261104/native_confirm/baseline.npz')['action']);np.savez(root/'old_speed.npz',action=a,normalized=n)
(root/'completed.json').write_text(json.dumps(dict(native_cases=len(rows),native_exact=True,accuracy_exact=True,rollback_exact=True),indent=2));print('FINAL VALIDATION COMPLETE',flush=True)
