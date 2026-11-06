"""Compare saved benchmark I/O against actual original RoboTwin policy methods.

Extract only independent methods with AST to avoid importing simulator/dataset
training dependencies. No method implementation is rewritten in this check.
"""
import ast,json,sys
from pathlib import Path
from typing import Dict,Any
import numpy as np
import torch
from PIL import Image
from hydra.utils import instantiate
from omegaconf import OmegaConf
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from imagewam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json

out=Path(sys.argv[1]).resolve()
path=ROOT/'experiments/robotwin/imagewam_policy/deploy_policy.py'
tree=ast.parse(path.read_text())
functions={'_resize_rgb','_robotwin_camera_sizes'}
methods={'_normalize_state','_denormalize_action','_build_robotwin_image_tensor'}
nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in functions]
original=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='WorldActionRobotWinPolicy')
original.body=[n for n in original.body if isinstance(n,ast.FunctionDef) and n.name in methods]
nodes.append(original)
exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(path),'exec'))
policy=WorldActionRobotWinPolicy()
policy.model=type('Model',(),{'device':'mps','torch_dtype':torch.bfloat16})()
policy.robotwin_camera_layout='compact_288x256';policy.apply_action_transform_backward=False
cfg=OmegaConf.create({'data':OmegaConf.load(ROOT/'configs/data/robotwin_omnigen2.yaml')})
policy.processor=instantiate(cfg.data.train.processor).eval()
policy.processor.set_normalizer_from_stats(load_dataset_stats_from_json(str(out/'dataset_stats.json')))
inputs=np.load(out/'inputs.npz')
obs={'observation':{dst:{'rgb':inputs[src]} for src,dst in [('cam_high','head_camera'),('cam_left_wrist','left_camera'),('cam_right_wrist','right_camera')]}}
image=policy._build_robotwin_image_tensor(obs).float().cpu().numpy()
state=policy._normalize_state(inputs['state']).numpy()
action=policy._denormalize_action(torch.from_numpy(np.load(out/'normalized_first_000.npy')))[0]
results={}
for name,actual,filename in [('image',image,'model_input_bf16_as_fp32.npy'),('state',state,'proprio_normalized.npy'),('action',action,'reference_action.npy')]:
 expected=np.load(out/filename)
 results[name]={'bitwise_equal':bool(np.array_equal(actual,expected)),'max_abs_diff':float(np.max(np.abs(actual-expected)))}
 assert results[name]['bitwise_equal'],name
(out/'original_policy_io_validation.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(results))
