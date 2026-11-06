import copy
from pathlib import Path
import json
import numpy as np
import mlx.core as mx
from imagewam_mlx import ImageWAM
from imagewam_mlx.model import ImageWAM as Base
from imagewam_mlx.workspace import workspace_step
from imagewam_mlx.device_visual import DeviceVisual
from imagewam_mlx.group_visual import GroupVisual
ROOT=Path('records/roadmap_20261105')
def inputs():
 p=Path('examples/robotwin_ep0_frame0');d=np.load(p/'inputs.npz')
 return dict(images={k:d[k] for k in ('cam_high','cam_left_wrist','cam_right_wrist')},state=d['state'],prompt=(p/'prompt.txt').read_text(),noise=np.load(p/'initial_noise_fp32.npy'))
class SelectVisual(GroupVisual):
 def __init__(self,orig,shapes):
  super().__init__(orig,64);self.selected={tuple(v) for v in shapes}
 def mm(self,**kw):
  t=dict(kw['template']);shape=(t['ROWS'],kw['inputs'][0].shape[0],t['XK'])
  return (GroupVisual.mm if shape in self.selected else DeviceVisual.mm)(self,**kw)
def install(base,mode):
 if mode=='baseline':return base
 m=copy.copy(base)
 if mode.startswith('split'):
  m.action=copy.copy(base.action);m.action.split_target=int(mode[5:])
  m._compiled_workspace_step=mx.compile(lambda c,x,t,d,p:workspace_step(m,c,x,t,d,p))
 elif mode=='visual_selected':
  m.visual=SelectVisual(base.visual,json.loads((ROOT/'diagnostic.json').read_text())['selected_visual_shapes'])
  m.prefill=mx.compile(lambda h,l,s:Base.prefill(m,h,l,s))
 else:raise ValueError(mode)
 return m
