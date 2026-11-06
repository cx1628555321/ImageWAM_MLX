from pathlib import Path
import ast
r=Path(__file__).parent;root=r.parents[1]
t=ast.parse((root/'imagewam_mlx/device_visual.py').read_text());node=next(n.value for n in t.body if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='full_kernel' for x in n.targets));src=ast.literal_eval(next(k.value for k in node.keywords if k.arg=='source'))
text='import mlx.core as mx\nfrom imagewam_mlx.device_visual import DeviceVisual,sv\nKERNELS={}\n'
for mode,bm in [('g2_32',32),('g2_64',64)]:
 s=src.replace('constexpr int BM = 64;',f'constexpr int BM = {bm};').replace('constexpr int WM = 2;','constexpr int WM = 1;')
 if bm==32:
  s=s.replace('(size_t(tid.y) * (K / BK) * BM + tm) * BK','(size_t(tid.y / 2) * (K / BK) * 64 + (tid.y % 2) * 32 + tm) * BK').replace('x8 += BM * BK;','x8 += 64 * BK;')
 text+=f'KERNELS[{mode!r}]=mx.fast.metal_kernel(name="wam_{mode}",input_names=["w","scales","biases","w_inv","w_scale","x8","x_scale","x_exp"],output_names=["y"],header=sv.header,source={s!r})\n'
text+='''class Group2(DeviceVisual):
 def __init__(self,orig,mode):self.__dict__.update(orig.__dict__);self.mode=mode
 def mm(self,**kw):
  n=kw['inputs'][0].shape[0]
  if n%64:return super().mm(**kw)
  rows=dict(kw['template'])['ROWS'];bm=32 if self.mode=='g2_32' else 64
  kw['threadgroup']=(64,1,1);kw['grid']=((n+63)//64*64,(rows+bm-1)//bm,1)
  return KERNELS[self.mode](**kw)
'''
(r/'group2.py').write_text(text)
p=r/'common.py';s=p.read_text().replace('m=copy.copy(base);m.visual=Candidate(base.visual,mode)', '''m=copy.copy(base)
 if mode.startswith('g2_'):
  from group2 import Group2
  m.visual=Group2(base.visual,mode)
 else:m.visual=Candidate(base.visual,mode)''');p.write_text(s)
s=(r/'screen.py').read_text().replace('from common import MODES,install',"from common import install\nMODES=['baseline','g2_32','g2_64']").replace("device_layout_20261105/screen'","device_layout_20261105/group2'");(r/'group2_test.py').write_text(s)
