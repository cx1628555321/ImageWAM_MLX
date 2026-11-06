import ast
from pathlib import Path
r=Path(__file__).resolve().parents[2];out=Path(__file__).parent
tr=ast.parse((r/'imagewam_mlx/device_visual.py').read_text());src={}
for n in tr.body:
 if isinstance(n,ast.Assign) and isinstance(n.value,ast.Call) and any(isinstance(t,ast.Name) and t.id in ('full_kernel','tail_kernel') for t in n.targets):
  src[n.targets[0].id]=ast.literal_eval(next(k.value for k in n.value.keywords if k.arg=='source'))
a=src['full_kernel'];b=src['tail_kernel'];prefix,body=b.split('} else {\nuint3 tid=',1);body='uint3 tid='+body
variants={}
for name,wm,wn in [('m4n1',4,1),('m1n4',1,4)]:
 def change(s):return s.replace('constexpr int WM = 2;',f'constexpr int WM = {wm};').replace('constexpr int WN = 2;',f'constexpr int WN = {wn};')
 variants[name]=(change(a),prefix+'} else {\n'+change(body))
def unroll(s):return s.replace('  for (int k = 0; k < K; k += BK) {','  #pragma clang loop unroll_count(2)\n  for (int k = 0; k < K; k += BK) {')
variants['k2']=(unroll(a),unroll(b))
s='import mlx.core as mx\nfrom imagewam_mlx.device_visual import DeviceVisual,sv\nKERNELS={}\n'
for name,pair in variants.items():
 s+=f'KERNELS[{name!r}]=[]\n'
 for i,code in enumerate(pair):s+=f'KERNELS[{name!r}].append(mx.fast.metal_kernel(name="wam_layout_{name}_{i}",input_names=["w","scales","biases","w_inv","w_scale","x8","x_scale","x_exp"],output_names=["y"],header=sv.header,source={code!r}))\n'
s+='''class Candidate(DeviceVisual):
 def __init__(self,orig,mode):self.__dict__.update(orig.__dict__);self.mode=mode
 def mm(self,**kwargs):
  if kwargs['inputs'][0].shape[0]%64:return super().mm(**kwargs)
  rows=dict(kwargs['template'])['ROWS']
  return KERNELS[self.mode][int(0<rows%64<=32)](**kwargs)
'''
(out/'candidate.py').write_text(s)
