from pathlib import Path
import ast
r=Path(__file__).parent
p=r/'candidate.py';s=p.read_text();tr=ast.parse(s)
# Get m1n4 source literals without importing MLX.
values=[]
for n in tr.body:
 if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call):
  f=n.value.func
  if isinstance(f,ast.Attribute) and isinstance(f.value,ast.Subscript) and ast.literal_eval(f.value.slice)=='m1n4':
   values.append(ast.literal_eval(next(k.value for k in n.value.args[0].keywords if k.arg=='source')))
assert len(values)==2
extra='''
# The two-M-fragment overload must describe a 32x16 result, not 16x32.
_h0,_h1=sv.header.split('thread dtype_frag_t<CType>& Cm0,',1)
assert '        16,\\n        32,\\n        16,' in _h1
fixed_header=_h0+'thread dtype_frag_t<CType>& Cm0,'+_h1.replace('        16,\\n        32,\\n        16,','        32,\\n        16,\\n        16,',1)
KERNELS['fixed']=[]
'''
for i,code in enumerate(values):extra+=f'KERNELS["fixed"].append(mx.fast.metal_kernel(name="wam_layout_fixed_{i}",input_names=["w","scales","biases","w_inv","w_scale","x8","x_scale","x_exp"],output_names=["y"],header=fixed_header,source={code!r}))\n'
pos=s.index('class Candidate');p.write_text(s[:pos]+extra+s[pos:])
s=(r/'screen.py').read_text().replace('from common import MODES,install',"from common import install\nMODES=['baseline','fixed']").replace("device_layout_20261105/screen'","device_layout_20261105/fixed'")
(r/'fixed.py').write_text(s)
