import runpy
from pathlib import Path
r=Path(__file__).parent;d=runpy.run_path(str(r/'build.py'));variants=d['variants'];base=variants['m1n4']
expr='y[size_t(r) * N + c] =\n                static_cast<T>(float(frag[e]) * x_scale[r] * w_scale[c]);'
for mode,inner,outer in [('left','float(frag[e])*x_scale[r]','v0*w_scale[c]'),('right','x_scale[r]*w_scale[c]','float(frag[e])*v0')]:
 code=[]
 for s in base:
  assert expr in s
  code.append(s.replace(expr,f'volatile float v0={inner}; y[size_t(r)*N+c]=static_cast<T>({outer});'))
 variants[mode]=code
s=(r/'candidate.py').read_text();pos=s.index('class Candidate')
extra=''
for mode in ('left','right'):
 extra+=f'KERNELS[{mode!r}]=[]\n'
 for i,code in enumerate(variants[mode]):extra+=f'KERNELS[{mode!r}].append(mx.fast.metal_kernel(name="wam_layout_{mode}_{i}",input_names=["w","scales","biases","w_inv","w_scale","x8","x_scale","x_exp"],output_names=["y"],header=sv.header,source={code!r}))\n'
(r/'candidate.py').write_text(s[:pos]+extra+s[pos:])
