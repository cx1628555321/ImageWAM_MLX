import ast
from pathlib import Path
root=Path(__file__).resolve().parents[2]
out=Path(__file__).parent

def source(name):
 tree=ast.parse((root/'imagewam_mlx'/name).read_text())
 call=next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='kernel' for t in n.targets))
 return ast.literal_eval(next(k.value for k in call.keywords if k.arg=='source'))
def direct(s):
 start=s.index('  threadgroup uint Ws_storage')
 end=s.index('  constexpr short SM',start)
 s=s[:start]+s[end:]
 start=s.index('  // The part of the weight tile')
 end=s.index('  // x8 has',start)
 s=s[:start]+s[end:]
 start=s.index('    threadgroup_barrier(mem_flags::mem_threadgroup);')
 end=s.index('    // Move the accumulator',start)
 s=s[:start]+s[end:]
 s=s.replace('Btile.template load<int8_t, BK_padded, 1>(Ws + tn * BK_padded + kk1);','Btile.load((const device int8_t*)w + size_t(y_col + tn) * K + k + kk1, K);')
 return s
normal=source('prepacked_visual.py');tail=source('tail_visual.py')
a,b=tail.split('} else {\nuint3 tid=',1);b='uint3 tid='+b
# Tail wrapper shares integer arithmetic and quantization with current production.
variants={'direct':(direct(normal),direct(a)+'} else {\n'+direct(b)),
'combined':(direct(normal).replace('STEEL_PRAGMA_NO_UNROLL','STEEL_PRAGMA_UNROLL'),(direct(a)+'} else {\n'+direct(b)).replace('STEEL_PRAGMA_NO_UNROLL','STEEL_PRAGMA_UNROLL')),
'direct_tail':(normal,direct(a)+'} else {\n'+b),
'unroll':(normal.replace('STEEL_PRAGMA_NO_UNROLL','STEEL_PRAGMA_UNROLL'),tail.replace('STEEL_PRAGMA_NO_UNROLL','STEEL_PRAGMA_UNROLL'))}
text='import mlx.core as mx\nfrom imagewam_mlx.tail_visual import TailVisual,sv\nKERNELS={}\n'
for mode,pair in variants.items():
 text+=f'KERNELS[{mode!r}]=[]\n'
 for i,s in enumerate(pair):
  text+=f'KERNELS[{mode!r}].append(mx.fast.metal_kernel(name="wam_{mode}_{i}",input_names=["w","scales","biases","w_inv","w_scale","x8","x_scale","x_exp"],output_names=["y"],header=sv.header,source={s!r}))\n'
text+='''class Candidate(TailVisual):
 def __init__(self,orig,mode):
  self.__dict__.update(orig.__dict__);self.mode=mode
 def mm(self,**kwargs):
  rows=dict(kwargs['template'])['ROWS']
  # Device loads require a complete N tile. Production shapes satisfy this;
  # unsupported shapes retain the safe original implementation.
  if kwargs['inputs'][0].shape[0]%64:return super().mm(**kwargs)
  return KERNELS[self.mode][int(0<rows%64<=32)](**kwargs)
'''
(out/'candidate.py').write_text(text)
