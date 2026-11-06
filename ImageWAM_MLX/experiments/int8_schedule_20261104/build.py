from pathlib import Path
import ast
p=Path('imagewam_mlx_work/project');out=p/'experiments/int8_schedule_20261104'
s=(p/'imagewam_mlx/prepacked_visual.py').read_text();tree=ast.parse(s)
c=next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='kernel' for t in n.targets))
src=ast.literal_eval(next(k.value for k in c.keywords if k.arg=='source'))
src=src.replace('uint3 tid=threadgroup_position_in_grid;', 'uint3 tid=threadgroup_position_in_grid; uint logical_m=tid.y*SWIZZLE+tid.x%SWIZZLE;tid.x=tid.x/SWIZZLE;tid.y=logical_m;if(tid.y>=(ROWS+63)/64)return;')
header='''import mlx.core as mx
from imagewam_mlx.gate_quant_visual import GateQuantVisual,q
from imagewam_mlx import prepacked_visual as pv
sv=pv.sv
'''
header+=f"kernel=mx.fast.metal_kernel(name='wam_int8_swizzle',input_names=['w','scales','biases','w_inv','w_scale','x8','x_scale','x_exp'],output_names=['y'],header=sv.header,source={src!r})\n"
header+='''class Candidate(GateQuantVisual):
    def __init__(self,orig,swizzle=2):
        self.__dict__.update(orig.__dict__)
        self.swizzle=swizzle
    def mm(self,**kwargs):
        kwargs['template']=[*kwargs['template'],('SWIZZLE',self.swizzle)]
        gx,gy,gz=kwargs['grid']
        kwargs['grid']=(gx*self.swizzle,(gy+self.swizzle-1)//self.swizzle,gz)
        return kernel(**kwargs)
'''
for file,cls,method in [('prepacked_visual.py','PrepackedVisual','linear'),('gate_quant_visual.py','GateQuantVisual','single_post')]:
 s=(p/'imagewam_mlx'/file).read_text();t=ast.parse(s);cl=next(n for n in t.body if isinstance(n,ast.ClassDef) and n.name==cls);f=next(n for n in cl.body if isinstance(n,ast.FunctionDef) and n.name==method)
 code='\n'.join(s.splitlines()[f.lineno-1:f.end_lineno]);code=code.replace('pv.kernel(', 'self.mm(').replace('y = kernel(', 'y = self.mm(')
 header+=code+'\n'
(out/'candidate.py').write_text(header)
f=(p/'experiments/mod_quant_20261104/full.py').read_text().replace('records/mod_quant_20261104/full','records/int8_schedule_20261104/full')
a=f.index('base=ImageWAM');b=f.index("ex=Path",a)
f=f[:a]+'''base=ImageWAM('weights/bf16','assets',profile='speed');models={'baseline':base}
for sw in [2,4,8]:
 c=copy.copy(base);c.visual=Candidate(base.visual,sw);c.prefill=mx.compile(lambda h,l,s,c=c:Base.prefill(c,h,l,s));models['sw'+str(sw)]=c
kernels={k:None for k in models if k!='baseline'}
'''+f[b:]
f=f.replace('range(200)','range(100)')
(out/'full.py').write_text(f)
