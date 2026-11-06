from pathlib import Path
import ast
p=Path('imagewam_mlx_work/project');out=p/'experiments/int8_wide_20261104'
s=(p/'imagewam_mlx/prepacked_visual.py').read_text();tree=ast.parse(s)
c=next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='kernel' for t in n.targets))
src=ast.literal_eval(next(k.value for k in c.keywords if k.arg=='source'))
assert 'constexpr int BN = 64;' in src and 'constexpr int WN = 2;' in src
src=src.replace('constexpr int BN = 64;','constexpr int BN = 128;').replace('constexpr int WN = 2;','constexpr int WN = TILE_WN;')
header='''import mlx.core as mx
from imagewam_mlx.gate_quant_visual import GateQuantVisual,q
from imagewam_mlx import prepacked_visual as pv
sv=pv.sv
'''
header+=f"kernel=mx.fast.metal_kernel(name='wam_int8_wide',input_names=['w','scales','biases','w_inv','w_scale','x8','x_scale','x_exp'],output_names=['y'],header=sv.header,source={src!r})\n"
header+='''class Candidate(GateQuantVisual):
    def __init__(self,orig,wn=2):
        self.__dict__.update(orig.__dict__)
        if wn not in (2,4):raise ValueError('wn must be 2 or 4')
        self.wn=wn
    def mm(self,**kwargs):
        kwargs['template']=[*kwargs['template'],('TILE_WN',self.wn)]
        n=kwargs['inputs'][0].shape[0]
        gx,gy,gz=kwargs['grid'];threads=64*self.wn
        kwargs['grid']=((n+127)//128*threads,gy,gz)
        kwargs['threadgroup']=(threads,1,1)
        return kernel(**kwargs)
'''
for file,cls,method in [('prepacked_visual.py','PrepackedVisual','linear'),('gate_quant_visual.py','GateQuantVisual','single_post')]:
 s=(p/'imagewam_mlx'/file).read_text();t=ast.parse(s);cl=next(n for n in t.body if isinstance(n,ast.ClassDef) and n.name==cls);f=next(n for n in cl.body if isinstance(n,ast.FunctionDef) and n.name==method)
 code='\n'.join(s.splitlines()[f.lineno-1:f.end_lineno]);code=code.replace('pv.kernel(', 'self.mm(').replace('y = kernel(', 'y = self.mm(')
 header+=code+'\n'
(out/'candidate.py').write_text(header)
old=p/'experiments/int8_schedule_20261104'
for name in ('full.py','abba.py','dynamic.py','audit.py'):
 s=(old/name).read_text().replace('int8_schedule_20261104','int8_wide_20261104')
 if name=='full.py':s=s.replace('for sw in [2,4,8]:','for sw in [2,4]:').replace("'sw'+str(sw)","'wn'+str(sw)")
 if name=='abba.py':s=s.replace("mode.startswith('sw')","mode.startswith('wn')")
 (out/name).write_text(s)
