import ast
from pathlib import Path
p=Path('imagewam_mlx_work/project');out=p/'experiments/int8_tail_20261105'
s=(p/'imagewam_mlx/prepacked_visual.py').read_text();tree=ast.parse(s)
c=next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='kernel' for t in n.targets))
src=ast.literal_eval(next(k.value for k in c.keywords if k.arg=='source'))
small=src.replace('constexpr int BM = 64;','constexpr int BM = 32;').replace('tid.y * BM','tid.y * 64').replace('(K / BK) * BM','(K / BK) * 64').replace('x8 += BM * BK;','x8 += 64 * BK;')
# One dispatch, same 128 threads and 64-row activation layout. Uniform branch.
body='if (ROWS % 64 > 0 && ROWS % 64 <= 32 && threadgroup_position_in_grid.y == ROWS / 64) {\n'+small+'\n} else {\n'+src+'\n}'
base=(p/'experiments/int8_wide_20261104/candidate.py').read_text()
head=base[:base.index('kernel=')]
head+=f"kernel=mx.fast.metal_kernel(name='wam_int8_tail32',input_names=['w','scales','biases','w_inv','w_scale','x8','x_scale','x_exp'],output_names=['y'],header=sv.header,source={body!r})\n"
head+='''class Candidate(GateQuantVisual):
    def __init__(self,orig,scope='all'):
        self.__dict__.update(orig.__dict__)
        self.scope=scope
    def mm(self,**kwargs):
        rows=dict(kwargs['template'])['ROWS']
        if rows%64==0 or rows%64>32 or (self.scope=='linear2' and kwargs['inputs'][0].shape[0]!=3072):
            return pv.kernel(**kwargs)
        return kernel(**kwargs)
'''
head+=base[base.index('    def linear('):]
(out/'candidate.py').write_text(head)
