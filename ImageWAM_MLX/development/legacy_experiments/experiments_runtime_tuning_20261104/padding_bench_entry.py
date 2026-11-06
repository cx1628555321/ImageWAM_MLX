import os,sys,ast,runpy
os.environ['MLX_MAX_OPS_PER_BUFFER']='256'
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
pad=int(sys.argv.pop(1));assert pad in [64,80]
if pad!=80:
 import mlx.core as mx
 from imagewam_mlx import prepacked_visual as pv
 tree=ast.parse(Path(pv.__file__).read_text());call=next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='kernel' for t in n.targets));src=ast.literal_eval(next(k.value for k in call.keywords if k.arg=='source')).replace('constexpr int BK_padded = BK + 16;',f'constexpr int BK_padded = {pad};')
 pv.kernel=mx.fast.metal_kernel(name='wam_padding_confirm',input_names=['w','scales','biases','w_inv','w_scale','x8','x_scale','x_exp'],output_names=['y'],header=pv.sv.header,source=src)
runpy.run_path('scripts/benchmark.py',run_name='__main__')
