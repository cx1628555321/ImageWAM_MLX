import os,sys,json,importlib.metadata as md
from pathlib import Path
os.environ['MLX_MAX_OPS_PER_BUFFER']='256'
sys.path.insert(0,str(Path.cwd()))
import mlx.core as mx
mode=sys.argv[1]
if mode.startswith('math_'):
 original=mx.fast.metal_kernel
 def factory(*a,**kw):
  # Limit math mode to GEMM / quantization / epilogue, never masked softmax.
  name=kw.get('name','')
  if any(s in name for s in ('device_unroll','pr4722_x8','w1_tiled','split_epilogue','bg_bias')):kw['compile_options']={'math_mode':mode[5:]}
  return original(*a,**kw)
 mx.fast.metal_kernel=factory
if os.environ.get('BG_ALLOW_NEW_MLX')=='1':
 import imagewam_mlx,imagewam_mlx.runtime as rt
 source=Path(rt.__file__).read_text();assert "version('mlx')=='0.32.4'" in source
 # Process-local experimental compatibility override; never edit production files.
 source=source.replace("version('mlx')=='0.32.4'",'True');exec(compile(source,rt.__file__,'exec'),rt.__dict__);imagewam_mlx.ImageWAM=rt.ImageWAM
out=Path(sys.argv[2]);out.parent.mkdir(parents=True,exist_ok=True)
(out.parent/(out.name+'_environment.json')).write_text(json.dumps(dict(mlx=md.version('mlx'),python=sys.executable,mode=mode,experimental_version_override=os.environ.get('BG_ALLOW_NEW_MLX')=='1',device=mx.device_info()),indent=2,default=str))
source=Path('experiments/background_20261105/worker_template.py').read_text();source=source.replace("profile='speed',text_cache_size=1","profile='speed',text_cache_size=1,prefix_precision='all_int8'")
exec(compile(source,'background_worker_template','exec'))
