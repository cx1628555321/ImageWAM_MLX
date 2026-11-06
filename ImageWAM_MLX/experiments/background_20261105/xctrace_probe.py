import os,sys,time,json
from pathlib import Path
os.environ['MLX_MAX_OPS_PER_BUFFER']='256'
root=Path('/Users/qklj/work/cx/ImageWAM_MLX');os.chdir(root)
sys.path.insert(0,str(root));sys.path.insert(0,str(root/'experiments/roadmap_20261105'))
import mlx.core as mx
import imagewam_mlx,imagewam_mlx.runtime as rt
s=Path(rt.__file__).read_text();assert "version('mlx')=='0.32.4'" in s
exec(compile(s.replace("version('mlx')=='0.32.4'",'True'),rt.__file__,'exec'),rt.__dict__)
from common import inputs
m=rt.ImageWAM('weights/bf16','assets',profile='speed',text_cache_size=1,prefix_precision='all_int8')
args=inputs();m.prepare_text(args['prompt'])
for _ in range(10):m.predict(**args)
mx.synchronize();print('TRACE_STEADY_BEGIN',flush=True)
t=time.monotonic();count=0
while time.monotonic()-t<10:
 m.predict(**args);mx.synchronize();count+=1
print('TRACE_STEADY_END',count,flush=True)
