import os,sys,runpy
os.environ['MLX_MAX_OPS_PER_BUFFER']='256'
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
mode=sys.argv.pop(1)
if mode=='tiled':
 import mlx.core as mx
 import imagewam_mlx
 from imagewam_mlx.model import ImageWAM as Base
 from imagewam_mlx.tiled_visual import TiledVisual
 original=imagewam_mlx.ImageWAM
 def build(*a,**kw):
  m=original(*a,**kw);m.visual=TiledVisual(m.visual);m.quantization_info['visual']=m.visual.info;m.prefill=mx.compile(lambda h,l,s:Base.prefill(m,h,l,s));return m
 imagewam_mlx.ImageWAM=build
runpy.run_path('scripts/benchmark.py',run_name='__main__')
