import os,sys,runpy
os.environ['MLX_MAX_OPS_PER_BUFFER']='256'
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
mode=sys.argv.pop(1)
import mlx.core as mx
import imagewam_mlx
from imagewam_mlx.model import ImageWAM as Base
from fusion_candidates import SplitCandidate,CombinedCandidate
original=imagewam_mlx.ImageWAM
def build(*a,**kw):
 kw['fused_concat']=False
 m=original(*a,**kw)
 if mode!='baseline':
  m.visual=({'split':SplitCandidate,'combined':CombinedCandidate}[mode])(m.visual);m.prefill=mx.compile(lambda h,l,s:Base.prefill(m,h,l,s))
 m.quantization_info['visual']=dict(m.quantization_info['visual'],fusion_candidate=mode)
 return m
imagewam_mlx.ImageWAM=build
runpy.run_path('scripts/benchmark.py',run_name='__main__')
