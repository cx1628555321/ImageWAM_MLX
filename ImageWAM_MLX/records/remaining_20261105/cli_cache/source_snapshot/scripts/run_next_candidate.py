"""Run existing full validation/benchmark with an explicit isolated candidate."""
import sys,argparse,runpy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import imagewam_mlx
from next_optimization_candidates import BUILDERS
p=argparse.ArgumentParser();p.add_argument('--mode',choices=list(BUILDERS),required=True);p.add_argument('--entry',choices=['validate_cases','benchmark'],required=True);a,rest=p.parse_known_args()
Original=imagewam_mlx.ImageWAM
class Candidate(Original):
 def __init__(self,*args,**kwargs):
  kwargs.setdefault('fused_vae',False)
  super().__init__(*args,**kwargs)
  candidate=BUILDERS[a.mode](self)
  self.__dict__.update(candidate.__dict__)
imagewam_mlx.ImageWAM=Candidate
sys.argv=[a.entry]+rest
runpy.run_path(str(Path(__file__).with_name(a.entry+'.py')),run_name='__main__')
