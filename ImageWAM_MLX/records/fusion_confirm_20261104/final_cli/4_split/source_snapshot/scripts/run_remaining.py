"""Run a candidate's actual object so trace/profile belong to the returned model."""
import sys,argparse,runpy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import imagewam_mlx
from remaining_candidates import BUILDERS
p=argparse.ArgumentParser();p.add_argument('--mode',choices=list(BUILDERS),required=True);p.add_argument('--entry',choices=['validate_cases','benchmark'],required=True);a,rest=p.parse_known_args()
Original=imagewam_mlx.ImageWAM
def candidate(*args,**kwargs):
 kwargs.setdefault('fused_vae',True)
 kwargs.setdefault('overlap',False)
 kwargs.setdefault('qwen_rope',False)
 kwargs.setdefault('vae_group_inv',False)
 return BUILDERS[a.mode](Original(*args,**kwargs))
imagewam_mlx.ImageWAM=candidate
sys.argv=[a.entry]+rest
runpy.run_path(str(Path(__file__).with_name(a.entry+'.py')),run_name='__main__')
