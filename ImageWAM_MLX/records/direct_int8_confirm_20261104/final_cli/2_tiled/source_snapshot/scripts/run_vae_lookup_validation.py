"""Isolated candidate wrapper for existing full input regression."""
import sys,runpy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import imagewam_mlx
from imagewam_mlx.pow_vae import LookupPowVAE
class Candidate(imagewam_mlx.ImageWAM):
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs)
  self.vae=LookupPowVAE.from_vae(self.vae)
imagewam_mlx.ImageWAM=Candidate
runpy.run_path(str(Path(__file__).with_name('validate_cases.py')),run_name='__main__')
