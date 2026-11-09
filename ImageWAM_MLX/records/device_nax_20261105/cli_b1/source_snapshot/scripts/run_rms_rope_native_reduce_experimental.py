"""Opt-in final visual K/V projection experiment on the packaged default."""
import argparse,json,runpy,sys,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import imagewam_mlx.runtime as rt
from rms_rope_native_reduce_experiments import FusedCandidates
p=argparse.ArgumentParser(add_help=False);p.add_argument('--entry',choices=['infer','benchmark','validate_cases'],required=True);a,rest=p.parse_known_args();original=rt.ImageWAM.__init__
def init(self,*args,**kwargs):
 original(self,*args,**kwargs)
 if self.optimization!='default':raise ValueError('RMS/RoPE requires default runtime')
 self._rms_rope=FusedCandidates(self);self._rms_rope.set('rms_rope')
rt.ImageWAM.__init__=init
try:
 sys.argv=[str(ROOT/'scripts'/(a.entry+'.py'))]+rest;runpy.run_path(sys.argv[0],run_name='__main__');out=Path(rest[rest.index('--output')+1]);(out/'optimization.json').write_text(json.dumps({'mode':'rms_rope_experimental','source_sha256':{f:hashlib.sha256((ROOT/'scripts'/f).read_bytes()).hexdigest() for f in ['rms_rope_native_reduce_experiments.py','run_rms_rope_native_reduce_experimental.py']}},indent=2))
finally:rt.ImageWAM.__init__=original
