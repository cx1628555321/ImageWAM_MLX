"""Opt-in final visual K/V projection experiment on the packaged default."""
import argparse,json,runpy,sys,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import imagewam_mlx.runtime as rt
from selective_kv_experiments import SelectiveCandidates
p=argparse.ArgumentParser(add_help=False);p.add_argument('--entry',choices=['infer','benchmark','validate_cases'],required=True);a,rest=p.parse_known_args();original=rt.ImageWAM.__init__
def init(self,*args,**kwargs):
 original(self,*args,**kwargs)
 if self.optimization!='default':raise ValueError('Selective K/V requires default runtime')
 self._selective=SelectiveCandidates(self);self._selective.set('selective_kv')
rt.ImageWAM.__init__=init
try:
 sys.argv=[str(ROOT/'scripts'/(a.entry+'.py'))]+rest;runpy.run_path(sys.argv[0],run_name='__main__');out=Path(rest[rest.index('--output')+1]);(out/'optimization.json').write_text(json.dumps({'mode':'selective_kv_experimental','source_sha256':{f:hashlib.sha256((ROOT/'scripts'/f).read_bytes()).hexdigest() for f in ['selective_kv_experiments.py','run_selective_kv_experimental.py']}},indent=2))
finally:rt.ImageWAM.__init__=original
