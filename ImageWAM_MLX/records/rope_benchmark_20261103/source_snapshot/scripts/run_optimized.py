"""Explicit, reversible optimized inference/benchmark entry. Defaults stay intact."""
import argparse,json,runpy,sys,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from compile_experiments import Candidates
import imagewam_mlx.model as mm
p=argparse.ArgumentParser(add_help=False);p.add_argument('--mode',choices=['async','action_compile','visual_compile','both_compile'],required=True);p.add_argument('--entry',choices=['infer','benchmark'],required=True)
a,rest=p.parse_known_args();original=mm.ImageWAM.__init__
def init(self,*args,**kwargs):
 original(self,*args,**kwargs);self._compile_candidates=Candidates(self);self._compile_candidates.set(a.mode)
mm.ImageWAM.__init__=init
sys.argv=[str(Path(__file__).with_name(a.entry+'.py'))]+rest
runpy.run_path(sys.argv[0],run_name='__main__')
if '--output' in rest:
 out=Path(rest[rest.index('--output')+1]);f=Path(__file__).with_name('compile_experiments.py')
 (out/'optimization.json').write_text(json.dumps({'mode':a.mode,'entry':a.entry,'compiler_implementation_sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'policy':'no truncation, no feature caching, BF16 weights unchanged; shape-specific compile with actual tensor inputs'},indent=2))
