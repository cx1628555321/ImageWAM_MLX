"""Opt-in scheduling modes; unchanged weights and runtime inputs."""
import argparse,json,hashlib,runpy,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from conditioning_experiments import ConditioningCandidates
import imagewam_mlx.model as mm

def main():
 p=argparse.ArgumentParser(add_help=False);p.add_argument('--mode',choices=['conditioning'],required=True);p.add_argument('--entry',choices=['infer','benchmark','validate_cases'],required=True);a,rest=p.parse_known_args();original=mm.ImageWAM.__init__
 def init(self,*args,**kwargs):
  original(self,*args,**kwargs);self._graph_candidates=ConditioningCandidates(self);self._graph_candidates.set(a.mode)
 mm.ImageWAM.__init__=init
 try:
  sys.argv=[str(ROOT/'scripts'/(a.entry+'.py'))]+rest;runpy.run_path(sys.argv[0],run_name='__main__')
  out=Path(rest[rest.index('--output')+1]);files=['run_conditioning_optimized.py','conditioning_experiments.py','graph_experiments.py','compile_experiments.py']
  (out/'optimization.json').write_text(json.dumps({'mode':a.mode,'entry':a.entry,'trace_policy':'same conditioned step for trace and normal mode; fixed model weights; schedule LRU size 4; clear schedules or recreate adapter after weight changes','source_sha256':{f:hashlib.sha256((ROOT/'scripts'/f).read_bytes()).hexdigest() for f in files}},indent=2))
  if a.entry=='validate_cases':
   rows=[]
   for f in sorted(out.glob('*.npy')):
    ref=ROOT/'records/validation_v1'/f.name
    if ref.exists():
     x,y=np.load(f),np.load(ref);rows.append({'file':f.name,'bitwise_equal':bool(np.array_equal(x,y)),'max_abs':float(np.max(np.abs(x.astype(np.float64)-y.astype(np.float64))))})
   assert sum(r['file'].endswith('_action.npy') for r in rows)==51
   (out/'conditioning_vs_original_mlx.json').write_text(json.dumps(rows,indent=2))
 finally:mm.ImageWAM.__init__=original
if __name__=='__main__':main()
