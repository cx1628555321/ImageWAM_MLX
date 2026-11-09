"""Full representative-case validation for an explicitly selected compile mode."""
import json,runpy,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from compile_experiments import Candidates
import os
os.environ['IMAGEWAM_MLX_OPTIMIZATION']='legacy'
import imagewam_mlx.model as mm
index=sys.argv.index('--mode');mode=sys.argv[index+1];del sys.argv[index:index+2]
if mode not in ('action_compile','visual_compile','both_compile'):raise ValueError(mode)
original=mm.ImageWAM.__init__
def init(self,*args,**kwargs):
 original(self,*args,**kwargs);self._compile_candidates=Candidates(self);self._compile_candidates.set(mode)
mm.ImageWAM.__init__=init
runpy.run_path(str(Path(__file__).with_name('validate_cases.py')),run_name='__main__')
out=Path(sys.argv[sys.argv.index('--output')+1]);reference=Path('records/validation_v1');rows=[]
for f in sorted(out.glob('*_action.npy')):
 old=reference/f.name
 if old.exists():
  x,y=np.load(f),np.load(old);rows.append({'file':f.name,'bitwise_equal':bool(np.array_equal(x,y)),'max_abs':float(np.max(abs(x-y))),'mae':float(np.mean(abs(x-y)))})
assert len(rows)==51
(out/'compiled_vs_original_mlx.json').write_text(json.dumps({'mode':mode,'cases':rows,'all_bitwise_equal':all(r['bitwise_equal'] for r in rows),'max_abs':max(r['max_abs'] for r in rows)},indent=2))
print('COMPILE_VALIDATION',mode,'bitwise',all(r['bitwise_equal'] for r in rows),flush=True)
