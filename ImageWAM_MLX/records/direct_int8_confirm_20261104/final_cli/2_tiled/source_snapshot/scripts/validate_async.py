"""Opt-in scheduling-only validation across the existing 51 real input cases.
Derives the identical denoise function from the current source and substitutes
only mx.eval(x) with mx.async_eval(x). Does not modify the production files.
"""
import inspect,json,runpy,sys,textwrap,types
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import imagewam_mlx.model as mm
original=mm.ImageWAM.__init__
src=textwrap.dedent(inspect.getsource(mm.ImageWAM.denoise))
if src.count('mx.eval(x)')!=1:raise RuntimeError('Unexpected denoise implementation; re-review scheduling change')
ns=dict(vars(mm));exec(src.replace('mx.eval(x)','mx.async_eval(x)'),ns)
def init(self,*args,**kwargs):
 original(self,*args,**kwargs);self.denoise=types.MethodType(ns['denoise'],self)
mm.ImageWAM.__init__=init
runpy.run_path(str(Path(__file__).with_name('validate_cases.py')),run_name='__main__')
out=Path(sys.argv[sys.argv.index('--output')+1]);reference=Path('records/validation_v1');rows=[]
for f in sorted(out.glob('*_action.npy')):
 old=reference/f.name
 if old.exists():
  x,y=np.load(f),np.load(old);rows.append({'file':f.name,'bitwise_equal':bool(np.array_equal(x,y)),'max_abs':float(np.max(abs(x-y)))})
(out/'async_vs_original_mlx.json').write_text(json.dumps({'cases':rows,'all_bitwise_equal':all(r['bitwise_equal'] for r in rows),'note':'51 representative cases against saved original MLX; independent output files retained.'},indent=2))
assert len(rows)==51 and all(r['bitwise_equal'] for r in rows)
print('ASYNC_51_CASES_BITWISE_EQUAL',flush=True)
