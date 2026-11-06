import os
os.environ['MLX_MAX_OPS_PER_BUFFER']='256'
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
import mlx.core as mx
from imagewam_mlx.action_kernel import fused
r=Path('records/audit4_20261105/export');r.mkdir(exist_ok=False);result={}
# Sanity-check export/import before testing an actual hot custom primitive.
x=mx.array([1.,2.]);mx.export_function(str(r/'sanity.mlxfn'),lambda a:mx.sin(a)+a,x);f=mx.import_function(str(r/'sanity.mlxfn'));result['sanity_exact']=bool(mx.array_equal(f(x)[0],mx.sin(x)+x).item())
p=Path('records/roadmap_20261105/capture');d=mx.load(str(p/'projection_inputs.npz'));kw=json.loads((p/'projection_parameters.json').read_text());parts=[d[f'part_{i}'] for i in range(len([k for k in d if k.startswith('part_')]))]
def function(x,w,s,q,residual,gate,bias):return fused(x,[w,s,q],residual,gate,bias,**kw)
args=[d['x'],*parts,d['residual'],d['gate'],d['bias']]
try:
 mx.export_function(str(r/'real_projection.mlxfn'),function,*args);fn=mx.import_function(str(r/'real_projection.mlxfn'));a=function(*args);b=fn(*args)[0];mx.eval(a,b);result['projection_exact']=bool(mx.array_equal(a,b).item());result['projection_status']='export and import successful'
except Exception as e:result['projection_status']='blocked';result['projection_error']=repr(e)
(r/'results.json').write_text(json.dumps(result,indent=2));print(result,flush=True)
