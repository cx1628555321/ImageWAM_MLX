"""Independent random strided and boundary shape checks against original RoPE."""
import json,sys,argparse
from pathlib import Path
import numpy as np
import mlx.core as mx
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rope_experiments import fused_rotate
from imagewam_mlx.ops import rotate
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);mx.random.seed(123);rows=[]
for n in (1,8,16,32,64):
 for strided in (False,True):
  x=mx.random.normal((1,48,n*2,256)).astype(mx.bfloat16)[:,::2,::2,::2] if strided else mx.random.normal((1,24,n,128)).astype(mx.bfloat16)
  angles=mx.random.normal((1,1,n*2,64));c=mx.cos(angles)[:,:,::2,:];s=mx.sin(angles)[:,:,::2,:]
  expected=rotate(x,(c,s));actual=fused_rotate(x,(c,s));mx.eval(expected,actual)
  lhs,rhs=np.array(expected.astype(mx.float32)),np.array(actual.astype(mx.float32));diff=float(np.max(abs(lhs-rhs)));row={'n':n,'strided':strided,'max_abs':diff,'bitwise':bool(np.array_equal(lhs,rhs))};rows.append(row)
  np.save(a.output/f'{n}_{strided}_actual.npy',rhs);np.save(a.output/f'{n}_{strided}_expected.npy',lhs)
(a.output/'results.json').write_text(json.dumps({'cases':rows,'all_bitwise':all(r['bitwise'] for r in rows)},indent=2));print(rows)
