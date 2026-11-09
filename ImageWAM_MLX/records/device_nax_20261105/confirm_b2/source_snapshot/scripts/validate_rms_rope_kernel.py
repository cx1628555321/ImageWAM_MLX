"""Random contiguous/strided RMS+RoPE comparison against original operators."""
import argparse,json,sys,shutil
from pathlib import Path
import numpy as np
import mlx.core as mx
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from rms_rope_experiments import fused,rms,rotate
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);rows=[];mx.random.seed(42)
for n in (1,16,64,328):
 for strided in (False,True):
  x=mx.random.normal((1,48,n*2,256)).astype(mx.bfloat16)[:,::2,::2,::2] if strided else mx.random.normal((1,24,n,128)).astype(mx.bfloat16)
  w=mx.random.uniform(.5,1.5,(128,)).astype(mx.bfloat16);angles=mx.random.normal((1,1,n,64));pe=(mx.cos(angles),mx.sin(angles))
  expected,actual=rotate(rms(x,w),pe),fused(x,w,pe);mx.eval(expected,actual);expected,actual=np.array(expected.astype(mx.float32)),np.array(actual.astype(mx.float32));rows.append({'n':n,'strided':strided,'max_abs':float(abs(expected-actual).max()),'different':int(np.count_nonzero(expected!=actual)),'count':expected.size});np.save(a.output/f'{n}_{strided}_reference.npy',expected);np.save(a.output/f'{n}_{strided}_fused.npy',actual)
(a.output/'results.json').write_text(json.dumps(rows,indent=2));shutil.copyfile(__file__,a.output/Path(__file__).name);print(rows)
