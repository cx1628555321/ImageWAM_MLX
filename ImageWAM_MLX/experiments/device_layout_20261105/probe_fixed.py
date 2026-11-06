import os
os.environ.setdefault('MLX_MAX_OPS_PER_BUFFER','256')
import sys,ast,json,copy
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
import numpy as np
import mlx.core as mx
from imagewam_mlx import ImageWAM
from imagewam_mlx.model import ImageWAM as Base
from imagewam_mlx.device_visual import DeviceVisual,sv
from candidate import Candidate,fixed_header
r=Path('records/device_layout_20261105');rows=[]
# Extract the current production and experimental integer accumulator stores.
def sources(path,names):
 tr=ast.parse(Path(path).read_text());result=[]
 for n in tr.body:
  if isinstance(n,ast.Assign) and isinstance(n.value,ast.Call) and any(isinstance(t,ast.Name) and t.id in names for t in n.targets):result.append(ast.literal_eval(next(k.value for k in n.value.keywords if k.arg=='source')))
 return result
src=sources('imagewam_mlx/device_visual.py',['full_kernel','tail_kernel'])
kernels={}
for mode in ('base','fixed'):
 kernels[mode]=[]
 for i,s in enumerate(src):
  if mode=='fixed':
   if i==1:
    a,b=s.split('} else {\nuint3 tid=',1);s=a+'} else {\nuint3 tid='+b.replace('constexpr int WM = 2;','constexpr int WM = 1;').replace('constexpr int WN = 2;','constexpr int WN = 4;')
   else:s=s.replace('constexpr int WM = 2;','constexpr int WM = 1;').replace('constexpr int WN = 2;','constexpr int WN = 4;')
  s=s.replace('static_cast<T>(float(frag[e]) * x_scale[r] * w_scale[c])','frag[e]')
  kernels[mode].append(mx.fast.metal_kernel(name=f'probe_acc_{mode}_{i}',input_names=['w','scales','biases','w_inv','w_scale','x8','x_scale','x_exp'],output_names=['y'],header=fixed_header if mode=='fixed' else sv.header,source=s))
class Probe(DeviceVisual):
 def mm(self,**kw):
  y=super().mm(**kw);mx.eval(y);ref=np.array(y[0].astype(mx.float32));row={'shape':kw['output_shapes'][0],'errors':{}}
  for mode in ('fixed',):
   z=Candidate(self,mode).mm(**kw);mx.eval(z);v=np.array(z[0].astype(mx.float32));row['errors'][mode]={'equal':bool(np.array_equal(ref,v)),'different':int(np.count_nonzero(ref!=v)),'max_abs':float(np.abs(ref-v).max())}
  k=dict(kw);k['output_dtypes']=[mx.int32];m=dict(k['template'])['ROWS'];idx=int(0<m%64<=32)
  a=kernels['base'][idx](**k)[0];b=kernels['fixed'][idx](**k)[0];mx.eval(a,b);a=np.array(a);b=np.array(b);row['integer_acc_equal']=bool(np.array_equal(a,b));row['integer_acc_different']=int(np.count_nonzero(a!=b));rows.append(row)
  print(len(rows),row,flush=True)
  return y
base=ImageWAM('weights/bf16','assets',profile='speed',text_cache_size=0,device_visual=True);p=Path('examples/robotwin_ep0_frame0');raw=np.load(p/'inputs.npz');x,st=base.preprocess({k:raw[k] for k in ('cam_high','cam_left_wrist','cam_right_wrist')},raw['state']);hidden=base.qwen(mx.array(base.tokenize((p/'prompt.txt').read_text())));latent=base.vae(x);mx.eval(hidden,latent,st)
base.visual=Probe(base.visual);cache=Base.prefill(base,hidden,latent,st);mx.eval(cache)
(r/'probe_fixed.json').write_text(json.dumps(rows,indent=2))
