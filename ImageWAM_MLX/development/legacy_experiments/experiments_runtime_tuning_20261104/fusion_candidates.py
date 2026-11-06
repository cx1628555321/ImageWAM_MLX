import os
os.environ['MLX_MAX_OPS_PER_BUFFER']='256'
import ast,copy,json,time,sys
from pathlib import Path
import numpy as np
import mlx.core as mx
sys.path.insert(0,str(Path.cwd()))
from imagewam_mlx import ImageWAM
from imagewam_mlx.model import ImageWAM as Base
from imagewam_mlx import prepacked_visual as pv
from imagewam_mlx.ops import gated
sv=pv.sv
tree=ast.parse(Path(sv.__file__).read_text());call=next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='qx' for t in n.targets))
src=eval(compile(ast.Expression(next(k.value for k in call.keywords if k.arg=='source')),'source','eval'))
src=src.replace('const int K=x_shape[1],M=x_shape[0],max_exp=0,x_span=0;', 'const int KA=att_shape[1],KB=mlp_shape[1],K=KA+KB,M=att_shape[0],max_exp=0,x_span=0;')
needle='x += size_t(row) * K + lid * BK;';assert needle in src
src=src.replace(needle,'const device T* x = lid*BK < KA ? att + size_t(row)*KA + lid*BK : mlp + size_t(row)*KB + lid*BK-KA;')
qx=mx.fast.metal_kernel(name='wam_split_input_quant',input_names=['att','mlp'],output_names=['x8','x_scale','x_exp'],header=sv.header,source=src)
kernels={'split_quant':None}
class SplitCandidate(pv.PrepackedVisual):
 def __init__(self,orig):self.w=orig.w;self.packed=orig.packed
 def single_post(self,x,att,mlp,p,m):
  key=p+'.linear2'
  if key not in self.packed:return super().single_post(x,att,mlp,p,m)
  g=gated(mlp);a0=att.reshape(-1,att.shape[-1]);g=g.reshape(-1,g.shape[-1]);rows=a0.shape[0];k=a0.shape[1]+g.shape[1];mp=(rows+63)//64*64;nb=k//64;threads=(nb+31)//32*32;assert a0.shape[1]%64==0 and g.shape[1]%64==0
  a,s,e=qx(inputs=[a0,g],template=[('T',a0.dtype)],grid=(threads,mp,1),threadgroup=(threads,1,1),output_shapes=[(mp,k),(rows,),(mp,nb+1)],output_dtypes=[mx.int8,mx.float32,mx.uint8])
  parts=self.packed[key];n=parts[0].shape[0]
  y=pv.kernel(inputs=[*parts,a,s,e],template=[('T',x.dtype),('group_size',64),('bits',4),('x_span',0),('XK',k),('ROWS',rows)],grid=(((n+63)//64)*128,(rows+63)//64,1),threadgroup=(128,1,1),output_shapes=[(rows,n)],output_dtypes=[x.dtype])[0].reshape(*x.shape[:-1],n)
  bias=self.w.get(key+'.bias')
  if bias is not None:y=y+bias
  return x+m[2]*y
import os
os.environ['MLX_MAX_OPS_PER_BUFFER']='256'
import ast,copy,json,time,sys
from pathlib import Path
import numpy as np
import mlx.core as mx
sys.path.insert(0,str(Path.cwd()))
from imagewam_mlx import ImageWAM
from imagewam_mlx.model import ImageWAM as Base
from imagewam_mlx import prepacked_visual as pv
from imagewam_mlx.ops import gated
sv=pv.sv
tree=ast.parse(Path(pv.__file__).read_text());call=next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='kernel' for t in n.targets));src=ast.literal_eval(next(k.value for k in call.keywords if k.arg=='source'))
needle='y[size_t(r) * N + c] =\n                static_cast<T>(float(frag[e]) * x_scale[r] * w_scale[c]);'
replacement='T yy = static_cast<T>(float(frag[e]) * x_scale[r] * w_scale[c]); if(HAS_BIAS) yy=T(float(yy)+float(bias[c])); T scaled=T(float(yy)*float(alpha[c])); y[size_t(r)*N+c]=T(float(residual[size_t(r)*N+c])+float(scaled));'
assert src.count(needle)==2;src=src.replace(needle,replacement)
kernel=mx.fast.metal_kernel(name='wam_visual_residual_epilogue',input_names=['w','scales','biases','w_inv','w_scale','x8','x_scale','x_exp','alpha','residual','bias'],output_names=['y'],header=sv.header,source=src)
kernels={'epilogue':None}
class EpilogueCandidate(pv.PrepackedVisual):
 def __init__(self,orig):self.w=orig.w;self.packed=orig.packed
 def single_post(self,x,att,mlp,p,m):
  key=p+'.linear2'
  if key not in self.packed:return super().single_post(x,att,mlp,p,m)
  z=mx.concatenate([att,gated(mlp)],axis=-1).reshape(-1,att.shape[-1]+mlp.shape[-1]//2);rows,k=z.shape;parts=self.packed[key];n=parts[0].shape[0];a,s,e=sv.quantize(z);bias=self.w.get(key+'.bias');has=bias is not None
  if bias is None:bias=mx.zeros((n,),x.dtype)
  y=kernel(inputs=[*parts,a,s,e,m[2].reshape(-1),x.reshape(rows,n),bias],template=[('T',x.dtype),('group_size',64),('bits',4),('x_span',0),('XK',k),('ROWS',rows),('HAS_BIAS',has)],grid=(((n+63)//64)*128,(rows+63)//64,1),threadgroup=(128,1,1),output_shapes=[(rows,n)],output_dtypes=[x.dtype])[0].reshape(x.shape)
  return y
class CombinedCandidate(pv.PrepackedVisual):
 def __init__(self,orig):self.w=orig.w;self.packed=orig.packed
 def single_post(self,x,att,mlp,p,m):
  key=p+'.linear2'
  if key not in self.packed:return super().single_post(x,att,mlp,p,m)
  g=gated(mlp);a0=att.reshape(-1,att.shape[-1]);g=g.reshape(-1,g.shape[-1]);rows=a0.shape[0];k=a0.shape[1]+g.shape[1];mp=(rows+63)//64*64;nb=k//64;threads=(nb+31)//32*32;assert a0.shape[1]%64==0 and g.shape[1]%64==0
  a,s,e=qx(inputs=[a0,g],template=[('T',a0.dtype)],grid=(threads,mp,1),threadgroup=(threads,1,1),output_shapes=[(mp,k),(rows,),(mp,nb+1)],output_dtypes=[mx.int8,mx.float32,mx.uint8])
  parts=self.packed[key];n=parts[0].shape[0];bias=self.w.get(key+'.bias');has=bias is not None
  if bias is None:bias=mx.zeros((n,),x.dtype)
  y=kernel(inputs=[*parts,a,s,e,m[2].reshape(-1),x.reshape(rows,n),bias],template=[('T',x.dtype),('group_size',64),('bits',4),('x_span',0),('XK',k),('ROWS',rows),('HAS_BIAS',has)],grid=(((n+63)//64)*128,(rows+63)//64,1),threadgroup=(128,1,1),output_shapes=[(rows,n)],output_dtypes=[x.dtype])[0].reshape(*x.shape[:-1],n)
  return y
