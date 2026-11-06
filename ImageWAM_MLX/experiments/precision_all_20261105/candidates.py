import copy,types
from pathlib import Path
import mlx.core as mx
from imagewam_mlx.device_visual import DeviceVisual
from imagewam_mlx import staged_visual as sv,action_kernel as ak
from imagewam_mlx.model import ImageWAM as Base
from imagewam_mlx.workspace import workspace_step
from imagewam_mlx.ops import D,silu,norm,flux_rope
import numpy as np

fixed=mx.fast.metal_kernel(name='wam_fixed_x8',input_names=['x','calibrated_scale'],output_names=['q','scale'],source='''
uint i=thread_position_in_grid.x;if(i>=MP*K)return;uint row=i/K,col=i%K;
uint dest=((row/64)*(K/64)+col/64)*4096+(row%64)*64+col%64;
q[dest]=row<M?char(clamp(round(float(x[row*K+col])/calibrated_scale[0]),-127.0f,127.0f)):char(0);
if(col==0&&row<M)scale[row]=calibrated_scale[0];
''')
def project(layer,x,parts,scale=None):
 shape=x.shape;m=x.size//shape[-1];k=shape[-1];n=parts[0].shape[0];flat=x.reshape(m,k);mp=(m+63)//64*64
 if scale is None:q,s,e=sv.quantize(flat)
 else:
  q,s=fixed(inputs=[flat,mx.array([scale],dtype=mx.float32)],template=[('M',m),('MP',mp),('K',k)],grid=(mp*k,1,1),threadgroup=(256,1,1),output_shapes=[(mp,k),(m,)],output_dtypes=[mx.int8,mx.float32]);e=mx.zeros((mp,k//64+1),mx.uint8)
 y=layer.mm(inputs=[*parts,q,s,e],template=[('T',x.dtype),('group_size',64),('bits',4),('x_span',0),('XK',k),('ROWS',m)],grid=((n+63)//64*128,(m+63)//64,1),threadgroup=(128,1,1),output_shapes=[(m,n)],output_dtypes=[x.dtype])[0]
 return y.reshape(*shape[:-1],n)
class ExtraVisual(DeviceVisual):
 def __init__(self,orig,scope):
  self.__dict__=dict(orig.__dict__);self.w=dict(orig.w);self.extra={};self.inventory=[]
  for key,w in list(self.w.items()):
   if not key.endswith('.weight') or w.ndim!=2 or min(w.shape)<64 or w.shape[0]%64 or w.shape[1]%64:continue
   p=key[:-7];sel=(p.startswith('double_blocks.') and (scope=='all' or '_mlp.' in p)) or (scope=='all' and p in ('img_in','txt_in','single_blocks.19.kv'))
   if not sel:continue
   scale=mx.maximum(mx.max(mx.abs(w.astype(mx.float32)),axis=1)/127,1e-12);q=mx.clip(mx.round(w.astype(mx.float32)/scale[:,None]),-127,127).astype(mx.int8);dummy=mx.zeros((1,),mx.float32);mx.eval(q,scale)
   self.extra[p]=(q,dummy,dummy,dummy,scale);self.inventory.append(dict(name=p,shape=list(w.shape)));del self.w[key]
 def linear(self,x,p):
  if p not in self.extra:return super().linear(x,p)
  y=project(self,x,self.extra[p]);bias=self.w.get(p+'.bias');return y if bias is None else y+bias
class FixedVisual(DeviceVisual):
 def __init__(self,orig,scales):self.__dict__=dict(orig.__dict__);self.scales=scales
 def linear(self,x,p):
  if p not in self.scales:return super().linear(x,p)
  y=project(self,x,self.packed[p],self.scales[p]);bias=self.w.get(p+'.bias');return y if bias is None else y+bias

_SPLIT={}
def split(x,parts,residual,gate,bias,target=512,tile=(32,64,2,2),bits=1,partial='float'):
 key=(bits,partial)
 if key not in _SPLIT:
  head,tail=ak.header.split('METAL_FUNC void split_qmm(',1);tail=tail.replace('device float* y, const int splits',f'device {partial}* y, const int splits')
  _SPLIT[key]=mx.fast.metal_kernel(name=f'wam_bits{bits}_{partial}',input_names=['x','w','scales','qb'],output_names=['out'],header=head+'METAL_FUNC void split_qmm('+tail,source=f'''threadgroup T Ws[BN*72];int M=x_shape[1],N=w_shape[0],K=x_shape[2];split_qmm<T,64,{bits},true,BM,64,BN,WM,WN>(w,scales,qb,x,out,SPLITS,Ws,K,N,M,threadgroup_position_in_grid,thread_index_in_threadgroup,simdgroup_index_in_threadgroup,thread_index_in_simdgroup);''')
 bm,bn,wm,wn=tile;threads=32*wm*wn;m,n=residual.shape[1:];k=x.shape[-1];nt=(n+bn-1)//bn;mt=(m+bm-1)//bm;s=min(max(1,target//(nt*mt)),k//64)
 while s>1 and k%(s*64):s-=1
 p=_SPLIT[key](inputs=[x,*parts],template=[('T',x.dtype),('SPLITS',s),('BM',bm),('BN',bn),('WM',wm),('WN',wn)],grid=(nt*threads,mt,s),threadgroup=(threads,1,1),output_shapes=[(s,m,n)],output_dtypes=[{'float':mx.float32,'half':mx.float16,'bfloat':mx.bfloat16}[partial]])[0]
 return ak.reduce(inputs=[p,residual,gate,bias],template=[('T',x.dtype),('SPLITS',s)],grid=(residual.size,1,1),threadgroup=(256,1,1),output_shapes=[residual.shape],output_dtypes=[x.dtype])[0]

def action_copy(orig,bits=1,partial='float',raw_weights=None):
 a=copy.copy(orig);a.w=dict(orig.w);a.packed=dict(orig.packed)
 if bits!=1:
  # Quantize original BF16 weights, not already degraded W1 values.
  for p in a.packed:
   a.packed[p]=mx.quantize(raw_weights[p+'.weight'],group_size=64,bits=bits);mx.eval(a.packed[p])
  a.bits=bits;a.kind='w'+str(bits)
 def residual(self,x,p,res,gate):
  if p in self.packed and x.shape[0]==1 and x.shape[1]>13 and x.shape[-1]>4096:
   bias=self.w.get(p+'.bias',mx.zeros((res.shape[-1],),dtype=x.dtype));return split(x,self.packed[p],res,gate,bias,target=self.split_target,tile=(32,64,2,2) if x.shape[1]>16 else self.split_tile,bits=bits,partial=partial)
  return res+gate*self.linear(x,p)
 a.residual_linear=types.MethodType(residual,a);return a

def install(m,mode,raw=None,scales=None):
 if mode=='baseline':return []
 if mode.startswith('combined_'):
  m.visual=ExtraVisual(m.visual,'all');m.prefill=mx.compile(lambda h,l,s:Base.prefill(m,h,l,s))
  m.action=action_copy(m.action,bits=4,raw_weights=raw) if mode=='combined_w4' else action_copy(m.action,partial='bfloat')
  m._compiled_workspace_step=mx.compile(lambda c,x,t,d,p:workspace_step(m,c,x,t,d,p));return m.visual.inventory
 if mode.startswith('visual_'):
  m.visual=ExtraVisual(m.visual,mode[7:]);m.prefill=mx.compile(lambda h,l,s:Base.prefill(m,h,l,s));return m.visual.inventory
 if mode=='fixed_scale':
  m.visual=FixedVisual(m.visual,scales);m.prefill=mx.compile(lambda h,l,s:Base.prefill(m,h,l,s));return list(scales)
 if mode.startswith('action_w'):m.action=action_copy(m.action,bits=int(mode[-1]),raw_weights=raw)
 elif mode.startswith('partial_'):m.action=action_copy(m.action,partial=mode[8:])
 elif mode=='kv_int8':
  from kv_fused import denoise
  m.denoise=types.MethodType(denoise,m);return []
 else:raise ValueError(mode)
 m._compiled_workspace_step=mx.compile(lambda c,x,t,d,p:workspace_step(m,c,x,t,d,p));return []

def action_int8(orig,raw):
 a=copy.copy(orig);extra={};engine=DeviceVisual.__new__(DeviceVisual)
 for p in orig.packed:
  if not p.startswith(('double_blocks.','single_blocks.')):continue
  w=raw[p+'.weight'];s=mx.maximum(mx.max(mx.abs(w.astype(mx.float32)),axis=1)/127,1e-12);q=mx.clip(mx.round(w.astype(mx.float32)/s[:,None]),-127,127).astype(mx.int8);dummy=mx.zeros((1,),mx.float32);mx.eval(q,s);extra[p]=(q,dummy,dummy,dummy,s)
 oldlinear=orig.linear;oldres=orig.residual_linear
 def linear(self,x,p):
  if p not in extra or x.shape[-2]<14:return oldlinear(x,p)
  y=project(engine,x,extra[p]);bias=self.w.get(p+'.bias');return y if bias is None else y+bias
 def residual(self,x,p,res,gate):
  if p in extra and x.shape[-2]>=14:return res+gate*self.linear(x,p)
  return oldres(x,p,res,gate)
 a.linear=types.MethodType(linear,a);a.residual_linear=types.MethodType(residual,a);return a
