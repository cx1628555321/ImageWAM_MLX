"""Opt-in INT8 prefix projections; changes numerical outputs. M5/MLX 0.32.4."""
import mlx.core as mx
from .device_visual import DeviceVisual
from . import staged_visual as sv

def project(layer,x,parts):
 shape=x.shape;m=x.size//shape[-1];k=shape[-1];n=parts[0].shape[0];flat=x.reshape(m,k);mp=(m+63)//64*64
 q,s,e=sv.quantize(flat)
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
