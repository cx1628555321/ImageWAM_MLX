"""Opt-in complete-model experiments; no production changes until benchmarks pass."""
import types
import mlx.core as mx
from native_int8 import pack_weight,linear,int8_gemm
from fused_quant import quantize_fused,quantize_concat_gate
from quant_attention import quant_kv,append_kv,attend_quantized
from quant_conv import QuantConv
from imagewam_mlx.runtime import RopeWeights
from imagewam_mlx.qwen import Qwen
from imagewam_mlx.model import ImageWAM as Base
from imagewam_mlx.ops import attention,silu,norm,gated,rms

class QInput:
 def __init__(self,parts,shape):self.parts=parts;self.shape=shape

class NativeMixin:
 def setup(self,original,fused=False):
  self.w=dict(original.w);self.packed={};self.fused=fused
  for k,w in list(self.w.items()):
   if k.endswith('.weight') and w.ndim==2 and 'embed_tokens' not in k and w.shape[1]>=256 and w.shape[0]>=64:
    self.packed[k[:-7]]=pack_weight(w);del self.w[k]
  return self
 def linear(self,x,p):
  if p not in self.packed:return super().linear(x,p)
  w,sw=self.packed[p];bias=self.w.get(p+'.bias',mx.zeros((w.shape[0],),dtype=mx.bfloat16))
  if isinstance(x,QInput):
   q,s=x.parts;return int8_gemm(q,w,s,sw,bias).reshape(*x.shape[:-1],w.shape[0])
  return linear(x,(w,sw),bias)

class NativeWeights(NativeMixin,RopeWeights):
 def norm_input(self,x,m):
  return QInput(quantize_fused(x,'layernorm',1+m[1],m[0]),x.shape)
 def double_prepare(self,x,p,m,pe,stream='img'):
  if not self.fused:return super().double_prepare(x,p,m,pe,stream)
  return self.qkv(self.linear(self.norm_input(x,m),p+'.'+stream+'_attn.qkv'),p+'.'+stream+'_attn.norm',pe)
 def double_post(self,x,att,p,m,stream='img'):
  if not self.fused:return super().double_post(x,att,p,m,stream)
  x=x+m[2]*self.linear(att,p+'.'+stream+'_attn.proj')
  z=QInput(quantize_fused(x,'layernorm',1+m[4],m[3]),x.shape)
  z=self.linear(z,p+'.'+stream+'_mlp.0')
  return x+m[5]*self.linear(QInput(quantize_fused(z,'gated'),(*z.shape[:-1],z.shape[-1]//2)),p+'.'+stream+'_mlp.2')
 def single_prepare(self,x,p,m,pe):
  if not self.fused:return super().single_prepare(x,p,m,pe)
  z=self.linear(self.norm_input(x,m),p+'.linear1');q,k,v=self.qkv(z[...,:9216],p+'.norm',pe)
  return q,k,v,z[...,9216:]
 def single_post(self,x,att,mlp,p,m):
  if not self.fused:return super().single_post(x,att,mlp,p,m)
  z=QInput(quantize_concat_gate(att,mlp),(*att.shape[:-1],att.shape[-1]+mlp.shape[-1]//2))
  return x+m[2]*self.linear(z,p+'.linear2')

class NativeQwen(NativeMixin,Qwen):pass

def quant_step(self,cache,x,t,delta,pe):
 a=self.action;vec=a.embed_time(t);md=a.modulation(vec,'double_stream_modulation_img',True);ms=a.modulation(vec,'single_stream_modulation');z=a.linear(x,'action_encoder')
 for i in range(5):
  p=f'double_blocks.{i}';q,k,v=a.double_prepare(z,p,md,pe)
  z=a.double_post(z,attend_quantized(q,append_kv(cache[i],k,v)),p,md)
 for i in range(20):
  p=f'single_blocks.{i}';q,k,v,mlp=a.single_prepare(z,p,ms,pe)
  z=a.single_post(z,attend_quantized(q,append_kv(cache[i+5],k,v)),mlp,p,ms)
 shift,scale=mx.split(a.linear(silu(vec),'head.adaLN_modulation.1'),2,axis=-1)
 pred=a.linear((1+scale[:,None])*norm(z)+shift[:,None],'head.linear')
 return x+pred*delta

def install(model,components=(),fused=False,kv=False,conv=False):
 for name in components:
  cls=NativeQwen if name=='qwen' else NativeWeights
  setattr(model,name,cls.__new__(cls).setup(getattr(model,name),fused))
 if conv:
  v=model.vae;packed={k[:-7]:QuantConv(w,v.w[k[:-7]+'.bias']) for k,w in v.w.items() if k.endswith('.weight') and w.ndim==4}
  def call(self,x,p,stride=1,pad=0):return packed[p](x,stride,pad)
  v.conv=types.MethodType(call,v)
  if hasattr(v,'_forward'):v._forward=mx.compile(v._encode)
 if kv:
  def pre(self,h,l,s):return [quant_kv(k,v) for k,v in Base.prefill(self,h,l,s)]
  model.prefill=mx.compile(types.MethodType(pre,model));model._compiled_step=mx.compile(types.MethodType(quant_step,model))
 else:
  model.prefill=mx.compile(types.MethodType(Base.prefill,model));model._compiled_step=mx.compile(model._step)
 model.experimental_options=dict(components=components,fused=fused,kv=kv,conv=conv)
 return model
