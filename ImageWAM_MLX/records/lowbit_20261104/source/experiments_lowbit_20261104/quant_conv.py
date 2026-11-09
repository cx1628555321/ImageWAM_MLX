"""Prototype INT8 convolution via im2col + native GEMM. Includes packing traffic."""
import mlx.core as mx
from native_int8 import pack_weight,linear
class QuantConv:
 def __init__(self,w,bias):
  # NHWC convolution weight [O,KH,KW,I].
  self.kh,self.kw=w.shape[1:3];self.weight=pack_weight(w.reshape(w.shape[0],-1));self.bias=bias
 def __call__(self,x,stride=1,pad=0):
  b,h,w,c=x.shape;kh,kw=self.kh,self.kw
  z=mx.pad(x,[(0,0),(pad,pad),(pad,pad),(0,0)]) if pad else x
  oh=(h+2*pad-kh)//stride+1;ow=(w+2*pad-kw)//stride+1
  patches=mx.concatenate([z[:,i:i+oh*stride:stride,j:j+ow*stride:stride,:] for i in range(kh) for j in range(kw)],axis=-1)
  return linear(patches,self.weight,self.bias)
