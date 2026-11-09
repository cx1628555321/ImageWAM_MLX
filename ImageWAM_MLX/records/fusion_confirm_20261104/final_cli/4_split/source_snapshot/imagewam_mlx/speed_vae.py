"""Accuracy-relaxed speed branch only: fused group LayerNorm and compiled VAE."""
import mlx.core as mx
from .vae import VAE

class SpeedVAE(VAE):
    @classmethod
    def from_vae(cls,original,dtype=mx.bfloat16):
        obj=cls.__new__(cls)
        obj.w={k:v.astype(dtype) for k,v in original.w.items()}
        mx.eval(obj.w);obj.dtype=dtype;obj._forward=mx.compile(obj._encode)
        return obj
    def gn(self,x,p):
        b,h,w,c=x.shape
        z=x.reshape(b,h*w,32,c//32).transpose(0,2,1,3).reshape(b,32,-1)
        z=mx.fast.layer_norm(z,None,None,eps=1e-6)
        z=z.reshape(b,32,h*w,c//32).transpose(0,2,1,3).reshape(x.shape)
        return z*self.w[p+'.weight']+self.w[p+'.bias']
    def _encode(self,x):return super().__call__(x)
    def __call__(self,x):return self._forward(x.astype(self.dtype)).astype(mx.bfloat16)
