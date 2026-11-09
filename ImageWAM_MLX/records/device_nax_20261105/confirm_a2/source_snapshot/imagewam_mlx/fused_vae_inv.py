"""Reuse inverse standard deviation within each current-input VAE group."""
import mlx.core as mx
from .fused_vae import FusedVAE

_inv=mx.fast.metal_kernel(name='wam_group_inv_reuse',input_names=['x','mean','inv','w','bias'],output_names=['out'],ensure_row_contiguous=False,source=r'''
 uint i=thread_position_in_grid.x;if(i>=x_shape[0]*x_shape[1]*x_shape[2]*x_shape[3])return;
 uint c=i%x_shape[3],b=i/(x_shape[1]*x_shape[2]*x_shape[3]);uint g=c/(x_shape[3]/32);
 volatile float centered=float(x[i])-mean[b*32+g];
 volatile float normalized=centered*inv[b*32+g];
 volatile float scaled=normalized*float(w[c]);T z=T(scaled+float(bias[c]));
 auto sy=1/(1+metal::precise::exp(metal::abs(z)));T sig=(z<0)?sy:1-sy;
 out[i]=T(float(z)*float(sig));
''')
class GroupInverseVAE(FusedVAE):
 def gn_silu(self,x,p):
  b,h,w,c=x.shape;z=x.astype(mx.float32).reshape(b,h,w,32,c//32)
  mean=mx.mean(z,axis=(1,2,4),keepdims=True)
  var=mx.mean((z-mean)**2,axis=(1,2,4),keepdims=True)
  inv=mx.rsqrt(var+1e-6)
  return _inv(inputs=[mx.contiguous(x),mean,inv,self.w[p+'.weight'],self.w[p+'.bias']],template=[('T',x.dtype)],grid=(x.size,1,1),threadgroup=(256,1,1),output_shapes=[x.shape],output_dtypes=[x.dtype])[0]
