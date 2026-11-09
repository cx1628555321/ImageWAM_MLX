"""VAE residual-block GN affine + SiLU fusion; original reductions and BF16 boundaries.

Matches MLX Sigmoid's stable branch and precise exponential. No learned values
or features are cached; original VAE is retained for legacy/opt-out execution.
"""
import mlx.core as mx
from .vae import VAE

_gn=mx.fast.metal_kernel(name='wam_gn_affine_silu_precise',input_names=['x','mean','var','w','bias'],output_names=['out'],ensure_row_contiguous=False,source=r'''
 uint i=thread_position_in_grid.x;if(i>=x_shape[0]*x_shape[1]*x_shape[2]*x_shape[3])return;
 uint c=i%x_shape[3],b=i/(x_shape[1]*x_shape[2]*x_shape[3]);uint g=c/(x_shape[3]/32);
 volatile float centered=float(x[i])-mean[b*32+g];
 volatile float inv=metal::precise::rsqrt(var[b*32+g]+1e-6f);
 volatile float normalized=centered*inv;
 volatile float scaled=normalized*float(w[c]);
 T z=T(scaled+float(bias[c]));
 auto sy=1/(1+metal::precise::exp(metal::abs(z))); T sig=(z<0)?sy:1-sy;
 out[i]=T(float(z)*float(sig));
''')
class FusedVAE(VAE):
    @classmethod
    def from_vae(cls,original):
        # original.w is already NHWC; do not transpose the kernels again.
        result=cls.__new__(cls)
        result.w=original.w
        return result
    def gn_silu(self,x,p):
        b,h,w,c=x.shape;z=x.astype(mx.float32).reshape(b,h,w,32,c//32)
        mean=mx.mean(z,axis=(1,2,4),keepdims=True)
        var=mx.mean((z-mean)**2,axis=(1,2,4),keepdims=True)
        return _gn(inputs=[mx.contiguous(x),mean,var,self.w[p+'.weight'],self.w[p+'.bias']],template=[('T',x.dtype)],grid=(x.size,1,1),threadgroup=(256,1,1),output_shapes=[x.shape],output_dtypes=[x.dtype])[0]
    def res(self,x,p):
        y=self.conv(self.gn_silu(x,p+'.norm1'),p+'.conv1',pad=1)
        y=self.conv(self.gn_silu(y,p+'.norm2'),p+'.conv2',pad=1)
        if p+'.nin_shortcut.weight' in self.w:x=self.conv(x,p+'.nin_shortcut')
        return x+y

