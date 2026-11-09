"""Exact-power centered-square kernel: keep original reduction ordering."""
import mlx.core as mx
from .fused_vae import FusedVAE,_gn

_centered_pow=mx.fast.metal_kernel(name='wam_centered_original_precise_pow',input_names=['x','mean','exponent'],output_names=['out'],source=r'''
 uint i=thread_position_in_grid.x;
 uint C=x_shape[3],HW=x_shape[1]*x_shape[2];
 if(i>=x_shape[0]*HW*C)return;
 uint b=i/(HW*C),g=(i%C)/(C/32);
 volatile float z=float(x[i])-mean[b*32+g];
 out[i]=metal::precise::pow(z,exponent[0]);
''')
class PowVAE(FusedVAE):
    @classmethod
    def from_vae(cls,original):
        obj=super().from_vae(original)
        # Runtime buffer prevents the compiler replacing pow(x,2) by x*x.
        obj._exponent=mx.array([2.0],dtype=mx.float32)
        return obj
    def squared(self,x,mean):
        return _centered_pow(inputs=[x,mean,self._exponent],grid=(x.size,1,1),threadgroup=(256,1,1),output_shapes=[x.shape],output_dtypes=[mx.float32])[0]
    def gn_silu(self,x,p):
        b,h,w,c=x.shape;z=x.astype(mx.float32).reshape(b,h,w,32,c//32)
        mean=mx.mean(z,axis=(1,2,4),keepdims=True)
        sq=self.squared(x,mean).reshape(b,h,w,32,c//32)
        var=mx.mean(sq,axis=(1,2,4),keepdims=True)
        return _gn(inputs=[mx.contiguous(x),mean,var,self.w[p+'.weight'],self.w[p+'.bias']],template=[('T',x.dtype)],grid=(x.size,1,1),threadgroup=(256,1,1),output_shapes=[x.shape],output_dtypes=[x.dtype])[0]

_table=mx.fast.metal_kernel(name='wam_bf16_centered_pow_table',input_names=['mean','exponent'],output_names=['out'],source=r'''
 uint i=thread_position_in_grid.x;
 uint bits=i%65536,g=i/65536;
 float x=as_type<float>(bits<<16);
 volatile float z=x-mean[g];
 out[i]=metal::precise::pow(z,exponent[0]);
''')
_lookup=mx.fast.metal_kernel(name='wam_bf16_centered_pow_lookup',input_names=['x','table'],output_names=['out'],source=r'''
 uint i=thread_position_in_grid.x,C=x_shape[3],HW=x_shape[1]*x_shape[2];
 if(i>=x_shape[0]*HW*C)return;
 uint g=i/(HW*C)*32+(i%C)/(C/32);
 device const ushort* bits=reinterpret_cast<device const ushort*>(x);
 out[i]=table[g*65536+uint(bits[i])];
''')
class LookupPowVAE(PowVAE):
    def squared(self,x,mean):
        if x.dtype!=mx.bfloat16 or x.size<8_000_000:
            return super().squared(x,mean)
        # Rebuild from this observation's means; never reuse across requests.
        table=_table(inputs=[mean,self._exponent],grid=(mean.size*65536,1,1),threadgroup=(256,1,1),output_shapes=[(mean.size,65536)],output_dtypes=[mx.float32])[0]
        return _lookup(inputs=[mx.contiguous(x),table],grid=(x.size,1,1),threadgroup=(256,1,1),output_shapes=[x.shape],output_dtypes=[mx.float32])[0]
