"""Non-interleaved Qwen RoPE with original BF16 multiplication/addition boundaries."""
import mlx.core as mx
from .qwen import Qwen

_qrope=mx.fast.metal_kernel(name='wam_qwen_rope_exact',input_names=['x','c','s'],output_names=['out'],ensure_row_contiguous=False,source=r'''
 uint i=thread_position_in_grid.x;if(i>=x_shape[0]*x_shape[1]*x_shape[2]*128)return;
 uint d=i%128,t=(i/128)%x_shape[2],h=(i/128/x_shape[2])%x_shape[1],b=i/(128*x_shape[2]*x_shape[1]);
 size_t base=b*x_strides[0]+h*x_strides[1]+t*x_strides[2];
 T a=x[base+d*x_strides[3]],v=x[base+((d+64)%128)*x_strides[3]];
 if(d<64)v=T(-float(v));
 T left=T(float(a)*float(c[t*c_strides[2]+d*c_strides[3]]));
 T right=T(float(v)*float(s[t*s_strides[2]+d*s_strides[3]]));
 out[i]=T(float(left)+float(right));
''')
def qrope(x,c,s):
 return _qrope(inputs=[x,c,s],template=[('T',x.dtype)],grid=(x.size,1,1),threadgroup=(256,1,1),output_shapes=[x.shape],output_dtypes=[x.dtype])[0]

class FusedRopeQwen(Qwen):
    @classmethod
    def from_qwen(cls,original):
        result=cls.__new__(cls);result.w=original.w;return result
    def rope(self,z,c,s):return qrope(z,c,s)
