import mlx.core as mx
kernel=mx.fast.metal_kernel(name='imagewam_gate_concat_exact',input_names=['att','mlp'],output_names=['out'],ensure_row_contiguous=False,source='''
uint i=thread_position_in_grid.x;
uint a=att_shape[2], d=mlp_shape[2]/2, n=a+d;
if(i>=att_shape[0]*att_shape[1]*n)return;
uint col=i%n,row=(i/n)%att_shape[1],batch=i/n/att_shape[1];
if(col<a){out[i]=att[batch*att_strides[0]+row*att_strides[1]+col*att_strides[2]];return;}
uint j=col-a;size_t off=batch*mlp_strides[0]+row*mlp_strides[1]+j*mlp_strides[2];
T x=mlp[off],y=mlp[off+d*mlp_strides[2]];
auto sy=1 / (1 + metal::precise::exp(metal::abs(x)));
T sig=(x < 0) ? sy : 1-sy;
T act=T(x*sig);out[i]=T(act*y);
''')
def gate_concat(att,mlp):
 shape=(*att.shape[:-1],att.shape[-1]+mlp.shape[-1]//2)
 return kernel(inputs=[att,mlp],template=[('T',att.dtype)],grid=(shape[0]*shape[1]*shape[2],1,1),threadgroup=(256,1,1),output_shapes=[shape],output_dtypes=[att.dtype])[0]
