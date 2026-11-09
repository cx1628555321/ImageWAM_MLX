"""One kernel produces INT8 activation + FP32 scale from normalized/gated input."""
import mlx.core as mx
_kernel=mx.fast.metal_kernel(name='wam_fused_norm_quant',input_names=['x','gamma','beta'],output_names=['q','scales'],source=r'''
 uint row=threadgroup_position_in_grid.x,tid=thread_position_in_threadgroup.x;
 threadgroup float red[8];threadgroup float avg,inv,scale;
 if(MODE==1||MODE==2){
 float s=0;for(uint j=tid;j<K;j+=256)s+=float(x[row*IK+j]);
 s=simd_sum(s);if(tid%32==0)red[tid/32]=s;threadgroup_barrier(mem_flags::mem_threadgroup);
 if(tid<32){float z=tid<8?red[tid]:0;z=simd_sum(z);if(tid==0)avg=MODE==1?z/float(K):0;}
 threadgroup_barrier(mem_flags::mem_threadgroup);
 s=0;for(uint j=tid;j<K;j+=256){float z=float(x[row*IK+j])-avg;s+=z*z;}
 s=simd_sum(s);if(tid%32==0)red[tid/32]=s;threadgroup_barrier(mem_flags::mem_threadgroup);
 if(tid<32){float z=tid<8?red[tid]:0;z=simd_sum(z);if(tid==0)inv=(MODE==1||MODE==2)?rsqrt(z/float(K)+1e-6f):1;}
 threadgroup_barrier(mem_flags::mem_threadgroup);
 }else{if(tid==0){avg=0;inv=1;}threadgroup_barrier(mem_flags::mem_threadgroup);}
 float a=0;
 for(uint j=tid;j<K;j+=256){float z=(float(x[row*IK+j])-avg)*inv;
  if(MODE==1||MODE==2)z=z*float(gamma[j])+float(beta[j]);
  if(MODE==3||MODE==4)z=z/(1+exp(-z));
  if(MODE==4)z*=float(x[row*IK+K+j]);a=max(a,abs(z));}
 a=simd_max(a);if(tid%32==0)red[tid/32]=a;threadgroup_barrier(mem_flags::mem_threadgroup);
 if(tid<32){float z=tid<8?red[tid]:0;z=simd_max(z);if(tid==0){scale=max(z/127.0f,1e-12f);scales[row]=scale;}}
 threadgroup_barrier(mem_flags::mem_threadgroup);
 for(uint j=tid;j<K;j+=256){float z=(float(x[row*IK+j])-avg)*inv;
  if(MODE==1||MODE==2)z=z*float(gamma[j])+float(beta[j]);
  if(MODE==3||MODE==4)z=z/(1+exp(-z));
  if(MODE==4)z*=float(x[row*IK+K+j]);
  q[row*K+j]=char(clamp(rint(z/scale),-127.0f,127.0f));}
''')
def quantize_fused(x,mode='layernorm',gamma=None,beta=None,eps=1e-6):
 if eps!=1e-6:raise ValueError('This prototype supports eps=1e-6')
 modes={'layernorm':1,'rms':2,'silu':3,'gated':4};code=modes[mode];ik=x.shape[-1];k=ik//2 if code==4 else ik;m=x.size//ik
 x=mx.contiguous(x.reshape(m,ik));gamma=mx.ones((k,),dtype=x.dtype) if gamma is None else gamma.reshape(-1);beta=mx.zeros((k,),dtype=x.dtype) if beta is None else beta.reshape(-1)
 return tuple(_kernel(inputs=[x,gamma,beta],template=[('MODE',code),('K',k),('IK',ik)],grid=(m*256,1,1),threadgroup=(256,1,1),output_shapes=[(m,k),(m,)],output_dtypes=[mx.int8,mx.float32]))

_concat=mx.fast.metal_kernel(name='wam_gate_concat_quant',input_names=['a','z'],output_names=['q','scales'],source=r'''
 uint row=threadgroup_position_in_grid.x,tid=thread_position_in_threadgroup.x;
 threadgroup float red[8];threadgroup float scale;
 float vmax=0;
 for(uint j=tid;j<A+G;j+=256){float v;if(j<A)v=float(a[row*A+j]);else{float f=float(z[row*G*2+j-A]);v=f/(1+exp(-f))*float(z[row*G*2+G+j-A]);}vmax=max(vmax,abs(v));}
 vmax=simd_max(vmax);if(tid%32==0)red[tid/32]=vmax;threadgroup_barrier(mem_flags::mem_threadgroup);
 if(tid<32){float v=tid<8?red[tid]:0;v=simd_max(v);if(tid==0){scale=max(v/127.0f,1e-12f);scales[row]=scale;}}
 threadgroup_barrier(mem_flags::mem_threadgroup);
 for(uint j=tid;j<A+G;j+=256){float v;if(j<A)v=float(a[row*A+j]);else{float f=float(z[row*G*2+j-A]);v=f/(1+exp(-f))*float(z[row*G*2+G+j-A]);}q[row*(A+G)+j]=char(clamp(rint(v/scale),-127.0f,127.0f));}
''')
def quantize_concat_gate(a,z):
 ac=a.shape[-1];gc=z.shape[-1]//2;m=a.size//ac
 return tuple(_concat(inputs=[mx.contiguous(a),mx.contiguous(z)],template=[('A',ac),('G',gc)],grid=(m*256,1,1),threadgroup=(256,1,1),output_shapes=[(m,ac+gc),(m,)],output_dtypes=[mx.int8,mx.float32]))
