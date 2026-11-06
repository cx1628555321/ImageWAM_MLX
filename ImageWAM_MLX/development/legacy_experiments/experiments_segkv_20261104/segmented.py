"""Experimental single-kernel dual-source KV attention; FP32 logits/softmax/PV.
No concatenation, cache mutation, or cross-request reuse. D=128 only.
"""
import mlx.core as mx
_KERNEL=mx.fast.metal_kernel(name='wam_segmented_sdpa',input_names=['q','pk','pv','ak','av'],output_names=['out'],ensure_row_contiguous=False,source=r'''
 uint tid=thread_position_in_threadgroup.x;
 uint row=threadgroup_position_in_grid.x, h=row/Q, qi=row%Q;
 threadgroup float logits[L];threadgroup float reduced[4];threadgroup float mxval,denom;
 float top=-INFINITY;
 for(uint j=tid;j<L;j+=128){
  float dot=0;
  for(uint d=0;d<128;d++){
   float a=float(q[h*q_strides[1]+qi*q_strides[2]+d*q_strides[3]]);
   float b=j<P?float(pk[h*pk_strides[1]+j*pk_strides[2]+d*pk_strides[3]]):float(ak[h*ak_strides[1]+(j-P)*ak_strides[2]+d*ak_strides[3]]);
   dot+=a*b;
  }
  dot*=0.08838834764831845f;logits[j]=dot;top=max(top,dot);
 }
 top=simd_max(top);if(tid%32==0)reduced[tid/32]=top;threadgroup_barrier(mem_flags::mem_threadgroup);
 if(tid==0)mxval=max(max(reduced[0],reduced[1]),max(reduced[2],reduced[3]));threadgroup_barrier(mem_flags::mem_threadgroup);
 float sum=0;for(uint j=tid;j<L;j+=128){float p=exp(logits[j]-mxval);logits[j]=p;sum+=p;}
 sum=simd_sum(sum);if(tid%32==0)reduced[tid/32]=sum;threadgroup_barrier(mem_flags::mem_threadgroup);
 if(tid==0)denom=reduced[0]+reduced[1]+reduced[2]+reduced[3];threadgroup_barrier(mem_flags::mem_threadgroup);
 float acc=0;
 for(uint j=0;j<L;j++){
  float v=j<P?float(pv[h*pv_strides[1]+j*pv_strides[2]+tid*pv_strides[3]]):float(av[h*av_strides[1]+(j-P)*av_strides[2]+tid*av_strides[3]]);
  acc+=logits[j]*v;
 }
 out[(qi*H+h)*128+tid]=T(acc/denom);
''')
def segmented(q,pk,pv,ak,av):
 b,h,n,d=q.shape
 if b!=1 or d!=128:raise ValueError('Experimental kernel requires batch 1 and head dim 128')
 if pk.shape[:2]!=(b,h) or pk.shape!=pv.shape or ak.shape!=av.shape:raise ValueError('KV shape mismatch')
 p=pk.shape[2];length=p+ak.shape[2]
 return _KERNEL(inputs=[q,pk,pv,ak,av],template=[('Q',n),('P',p),('L',length),('H',h),('T',q.dtype)],grid=(h*n*128,1,1),threadgroup=(128,1,1),output_shapes=[(b,n,h*d)],output_dtypes=[q.dtype])[0]
