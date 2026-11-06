from pathlib import Path
import mlx.core as mx
HEADER=(Path(__file__).parent/'segmented_steel.h').read_text()
KERNEL=mx.fast.metal_kernel(name='wam_steel_dual_kv',input_names=['q','pk','pv','ak','av'],output_names=['out'],ensure_row_contiguous=False,header=HEADER,source=r'''
 threadgroup T qmem[1];threadgroup T kvmem[5120];
 AttnParams p;
 p.B=1;p.H=H;p.D=128;p.qL=Q;p.kL=L;p.gqa_factor=1;p.scale=0.08838834764831845f;
 p.NQ=(Q+15)/16;p.NK=(L+31)/32;p.NQ_aligned=Q/16;p.NK_aligned=L/32;p.qL_rem=Q%16;p.kL_rem=L%32;p.qL_off=L-Q;
 for(int i=0;i<3;i++){p.Q_strides[i]=q_strides[i];p.K_strides[i]=pk_strides[i];p.V_strides[i]=pv_strides[i];}
 p.O_strides[0]=Q*H*128;p.O_strides[1]=128;p.O_strides[2]=H*128;
 segmented_steel<T,16,32,128,128,2,2,float,float>(q,pk,pv,out,&p,nullptr,nullptr,nullptr,nullptr,
 thread_position_in_threadgroup.x%32,thread_position_in_threadgroup.x/32,
 uint3(threadgroup_position_in_grid.x,threadgroup_position_in_grid.y,0),thread_position_in_threadgroup,
 ak,av,P,ak_strides[2],av_strides[2],ak_strides[1],av_strides[1],qmem,kvmem);
''')
def segmented(q,pk,pv,ak,av):
 b,h,n,d=q.shape
 if b!=1 or d!=128:raise ValueError('batch1/head_dim128 required')
 p=pk.shape[2]
 return KERNEL(inputs=[q,pk,pv,ak,av],template=[('T',q.dtype),('H',h),('Q',n),('P',p),('L',p+ak.shape[2])],grid=(((n+15)//16)*128,h,1),threadgroup=(128,1,1),output_shapes=[(1,n,h*d)],output_dtypes=[q.dtype])[0]
