from pathlib import Path
import mlx.core as mx
header=Path(__file__).with_name('epilogue.h').read_text()
kernel=mx.fast.metal_kernel(name='imagewam_w1_split',input_names=['x','w','scales','qb'],output_names=['out'],header=header,source='''
threadgroup T Ws[64*72];
int M=x_shape[1],N=w_shape[0],K=x_shape[2];
split_qmm<T,64,1,true,32,64,64,2,2>(w,scales,qb,x,out,SPLITS,Ws,K,N,M,threadgroup_position_in_grid,thread_index_in_threadgroup,simdgroup_index_in_threadgroup,thread_index_in_simdgroup);
''')
reduce=mx.fast.metal_kernel(name='imagewam_split_epilogue',input_names=['partial','residual','gate','bias'],output_names=['out'],source='''
uint i=thread_position_in_grid.x;uint n=residual_shape[2],total=residual_shape[1]*n;
if(i>=total)return;
float v=0; for(int j=0;j<SPLITS;j++)v+=partial[j*total+i];
T z=T(v);volatile float vb=float(z)+float(bias[i%n]);z=T(vb);
volatile float pg=float(z)*float(gate[i%n]);z=T(pg);
volatile float result=float(residual[i])+float(z);out[i]=T(result);
''')
def fused(x,parts,residual,gate,bias):
 m,n=residual.shape[1:];k=x.shape[-1];nt=(n+63)//64;mt=(m+31)//32;s=min(max(1,512//(nt*mt)),k//64)
 while s>1 and k%(s*64):s-=1
 p=kernel(inputs=[x,*parts],template=[('T',x.dtype),('SPLITS',s)],grid=(nt*128,mt,s),threadgroup=(128,1,1),output_shapes=[(s,m,n)],output_dtypes=[mx.float32])[0]
 return reduce(inputs=[p,residual,gate,bias],template=[('T',x.dtype),('SPLITS',s)],grid=(residual.size,1,1),threadgroup=(256,1,1),output_shapes=[residual.shape],output_dtypes=[x.dtype])[0]
