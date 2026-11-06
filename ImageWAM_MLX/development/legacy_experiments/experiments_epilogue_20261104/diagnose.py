import mlx.core as mx
import numpy as np
from split import kernel
mx.random.seed(8)
m,n,k=16,64,256
x=mx.random.normal((1,m,k)).astype(mx.bfloat16);w=mx.random.normal((n,k)).astype(mx.bfloat16);parts=mx.quantize(w,bits=1,group_size=64)
for s in (1,2,4):
 p=kernel(inputs=[x,*parts],template=[('T',x.dtype),('SPLITS',s)],grid=((n//64)*128,1,s),threadgroup=(128,1,1),output_shapes=[(s,m,n)],output_dtypes=[mx.float32])[0]
 mx.eval(p)
 ref=mx.quantized_matmul(x,*parts,transpose=True,bits=1,group_size=64);mx.eval(ref)
 print(s,'partial',np.array(p)[:,0,:8],'sum',np.array(p).sum(0)[0,:8],'ref',np.array(ref.astype(mx.float32))[0,0,:8],flush=True)
