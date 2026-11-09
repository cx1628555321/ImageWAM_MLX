"""Native INT8 QK/PV with BF16/FP32 softmax island. Tokenwise INT8 KV reusable across steps."""
import mlx.core as mx
from native_int8 import quantize_rows,batched_gemm

def quant_kv(k,v):
 b,h,s,d=k.shape
 kq,ks=quantize_rows(k);vq,vs=quantize_rows(v)
 return kq.reshape(b*h,s,d),ks.reshape(b*h,s),vq.reshape(b*h,s,d),vs.reshape(b*h,s)

def attend_quantized(q,cache):
 b,h,m,d=q.shape;kq,ks,vq,vs=cache;bh=b*h;s=kq.shape[1]
 qq,qs=quantize_rows(q)
 logits=batched_gemm(qq.reshape(bh,m,d),kq,qs.reshape(bh,m),ks)
 p=mx.softmax(logits.astype(mx.float32)*(d**-.5),axis=-1)
 # V scale is per token; fold it into P before integer PV multiplication.
 pq,ps=quantize_rows(p*vs[:,None,:])
 y=batched_gemm(pq.reshape(bh,m,s),vq.transpose(0,2,1),ps.reshape(bh,m),mx.ones((bh,d),dtype=mx.float32))
 return y.reshape(b,h,m,d).transpose(0,2,1,3).reshape(b,m,h*d)

def append_kv(prefix,k,v):
 kq,ks,vq,vs=quant_kv(k,v)
 return tuple(mx.concatenate([a,z],axis=1) for a,z in zip(prefix,(kq,ks,vq,vs)))
