import numpy as np
import mlx.core as mx
from imagewam_mlx.ops import D,silu,norm,flux_rope
kernel=mx.fast.metal_kernel(name='wam_kv8_fused_attention',input_names=['q','pk','ks','pv','vs','ak','av'],output_names=['out'],source=r'''
uint tid=thread_index_in_threadgroup,bh=threadgroup_position_in_grid.y,row=threadgroup_position_in_grid.x;
threadgroup float scores[P+M];threadgroup float red[4];threadgroup float maximum,denom;
float localmax=-INFINITY;
for(uint j=tid;j<P+M;j+=128){
 float dot=0;
 for(uint c=0;c<D;c++){
  float kval=j<P?float(pk[(bh*P+j)*D+c])*ks[bh*P+j]:float(ak[(bh*M+j-P)*D+c]);
  dot+=float(q[(bh*M+row)*D+c])*kval;
 }
 dot*=rsqrt(float(D));scores[j]=dot;localmax=max(localmax,dot);
}
localmax=simd_max(localmax);if(tid%32==0)red[tid/32]=localmax;threadgroup_barrier(mem_flags::mem_threadgroup);
if(tid<32){float v=tid<4?red[tid]:-INFINITY;v=simd_max(v);if(tid==0)maximum=v;}threadgroup_barrier(mem_flags::mem_threadgroup);
float localsum=0;for(uint j=tid;j<P+M;j+=128){float v=exp(scores[j]-maximum);scores[j]=v;localsum+=v;}
localsum=simd_sum(localsum);if(tid%32==0)red[tid/32]=localsum;threadgroup_barrier(mem_flags::mem_threadgroup);
if(tid<32){float v=tid<4?red[tid]:0;v=simd_sum(v);if(tid==0)denom=v;}threadgroup_barrier(mem_flags::mem_threadgroup);
for(uint c=tid;c<D;c+=128){float value=0;for(uint j=0;j<P+M;j++){
 float vv=j<P?float(pv[(bh*P+j)*D+c])*vs[bh*P+j]:float(av[(bh*M+j-P)*D+c]);value+=scores[j]*vv;
}out[(row*H+bh)*D+c]=T(value/denom);}
''')
def quant(k,v):
 out=[]
 for x in (k,v):
  s=mx.maximum(mx.max(mx.abs(x.astype(mx.float32)),axis=-1)/127,1e-12);q=mx.clip(mx.round(x.astype(mx.float32)/s[...,None]),-127,127).astype(mx.int8);out.extend((q,s))
 return tuple(out)
def attention(q,cache,k,v):
 pk,ks,pv,vs=cache;b,h,m,d=q.shape;p=pk.shape[2];assert b==1
 return kernel(inputs=[q,pk,ks,pv,vs,k,v],template=[('T',q.dtype),('P',p),('M',m),('D',d),('H',h)],grid=(m*128,h,1),threadgroup=(128,1,1),output_shapes=[(1,m,h*d)],output_dtypes=[q.dtype])[0]
def step(self,cache,x,t,delta,pe):
 a=self.action;vec=a.embed_time(t);md=a.modulation(vec,'double_stream_modulation_img',True);ms=a.modulation(vec,'single_stream_modulation');z=a.linear(x,'action_encoder')
 for i in range(5):
  p=f'double_blocks.{i}';q,k,v=a.double_prepare(z,p,md,pe);z=a.double_post(z,attention(q,cache[i],k,v),p,md)
 for i in range(20):
  p=f'single_blocks.{i}';q,k,v,mlp=a.single_prepare(z,p,ms,pe);z=a.single_post(z,attention(q,cache[i+5],k,v),mlp,p,ms)
 shift,scale=mx.split(a.linear(silu(vec),'head.adaLN_modulation.1'),2,axis=-1);pred=a.linear((1+scale[:,None])*norm(z)+shift[:,None],'head.linear');return x+pred*delta

def denoise(self,cache,noise,steps=10,trace=False):
 if noise.ndim!=3 or noise.shape[0]!=1 or noise.shape[2]!=14 or not 1<=noise.shape[1]<=64 or steps<1:raise ValueError('invalid noise or steps')
 c=[quant(k,v) for k,v in cache];x=mx.array(noise).astype(D);u=mx.linspace(1,0,steps+1,dtype=mx.float32);sig=5*u/(1+4*u);times=(sig[:-1]*1000).astype(D);delta=(sig[1:]-sig[:-1]).astype(D)
 ids=np.zeros((1,x.shape[1],4),np.float32);ids[...,0]=2;ids[0,:,1]=np.arange(x.shape[1]);pe=flux_rope(mx.array(ids).astype(D))
 if not hasattr(self,'_kv8_step'):self._kv8_step=mx.compile(lambda c,x,t,d,p:step(self,c,x,t,d,p))
 for i in range(steps):
  x=self._kv8_step(c,x,(times[i:i+1]/1000).astype(D),delta[i],pe);mx.async_eval(x)
  if trace:self.trace[f'action_step_{i}']=np.array(x.astype(mx.float32))
 return x
