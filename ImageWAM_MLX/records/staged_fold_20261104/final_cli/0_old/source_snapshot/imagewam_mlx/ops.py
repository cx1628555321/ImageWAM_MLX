"""MLX operations matching the source's explicit BF16 rounding boundaries."""
import math
import mlx.core as mx
D=mx.bfloat16

def silu(x): return x * mx.sigmoid(x)
def gated(x):
    a,b=mx.split(x,2,axis=-1)
    return silu(a)*b

def norm(x): return mx.fast.layer_norm(x,None,None,eps=1e-6)
def rms(x,w,eps=1e-6):
    y=x.astype(mx.float32)
    return (y*mx.rsqrt(mx.mean(y*y,axis=-1,keepdims=True)+eps)).astype(x.dtype)*w

def attention(q,k,v,mask=None):
    y=mx.fast.scaled_dot_product_attention(q,k,v,scale=q.shape[-1]**-0.5,mask=mask)
    return y.transpose(0,2,1,3).reshape(y.shape[0],y.shape[2],-1)

def flux_rope(ids):
    # The source computes frequencies, products and trigonometry in BF16.
    scale=mx.arange(0,32,2,dtype=D)/32
    omega=(1/(mx.array(2000,dtype=D)**scale)).astype(D)
    angles=(ids[...,None]*omega).reshape(ids.shape[0],ids.shape[1],64)
    return mx.cos(angles).astype(mx.float32)[:,None],mx.sin(angles).astype(mx.float32)[:,None]

def rotate(x,pe):
    c,s=pe
    y=x.astype(mx.float32).reshape(*x.shape[:-1],-1,2)
    a,b=y[...,0],y[...,1]
    return mx.stack([c*a-s*b,s*a+c*b],axis=-1).reshape(x.shape).astype(x.dtype)

def time_embedding(t):
    t=(t*1000).astype(D)
    f=mx.exp(-math.log(10000)*mx.arange(128,dtype=mx.float32)/128)
    a=t.astype(mx.float32)[:,None]*f[None]
    return mx.concatenate([mx.cos(a),mx.sin(a)],axis=-1).astype(D)

class Weights:
    def __init__(self,w): self.w=w
    def rotate_key(self,x,pe):return rotate(x,pe)
    def norm_rotate_key(self,x,w,pe):return self.rotate_key(rms(x,w),pe)
    def linear(self,x,p):
        y=x@self.w[p+'.weight'].T
        if p+'.bias' in self.w:y=y+self.w[p+'.bias']
        return y
    def embed_time(self,t):
        return self.linear(silu(self.linear(time_embedding(t),'time_in.in_layer')),'time_in.out_layer')
    def modulation(self,vec,p,double=False):
        return mx.split(self.linear(silu(vec),p+'.lin')[:,None],6 if double else 3,axis=-1)
    def qkv(self,x,p,pe):
        q,k,v=mx.split(x.reshape(x.shape[0],x.shape[1],3,24,128).transpose(2,0,3,1,4),3,axis=0)
        q,k,v=q[0],k[0],v[0]
        q=rms(q,self.w[p+'.query_norm.scale']); k=rms(k,self.w[p+'.key_norm.scale'])
        return rotate(q,pe),rotate(k,pe),v
    def double_prepare(self,x,p,m,pe,stream='img'):
        z=(1+m[1])*norm(x)+m[0]
        return self.qkv(self.linear(z,p+'.'+stream+'_attn.qkv'),p+'.'+stream+'_attn.norm',pe)
    def double_post(self,x,att,p,m,stream='img'):
        x=x+m[2]*self.linear(att,p+'.'+stream+'_attn.proj')
        z=(1+m[4])*norm(x)+m[3]
        return x+m[5]*self.linear(gated(self.linear(z,p+'.'+stream+'_mlp.0')),p+'.'+stream+'_mlp.2')
    def single_prepare(self,x,p,m,pe):
        z=self.linear((1+m[1])*norm(x)+m[0],p+'.linear1')
        q,k,v=self.qkv(z[...,:9216],p+'.norm',pe)
        return q,k,v,z[...,9216:]
    def single_post(self,x,att,mlp,p,m):
        return x+m[2]*self.linear(mx.concatenate([att,gated(mlp)],axis=-1),p+'.linear2')
