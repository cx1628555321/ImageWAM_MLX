"""Qwen3 first 27 layers, raw hidden outputs 9/18/27; no LM head."""
import mlx.core as mx
from .ops import Weights,rms,silu,attention
class Qwen(Weights):
    def rope(self,z,c,s):
        first,last=mx.split(z,2,axis=-1)
        return z*c+mx.concatenate([-last,first],axis=-1)*s
    def __call__(self,ids):
        x=self.w['embed_tokens.weight'][ids]
        n=ids.shape[1]; hidden=[]
        freq=1/(1000000**(mx.arange(0,128,2,dtype=mx.float32)/128))
        a=mx.arange(n,dtype=mx.float32)[:,None]*freq[None]
        c=mx.concatenate([mx.cos(a),mx.cos(a)],axis=-1).astype(x.dtype)[None,None]
        s=mx.concatenate([mx.sin(a),mx.sin(a)],axis=-1).astype(x.dtype)[None,None]
        def rope(z):return self.rope(z,c,s)
        for i in range(27):
            p=f'layers.{i}'; y=rms(x,self.w[p+'.input_layernorm.weight'])
            q=self.linear(y,p+'.self_attn.q_proj').reshape(1,n,32,128).transpose(0,2,1,3)
            k=self.linear(y,p+'.self_attn.k_proj').reshape(1,n,8,128).transpose(0,2,1,3)
            v=self.linear(y,p+'.self_attn.v_proj').reshape(1,n,8,128).transpose(0,2,1,3)
            q=rope(rms(q,self.w[p+'.self_attn.q_norm.weight']))
            k=rope(rms(k,self.w[p+'.self_attn.k_norm.weight']))
            x=x+self.linear(attention(q,k,v,'causal'),p+'.self_attn.o_proj')
            y=rms(x,self.w[p+'.post_attention_layernorm.weight'])
            x=x+self.linear(silu(self.linear(y,p+'.mlp.gate_proj'))*self.linear(y,p+'.mlp.up_proj'),p+'.mlp.down_proj')
            if i+1 in (9,18,27):hidden.append(x)
        return mx.concatenate(hidden,axis=-1)
