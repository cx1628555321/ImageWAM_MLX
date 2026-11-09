"""Speed-only merged projections, retaining dynamic sequence lengths and taps."""
import mlx.core as mx
from .qwen import Qwen
from .quantized import PackedMixin
from .ops import rms,silu,attention

class FusedPackedQwen(PackedMixin,Qwen):
    @classmethod
    def from_weights(cls,original,kind):
        obj=cls.__new__(cls);w=dict(original.w)
        for i in range(27):
            p=f'layers.{i}'
            for target,sources in ((p+'.self_attn.qkv',[p+'.self_attn.'+k+'_proj' for k in ('q','k','v')]),
                                   (p+'.mlp.gate_up',[p+'.mlp.gate_proj',p+'.mlp.up_proj'])):
                w[target+'.weight']=mx.concatenate([w[s+'.weight'] for s in sources],axis=0)
                mx.eval(w[target+'.weight'])
                for s in sources:del w[s+'.weight']
        if kind=='bf16':obj.w=w;obj.packed={};obj.quantization_info={'kind':'bf16','fused_projections':True}
        else:
            temp=Qwen(w);obj.pack(temp,kind);obj.quantization_info['fused_projections']=True
        return obj
    def __call__(self,ids):
        x=self.w['embed_tokens.weight'][ids];n=ids.shape[1];hidden=[]
        freq=1/(1000000**(mx.arange(0,128,2,dtype=mx.float32)/128))
        a=mx.arange(n,dtype=mx.float32)[:,None]*freq[None]
        c=mx.concatenate([mx.cos(a),mx.cos(a)],axis=-1).astype(x.dtype)[None,None]
        s=mx.concatenate([mx.sin(a),mx.sin(a)],axis=-1).astype(x.dtype)[None,None]
        for i in range(27):
            p=f'layers.{i}';y=rms(x,self.w[p+'.input_layernorm.weight'])
            q,k,v=mx.split(self.linear(y,p+'.self_attn.qkv'),[4096,5120],axis=-1)
            q=q.reshape(1,n,32,128).transpose(0,2,1,3)
            k=k.reshape(1,n,8,128).transpose(0,2,1,3)
            v=v.reshape(1,n,8,128).transpose(0,2,1,3)
            q=self.rope(rms(q,self.w[p+'.self_attn.q_norm.weight']),c,s)
            k=self.rope(rms(k,self.w[p+'.self_attn.k_norm.weight']),c,s)
            x=x+self.linear(attention(q,k,v,'causal'),p+'.self_attn.o_proj')
            y=rms(x,self.w[p+'.post_attention_layernorm.weight'])
            gate,up=mx.split(self.linear(y,p+'.mlp.gate_up'),2,axis=-1)
            x=x+self.linear(silu(gate)*up,p+'.mlp.down_proj')
            if i+1 in (9,18,27):hidden.append(x)
        return mx.concatenate(hidden,axis=-1)
