import mlx.core as mx
from imagewam_mlx.speed_vae import SpeedVAE
from imagewam_mlx.ops import silu,attention
class CandidateVAE(SpeedVAE):
    def __init__(self,original,asym=False,qkv=False):
        self.w=dict(original.w);self.dtype=original.dtype;self.asym=asym;self.qkv=qkv
        if qkv:
            p='encoder.mid.attn_1'
            for suffix in ('.weight','.bias'):
                self.w[p+'.qkv'+suffix]=mx.concatenate([self.w[p+'.'+a+suffix] for a in ('q','k','v')],axis=0)
                mx.eval(self.w[p+'.qkv'+suffix])
        self._forward=mx.compile(self._encode)
    def _encode(self,x):
        x=self.conv(x,'encoder.conv_in',pad=1)
        for i in range(4):
            for j in range(2):x=self.res(x,f'encoder.down.{i}.block.{j}')
            if i<3:
                key=f'encoder.down.{i}.downsample.conv'
                if self.asym:x=mx.conv_general(x,self.w[key+'.weight'],stride=2,padding=((0,0),(1,1)))+self.w[key+'.bias']
                else:x=self.conv(mx.pad(x,[(0,0),(0,1),(0,1),(0,0)]),key,stride=2)
        x=self.res(x,'encoder.mid.block_1'); p='encoder.mid.attn_1'; z=self.gn(x,p+'.norm')
        b,h,w,c=z.shape
        if self.qkv:
            q,k,v=[a.reshape(b,1,h*w,c) for a in mx.split(self.conv(z,p+'.qkv'),3,axis=-1)]
        else:q,k,v=[self.conv(z,p+'.'+a).reshape(b,1,h*w,c) for a in ('q','k','v')]
        x=x+self.conv(attention(q,k,v).reshape(b,h,w,c),p+'.proj_out')
        x=self.res(x,'encoder.mid.block_2')
        x=self.conv(silu(self.gn(x,'encoder.norm_out')),'encoder.conv_out',pad=1)
        x=self.conv(x,'encoder.quant_conv')[...,:32]
        b,h,w,c=x.shape
        x=x.reshape(b,h//2,2,w//2,2,c).transpose(0,1,3,5,2,4).reshape(b,h//2,w//2,128)
        # BatchNorm eval performs its arithmetic in float32 with BF16 buffers.
        x=((x.astype(mx.float32)-self.w['bn.running_mean'].astype(mx.float32))*mx.rsqrt(self.w['bn.running_var'].astype(mx.float32)+1e-4)).astype(x.dtype)
        return x
