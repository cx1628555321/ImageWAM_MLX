"""Encoder-only FLUX.2 VAE; native NHWC convolutions, PyTorch group order."""
import mlx.core as mx
from .ops import Weights,silu,attention
class VAE(Weights):
    def __init__(self,w):
        super().__init__({k:(v.transpose(0,2,3,1) if v.ndim==4 else v) for k,v in w.items()})
    def conv(self,x,p,stride=1,pad=0):
        return mx.conv2d(x,self.w[p+'.weight'],stride=stride,padding=pad)+self.w[p+'.bias']
    def gn(self,x,p):
        b,h,w,c=x.shape
        z=x.astype(mx.float32).reshape(b,h,w,32,c//32)
        mean=mx.mean(z,axis=(1,2,4),keepdims=True)
        var=mx.mean((z-mean)**2,axis=(1,2,4),keepdims=True)
        z=((z-mean)*mx.rsqrt(var+1e-6)).reshape(x.shape)
        return (z*self.w[p+'.weight'].astype(mx.float32)+self.w[p+'.bias'].astype(mx.float32)).astype(x.dtype)
    def res(self,x,p):
        y=self.conv(silu(self.gn(x,p+'.norm1')),p+'.conv1',pad=1)
        y=self.conv(silu(self.gn(y,p+'.norm2')),p+'.conv2',pad=1)
        if p+'.nin_shortcut.weight' in self.w:x=self.conv(x,p+'.nin_shortcut')
        return x+y
    def __call__(self,x):
        x=self.conv(x,'encoder.conv_in',pad=1)
        for i in range(4):
            for j in range(2):x=self.res(x,f'encoder.down.{i}.block.{j}')
            if i<3:x=self.conv(mx.pad(x,[(0,0),(0,1),(0,1),(0,0)]),f'encoder.down.{i}.downsample.conv',stride=2)
        x=self.res(x,'encoder.mid.block_1'); p='encoder.mid.attn_1'; z=self.gn(x,p+'.norm')
        b,h,w,c=z.shape
        q,k,v=[self.conv(z,p+'.'+a).reshape(b,1,h*w,c) for a in ('q','k','v')]
        x=x+self.conv(attention(q,k,v).reshape(b,h,w,c),p+'.proj_out')
        x=self.res(x,'encoder.mid.block_2')
        x=self.conv(silu(self.gn(x,'encoder.norm_out')),'encoder.conv_out',pad=1)
        x=self.conv(x,'encoder.quant_conv')[...,:32]
        b,h,w,c=x.shape
        x=x.reshape(b,h//2,2,w//2,2,c).transpose(0,1,3,5,2,4).reshape(b,h//2,w//2,128)
        # BatchNorm eval performs its arithmetic in float32 with BF16 buffers.
        x=((x.astype(mx.float32)-self.w['bn.running_mean'].astype(mx.float32))*mx.rsqrt(self.w['bn.running_var'].astype(mx.float32)+1e-4)).astype(x.dtype)
        return x
