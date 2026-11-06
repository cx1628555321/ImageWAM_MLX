from pathlib import Path
import ast
p=Path('imagewam_mlx_work/project');out=p/'experiments/online10_20261105'
s=(p/'imagewam_mlx/vae.py').read_text();t=ast.parse(s);cls=next(n for n in t.body if isinstance(n,ast.ClassDef));f=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__call__');code='\n'.join(s.splitlines()[f.lineno-1:f.end_lineno]).replace('def __call__(self,x):','def _encode(self,x):')
old="if i<3:x=self.conv(mx.pad(x,[(0,0),(0,1),(0,1),(0,0)]),f'encoder.down.{i}.downsample.conv',stride=2)"
new="""if i<3:
                key=f'encoder.down.{i}.downsample.conv'
                if self.asym:x=mx.conv_general(x,self.w[key+'.weight'],stride=2,padding=((0,0),(1,1)))+self.w[key+'.bias']
                else:x=self.conv(mx.pad(x,[(0,0),(0,1),(0,1),(0,0)]),key,stride=2)"""
assert old in code;code=code.replace(old,new)
old="q,k,v=[self.conv(z,p+'.'+a).reshape(b,1,h*w,c) for a in ('q','k','v')]"
new="""if self.qkv:
            q,k,v=[a.reshape(b,1,h*w,c) for a in mx.split(self.conv(z,p+'.qkv'),3,axis=-1)]
        else:q,k,v=[self.conv(z,p+'.'+a).reshape(b,1,h*w,c) for a in ('q','k','v')]"""
assert old in code;code=code.replace(old,new)
head='''import mlx.core as mx
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
'''
(out/'vae_candidate.py').write_text(head+code+'\n')
