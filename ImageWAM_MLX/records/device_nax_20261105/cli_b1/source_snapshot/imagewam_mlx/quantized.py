"""Speed-only packed weight kernels. Never dequantize full weights in forward."""
import mlx.core as mx
from .runtime import RopeWeights
from .qwen import Qwen

# Selected from end-to-end sweeps; aggressive numerical degradation is intentional.
SPEED_POLICY={'visual':'bf16','action':'w1','qwen':'bf16'}
SPEED_OPTIONS={'qwen_fused':False,'vae':'fast_bf16'}
MODES=['bf16','w1','w2','w3','w4','w6','w8','mxfp4','mxfp8','nvfp4','mxfp8a','nvfp4a'] + [f'w{b}g{g}' for b in (1,2,4,8) for g in (32,128)] + [f'w{b}_conditioning' for b in (1,2,4)] + ['w1m8','w4m8']

class PackedMixin:
    def pack(self,original,kind):
        if kind not in MODES or kind=='bf16':raise ValueError('Invalid packed linear mode: '+kind)
        self.w=dict(original.w);self.packed={};self.kind=kind
        self.row_chunk=8 if kind.endswith('m8') else None
        kind=kind.removesuffix('m8')
        conditioning_only=kind.endswith('_conditioning')
        kind=kind.removesuffix('_conditioning')
        self.quantize_input=kind in ('mxfp8a','nvfp4a')
        if kind.startswith('w'):
            self.mode='affine';spec=kind[1:].split('g');self.bits=int(spec[0]);self.group_size=int(spec[1]) if len(spec)>1 else 64
        else:
            self.mode=kind[:-1] if self.quantize_input else kind
            self.bits=8 if self.mode=='mxfp8' else 4
            self.group_size=16 if self.mode=='nvfp4' else 32
        original_bytes=packed_bytes=0
        for key in list(self.w):
            w=self.w[key]
            if conditioning_only and not (key.startswith('time_in.') or 'modulation' in key):continue
            if not (key.endswith('.weight') and w.ndim==2 and 'embed_tokens' not in key
                    and w.shape[0]>=64 and w.shape[1]>=256 and w.shape[1]%self.group_size==0):continue
            parts=mx.quantize(w,group_size=self.group_size,bits=self.bits,mode=self.mode)
            mx.eval(parts)
            self.packed[key[:-7]]=parts
            original_bytes+=w.nbytes;packed_bytes+=sum(v.nbytes for v in parts)
            del self.w[key]
        self.quantization_info={'kind':self.kind,'scope':'conditioning' if conditioning_only else 'eligible_linear','row_chunk':self.row_chunk,'mode':self.mode,'bits':self.bits,'group_size':self.group_size,
            'quantize_input':self.quantize_input,'metal_activation_quantize_dequantize':self.quantize_input,'linear_count':len(self.packed),
            'original_linear_bytes':original_bytes,'packed_linear_bytes':packed_bytes,
            'retained_bytes':sum(x.nbytes for x in self.w.values()),'explicit_dequantize_in_forward':False}
        return self

    def linear(self,x,p):
        if p not in self.packed:return super().linear(x,p)
        parts=self.packed[p]
        if self.quantize_input:
            y=mx.qqmm(x,parts[0],scales=parts[1],group_size=self.group_size,bits=self.bits,mode=self.mode)
        else:
            def qmm(z):return mx.quantized_matmul(z,*parts,transpose=True,group_size=self.group_size,bits=self.bits,mode=self.mode)
            if self.row_chunk and 8<x.size//x.shape[-1]<=32:
                flat=x.reshape(-1,x.shape[-1]);y=mx.concatenate([qmm(flat[i:i+self.row_chunk]) for i in range(0,flat.shape[0],self.row_chunk)],axis=0).reshape(*x.shape[:-1],-1)
            else:y=qmm(x)
        if p+'.bias' in self.w:y=y+self.w[p+'.bias']
        return y

class PackedRopeWeights(PackedMixin,RopeWeights):
    @classmethod
    def from_weights(cls,original,kind):return cls.__new__(cls).pack(original,kind)

class PackedQwen(PackedMixin,Qwen):
    @classmethod
    def from_weights(cls,original,kind):return cls.__new__(cls).pack(original,kind)

def configure(model,policy,qwen_fused=False):
    if set(policy)-{'visual','action','qwen'}:raise ValueError('Unknown quantized component')
    info={}
    for name in ('visual','action','qwen'):
        kind=policy.get(name,'bf16')
        if kind not in MODES:raise ValueError('Unsupported quantization mode '+kind)
        if kind=='bf16' and not(name=='qwen' and qwen_fused):info[name]={'kind':'bf16'};continue
        cls=PackedQwen if name=='qwen' else PackedRopeWeights
        if name=='qwen' and qwen_fused:
            from .speed_qwen import FusedPackedQwen
            cls=FusedPackedQwen
        layer=cls.from_weights(getattr(model,name),kind)
        setattr(model,name,layer);info[name]=layer.quantization_info
    return info
