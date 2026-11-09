"""Isolated candidates for compact ImageWAM; never alter production defaults."""
import copy,types
import mlx.core as mx
from imagewam_mlx.model import ImageWAM as Base
from imagewam_mlx.runtime import RopeWeights
from imagewam_mlx.ops import norm,rms,silu,attention
from imagewam_mlx.vae import VAE
from rms_rope_aligned_precise_experiments import FusedWeights,fused

class FullFusedWeights(FusedWeights):
    def norm_rotate_key(self,x,w,pe):return fused(x,w,pe)

def rms_models(base,visual=False):
    m=copy.copy(base);m.action=FusedWeights(base.action.w)
    if visual:
        m.visual=FullFusedWeights(base.visual.w)
        # The compact final K-only projection uses a separate path.
        m.prefill=mx.compile(types.MethodType(Base.prefill,m))
    m._compiled_step=mx.compile(m._step)
    return m

# Keep the original reductions. Fuse only affine normalization and BF16 SiLU.
_gn=mx.fast.metal_kernel(name='wam_gn_affine_silu_precise',input_names=['x','mean','var','w','bias'],output_names=['out'],ensure_row_contiguous=False,source=r'''
 uint i=thread_position_in_grid.x;if(i>=x_shape[0]*x_shape[1]*x_shape[2]*x_shape[3])return;
 uint c=i%x_shape[3],b=i/(x_shape[1]*x_shape[2]*x_shape[3]);uint g=c/(x_shape[3]/32);
 volatile float centered=float(x[i])-mean[b*32+g];
 volatile float inv=metal::precise::rsqrt(var[b*32+g]+1e-6f);
 volatile float normalized=centered*inv;
 volatile float scaled=normalized*float(w[c]);
 T z=T(scaled+float(bias[c]));
 auto sy=1/(1+metal::precise::exp(metal::abs(z))); T sig=(z<0)?sy:1-sy;
 out[i]=T(float(z)*float(sig));
''')
class FusedVAE(VAE):
    def gn_silu(self,x,p):
        b,h,w,c=x.shape;z=x.astype(mx.float32).reshape(b,h,w,32,c//32)
        mean=mx.mean(z,axis=(1,2,4),keepdims=True)
        var=mx.mean((z-mean)**2,axis=(1,2,4),keepdims=True)
        return _gn(inputs=[mx.contiguous(x),mean,var,self.w[p+'.weight'],self.w[p+'.bias']],template=[('T',x.dtype)],grid=(x.size,1,1),threadgroup=(256,1,1),output_shapes=[x.shape],output_dtypes=[x.dtype])[0]
    def res(self,x,p):
        y=self.conv(self.gn_silu(x,p+'.norm1'),p+'.conv1',pad=1)
        y=self.conv(self.gn_silu(y,p+'.norm2'),p+'.conv2',pad=1)
        if p+'.nin_shortcut.weight' in self.w:x=self.conv(x,p+'.nin_shortcut')
        return x+y

def vae_model(base):
    m=copy.copy(base);v=FusedVAE.__new__(FusedVAE);v.w=base.vae.w;m.vae=v;return m

# One threadgroup per query/head. Correct joint softmax across both segments.
# Prototype emphasizes removing concatenation, not matching native SDPA tiling.
_seg=mx.fast.metal_kernel(name='wam_segmented_attention',input_names=['q','pk','pv','ak','av'],output_names=['out'],ensure_row_contiguous=False,source=r'''
 uint r=threadgroup_position_in_grid.x,d=thread_position_in_threadgroup.x;
 uint nq=q_shape[2],h=(r/nq)%q_shape[1],b=r/(nq*q_shape[1]),t=r%nq;
 uint np=pk_shape[2],na=ak_shape[2],n=np+na;
 threadgroup float scores[1024];threadgroup float sums[256];
 for(uint j=d;j<n;j+=256){
  float s=0.0f;
  for(uint k=0;k<128;k++){
   size_t qo=b*q_strides[0]+h*q_strides[1]+t*q_strides[2]+k*q_strides[3];
   float v;
   if(j<np)v=float(pk[b*pk_strides[0]+h*pk_strides[1]+j*pk_strides[2]+k*pk_strides[3]]);
   else v=float(ak[b*ak_strides[0]+h*ak_strides[1]+(j-np)*ak_strides[2]+k*ak_strides[3]]);
   s+=float(q[qo])*v;
  }scores[j]=s*0.08838834764831844f;
 }
 threadgroup_barrier(mem_flags::mem_threadgroup);
 float m=-INFINITY;for(uint j=d;j<n;j+=256)m=max(m,scores[j]);sums[d]=m;
 threadgroup_barrier(mem_flags::mem_threadgroup);
 for(uint stride=128;stride;stride/=2){if(d<stride)sums[d]=max(sums[d],sums[d+stride]);threadgroup_barrier(mem_flags::mem_threadgroup);}
 m=sums[0];threadgroup_barrier(mem_flags::mem_threadgroup);
 float total=0.0f;for(uint j=d;j<n;j+=256){scores[j]=metal::exp(scores[j]-m);total+=scores[j];}sums[d]=total;
 threadgroup_barrier(mem_flags::mem_threadgroup);
 for(uint stride=128;stride;stride/=2){if(d<stride)sums[d]+=sums[d+stride];threadgroup_barrier(mem_flags::mem_threadgroup);}
 if(d<128){float v=0.0f;for(uint j=0;j<n;j++){
  float z;if(j<np)z=float(pv[b*pv_strides[0]+h*pv_strides[1]+j*pv_strides[2]+d*pv_strides[3]]);
  else z=float(av[b*av_strides[0]+h*av_strides[1]+(j-np)*av_strides[2]+d*av_strides[3]]);
  v+=scores[j]/sums[0]*z;
 }out[((b*nq+t)*q_shape[1]+h)*128+d]=T(v);}
''')
def segmented(q,pk,pv,k,v):
    if pk.shape[2]+k.shape[2]>1024:
        return attention(q,mx.concatenate([pk,k],axis=2),mx.concatenate([pv,v],axis=2))
    shape=(q.shape[0],q.shape[2],q.shape[1]*128)
    return _seg(inputs=[q,pk,pv,k,v],template=[('T',q.dtype)],grid=(q.shape[0]*q.shape[1]*q.shape[2]*256,1,1),threadgroup=(256,1,1),output_shapes=[shape],output_dtypes=[q.dtype])[0]

def segmented_step(self,cache,x,t,delta,pe):
    a=self.action;vec=a.embed_time(t);md=a.modulation(vec,'double_stream_modulation_img',True);ms=a.modulation(vec,'single_stream_modulation')
    z=a.linear(x,'action_encoder')
    for i in range(5):
        p=f'double_blocks.{i}';q,k,v=a.double_prepare(z,p,md,pe)
        z=a.double_post(z,segmented(q,*cache[i],k,v),p,md)
    for i in range(20):
        p=f'single_blocks.{i}';q,k,v,mlp=a.single_prepare(z,p,ms,pe)
        z=a.single_post(z,segmented(q,*cache[i+5],k,v),mlp,p,ms)
    shift,scale=mx.split(a.linear(silu(vec),'head.adaLN_modulation.1'),2,axis=-1)
    pred=a.linear((1+scale[:,None])*norm(z)+shift[:,None],'head.linear')
    return x+pred*delta

def segmented_model(base):
    m=copy.copy(base);m._compiled_step=mx.compile(types.MethodType(segmented_step,m));return m

_gate=mx.fast.metal_kernel(name='wam_gate_concat_precise',input_names=['a','z'],output_names=['out'],ensure_row_contiguous=False,source=r'''
 uint i=thread_position_in_grid.x;uint width=a_shape[2]+z_shape[2]/2;
 uint rows=a_shape[0]*a_shape[1];if(i>=rows*width)return;
 uint row=i/width,d=i%width,b=row/a_shape[1],t=row%a_shape[1];
 if(d<a_shape[2])out[i]=a[b*a_strides[0]+t*a_strides[1]+d*a_strides[2]];
 else{uint c=d-a_shape[2];size_t p=b*z_strides[0]+t*z_strides[1]+c*z_strides[2];
 T x=z[p],y=z[p+z_shape[2]/2*z_strides[2]];
 auto sy=1/(1+metal::precise::exp(metal::abs(x))); T sig=(x<0)?sy:1-sy;T activated=T(float(x)*float(sig));out[i]=T(float(activated)*float(y));}
''')
class GateWeights(RopeWeights):
    def single_post(self,x,att,mlp,p,m):
        shape=(*att.shape[:-1],att.shape[-1]+mlp.shape[-1]//2)
        y=_gate(inputs=[att,mlp],template=[('T',att.dtype)],grid=(shape[0]*shape[1]*shape[2],1,1),threadgroup=(256,1,1),output_shapes=[shape],output_dtypes=[att.dtype])[0]
        return x+m[2]*self.linear(y,p+'.linear2')

def gate_model(base):
    m=copy.copy(base);m.visual=GateWeights(base.visual.w)
    m.prefill=mx.compile(types.MethodType(Base.prefill,m));return m

BUILDERS={'rms_action':lambda b:rms_models(b),'rms_both':lambda b:rms_models(b,True),'vae_gn_silu':vae_model,'segmented_attention':segmented_model,'visual_gate_concat':gate_model}
