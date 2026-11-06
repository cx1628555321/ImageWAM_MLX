import os
os.environ['MLX_MAX_OPS_PER_BUFFER']='256';os.environ['MTL_CAPTURE_ENABLED']='1'
import sys,time,json
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
from common import *
from imagewam_mlx.ops import D,flux_rope
import imagewam_mlx.action_epilogue as ae
import imagewam_mlx.action_kernel as ak
r=ROOT/'capture';r.mkdir(exist_ok=True)
m=ImageWAM('weights/bf16','assets',profile='speed',text_cache_size=0);args=inputs();x,st=m.preprocess(args['images'],args['state']);hidden=m.qwen(mx.array(m.tokenize(args['prompt'])));latent=m.vae(x);c=m.prefill(hidden,latent,st);mx.eval(c)
original=ae.fused;saved=[]
def hook(x,parts,residual,gate,bias,**kw):
 if not saved:mx.eval(x,parts,residual,gate,bias);saved.append((x,parts,residual,gate,bias,kw))
 return original(x,parts,residual,gate,bias,**kw)
ae.fused=hook;h=args['noise'].shape[1];ids=np.zeros((1,h,4),np.float32);ids[...,0]=2;ids[0,:,1]=np.arange(h);pe=flux_rope(mx.array(ids).astype(D));c=[(mx.pad(k,[(0,0),(0,0),(0,h),(0,0)]),mx.pad(v,[(0,0),(0,0),(0,h),(0,0)])) for k,v in c]
y=workspace_step(m,c,mx.array(args['noise']).astype(D),mx.array([1.]).astype(D),mx.array(-.02).astype(D),pe);mx.eval(y);ae.fused=original
x,parts,residual,gate,bias,kw=saved[0];bm,bn,wm,wn=kw['tile'];M,N=residual.shape[1:];K=x.shape[-1];nt=(N+bn-1)//bn;mt=(M+bm-1)//bm;s=min(max(1,kw['target']//(nt*mt)),K//64)
while s>1 and K%(s*64):s-=1
threads=32*wm*wn
fn=lambda:ak.kernel(inputs=[x,*parts],template=[('T',x.dtype),('SPLITS',s),('BM',bm),('BN',bn),('WM',wm),('WN',wn)],grid=(nt*threads,mt,s),threadgroup=(threads,1,1),output_shapes=[(s,M,N)],output_dtypes=[mx.float32])[0]
p=fn();mx.eval(p)
rf=lambda:ak.reduce(inputs=[p,residual,gate,bias],template=[('T',x.dtype),('SPLITS',s)],grid=(residual.size,1,1),threadgroup=(256,1,1),output_shapes=[residual.shape],output_dtypes=[x.dtype])[0]

mx.savez(str(r/"projection_inputs.npz"),x=x,residual=residual,gate=gate,bias=bias,**{f"part_{i}":v for i,v in enumerate(parts)})
(r/"projection_parameters.json").write_text(json.dumps(kw))
print("exported real projection inputs",flush=True)
