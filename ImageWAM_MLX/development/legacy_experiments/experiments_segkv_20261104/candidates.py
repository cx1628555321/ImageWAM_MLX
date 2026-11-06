import types
import mlx.core as mx
from imagewam_mlx.runtime import ImageWAM
from imagewam_mlx.ops import silu,norm
from segmented import segmented
class Segmented(ImageWAM):
    def _step(self,cache,x,t,delta,pe):
        a=self.action
        vec=a.embed_time(t)
        md=a.modulation(vec,'double_stream_modulation_img',True); ms=a.modulation(vec,'single_stream_modulation')
        z=a.linear(x,'action_encoder')
        for i in range(5):
            p=f'double_blocks.{i}'; q,k,v=a.double_prepare(z,p,md,pe)
            out=segmented(q,cache[i][0],cache[i][1],k,v)
            z=a.double_post(z,out,p,md)
        for i in range(20):
            p=f'single_blocks.{i}'; q,k,v,mlp=a.single_prepare(z,p,ms,pe)
            out=segmented(q,cache[i+5][0],cache[i+5][1],k,v)
            z=a.single_post(z,out,mlp,p,ms)
        shift,scale=mx.split(a.linear(silu(vec),'head.adaLN_modulation.1'),2,axis=-1)
        pred=a.linear((1+scale[:,None])*norm(z)+shift[:,None],'head.linear')
        x=x+pred*delta
        return x


def install(m,mode):
 if mode=='segmented':
  m.kv_workspace=False;m._compiled_step=mx.compile(types.MethodType(Segmented._step,m))
 elif mode!='baseline':raise ValueError(mode)
 return m
