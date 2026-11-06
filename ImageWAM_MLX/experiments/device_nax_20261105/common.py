import copy
import mlx.core as mx
from imagewam_mlx.model import ImageWAM as Base
from candidate import Candidate
MODES=['baseline','direct','direct_tail','unroll']
def install(base,mode):
 if mode=='baseline':return base
 m=copy.copy(base)
 if mode in ('reduce4','combo'):
  from reduce4 import Action4
  from imagewam_mlx.workspace import workspace_step
  m.action=Action4(base.action)
  m._compiled_workspace_step=mx.compile(lambda cache,x,t,delta,pe:workspace_step(m,cache,x,t,delta,pe))
 if mode!='reduce4':m.visual=Candidate(base.visual,'combined' if mode=='combo' else mode)
 m.prefill=mx.compile(lambda h,l,s:Base.prefill(m,h,l,s))
 return m

def predict(m,mode,steps,args):
 assert steps==10 and m.text_cache_size==0
 return m.predict(**args,steps=10)
