import copy
import mlx.core as mx
from imagewam_mlx.model import ImageWAM as Base
from candidate import Candidate
MODES=['baseline','m4n1','m1n4','k2']
def install(base,mode):
 if mode=='baseline':return base
 m=copy.copy(base)
 if mode.startswith('g2_'):
  from group2 import Group2
  m.visual=Group2(base.visual,mode)
 else:m.visual=Candidate(base.visual,mode)
 m.prefill=mx.compile(lambda h,l,s:Base.prefill(m,h,l,s))
 return m

def predict(m,mode,steps,args):
 assert steps==10 and m.text_cache_size==0
 return m.predict(**args,steps=10)
