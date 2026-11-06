import copy,types
import mlx.core as mx
from candidate import conditions,condition_step,condition_denoise
MODES=['baseline','unrolled','batched']
def install(base,mode):
 if mode=='baseline':return base
 if mode.startswith('attention_'):
  from attention import install as ai
  return ai(base,mode)
 m=copy.copy(base)
 m._request_conditions=mx.compile(lambda t:conditions(m,t,mode=='batched'))
 m._condition_step=mx.compile(lambda cache,x,c,d,pe:condition_step(m,cache,x,c,d,pe))
 m.denoise=types.MethodType(condition_denoise,m)
 if mode in ('prepare','prepare_unrolled'):
  from prepare import prepare,denoise
  m._hoist=mode=='prepare_unrolled'
  m._prepared_request=mx.compile(lambda cache,noise,u:prepare(m,cache,noise,u,m._hoist))
  m.denoise=types.MethodType(denoise,m)
 return m

def predict(m,mode,steps,args):
 assert steps==10 and m.text_cache_size==0
 return m.predict(**args,steps=10)
