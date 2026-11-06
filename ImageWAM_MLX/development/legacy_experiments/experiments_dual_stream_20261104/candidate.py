"""Request-local inputs on two reusable streams; frozen model weights shared."""
import copy,types
import mlx.core as mx
import numpy as np

def predict(self,images,state,prompt,noise,steps=10,profile=False,trace=False):
 if profile:raise ValueError('Per-module synchronization invalidates stream overlap; measure full requests instead')
 self.trace={};self.profile={}
 ids=self.tokenize(prompt)
 with mx.stream(self.text_stream):
  hidden=self.qwen(mx.array(ids));mx.async_eval(hidden)
 # Preprocessing GPU operations must also use the image stream, otherwise
 # they queue behind Qwen and prevent the intended encoder overlap.
 with mx.stream(self.image_stream):
  x,st=self.preprocess(images,state)
  latent=self.vae(x);mx.async_eval(latent,st)
 with mx.stream(self.text_stream):
  cache=self.prefill(hidden,latent,st)
  out=self.denoise(cache,noise,steps,trace)
  normalized=np.array(out[0].astype(mx.float32))
 scale,offset=self.norms['action'];result=(normalized-offset)/scale
 if trace:
  self.trace.update(image=np.array(x.astype(mx.float32)),state=np.array(st.astype(mx.float32)),text=np.array(hidden.astype(mx.float32)),latent=np.array(latent.astype(mx.float32)),token_ids=ids)
  for i,(k,v) in enumerate(cache):
   self.trace[f'cache_k_{i}']=np.array(k.astype(mx.float32));self.trace[f'cache_v_{i}']=np.array(v.astype(mx.float32))
 return result,normalized

def install(base,dual=True):
 c=copy.copy(base);c.vae=copy.copy(base.vae)
 c.text_stream=mx.default_stream(mx.gpu)
 c.image_stream=mx.new_stream(mx.gpu) if dual else c.text_stream
 # Separate compiled wrapper, first traced on its intended stream.
 with mx.stream(c.image_stream):c.vae._forward=mx.compile(c.vae._encode)
 c.predict=types.MethodType(predict,c)
 return c

def synchronize(model):
 mx.synchronize()
 if hasattr(model,'image_stream'):mx.synchronize(model.image_stream)
