import numpy as np,json
from text_cache import TextCache
class Encoder:
 def __init__(self):self.calls=0;self.scale=1
 def __call__(self,x):self.calls+=1;return x.astype(np.float32)*self.scale
enc=Encoder();c=TextCache(enc,capacity=2,max_bytes=40)
a=np.array([[1,2]],np.int32);b=np.array([[3,4]],np.int32);d=np.array([[5,6]],np.int32)
assert np.array_equal(c(a),a);c(a);assert enc.calls==1
c(b);c(a);c(d);assert len(c.entries)==2
c(b);assert enc.calls==4
c.clear();enc.scale=2;assert np.array_equal(c(a),a*2)
long=np.arange(100).reshape(1,-1).astype(np.int32);c(long);assert c.bytes<=40 and len(c.entries)<=2
c(np.array([[1,2]],np.int64));assert len(c.entries)==2
print(json.dumps(dict(lru=True,clear_invalidation=True,oversized_bypass=True,dtype_key=True,byte_cap=True,hits=c.hits,misses=c.misses)))
