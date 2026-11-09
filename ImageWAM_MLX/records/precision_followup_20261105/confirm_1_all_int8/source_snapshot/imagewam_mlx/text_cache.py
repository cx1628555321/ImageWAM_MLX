"""Opt-in per-model bounded cache; clear after changing weights/template policy."""
from collections import OrderedDict
import numpy as np
class TextCache:
    def __init__(self,encoder,capacity=8,max_bytes=64*1024*1024):
        if capacity<1 or max_bytes<1:raise ValueError('positive capacity and byte limit required')
        self.encoder=encoder;self.capacity=capacity;self.max_bytes=max_bytes
        self.entries=OrderedDict();self.bytes=0;self.hits=0;self.misses=0
    def clear(self):self.entries.clear();self.bytes=0
    def __getattr__(self,name):return getattr(self.encoder,name)
    def __call__(self,ids):
        a=np.asarray(ids);key=(a.dtype.str,a.shape,a.tobytes())
        if key in self.entries:
            self.hits+=1;y=self.entries.pop(key);self.entries[key]=y;return y
        self.misses+=1;y=self.encoder(ids)
        if y.nbytes<=self.max_bytes:
            while self.entries and (len(self.entries)>=self.capacity or self.bytes+y.nbytes>self.max_bytes):
                _,v=self.entries.popitem(last=False);self.bytes-=v.nbytes
            self.entries[key]=y;self.bytes+=y.nbytes
        return y
