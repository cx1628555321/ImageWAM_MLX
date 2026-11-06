import copy,types
from vae_candidate import CandidateVAE
from preprocess import preprocess
MODES=['baseline','normalize','asym','qkv','both']
def install(base,mode):
 m=copy.copy(base)
 if mode=='normalize':m.preprocess=types.MethodType(preprocess,m)
 elif mode in ('asym','qkv','both'):m.vae=CandidateVAE(base.vae,mode in ('asym','both'),mode in ('qkv','both'))
 elif mode!='baseline':raise ValueError(mode)
 return m

def predict(m,mode,steps,args):
 assert steps==10 and m.text_cache_size==0
 return m.predict(**args,steps=10)
