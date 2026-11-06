import copy,types
import mlx.core as mx
from text_cache import TextCache
MODES=['baseline','static','packed','cache_hit','cache_miss','steps8','steps6','steps4']
def install(base,mode):
    m=copy.copy(base);steps=10
    if mode in ('static','packed'):
        mod=__import__(mode+'_workspace');m._compiled_workspace_step=mx.compile(lambda cache,x,t,d,pe:mod.workspace_step(m,cache,x,t,d,pe));m.denoise=types.MethodType(mod.workspace_denoise,m)
    elif mode.startswith('cache'):
        m.qwen=TextCache(base.qwen)
        if '_steps' in mode:steps=int(mode.split('_steps')[1])
    elif mode.startswith('steps'):steps=int(mode[5:])
    elif mode!='baseline':raise ValueError(mode)
    return m,steps

def predict(m,mode,steps,args):
    if mode=='cache_miss':m.qwen.clear()
    return m.predict(**args,steps=steps)
