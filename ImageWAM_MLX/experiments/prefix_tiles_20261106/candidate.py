import ast,json
from pathlib import Path
import mlx.core as mx
from imagewam_mlx import staged_visual as sv
from candidates import install as original_install

def make_kernel(wn):
    tree=ast.parse(Path('imagewam_mlx/device_visual.py').read_text())
    node=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='full_kernel' for t in n.targets))
    kw={k.arg:ast.literal_eval(k.value) for k in node.value.keywords if k.arg in ('source','input_names','output_names')}
    assert kw['source'].count('constexpr int BN = 64;')==1
    kw['source']=kw['source'].replace('constexpr int BN = 64;','constexpr int BN = 128;').replace('constexpr int WN = 2;',f'constexpr int WN = {wn};')
    return mx.fast.metal_kernel(name=f'wam_device_wide_wn{wn}',header=sv.header,**kw)

def shape(kw):
    t=dict(kw['template']);return (t['ROWS'],kw['inputs'][0].shape[0],t['XK'])

def wide(kernel,wn,kw):
    kw=dict(kw);m,n,k=shape(kw);threads=64*wn
    kw['grid']=((n+127)//128*threads,(m+63)//64,1)
    kw['threadgroup']=(threads,1,1)
    return kernel(**kw)

def install(model,mode,raw=None,scales=None):
    info=original_install(model,'native_w1')
    if mode=='baseline':return info
    assert mode=='selected_tiles'
    data=json.loads(Path('records/prefix_tiles_20261106/diagnostic/selection.json').read_text())
    selection={tuple(x['shape']):x['wn'] for x in data}
    kernels={wn:make_kernel(wn) for wn in set(selection.values())};old=model.visual.mm
    def mm(**kw):
        key=shape(kw)
        if key not in selection or key[1]%128:return old(**kw)
        wn=selection[key];return wide(kernels[wn],wn,kw)
    model.visual.mm=mm
    return info+[{'tile_selection':data,'fallback':'original for every unselected shape'}]
