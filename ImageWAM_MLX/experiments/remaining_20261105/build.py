from pathlib import Path
p=Path('imagewam_mlx_work/project');out=p/'experiments/remaining_20261105'
s=(p/'imagewam_mlx/workspace.py').read_text().replace('from .ops import','from imagewam_mlx.ops import')
# Fixed padded prefix, without feeding updated caches back into the next step.
t=s.replace('x,cache=self._compiled_workspace_step(cache,x,t,deltas[i],pe)','x,_updated=self._compiled_workspace_step(cache,x,t,deltas[i],pe)').replace('mx.async_eval(x,cache)','mx.async_eval(x)')
(out/'static_workspace.py').write_text(t)
# Stack K/V and update once per layer, retaining functional ownership semantics.
t=s
for index in ('i','i+5'):
 old=f"offset=cache[{index}][0].shape[2]-k.shape[2]\n        k=mx.slice_update(cache[{index}][0],k,mx.array([offset]),axes=(2,)); v=mx.slice_update(cache[{index}][1],v,mx.array([offset]),axes=(2,))\n        updated.append((k,v))"
 new=f"offset=cache[{index}].shape[3]-k.shape[2]\n        kv=mx.slice_update(cache[{index}],mx.stack([k,v]),mx.array([offset]),axes=(3,))\n        updated.append(kv)\n        k,v=kv[0],kv[1]"
 assert old in t;t=t.replace(old,new)
old='cache=[(mx.pad(k,[(0,0),(0,0),(0,x.shape[1]),(0,0)]),mx.pad(v,[(0,0),(0,0),(0,x.shape[1]),(0,0)])) for k,v in cache]'
assert old in t;t=t.replace(old,'cache=[mx.pad(mx.stack([k,v]),[(0,0),(0,0),(0,0),(0,x.shape[1]),(0,0)]) for k,v in cache]')
(out/'packed_workspace.py').write_text(t)
