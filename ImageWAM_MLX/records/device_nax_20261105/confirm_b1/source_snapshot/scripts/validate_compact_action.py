"""Compare compact and unpruned default kernels, including intermediate traces."""
import sys,json,gc,time
from pathlib import Path
import numpy as np
import mlx.core as mx
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from imagewam_mlx import ImageWAM
out=Path('records/compact_action_20261103');results={'saved_outputs':[], 'cases':[]}
for f in sorted(Path('records/compact_action_validation_20261103').glob('*.npy')):
    a=np.load(f);b=np.load(Path('records/redundancy_validation_20261103')/f.name)
    results['saved_outputs'].append({'file':f.name,'equal':bool(np.array_equal(a,b)),'max_abs':float(np.max(np.abs(a.astype(float)-b.astype(float))))})
assert len(results['saved_outputs'])==162
ex=Path('examples/robotwin_ep0_frame0');raw=np.load(ex/'inputs.npz');images={k:raw[k] for k in ('cam_high','cam_left_wrist','cam_right_wrist')};prompt=(ex/'prompt.txt').read_text()
for compact in (False,True):
    tag='compact' if compact else 'full'
    start=time.perf_counter();m=ImageWAM('weights/bf16','assets',compact=compact);mx.synchronize()
    results[tag]={'load_s':time.perf_counter()-start,'weights_path':m.weights_path,'compact':m.compact}
    assert m.compact==compact
    for h,steps in ((1,1),(8,3),(16,7),(16,10),(32,10),(64,2)):
        noise=np.random.default_rng(h).standard_normal((1,h,14)).astype(np.float32)
        action,normalized=m.predict(images,raw['state'],prompt,noise,steps=steps,trace=True)
        path=out/f'{tag}_h{h}_s{steps}.npz';np.savez(path,action=action,normalized=normalized,**m.trace)
        if compact:
            ref=np.load(out/f'full_h{h}_s{steps}.npz');current=np.load(path)
            diffs={k:float(np.max(np.abs(current[k].astype(float)-ref[k].astype(float)))) for k in current.files}
            results['cases'].append({'horizon':h,'steps':steps,'max_abs_by_tensor':diffs,'all_equal':all(np.array_equal(current[k],ref[k]) for k in current.files)})
    del m;gc.collect();mx.clear_cache()
results['all_equal']=all(r['equal'] for r in results['saved_outputs']) and all(r['all_equal'] for r in results['cases'])
(out/'precision.json').write_text(json.dumps(results,indent=2))
print('ALL_EQUAL',results['all_equal']);print(results['cases'])
assert results['all_equal']
