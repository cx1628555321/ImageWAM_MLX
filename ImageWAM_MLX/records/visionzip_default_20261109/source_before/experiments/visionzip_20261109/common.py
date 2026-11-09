"""Original-project loader and experiment utilities; no embody_edge imports."""
import os
os.environ.setdefault('MLX_MAX_OPS_PER_BUFFER','256')
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import mlx.core as mx
from imagewam_mlx.visionzip import PRESETS, Config, install, build_prefill, describe, incoming_scores, select_tokens

CAMS=('cam_high','cam_left_wrist','cam_right_wrist')
OLD=ROOT/'records/robotwin_accuracy_20261106'
RECORDS=ROOT/'records/visionzip_20261109'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def load_model():
    import imagewam_mlx.runtime as rt
    from imagewam_mlx.quantized import PackedRopeWeights
    from imagewam_mlx.workspace import workspace_step
    assert importlib.metadata.version('mlx')=='0.32.5.dev20261106'
    # Same bounded runtime compatibility override as scripts/infer_version.py.
    source=Path(rt.__file__).read_text();guard="version('mlx')=='0.32.4'"
    assert source.count(guard)==3
    exec(compile(source.replace(guard,'True'),rt.__file__,'exec'),rt.__dict__)
    mx.set_cache_limit(64*1024**2);mx.set_memory_limit(19*1024**3)
    started=time.perf_counter()
    model=rt.ImageWAM(ROOT/'weights/bf16',ROOT/'assets',profile='speed',prefix_precision='all_int8',text_cache_size=1,vae_asymmetric=False)
    weights=mx.load(str(Path(model.weights_path)/'action.safetensors'));mx.eval(weights)
    model.action=PackedRopeWeights.from_weights(rt.RopeWeights(weights),'w4')
    del weights
    model._compiled_workspace_step=mx.compile(lambda c,x,t,d,pe:workspace_step(model,c,x,t,d,pe))
    model.action_epilogue=False
    mx.synchronize();mx.clear_cache()
    # 256 MiB slowed the VAE in the original study. Record and make the study's
    # allocator setting explicit; this is buffer reuse, not observation caching.
    steady_cache_mib=int(os.environ.get('IMAGEWAM_STUDY_CACHE_MIB','1024'))
    if steady_cache_mib<0:raise ValueError('IMAGEWAM_STUDY_CACHE_MIB must be nonnegative')
    mx.set_cache_limit(steady_cache_mib*1024**2);mx.set_memory_limit(13*1024**3)
    model.study_info={'variant':'native-w4','mlx':mx.__version__,'mlx_core':mx.__file__,
        'weights':str(model.weights_path),'load_s':time.perf_counter()-started,
        'runtime_sha256':sha(rt.__file__),'visionzip_sha256':sha(ROOT/'imagewam_mlx/visionzip.py'),
        'original_weight_manifest_sha256':sha(ROOT/'weights/bf16/manifest.json'),
        'device':mx.device_info(),'cache_limit_bytes':steady_cache_mib*1024**2,'steady_memory_limit_bytes':13*1024**3}
    print('MODEL_LOADED',model.study_info['load_s'],flush=True)
    return model

def cases(expanded=True):
    result=[]
    for cohort,base in [('replay',OLD)]+([('new_observations',RECORDS/'dataset')] if expanded else []):
        manifest=json.loads((base/'manifest.json').read_text())
        for case in manifest['cases']:
            result.append({**case,'cohort':cohort,'base':str(base),'seeds':manifest['seeds']})
    return result

def sample(case,seed):
    base=Path(case['base']);path=base/'inputs'/(case['label']+'.npz');assert sha(path)==case['sha256']
    with np.load(path) as d:
        inputs=dict(images={k:d[k].copy() for k in CAMS},state=d['state'].copy(),prompt=case['prompt'],
            noise=np.load(base/'inputs'/f'noise_{seed}.npy'),steps=10)
        target=d['target'].copy()
    return inputs,target

def metrics(actual,expected):
    a=actual.astype(np.float64);b=expected.astype(np.float64);d=a-b
    return {'sum_sq':float(np.sum(d*d)),'reference_sum_sq':float(np.sum(b*b)),'count':d.size,
        'rmse':float(np.sqrt(np.mean(d*d))),'max_abs':float(np.max(np.abs(d))),
        'mae':float(np.mean(np.abs(d))),'per_dimension_sumsq':np.sum(d*d,axis=0).tolist(),
        'per_step_sumsq':np.sum(d*d,axis=1).tolist(),'per_dimension_max_abs':np.max(np.abs(d),axis=0).tolist()}

def aggregate(rows):
    count=sum(x['count'] for x in rows);sq=sum(x['sum_sq'] for x in rows)
    return {'count':len(rows),'elements':count,'rmse':float(np.sqrt(sq/count)),
        'relative_l2':float(np.sqrt(sq/max(sum(x['reference_sum_sq'] for x in rows),1e-12))),
        'mae':float(np.mean([x['mae'] for x in rows])),
        'worst_max_abs':max(x['max_abs'] for x in rows),
        'case_rmse_p50':float(np.percentile([x['rmse'] for x in rows],50)),
        'case_rmse_p95':float(np.percentile([x['rmse'] for x in rows],95)),
        'per_dimension_rmse':np.sqrt(np.sum([x['per_dimension_sumsq'] for x in rows],axis=0)/(len(rows)*16)).tolist(),
        'per_step_rmse':np.sqrt(np.sum([x['per_step_sumsq'] for x in rows],axis=0)/(len(rows)*14)).tolist(),
        'per_dimension_max_abs':np.max([x['per_dimension_max_abs'] for x in rows],axis=0).tolist()}

def snapshot(out):
    result={}
    for f in [ROOT/'imagewam_mlx/visionzip.py',ROOT/'scripts/infer_version.py',*Path(__file__).parent.glob('*.py')]:
        dest=out/'source'/f.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes(f.read_bytes());result[str(f.relative_to(ROOT))]=sha(f)
    return result
