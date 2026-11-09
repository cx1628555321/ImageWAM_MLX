"""Compose audited per-preset processes without pooling incompatible inputs."""
import argparse
import json
import shutil
from pathlib import Path
from common import PRESETS, snapshot, sha, save


def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args()
    for category,stem in [('validation','accuracy_'),('benchmark','perf_')]:
        source={}
        provenance={}
        for name in PRESETS:
            directory=a.root/(stem+name)
            raw=directory/'results.json';r=json.loads(raw.read_text())
            audit=json.loads((directory/'independent_audit.json').read_text())
            assert r['status']=='complete' and audit['passed'] and audit['results_sha256']==sha(raw)
            source[name]=r;provenance[name]={'path':str(raw),'sha256':sha(raw)}
        first=source[next(iter(PRESETS))]
        dest=a.root/category;dest.mkdir(exist_ok=False)
        r={**first,'sources':snapshot(dest),'composition':provenance,
           'models_by_preset':{k:v['model'] for k,v in source.items()},
           'process_protocol':'Each preset measured in a fresh process; default allocator; no cache clearing between requests or input groups.'}
        if category=='validation':
            r['variants']={k:v['variants'][k] for k,v in source.items()}
            assert all(v['baseline']==first['baseline'] and v['observations']==first['observations'] for v in source.values())
            assert all(v['off_restores_baseline'] and v['variants'][k]['restricted_allocator_reference_byte_equal_count']==360 for k,v in source.items())
            shutil.copytree(a.root/('accuracy_'+next(iter(PRESETS)))/'baseline',dest/'baseline')
            for name in PRESETS:shutil.copytree(a.root/('accuracy_'+name)/name,dest/name)
        else:
            assert all(v['selected']==first['selected'] and v['validation_sha256']==first['validation_sha256'] for v in source.values())
            assert all(v['all_timed_outputs_byte_equal'] for v in source.values())
            r['enabled_presets']=list(PRESETS)
            for field in ('e2e','modules','cold_text','micro','worst_case_traces'):
                r[field]={k:v[field][k] for k,v in source.items()}
            (dest/'traces').mkdir()
            for name in PRESETS:
                for path in (a.root/('perf_'+name)/'traces').glob('*.npz'):
                    shutil.copy2(path,dest/'traces'/path.name)
            r['fresh_default_validation_sha256']=sha(a.root/'validation/results.json')
        save(dest/'results.json',r)
    print('COMPOSED',flush=True)


if __name__=='__main__':main()
