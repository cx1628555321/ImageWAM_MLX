"""Compare fresh default-allocator outputs with frozen restricted-cache outputs."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--before',type=Path,required=True)
    p.add_argument('--after',type=Path,required=True)
    args=p.parse_args()
    records=[]
    for root in (args.before,args.after):
        result=json.loads((root/'results.json').read_text())
        audit=json.loads((root/'independent_audit.json').read_text())
        assert result['status']=='complete' and audit['passed']
        assert hashlib.sha256((root/'results.json').read_bytes()).hexdigest()==audit['results_sha256']
        records.append(result)
    assert set(records[0]['variants'])==set(records[1]['variants'])
    report={'passed':True,'before':str(args.before),'after':str(args.after),'variants':{}}
    for name in ['baseline',*records[0]['variants']]:
        old={x.name:x for x in (args.before/name).glob('*.npz')}
        new={x.name:x for x in (args.after/name).glob('*.npz')}
        assert old.keys()==new.keys() and len(new)==360
        mismatches=[]
        for filename in sorted(old):
            with np.load(old[filename]) as a,np.load(new[filename]) as b:
                for field in ('action','normalized'):
                    if a[field].dtype!=b[field].dtype or a[field].shape!=b[field].shape or a[field].tobytes()!=b[field].tobytes():
                        mismatches.append({'case':filename,'field':field})
        report['variants'][name]={'calls':len(new),'all_byte_equal':not mismatches,'mismatches':mismatches}
        report['passed'] &= not mismatches
    (args.after.parent/'allocator_output_comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    assert report['passed']


if __name__=='__main__':main()
