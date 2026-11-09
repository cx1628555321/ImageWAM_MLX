import json
from pathlib import Path
import numpy as np
root=Path('records/remaining_20261103');report={'performance':{},'precision':{}}
for name in ('overlap','overlap_repeat','group_inv','group_inv_repeat','qwen_rope','qwen_rope_repeat','combined','qwen_incremental'):
 p=root/name/'results.json'
 if p.exists():
  r=json.loads(p.read_text());report['performance'][name]={k:v for k,v in r.items() if k not in ('pairs','checks')}
for folder in root.glob('*validation'):
 rows=[]
 for p in sorted(folder.glob('*.npy')):
  a=np.load(p);b=np.load(Path('records/next_optimizations_20261103/default_validation')/p.name)
  rows.append({'file':p.name,'equal':bool(np.array_equal(a,b)),'bytes_equal':a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()})
 report['precision'][folder.name]={'count':len(rows),'all_equal':len(rows)==162 and all(x['equal'] for x in rows),'all_bytes_equal':len(rows)==162 and all(x['bytes_equal'] for x in rows),'files':rows}
(root/'summary.json').write_text(json.dumps(report,indent=2));print({k:{a:b for a,b in v.items() if a!='files'} for k,v in report['precision'].items()})
