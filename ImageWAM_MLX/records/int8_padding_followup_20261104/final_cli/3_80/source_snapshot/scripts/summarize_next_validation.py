import json
from pathlib import Path
import numpy as np
root=Path('records/next_optimizations_20261103');r={}
for folder in sorted(root.glob('*validation')):
 files=sorted(folder.glob('*.npy'));rows=[]
 for f in files:
  ref=Path('records/compact_action_validation_20261103')/f.name
  a,b=np.load(f),np.load(ref)
  rows.append({'file':f.name,'equal':bool(np.array_equal(a,b)),'max_abs':float(np.max(np.abs(a.astype(float)-b.astype(float))))})
 r[folder.name]={'count':len(files),'all_equal':len(files)==162 and all(x['equal'] for x in rows),'max_abs':max((x['max_abs'] for x in rows),default=None),'files':rows}
(root/'precision_summary.json').write_text(json.dumps(r,indent=2));print({k:{a:b for a,b in v.items() if a!='files'} for k,v in r.items()})
