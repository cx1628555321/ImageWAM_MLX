"""Finish candidates sequentially; record full precision and repeat positive screens."""
import sys,subprocess,json
from pathlib import Path
import numpy as np
root=Path('records/remaining_20261103');summary={}
for mode in ('overlap','group_inv','qwen_rope'):
 r=json.loads((root/mode/'results.json').read_text())
 if r['status']!='completed' or not r['all_equal']:
  summary[mode]={'initial_precision_pass':False};continue
 target=root/(mode+'_validation')
 cmd=[sys.executable,'scripts/run_remaining.py','--mode',mode,'--entry','validate_cases','--output',str(target)]
 print('VALIDATE',mode,flush=True)
 with (root/(mode+'_validation.log')).open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=300)
 rows=[]
 for file in sorted(target.glob('*.npy')):
  x=np.load(file);y=np.load(Path('records/next_optimizations_20261103/default_validation')/file.name)
  rows.append({'file':file.name,'equal':bool(np.array_equal(x,y)),'max_abs':float(np.max(np.abs(x.astype(float)-y.astype(float))))})
 summary[mode]={'files':rows,'count':len(rows),'all_equal':len(rows)==162 and all(x['equal'] for x in rows)}
 (root/'precision.json').write_text(json.dumps(summary,indent=2))
 if summary[mode]['all_equal'] and r['saving_ci95'][0]>0:
  print('REPEAT',mode,flush=True)
  with (root/(mode+'_repeat.log')).open('w') as f:subprocess.run([sys.executable,'scripts/test_remaining.py','--mode',mode,'--pairs','80','--output',str(root/(mode+'_repeat'))],stdout=f,stderr=subprocess.STDOUT,check=True,timeout=300)
print('COMPLETE',flush=True)
