"""Exercise the real user-facing CLI, including default-off and output metadata."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser();p.add_argument('--records',type=Path,required=True)
p.add_argument('--modes',nargs='+',choices=['default','merge252','late_align32','prune252','prune216'],default=['default','merge252','late_align32','prune252','prune216'])
a=p.parse_args();OUT=a.records.resolve()
validation=json.loads((OUT/'validation/results.json').read_text());assert validation['status']=='complete'
case=validation['observations'][0];key=case['label']+'_s42'
prompt=OUT/'cli_prompt.txt';prompt.write_text(case['prompt'])
report={'status':'running','checks':{}};path=OUT/'cli_checks.json'
def save():path.write_text(json.dumps(report,indent=2)+'\n')
save()
for mode,preset in [('default','baseline'),('merge252','merge252'),('late_align32','late_align32'),('prune252','prune252'),('prune216','prune216')]:
    if mode not in a.modes:continue
    dest=OUT/('cli_'+mode)
    args=[sys.executable,str(ROOT/'scripts/infer_version.py'),'--variant','native-w4',
        '--inputs',str(Path(case['base'])/'inputs'/(case['label']+'.npz')),
        '--prompt-file',str(prompt),'--noise',str(Path(case['base'])/'inputs/noise_42.npy'),
        '--steps','10','--warmup','2','--output',str(dest)]
    if mode!='default':args+=['--visionzip',mode]
    started=time.monotonic()
    with (OUT/('cli_'+mode+'.log')).open('x') as log:
        run=subprocess.run(args,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=300)
    report['checks'][mode]={'command':args,'returncode':run.returncode,'elapsed_s':time.monotonic()-started};save()
    assert run.returncode==0,(mode,run.returncode)
    with np.load(OUT/'validation'/preset/(key+'.npz')) as ref:
        for field in ('action','normalized'):assert np.load(dest/(field+'.npy')).tobytes()==ref[field].tobytes()
    record=json.loads((dest/'run.json').read_text())
    if mode!='default':assert record['tested_version']['visionzip']['preset']==mode
    else:assert not record['tested_version'].get('visionzip',{}).get('enabled',False)
    report['checks'][mode].update(byte_equal_validation=True,metadata_correct=True);save();print('CLI_OK',mode,flush=True)
for args,message in [(['--variant','action-bf16','--visionzip','merge252'],'requires --variant native-w4'),
                     (['--variant','native-w4','--visionzip','invalid'],'invalid choice')]:
    run=subprocess.run([sys.executable,str(ROOT/'scripts/infer_version.py'),*args],cwd=ROOT,capture_output=True,text=True,timeout=30)
    assert run.returncode==2 and message in run.stderr
report.update(status='complete',invalid_arguments_rejected=True);save();print('COMPLETE',flush=True)
