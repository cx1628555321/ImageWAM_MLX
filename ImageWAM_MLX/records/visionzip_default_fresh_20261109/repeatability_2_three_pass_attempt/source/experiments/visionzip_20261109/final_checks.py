"""Run the remaining checks serially after the main benchmark has finished."""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).parent;OUT=ROOT/'records/visionzip_20261109'
assert json.loads((OUT/'benchmark/results.json').read_text())['status']=='complete'
state={'status':'running','pid':__import__('os').getpid(),'jobs':[]};path=OUT/'final_checks.json'
jobs=[('benchmark_audit',['audit.py',str(OUT/'benchmark'),'--benchmark']),
      ('modules_direct',['modules_direct.py','--output',str(OUT/'modules_direct')]),
      ('modules_direct_audit',['audit.py',str(OUT/'modules_direct'),'--direct']),
      ('cli',['cli_check.py']),('summary',['summarize.py'])]
for name,args in jobs:
    job={'name':name,'status':'running'};state['jobs'].append(job);path.write_text(json.dumps(state,indent=2))
    started=time.monotonic()
    with (OUT/(name+'.log')).open('x') as f:
        run=subprocess.run([sys.executable,'-u',str(HERE/args[0]),*args[1:]],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,timeout=900)
    job.update(returncode=run.returncode,elapsed_s=time.monotonic()-started,status='complete' if run.returncode==0 else 'failed')
    if run.returncode!=0:
        state['status']='failed';path.write_text(json.dumps(state,indent=2));raise SystemExit(run.returncode)
    path.write_text(json.dumps(state,indent=2));print('DONE',name,flush=True)
state['status']='complete';path.write_text(json.dumps(state,indent=2));print('COMPLETE',flush=True)
