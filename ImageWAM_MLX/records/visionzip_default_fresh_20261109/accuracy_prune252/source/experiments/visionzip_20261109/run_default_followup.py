"""Wait for the serial suite, then check historical-input speed and real CLI."""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
SCRIPTS=Path(__file__).resolve().parent
OUT=ROOT/'records/visionzip_default_20261109'


def main():
    deadline=time.monotonic()+7200
    while True:
        state=json.loads((OUT/'suite.json').read_text())
        if state['status']=='complete':break
        if state['status']=='failed':raise SystemExit('Suite failed; followup not started')
        if time.monotonic()>deadline:raise SystemExit('Timed out waiting for suite')
        time.sleep(10)
    report={'status':'running','stages':[]}
    def save():
        path=OUT/'followup.json';tmp=path.with_suffix('.tmp')
        tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(path)
    jobs=[
        ('historical_default',[SCRIPTS/'baseline_probe.py','--output',OUT/'historical_default']),
        ('allocator_output_comparison',[SCRIPTS/'compare_allocator_outputs.py',
            '--before',ROOT/'records/visionzip_20261109/validation','--after',OUT/'validation']),
        ('cli_check',[SCRIPTS/'cli_check.py','--records',OUT]),
    ]
    for name,args in jobs:
        record={'name':name,'status':'running','started_unix':time.time()}
        report['stages'].append(record);save()
        with (OUT/(name+'.log')).open('x') as log:
            child=subprocess.Popen([sys.executable,'-u',*map(str,args)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            record['pid']=child.pid;save();code=child.wait()
        record.update(returncode=code,elapsed_s=time.time()-record['started_unix'],status='complete' if code==0 else 'failed')
        if code:report['status']='failed'
        save();print(name,record['status'],flush=True)
        if code:raise SystemExit(1)
    report['status']='complete';save()


if __name__=='__main__':main()
