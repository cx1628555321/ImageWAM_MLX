"""Resume after the archived compression microbenchmark compilation failure."""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
SCRIPTS=Path(__file__).resolve().parent
OUT=ROOT/'records/visionzip_default_fresh_20261109'
PRESETS=['merge252','late_align32','prune252','prune216']


def main():
    path=OUT/'suite.json';state=json.loads(path.read_text())
    assert state['status']=='failed' and state['stages'][-1]['name']=='zip_cost'
    job=json.loads((OUT/'zip_cost.job.json').read_text());assert job['status']=='failed'
    for suffix in ('','.log','.job.json'):
        source=OUT/('zip_cost'+suffix);dest=OUT/('zip_cost_failed_attempt1'+suffix)
        assert not dest.exists();source.rename(dest)
    state['stages'][-1].update(name='zip_cost_failed_attempt1',
        archived_output=str(OUT/'zip_cost_failed_attempt1'),
        note='Microbenchmark closure inputs were implicit; retry passes all intermediate tensors explicitly. Forward inference code unchanged.')
    state.update(status='running',resumed_unix=time.time())
    def save():
        tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(path)
    save()
    def run(name,args):
        rec={'name':name,'status':'running','started_unix':time.time(),'args':list(map(str,args))}
        state['stages'].append(rec);save()
        with (OUT/(name+'.log')).open('x') as log:
            child=subprocess.Popen([sys.executable,'-u',*map(str,args)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            rec['pid']=child.pid;save();code=child.wait()
        rec.update(returncode=code,elapsed_s=time.time()-rec['started_unix'],status='complete' if code==0 else 'failed')
        if code:state['status']='failed'
        save();print(name,rec['status'],flush=True)
        if code:raise SystemExit(1)
    def study(name,runner,args,mode):
        run(name,[SCRIPTS/'supervise.py',runner,'--output',OUT/name,*args])
        checks=[SCRIPTS/'audit.py',OUT/name,'--'+mode]
        if mode=='repeatability':checks.extend(['--validation',OUT/'validation'])
        run(name+'_audit',checks)
    shared=['--validation',OUT/'validation','--benchmark',OUT/'benchmark','--presets',*PRESETS]
    study('zip_cost','zip_cost.py',[*shared,'--pairs','100'],'zip')
    for index in (1,2):study('repeatability_'+str(index),'repeatability.py',[*shared,'--passes','3'],'repeatability')
    run('stability',[SCRIPTS/'stability.py',OUT])
    run('allocator_output_comparison',[SCRIPTS/'compare_allocator_outputs.py',
        '--before',ROOT/'records/visionzip_20261109/validation','--after',OUT/'validation'])
    run('cli_check',[SCRIPTS/'cli_check.py','--records',OUT])
    state.update(status='complete',elapsed_s=time.time()-state['started_unix']);save()
    run('summary',[SCRIPTS/'summarize_default.py',OUT])


if __name__=='__main__':main()
