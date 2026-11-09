"""Serial measurements with a fresh model process for each preset."""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
SCRIPTS=Path(__file__).resolve().parent
OUT=ROOT/'records/visionzip_default_fresh_20261109'
PRESETS=['merge252','late_align32','prune252','prune216']
OLD=ROOT/'records/visionzip_20261109/validation'
BASE=ROOT/'records/visionzip_default_20261109/validation'


def main():
    state={'status':'waiting_for_first_benchmark','stages':[],'started_unix':time.time()}
    path=OUT/'suite.json'
    def save():
        tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(path)
    save()
    while True:
        first=json.loads((OUT/'perf_merge252.job.json').read_text())
        if first['status']=='complete':break
        if first['status']=='failed':state['status']='failed';save();raise SystemExit('Initial benchmark failed')
        time.sleep(10)
    state['status']='running';save()
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
    def audit(name,mode=None):
        args=[SCRIPTS/'audit.py',OUT/name]
        if mode:args.append('--'+mode)
        if mode=='repeatability':args.extend(['--validation',OUT/'validation'])
        run(name+'_audit',args)
    def study(name,runner,args,mode=None):
        run(name,[SCRIPTS/'supervise.py',runner,'--output',OUT/name,*args]);audit(name,mode)
    audit('perf_merge252','benchmark')
    for name in PRESETS[1:]:
        study('perf_'+name,'benchmark.py',['--validation',OLD,'--presets',name,
            '--selection-presets',*PRESETS,'--pairs','50'],'benchmark')
    for name in PRESETS:
        study('accuracy_'+name,'validate_variant.py',['--baseline-validation',BASE,
            '--historical-validation',OLD,'--preset',name])
    run('combine',[SCRIPTS/'combine_fresh_runs.py',OUT])
    audit('validation');audit('benchmark','benchmark')
    shared=['--validation',OUT/'validation','--benchmark',OUT/'benchmark','--presets',*PRESETS]
    study('modules_direct','modules_direct.py',shared,'direct')
    study('zip_cost','zip_cost.py',[*shared,'--pairs','100'],'zip')
    for index in (1,2):study('repeatability_'+str(index),'repeatability.py',[*shared,'--passes','3'],'repeatability')
    run('stability',[SCRIPTS/'stability.py',OUT])
    run('allocator_output_comparison',[SCRIPTS/'compare_allocator_outputs.py','--before',OLD,'--after',OUT/'validation'])
    run('cli_check',[SCRIPTS/'cli_check.py','--records',OUT])
    state.update(status='complete',elapsed_s=time.time()-state['started_unix']);save()
    run('summary',[SCRIPTS/'summarize_default.py',OUT])


if __name__=='__main__':main()
