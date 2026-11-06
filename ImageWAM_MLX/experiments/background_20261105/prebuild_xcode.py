import os,subprocess,json,time
from pathlib import Path
r=Path('records/background_20261106_xcode')
state={'status':'running','jobs':[]}
def save():
 state['updated']=time.strftime('%Y-%m-%d %H:%M:%S');(r/'prebuild_status.json').write_text(json.dumps(state,indent=2))
for name,ref in [('route4796','refs/pull/4796/head'),('affine4802','refs/pull/4802/head')]:
 state['active']=name;save()
 with (r/('prebuild_'+name+'.log')).open('w') as f:
  result=subprocess.run(['.venv/bin/python','experiments/background_20261105/setup_xcode.py',name,ref],env=dict(os.environ,BG_BUILD_ONLY='1'),stdout=f,stderr=f)
 state['jobs'].append({'name':name,'returncode':result.returncode});save()
state['status']='completed' if all(x['returncode']==0 for x in state['jobs']) else 'finished_with_failures'
state['active']=None;save()
