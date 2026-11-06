import sys,subprocess,os,json,site
from pathlib import Path
name,ref=sys.argv[1:];root=Path('records/background_20261105/builds')/name;root.mkdir(parents=True,exist_ok=True);repo=root/'mlx';venv=root/'venv';log=dict(name=name,ref=ref,status='building');record=root/'build.json'
def save():record.write_text(json.dumps(log,indent=2))
def run(args,**kw):subprocess.run(args,check=True,**kw)
save()
try:
 if not (repo/'.git').exists():run(['git','clone','--filter=blob:none','--no-checkout','https://github.com/ml-explore/mlx.git',str(repo)])
 run(['git','-C',str(repo),'fetch','--depth','1','origin',ref]);run(['git','-C',str(repo),'checkout','--detach','FETCH_HEAD']);log['commit']=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip();save()
 run([sys.executable,'-m','venv',str(venv)]);py=str(venv/'bin/python');run([py,'-m','pip','install','cmake','ninja']);env=dict(os.environ,CMAKE_BUILD_PARALLEL_LEVEL='4');run([py,'-m','pip','install','--no-deps',str(repo.resolve())],env=env)
 dest=Path(subprocess.check_output([py,'-c','import site;print(site.getsitepackages()[0])'],text=True).strip());(dest/'imagewam_existing_dependencies.pth').write_text('\n'.join(site.getsitepackages())+'\n')
 check=subprocess.check_output([py,'-c','import mlx.core as m,importlib.metadata as d;print(d.version("mlx"));print(m.__file__);print(m.device_info());m.eval(m.ones(4)+1)'],text=True);log.update(status='ready',probe=check,python=py)
except Exception as e:log.update(status='blocked',error=repr(e));save();raise
save()
