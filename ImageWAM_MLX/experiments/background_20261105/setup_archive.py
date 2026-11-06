import sys,subprocess,os,json,site,time,shutil,urllib.request,tarfile,hashlib
from pathlib import Path
name,ref=sys.argv[1:];root=Path('records/background_20261106_archive/builds')/name;root.mkdir(parents=True,exist_ok=True);repo=root/'mlx';venv=root/'venv';(root/'tmp').mkdir(exist_ok=True);log=dict(name=name,ref=ref,status='building');record=root/'build.json'
def save():record.write_text(json.dumps(log,indent=2))
def run(args,**kw):
 for attempt in range(3):
  try:return subprocess.run(args,check=True,**kw)
  except subprocess.CalledProcessError:
   if args[0]!='git' or attempt==2:raise
   time.sleep(5)

save()
try:
 if not repo.exists():
  if name=='route4796':
   src=Path('records/background_20261105/builds/route4796/mlx');log['commit']=subprocess.check_output(['git','-C',str(src),'rev-parse','HEAD'],text=True).strip();shutil.copytree(src,repo,ignore=shutil.ignore_patterns('.git','build','*.egg-info','__pycache__'))
  else:
   req=urllib.request.Request('https://api.github.com/repos/ml-explore/mlx/pulls/4802',headers={'User-Agent':'imagewam-validation'})
   with urllib.request.urlopen(req,timeout=60) as f:meta=json.load(f)
   log['commit']=meta['head']['sha'];url='https://codeload.github.com/ml-explore/mlx/tar.gz/'+log['commit'];archive=root/'source.tar.gz'
   with urllib.request.urlopen(url,timeout=120) as f:archive.write_bytes(f.read())
   log['archive_sha256']=hashlib.sha256(archive.read_bytes()).hexdigest()
   with tarfile.open(archive) as t:
    names=t.getnames();prefix=names[0].split('/')[0];t.extractall(root,filter='data')
   (root/prefix).rename(repo)
 else:log['commit']=json.loads(record.read_text()).get('commit','see previous archive metadata')
 save()
 run([sys.executable,'-m','venv',str(venv)]);py=str(venv/'bin/python');run([py,'-m','pip','install','cmake','ninja']);env=dict(os.environ,CMAKE_BUILD_PARALLEL_LEVEL='4',CMAKE_ARGS='-DMLX_METAL_JIT=ON -DMLX_BUILD_GGUF=OFF',TMPDIR=str(root.resolve()/'tmp'));run([py,'-m','pip','install','--no-deps',str(repo.resolve())],env=env)
 dest=Path(subprocess.check_output([py,'-c','import site;print(site.getsitepackages()[0])'],text=True).strip());(dest/'imagewam_existing_dependencies.pth').write_text('\n'.join(site.getsitepackages())+'\n')
 check=subprocess.check_output([py,'-c','import mlx.core as m,importlib.metadata as d;print(d.version("mlx"));print(m.__file__);print(m.device_info());m.eval(m.ones(4)+1)'],text=True);log.update(status='ready',probe=check,python=py)
except Exception as e:log.update(status='blocked',error=repr(e));save();raise
save()
