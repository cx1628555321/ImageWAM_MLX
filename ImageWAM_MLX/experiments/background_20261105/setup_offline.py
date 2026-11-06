import sys,subprocess,os,json,site,shutil,tarfile,zipfile,hashlib
from pathlib import Path
name,ref=sys.argv[1:];root=Path('records/background_20261106_offline/builds')/name;root.mkdir(parents=True,exist_ok=True);repo=root/'mlx';venv=root/'venv';wheels=Path('experiments/background_20261105/build_wheels');log=dict(name=name,ref=ref,status='building',metal_jit=True,gguf=False);record=root/'build.json'
def save():record.write_text(json.dumps(log,indent=2))
def run(args,**kw):subprocess.run(args,check=True,**kw)
def unpack(filename,parent):
 with tarfile.open(filename) as t:
  prefix=t.getnames()[0].split('/')[0];t.extractall(parent,filter='data')
 return parent/prefix
save()
try:
 if name=='route4796':
  src=Path('records/background_20261105/builds/route4796/mlx');log['commit']=subprocess.check_output(['git','-C',str(src),'rev-parse','HEAD'],text=True).strip();shutil.copytree(src,repo,dirs_exist_ok=True,ignore=shutil.ignore_patterns('.git','build','*.egg-info','__pycache__'))
 else:
  log['archive']=json.loads((wheels/'affine4802.json').read_text());assert hashlib.sha256((wheels/'affine4802.tar.gz').read_bytes()).hexdigest()==log['archive']['sha256'];extracted=unpack(wheels/'affine4802.tar.gz',root);shutil.copytree(extracted,repo,dirs_exist_ok=True)
  with tarfile.open(wheels/'affine4802.tar.gz') as archive:log['commit']=archive.pax_headers.get('comment')
 fmt=unpack(wheels/'fmt.tar.gz',root);nb=unpack(wheels/'nanobind.tar.gz',root)
 with zipfile.ZipFile(next(wheels.glob('nanobind*.whl'))) as z:
  for n in z.namelist():
   if n.startswith('nanobind/ext/robin_map/') and not n.endswith('/'):
    dst=nb/n.removeprefix('nanobind/');dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(z.read(n))
 run([sys.executable,'-m','venv',str(venv)]);py=str(venv/'bin/python');run([py,'-m','pip','install','--no-index','--find-links',str(wheels),'setuptools','typing_extensions'])
 dest=Path(subprocess.check_output([py,'-c','import site;print(site.getsitepackages()[0])'],text=True).strip());old=Path('records/background_20261105/builds/route4796/venv')
 for p in (old/'lib/python3.12/site-packages').glob('*'):
  if p.name.startswith(('cmake','ninja')):
   if p.is_dir():shutil.copytree(p,dest/p.name,dirs_exist_ok=True)
   else:shutil.copy2(p,dest/p.name)
 deps=Path('records/background_20261105/builds/route4796/mlx/build/temp.macosx-11.0-arm64-cpython-312/mlx.core/_deps').resolve();tmp=root/'tmp';tmp.mkdir(exist_ok=True)
 flags=f'-DMLX_METAL_JIT=ON -DMLX_BUILD_GGUF=OFF -DFETCHCONTENT_SOURCE_DIR_JSON={deps}/json-src -DFETCHCONTENT_SOURCE_DIR_METAL_CPP={deps}/metal_cpp-src -DFETCHCONTENT_SOURCE_DIR_FMT={fmt.resolve()} -DFETCHCONTENT_SOURCE_DIR_NANOBIND={nb.resolve()}'
 env=dict(os.environ,DEV_RELEASE='1',CMAKE_BUILD_PARALLEL_LEVEL='4',CMAKE_ARGS=flags,TMPDIR=str(tmp.resolve()),PATH=str(old.resolve()/'bin')+os.pathsep+os.environ['PATH']);log['cmake_flags']=flags;save();run([py,'-m','pip','install','--no-index','--no-deps','--no-build-isolation',str(repo.resolve())],env=env)
 (dest/'imagewam_existing_dependencies.pth').write_text('\n'.join(site.getsitepackages())+'\n')
 check=subprocess.check_output([py,'-c','import mlx.core as m,importlib.metadata as d;print(d.version("mlx"));print(m.__file__);print(m.device_info());m.eval(m.ones(4)+1)'],text=True);log.update(status='ready',probe=check,python=py)
except Exception as e:log.update(status='blocked',error=repr(e));save();raise
save()
