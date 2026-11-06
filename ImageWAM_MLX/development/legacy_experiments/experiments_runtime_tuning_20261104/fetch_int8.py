import json,urllib.request,hashlib
from pathlib import Path
root=Path('development/legacy_experiments/experiments_runtime_tuning_20261104/upstream_int8');root.mkdir(exist_ok=False)
def get(url):
 return urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'ImageWAM-research'}),timeout=25).read()
try:
 info=json.loads(get('https://api.github.com/repos/ml-explore/mlx/pulls/4722'));sha=info['head']['sha'];(root/'pr.json').write_text(json.dumps(info,indent=2));manifest={}
 for path in ['mlx/backend/metal/kernels/quantized_nax.h','mlx/backend/metal/quantized.cpp','mlx/ops.cpp','CMakeLists.txt','LICENSE']:
  data=get('https://raw.githubusercontent.com/ml-explore/mlx/'+sha+'/'+path);dest=root/path;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data);manifest[path]=dict(sha256=hashlib.sha256(data).hexdigest(),bytes=len(data))
 (root/'manifest.json').write_text(json.dumps(dict(commit=sha,files=manifest),indent=2));print('FETCHED',sha,flush=True)
except Exception as e:
 (root/'error.txt').write_text(repr(e));print('FETCH FAILED',repr(e),flush=True)
