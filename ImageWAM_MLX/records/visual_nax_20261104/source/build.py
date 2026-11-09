from pathlib import Path
import re,json,hashlib,shutil
base=Path('.venv/lib/python3.12/site-packages/mlx/include');src=base/'mlx/backend/metal/kernels/steel/gemm/gemm_nax.h';seen=set()
def expand(p):
 p=p.resolve()
 if p in seen:return ''
 seen.add(p)
 def inc(m):
  name=m.group(1);q=base/name if name.startswith('mlx/') else p.parent/name
  return expand(q) if q.exists() else m.group(0)
 return re.sub(r'#include "([^"]+)"',inc,p.read_text().replace('#pragma once',''))
out=Path('experiments_visual_nax_20261104');(out/'gemm.h').write_text(expand(src));shutil.copy2('.venv/lib/python3.12/site-packages/mlx-0.32.4.dist-info/licenses/LICENSE',out/'LICENSE.mlx')
(out/'upstream.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in seen},indent=2))
