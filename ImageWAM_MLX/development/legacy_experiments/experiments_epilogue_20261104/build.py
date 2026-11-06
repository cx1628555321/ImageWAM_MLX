from pathlib import Path
import re,shutil,hashlib,json
base=Path('.venv/lib/python3.12/site-packages/mlx/include');src=base/'mlx/backend/metal/kernels/quantized_nax.h';raw=src.read_text()
start=raw.index('template <\n    typename T,\n    const int group_size,\n    const int bits,\n    const bool aligned_N,')
end=raw.index('\ntemplate <',start+10)
s=raw[:raw.index('template <typename T>\nMETAL_FUNC void adjust_matrix_offsets')]+raw[start:end]
s=re.sub(r'constant bool (\w+) \[\[function_constant\(\d+\)\]\];',r'constant constexpr bool \1 = false;',s)
s=s.replace('qmm_t_nax_tgp_impl','epilogue_qmm').replace('    device T* y,','    device T* y, const device T* residual, const device T* gate, const device T* bias,')
s=s.replace('const constant int&','const int').replace(' [[threadgroup_position_in_grid]]','').replace(' [[thread_index_in_threadgroup]]','').replace(' [[simdgroup_index_in_threadgroup]]','').replace(' [[thread_index_in_simdgroup]]','')
a=s.index('      if constexpr (kAlignedM.value && kAlignedN.value)');b=s.index('\n    });',a)
s=s[:a]+'''      for(short i=0;i<TM;i++) for(short j=0;j<TN;j++) {
        for(short e=0;e<8;e++) {
          short2 c=BaseNAXFrag::get_coord(e);
          int row=tm+i*16+c.y,col=tn+j*16+c.x;
          if(y_row+row<M && y_col+col<N) {
            int gc=y_col+col; long off=long(y_row+row)*N+gc;
            T v=T(Dtile.frag_at(i,j)[e]);
            volatile float vb=float(v)+float(bias[gc]); v=T(vb);
            volatile float pg=float(v)*float(gate[gc]); T prod=T(pg);
            volatile float result=float(residual[off])+float(prod);
            y[row*N+col]=T(result);
          }
        }
      }'''+s[b:]
s='#include "mlx/backend/metal/kernels/steel/gemm/nax.h"\n'+s
seen=set()
def expand(text,parent):
 def inc(m):
  name=m.group(1);p=(base/name if name.startswith('mlx/') else parent/name).resolve()
  if not p.exists():return m.group(0)
  if p in seen:return ''
  seen.add(p);return expand(p.read_text().replace('#pragma once',''),p.parent)
 return re.sub(r'#include "([^"]+)"',inc,text)
s=expand(s,src.parent)
out=Path('development/legacy_experiments/experiments_epilogue_20261104');(out/'epilogue.h').write_text(s)
shutil.copy2('.venv/lib/python3.12/site-packages/mlx-0.32.4.dist-info/licenses/LICENSE',out/'LICENSE.mlx')
(out/'upstream.json').write_text(json.dumps(dict(version='0.32.4',source=str(src),sha256=hashlib.sha256(src.read_bytes()).hexdigest()),indent=2))
