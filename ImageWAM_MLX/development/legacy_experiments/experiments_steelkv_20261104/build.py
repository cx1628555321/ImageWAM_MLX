from pathlib import Path
import re,shutil,hashlib,json
base=Path('.venv/lib/python3.12/site-packages/mlx/include');src=base/'mlx/backend/metal/kernels/steel/attn/kernels/steel_attention.h';s=src.read_text()
s=re.sub(r'constant bool (\w+) \[\[function_constant\(\d+\)\]\];',r'constant constexpr bool \1 = false;',s)
a=s.index('[[kernel,');b=s.index(' // clang-format on',a)
s=s[:a]+'''METAL_FUNC void segmented_steel(
 const device T* Q,const device T* K,const device T* V,device T* O,
 const thread AttnParams* params,const thread AttnMaskParams* mask_params,
 const device MaskType* mask,const device T* sinks,device float* lse,
 uint simd_lane_id,uint simd_group_id,uint3 tid,uint3 lid,
 const device T* AK,const device T* AV,int prefix,int aks,int avs,int akh,int avh,threadgroup T* Q_smem,threadgroup T* KV_smem) {'''+s[b:]
s=s.replace('  // Prepare threadgroup memory','  AK += ulong(tid.y)*akh; AV += ulong(tid.y)*avh;\n  // Prepare threadgroup memory')
for which,dim,ld,rowmajor in [('k','BD','LDK_tgp',False),('v','BV','LDV_tgp',True)]:
 a=s.index(f'    if (!align_K && kb == (params->NK_aligned)) {{\n      loader_{which}.load_safe')
 b=s.index('\n    }',s.index('} else {',a))+6
 orig=which.upper();dest='Vs' if rowmajor else 'Ks';index=f'rr*{ld}+dd' if rowmajor else f'dd*{ld}+rr'
 repl=f'''    for(int ii=int(simd_group_id)*32+int(simd_lane_id);ii<BK*{dim};ii+=WM*WN*32) {{
      int rr=ii/{dim},dd=ii%{dim},token=kb*BK+rr;
      T value=T(0);
      if(token<params->kL) value=token<prefix?{orig}[token*params->{orig}_strides[2]+dd]:A{orig}[(token-prefix)*a{which}s+dd];
      {dest}[{index}]=value;
    }}'''
 s=s[:a]+repl+s[b:]
s=s.replace('    loader_k.next();','').replace('    loader_v.next();','')
s=s.replace('  threadgroup T Q_smem[q_tgp_mem_s];','').replace('  threadgroup T KV_smem[tgp_mem_s];','')
# Inline local headers for Metal runtime compilation; preserve system headers.
seen=set()
def expand(text,parent):
 def inc(m):
  name=m.group(1);path=(base/name) if name.startswith('mlx/') else parent/name
  path=path.resolve()
  if not path.exists():return m.group(0)
  if path in seen:return ''
  seen.add(path);return expand(path.read_text().replace('#pragma once',''),path.parent)
 return re.sub(r'#include "([^"]+)"',inc,text)
s=expand(s,src.parent)
out=Path('development/legacy_experiments/experiments_steelkv_20261104');(out/'segmented_steel.h').write_text(s)
shutil.copy2('.venv/lib/python3.12/site-packages/mlx-0.32.4.dist-info/licenses/LICENSE',out/'LICENSE.mlx')
(out/'upstream.json').write_text(json.dumps(dict(version='0.32.4',source=str(src),sha256=hashlib.sha256(src.read_bytes()).hexdigest()),indent=2))
print(len(s))
