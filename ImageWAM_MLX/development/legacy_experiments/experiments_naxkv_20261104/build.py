from pathlib import Path
import re,shutil,hashlib,json
base=Path('.venv/lib/python3.12/site-packages/mlx/include');src=base/'mlx/backend/metal/kernels/steel/attn/kernels/steel_attention_nax.h';s=src.read_text().split('///////////////////////////////////////////////////////////////////////////////\n// Head-dim split attention kernel')[0]
s=re.sub(r'constant bool (\w+) \[\[function_constant\(\d+\)\]\];',r'constant constexpr bool \1 = false;',s)
a=s.index('[[kernel,');b=s.index(' // clang-format on',a)
s=s[:a]+'''METAL_FUNC void segmented_nax(
 const device T* Q,const device T* K,const device T* V,device T* O,
 const thread AttnParams* params,const thread AttnMaskParams* mask_params,
 const device MaskType* mask,const device T* sinks,device float* lse,
 uint simd_lane_id,uint simd_group_id,uint3 tid,uint3 lid,
 const device T* AK,const device T* AV,int prefix,int aks,int avs,int akh,int avh) {'''+s[b:]
s=s.replace('  const metal::uniform<float> scale2 =','  AK += ulong(tid.y)*akh; AV += ulong(tid.y)*avh;\n  const metal::uniform<float> scale2 =')
for letter,fr,fc in [('K',2,1),('V',1,2)]:
 start=s.index(f'          if (!align_K && is_last_k) {{\n            {letter}tile.load_rows(')
 end=s.index('\n          }',s.index('} else {',start))+12
 low=letter.lower();repl=f'''          int row0=kb*BK+ik*kU;
          if(row0+{fr}*kU<=prefix) {{
            {letter}tile.load({letter}+row0*params->{letter}_strides[2]+id*kU,int(params->{letter}_strides[2]));
          }} else if(row0>=prefix && row0+{fr}*kU<=params->kL) {{
            {letter}tile.load(A{letter}+(row0-prefix)*a{low}s+id*kU,a{low}s);
          }} else {{
            for(short rr=0;rr<{fr};rr++) for(short cc=0;cc<{fc};cc++) {{
              for(short e=0;e<8;e++) {{
                short2 c=BaseNAXFrag::get_coord(e);
                int row=row0+rr*kU+c.y,col=(id+cc)*kU+c.x;
                T value=T(0);
                if(row<params->kL) value=row<prefix?{letter}[row*params->{letter}_strides[2]+col]:A{letter}[(row-prefix)*a{low}s+col];
                {letter}tile.frag_at(rr,cc)[e]=value;
              }}
            }}
          }}'''
 s=s[:start]+repl+s[end:]
s=s.replace('    K += BK * int(params->K_strides[2]);','').replace('    V += BK * int(params->V_strides[2]);','')
seen=set()
def expand(text,parent):
 def inc(m):
  name=m.group(1);path=((base/name) if name.startswith('mlx/') else parent/name).resolve()
  if not path.exists():return m.group(0)
  if path in seen:return ''
  seen.add(path);return expand(path.read_text().replace('#pragma once',''),path.parent)
 return re.sub(r'#include "([^"]+)"',inc,text)
s=expand(s,src.parent);out=Path('development/legacy_experiments/experiments_naxkv_20261104');(out/'segmented_nax.h').write_text(s)
shutil.copy2('.venv/lib/python3.12/site-packages/mlx-0.32.4.dist-info/licenses/LICENSE',out/'LICENSE.mlx')
(out/'upstream.json').write_text(json.dumps(dict(version='0.32.4',source=str(src),sha256=hashlib.sha256(src.read_bytes()).hexdigest()),indent=2))
