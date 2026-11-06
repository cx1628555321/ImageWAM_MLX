from pathlib import Path
import re
base=Path('.venv/lib/python3.12/site-packages/mlx/include');src=base/'mlx/backend/metal/kernels/quantized_nax.h';raw=src.read_text();start=raw.index('template <\n    typename T,\n    const int group_size,\n    const int bits,\n    const bool aligned_N,');end=raw.index('\ntemplate <',start+10)
s=raw[start:end].replace('qmm_t_nax_tgp_impl','split_qmm').replace('device T* y,','device float* y, const int splits,')
s=s.replace('const constant int&','const int');s=re.sub(r' \[\[\w+\]\]','',s)
s=s.replace('  auto wl = (const device uint8_t*)w;', '''  const int kstart=tid.z*(K/splits);
  auto wl = (const device uint8_t*)w;
  x += kstart; wl += kstart*bytes_per_pack/pack_factor;
  scales += kstart/group_size; biases += kstart/group_size;
  y += long(tid.z)*M*N;''')
s=s.replace('k < K;', 'k < K/splits;')
s=s.replace('  constexpr short TN = SN / 16;', '  constexpr short TN = SN / 16;\n  static_assert(TM % 2 == 0 || TN % 2 == 0, "Unsupported NAX fragment combination");')
p=Path('development/legacy_experiments/experiments_epilogue_20261104/epilogue.h');p.write_text(p.read_text()+'\n'+s)
