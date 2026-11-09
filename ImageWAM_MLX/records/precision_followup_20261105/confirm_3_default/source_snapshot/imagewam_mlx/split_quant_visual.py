"""Fuse two-input concatenation into online INT8 quantization.
Preserves mode-2 arithmetic and BF16 gate rounding.
Derived from staged_visual, see metal/staged_int8_upstream.json and LICENSE.mlx.
"""
import mlx.core as mx
from . import prepacked_visual as pv
from .ops import gated
sv=pv.sv
qx=mx.fast.metal_kernel(name="wam_split_input_quant",input_names=["att","mlp"],output_names=["x8","x_scale","x_exp"],header=sv.header,source='uint3 tid=threadgroup_position_in_grid;uint lid=thread_index_in_threadgroup,simd_gid=simdgroup_index_in_threadgroup,simd_lid=thread_index_in_simdgroup,n_simd=simdgroups_per_threadgroup;const int KA=att_shape[1],KB=mlp_shape[1],K=KA+KB,M=att_shape[0],max_exp=0,x_span=0;\n\n  constexpr int BM = 64;\n  constexpr int BK = 64;\n\n  threadgroup float shared[33];\n\n  const int row = tid.y;\n  const int n_blocks = K / BK;\n  const bool active = lid < uint(n_blocks);\n  const size_t tile = size_t(row / BM) * n_blocks + lid;\n  device char4* q = (device char4*)(x8 + (tile * BM + row % BM) * BK);\n\n  // The exponent of the first block is stored twice so that the matmul can\n  // read the change from the block before\n  x_exp += size_t(row) * (n_blocks + 1);\n\n  if (row >= M) {\n    if (active) {\n      for (int i = 0; i < BK / 4; i++) {\n        q[i] = char4(0);\n      }\n      x_exp[lid + 1] = 0;\n      if (lid == 0) {\n        x_exp[0] = 0;\n      }\n    }\n    return;\n  }\n\n  const device T* x = lid*BK < KA ? att + size_t(row)*KA + lid*BK : mlp + size_t(row)*KB + lid*BK-KA;\n\n  float vals[BK];\n  float block_max = 0.0f;\n  if (active) {\n    for (int i = 0; i < BK; i++) {\n      vals[i] = float(x[i]);\n      block_max = max(block_max, fabs(vals[i]));\n    }\n  }\n  const float simd_row_max = simd_max(block_max);\n  if (simd_lid == 0) {\n    shared[simd_gid] = simd_row_max;\n  }\n\n  // The blocks that share an exponent are threads next to each other\n  float span_max = block_max;\n  if (x_span >= 2) {\n    span_max = max(span_max, simd_shuffle_xor(span_max, ushort(1)));\n  }\n  if (x_span >= 4) {\n    span_max = max(span_max, simd_shuffle_xor(span_max, ushort(2)));\n  }\n  threadgroup_barrier(mem_flags::mem_threadgroup);\n\n  if (lid == 0) {\n    float row_max = shared[0];\n    for (uint i = 1; i < n_simd; i++) {\n      row_max = max(row_max, shared[i]);\n    }\n    shared[32] = max(row_max * (1.0f / 127.0f), 1e-12f);\n  }\n  threadgroup_barrier(mem_flags::mem_threadgroup);\n\n  if (!active) {\n    return;\n  }\n\n  const float sx = shared[32];\n  const float inv = 1.0f / sx;\n\n  // Halve the scale of these blocks while their values still fit in int8\n  float t = span_max * inv;\n  int e = 0;\n  while (e < max_exp && t * 2.0f < 127.5f) {\n    t *= 2.0f;\n    e++;\n  }\n  const float gain = float(1 << e);\n\n  for (int i = 0; i < BK / 4; i++) {\n    const float4 v =\n        float4(vals[4 * i], vals[4 * i + 1], vals[4 * i + 2], vals[4 * i + 3]);\n    q[i] = char4(clamp(int4(round((v * inv) * gain)), -127, 127));\n  }\n\n  x_exp[lid + 1] = e;\n  if (lid == 0) {\n    x_exp[0] = e;\n  }\n  // The results are in the units of the last block\n  if (lid == n_blocks - 1) {\n    x_scale[row] = sx / gain;\n  }\n')
class SplitQuantVisual(pv.PrepackedVisual):

    def __init__(self, orig):
        self.w = orig.w
        self.packed = orig.packed
        self.info = dict(orig.info, concat_quant_fused=True)

    def single_post(self, x, att, mlp, p, m):
        key = p + '.linear2'
        if key not in self.packed:
            return super().single_post(x, att, mlp, p, m)
        g = gated(mlp)
        a0 = att.reshape(-1, att.shape[-1])
        g = g.reshape(-1, g.shape[-1])
        rows = a0.shape[0]
        k = a0.shape[1] + g.shape[1]
        mp = (rows + 63) // 64 * 64
        nb = k // 64
        threads = (nb + 31) // 32 * 32
        assert a0.shape[1] % 64 == 0 and g.shape[1] % 64 == 0
        a, s, e = qx(inputs=[a0, g], template=[('T', a0.dtype)], grid=(threads, mp, 1), threadgroup=(threads, 1, 1), output_shapes=[(mp, k), (rows,), (mp, nb + 1)], output_dtypes=[mx.int8, mx.float32, mx.uint8])
        parts = self.packed[key]
        n = parts[0].shape[0]
        y = pv.kernel(inputs=[*parts, a, s, e], template=[('T', x.dtype), ('group_size', 64), ('bits', 4), ('x_span', 0), ('XK', k), ('ROWS', rows)], grid=((n + 63) // 64 * 128, (rows + 63) // 64, 1), threadgroup=(128, 1, 1), output_shapes=[(rows, n)], output_dtypes=[x.dtype])[0].reshape(*x.shape[:-1], n)
        bias = self.w.get(key + '.bias')
        if bias is not None:
            y = y + bias
        return x + m[2] * y
