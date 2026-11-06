import mlx.core as mx
from imagewam_mlx.gate_quant_visual import GateQuantVisual,q
from imagewam_mlx import prepacked_visual as pv
sv=pv.sv
kernel=mx.fast.metal_kernel(name='wam_int8_wide',input_names=['w','scales','biases','w_inv','w_scale','x8','x_scale','x_exp'],output_names=['y'],header=sv.header,source='uint3 tid=threadgroup_position_in_grid;uint lid=thread_index_in_threadgroup,simd_gid=simdgroup_index_in_threadgroup;const int K=XK,N=w_shape[0],M=ROWS;\n\n  constexpr int BM = 64;\n  constexpr int BN = 128;\n  constexpr int BK = 64;\n  constexpr int WM = 2;\n  constexpr int WN = TILE_WN;\n  constexpr int BK_padded = BK + 16;\n\n  // Each thread folds VPT weights of one row, all in one quantization group\n  constexpr int VPT = (BN * BK) / (WM * WN * SIMD_SIZE);\n  static_assert(\n      VPT % 8 == 0 && BK % VPT == 0 && group_size % VPT == 0,\n      "A thread stages whole packs of the weights within one group");\n\n  constexpr int pack_factor = get_pack_factor<bits, 8>();\n  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();\n\n  threadgroup uint Ws_storage[(BN * BK_padded) / 4];\n  threadgroup int8_t* Ws = (threadgroup int8_t*)Ws_storage;\n\n  constexpr short SM = BM / WM;\n  constexpr short SN = BN / WN;\n  constexpr short SK = 32;\n\n  constexpr short TM = SM / 16;\n  constexpr short TN = SN / 16;\n  constexpr short TK = SK / 16;\n\n  const int y_row = tid.y * BM;\n  const int y_col = tid.x * BN;\n  const short tm = SM * (simd_gid / WN);\n  const short tn = SN * (simd_gid % WN);\n\n  // The part of the weight tile this thread stages\n  const int nl = lid / (BK / VPT);\n  const int kl = (lid % (BK / VPT)) * VPT;\n  const int n = y_col + nl;\n  const bool valid = n < N;\n\n  threadgroup int8_t* dst = Ws + nl * BK_padded + kl;\n  const device uint8_t* wl = (const device uint8_t*)w;\n  float inv = 0.0f;\n  if (valid) {\n    wl += size_t(n) * K;\n    scales += size_t(n) * (K / group_size);\n    biases += size_t(n) * (K / group_size);\n    inv = w_inv[n];\n  } else {\n    for (int i = 0; i < VPT / 4; i++) {\n      *(threadgroup uint*)(dst + 4 * i) = 0;\n    }\n  }\n\n  // x8 has one BM x BK block per row tile and BK columns of K\n  x8 += (size_t(tid.y) * (K / BK) * BM + tm) * BK;\n\n  // A thread has 8 values of the accumulator for each of 4 rows\n  constexpr int exp_ld = BaseNAXFrag::kElemRowsJump;\n  const int x_exp_stride = K / BK + 1;\n  x_exp += size_t(y_row + tm + BaseNAXFrag::get_coord().y) * x_exp_stride;\n\n  NAXTile<int32_t, TM, TN> Dtile;\n  Dtile.clear();\n\n  for (int k = 0; k < K; k += BK) {\n    threadgroup_barrier(mem_flags::mem_threadgroup);\n\n    if (valid) {\n      const int kk = k + kl;\n      const int a = int(scales[kk / group_size]);\n      const int b = int(biases[kk / group_size]);\n      const device uint8_t* p = wl + kk;\n\n      STEEL_PRAGMA_UNROLL\n      for (int i = 0; i < VPT / 4; i++) {\n        const int4 v = int4(*(const device char4*)(p + 4*i));\n        *(threadgroup uint*)(dst + 4 * i) = as_type<uint>(char4(v));\n      }\n    }\n\n    threadgroup_barrier(mem_flags::mem_threadgroup);\n\n    // Move the accumulator of each row to the scale of this block of x. A\n    // fragment has 4 values for each of 2 rows.\n    constexpr int shift_every = x_span > 0 ? x_span : 1;\n    const bool shift = x_span > 0 && (k / BK) % shift_every == 0;\n    STEEL_PRAGMA_UNROLL\n    for (short i = 0; shift && i < TM; i++) {\n      typedef metal::vec<int32_t, BaseNAXFrag::kElemsPerFrag> frag_t;\n      const device uint8_t* e0 =\n          x_exp + size_t(i * BaseNAXFrag::kFragRows) * x_exp_stride + k / BK;\n      const device uint8_t* e1 = e0 + size_t(exp_ld) * x_exp_stride;\n      const int s0 = int(e0[1]) - int(e0[0]);\n      const int s1 = int(e1[1]) - int(e1[0]);\n      const int l0 = max(s0, 0);\n      const int l1 = max(s1, 0);\n      const int r0 = max(-s0, 0);\n      const int r1 = max(-s1, 0);\n      const frag_t left = frag_t(l0, l0, l0, l0, l1, l1, l1, l1);\n      const frag_t right = frag_t(r0, r0, r0, r0, r1, r1, r1, r1);\n      const frag_t rounding = (frag_t(1) << right) >> 1;\n      STEEL_PRAGMA_UNROLL\n      for (short j = 0; j < TN; j++) {\n        Dtile.frag_at(i, j) =\n            ((Dtile.frag_at(i, j) << left) + rounding) >> right;\n      }\n    }\n\n    STEEL_PRAGMA_NO_UNROLL\n    for (int kk1 = 0; kk1 < BK; kk1 += SK) {\n      NAXTile<int8_t, TM, TK> Atile;\n      NAXTile<int8_t, TN, TK> Btile;\n\n      volatile int compiler_barrier;\n\n      Atile.load(x8 + kk1, BK);\n      Btile.template load<int8_t, BK_padded, 1>(Ws + tn * BK_padded + kk1);\n\n      tile_matmad_nax(\n          Dtile,\n          Atile,\n          metal::bool_constant<false>{},\n          Btile,\n          metal::bool_constant<true>{});\n\n      (void)compiler_barrier;\n    }\n\n    x8 += BM * BK;\n  }\n\n  // Guarded stores are slow, only use them at the edges of the output\n  const bool interior = (y_row + tm + SM <= M) && (y_col + tn + SN <= N);\n  dispatch_bool(interior, [&](auto kInterior) {\n    STEEL_PRAGMA_UNROLL\n    for (short i = 0; i < TM; i++) {\n      STEEL_PRAGMA_UNROLL\n      for (short j = 0; j < TN; j++) {\n        const auto frag = Dtile.frag_at(i, j);\n        STEEL_PRAGMA_UNROLL\n        for (short e = 0; e < BaseNAXFrag::kElemsPerFrag; e++) {\n          const short2 pos = BaseNAXFrag::get_coord(e);\n          const int r = y_row + tm + i * BaseNAXFrag::kFragRows + pos.y;\n          const int c = y_col + tn + j * BaseNAXFrag::kFragCols + pos.x;\n          if constexpr (kInterior.value) {\n            y[size_t(r) * N + c] =\n                static_cast<T>(float(frag[e]) * x_scale[r] * w_scale[c]);\n          } else if (r < M && c < N) {\n            y[size_t(r) * N + c] =\n                static_cast<T>(float(frag[e]) * x_scale[r] * w_scale[c]);\n          }\n        }\n      }\n    }\n  });\n')
class Candidate(GateQuantVisual):
    def __init__(self,orig,wn=2):
        self.__dict__.update(orig.__dict__)
        if wn not in (2,4):raise ValueError('wn must be 2 or 4')
        self.wn=wn
    def mm(self,**kwargs):
        kwargs['template']=[*kwargs['template'],('TILE_WN',self.wn)]
        n=kwargs['inputs'][0].shape[0]
        gx,gy,gz=kwargs['grid'];threads=64*self.wn
        kwargs['grid']=((n+127)//128*threads,gy,gz)
        kwargs['threadgroup']=(threads,1,1)
        return kernel(**kwargs)
    def linear(self, x, p):
        if p not in self.packed:
            return sv.RopeWeights.linear(self, x, p)
        shape = x.shape
        x = x.reshape(-1, shape[-1])
        m, k = x.shape
        parts = self.packed[p]
        n = parts[0].shape[0]
        a, s, e = sv.quantize(x)
        y = self.mm(inputs=[*parts, a, s, e], template=[('T', x.dtype), ('group_size', 64), ('bits', 4), ('x_span', 0), ('XK', k), ('ROWS', m)], grid=((n + 63) // 64 * 128, (m + 63) // 64, 1), threadgroup=(128, 1, 1), output_shapes=[(m, n)], output_dtypes=[x.dtype])[0].reshape(*shape[:-1], n)
        bias = self.w.get(p + '.bias')
        return y if bias is None else y + bias
    def single_post(self, x, att, mlp, p, m):
        key = p + '.linear2'
        if key not in self.packed:
            return super().single_post(x, att, mlp, p, m)
        a0 = att.reshape(-1, att.shape[-1])
        g = mlp.reshape(-1, mlp.shape[-1])
        rows = a0.shape[0]
        k = a0.shape[1] + g.shape[1] // 2
        mp = (rows + 63) // 64 * 64
        nb = k // 64
        a, s, e = q(inputs=[a0, g, self.lut], template=[('T', x.dtype), ('XK', k), ('THREADS', 1024)], grid=(1024, mp, 1), threadgroup=(1024, 1, 1), output_shapes=[(mp, k), (rows,), (mp, nb + 1)], output_dtypes=[mx.int8, mx.float32, mx.uint8])
        parts = self.packed[key]
        n = parts[0].shape[0]
        y = self.mm(inputs=[*parts, a, s, e], template=[('T', x.dtype), ('group_size', 64), ('bits', 4), ('x_span', 0), ('XK', k), ('ROWS', rows)], grid=((n + 63) // 64 * 128, (rows + 63) // 64, 1), threadgroup=(128, 1, 1), output_shapes=[(rows, n)], output_dtypes=[x.dtype])[0].reshape(x.shape)
        bias = self.w.get(key + '.bias')
        if bias is not None:
            y = y + bias
        return x + m[2] * y
