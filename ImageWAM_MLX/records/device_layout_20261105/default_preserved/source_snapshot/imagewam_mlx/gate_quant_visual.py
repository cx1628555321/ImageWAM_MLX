"""BF16 SiLU lookup and strided gate/INT8 quantization fusion.
A complete static function table, not a cache of request inputs.
"""
import mlx.core as mx
from . import split_quant_visual as sq
pv=sq.pv
q = mx.fast.metal_kernel(name='wam_gate_silu_vector', input_names=['att', 'mlp', 'lut'], output_names=['x8', 'x_scale', 'x_exp'], source='\n const uint row=threadgroup_position_in_grid.y,lid=thread_index_in_threadgroup;\n const uint lane=thread_index_in_simdgroup,sg=simdgroup_index_in_threadgroup;\n const int M=att_shape[0],KA=att_shape[1],KB=mlp_shape[1]/2,K=XK,nb=K/64;\n constexpr int VPT=(XK+THREADS-1)/THREADS;\n threadgroup float shared[33];float vals[VPT];float maxv=0.0f;\n for(int i=0;i<VPT;i++){\n  int j=lid+i*THREADS;float v=0.0f;\n  if(row<M && j<K){\n   if(j<KA)v=float(att[row*att_strides[0]+j*att_strides[1]]);\n   else {int idx=j-KA;T u=mlp[row*mlp_strides[0]+idx*mlp_strides[1]];T a=lut[as_type<ushort>(u)];v=float(T(float(a)*float(mlp[row*mlp_strides[0]+(idx+KB)*mlp_strides[1]])));}\n  }\n  vals[i]=v;maxv=max(maxv,fabs(v));\n }\n maxv=simd_max(maxv);if(lane==0)shared[sg]=maxv;\n threadgroup_barrier(mem_flags::mem_threadgroup);\n if(lid==0){float m=shared[0];for(int i=1;i<THREADS/32;i++)m=max(m,shared[i]);shared[32]=max(m*(1.0f/127.0f),1e-12f);if(row<M)x_scale[row]=shared[32];}\n threadgroup_barrier(mem_flags::mem_threadgroup);\n float inv=1.0f/shared[32];\n for(int i=0;i<VPT;i++){int j=lid+i*THREADS;if(j<K)x8[((row/64)*nb+j/64)*4096+(row%64)*64+j%64]=char(clamp(int(round(vals[i]*inv)),-127,127));}\n for(int j=lid;j<nb+1;j+=THREADS)x_exp[row*(nb+1)+j]=0;\n', ensure_row_contiguous=False)
class GateQuantVisual(sq.SplitQuantVisual):

    def __init__(self, orig):
        super().__init__(orig)
        self.info = dict(orig.info, gate_quant_fused=True, gate_lut_bytes=131072)
        keys = mx.arange(65536, dtype=mx.uint16).view(mx.bfloat16)
        self.lut = keys * mx.sigmoid(keys)
        mx.eval(self.lut)

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
        y = pv.kernel(inputs=[*parts, a, s, e], template=[('T', x.dtype), ('group_size', 64), ('bits', 4), ('x_span', 0), ('XK', k), ('ROWS', rows)], grid=((n + 63) // 64 * 128, (rows + 63) // 64, 1), threadgroup=(128, 1, 1), output_shapes=[(rows, n)], output_dtypes=[x.dtype])[0].reshape(x.shape)
        bias = self.w.get(key + '.bias')
        if bias is not None:
            y = y + bias
        return x + m[2] * y
