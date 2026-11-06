from pathlib import Path
import ast
p=Path('imagewam_mlx_work/project');out=p/'experiments/int8_pipeline_20261105'
s=(p/'imagewam_mlx/tail_visual.py').read_text();t=ast.parse(s)
c=next(n.value for n in t.body if isinstance(n,ast.Assign) and any(isinstance(a,ast.Name) and a.id=='kernel' for a in n.targets))
src=ast.literal_eval(next(k.value for k in c.keywords if k.arg=='source'))
start='  for (int k = 0; k < K; k += BK) {'
end='    // Move the accumulator'
# Transform both full and tail branches, preserving arithmetic and packed layout.
for _ in range(2):
 i=src.index(start);j=src.index(end,i)
 replacement='''  // Prime the first buffer, then stage the next tile while current compute is pending.
  if (valid) {
    STEEL_PRAGMA_UNROLL
    for (int i = 0; i < VPT / 4; i++)
      *(threadgroup uint*)(dst + 4*i) = *(const device uint*)(wl + kl + 4*i);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int k=0; k<K; k+=BK) {
    if (valid && k+BK<K) {
      threadgroup int8_t* next_dst=dst+(((k/BK+1)%2)*BN*BK_padded);
      STEEL_PRAGMA_UNROLL
      for (int i=0; i<VPT/4; i++)
        *(threadgroup uint*)(next_dst+4*i)=*(const device uint*)(wl+k+BK+kl+4*i);
    }

'''
 src=src[:i]+replacement+src[j:]
src=src.replace('Ws_storage[(BN * BK_padded) / 4]','Ws_storage[(2 * BN * BK_padded) / 4]')
src=src.replace('Ws + tn * BK_padded + kk1','Ws + ((k/BK)%2)*BN*BK_padded + tn * BK_padded + kk1')
src=src.replace('    x8 += 64 * BK;', '    threadgroup_barrier(mem_flags::mem_threadgroup);\n    x8 += 64 * BK;').replace('    x8 += BM * BK;', '    threadgroup_barrier(mem_flags::mem_threadgroup);\n    x8 += BM * BK;')
src=src.replace('*(threadgroup uint*)(dst + 4 * i) = 0;','*(threadgroup uint*)(dst + 4 * i) = 0;\n      *(threadgroup uint*)(dst + BN*BK_padded + 4*i) = 0;')
head=s[:s.index('kernel=')]
head+=f"kernel=mx.fast.metal_kernel(name='wam_int8_pipeline',input_names=['w','scales','biases','w_inv','w_scale','x8','x_scale','x_exp'],output_names=['y'],header=sv.header,source={src!r})\n"
head+=s[s.index('class TailVisual'):].replace('class TailVisual','class Candidate').replace('def __init__(self,orig):',"def __init__(self,orig,scope='all'):").replace('self.info=dict(self.info, tail_rows=32, tail_layout_rows=64)','self.scope=scope')
a=head.index('    def mm(');b=head.index('    def linear(',a)
head=head[:a]+'''    def mm(self,**kwargs):
        if self.scope=='linear2' and kwargs['inputs'][0].shape[0]!=3072:
            from imagewam_mlx.tail_visual import TailVisual
            return TailVisual.mm(self,**kwargs)
        return kernel(**kwargs)
'''+head[b:]
(out/'candidate.py').write_text(head)
old=p/'experiments/int8_tail_20261105'
for f in ('paired.py','worker.py','independent.py','dynamic.py'):
 text=(old/f).read_text().replace('int8_tail_20261105','int8_pipeline_20261105')
 (out/f).write_text(text)
