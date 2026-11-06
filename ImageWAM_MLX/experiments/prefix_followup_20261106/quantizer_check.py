import sys,json
from pathlib import Path
sys.path.insert(0,str(Path.cwd()));sys.path.insert(0,str(Path('experiments/background_20261105').resolve()));sys.path.insert(0,str(Path('experiments/prefix_followup_20261106').resolve()))
import mlx.core as mx
from candidate import slim_quantizer
from imagewam_mlx import staged_visual as sv
q=slim_quantizer();rows=[]
for m,k in [(1,64),(31,2048),(39,3072),(64,3072),(65,6144),(127,12288)]:
 x=mx.random.normal((m,k),key=mx.random.key(m)).astype(mx.bfloat16)
 a,s,e=sv.quantize(x);b,t,d=q(x);mx.eval(a,s,b,t)
 equal=bool(mx.array_equal(a,b).item() and mx.array_equal(s,t).item())
 rows.append(dict(m=m,k=k,exact=equal));assert equal,(m,k)
print(json.dumps(rows,indent=2))
