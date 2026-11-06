import sys,json,time
from pathlib import Path
import mlx.core as mx
import numpy as np
sys.path.insert(0,str(Path.cwd()))
from staged_int8 import prepare,project,quantize
root=Path('records/runtime_tuning_20261104/int8_probe');root.mkdir(exist_ok=False)
rng=np.random.default_rng(7);rows=[]
for m,k,n in [(64,256,128),(65,256,128),(328,512,128)]:
 x=mx.array(rng.normal(size=(m,k)).astype(np.float32)).astype(mx.bfloat16);w=mx.array(rng.normal(size=(n,k)).astype(np.float32)).astype(mx.bfloat16);parts=mx.quantize(w,group_size=64,bits=4);pp=prepare(parts);y=project(x,pp);mx.eval(y)
 q,s,e=quantize(x);mx.eval(q,s,e);mp=(m+63)//64*64;xi=np.array(q).reshape(mp//64,k//64,64,64).transpose(0,2,1,3).reshape(mp,k)[:m].astype(np.int32)
 codes=((np.array(parts[0])[...,None]>>np.arange(0,32,4,dtype=np.uint32))&15).reshape(n,k).astype(np.int32);sc=np.array(parts[1].astype(mx.float32));bs=np.array(parts[2].astype(mx.float32));inv=np.array(pp[3]);ws=np.array(pp[4]);a=np.rint(sc*inv[:,None]).astype(np.int32);b=np.rint(bs*inv[:,None]).astype(np.int32)+(1<<15);wi=((codes*np.repeat(a,64,axis=1)+np.repeat(b,64,axis=1))>>16).astype(np.int8).astype(np.int32)
 expected=(xi@wi.T).astype(np.float32)*np.array(s)[:,None]*ws[None,:];expected=np.array(mx.array(expected).astype(mx.bfloat16).astype(mx.float32));actual=np.array(y.astype(mx.float32));diff=actual-expected;assert np.array_equal(actual,expected),(m,float(abs(diff).max()))
 qref=np.array(mx.quantized_matmul(x,*parts,group_size=64,bits=4).astype(mx.float32));row=dict(m=m,k=k,n=n,integer_reference_exact=True,q4_rmse=float(np.sqrt(np.mean((actual-qref)**2))),q4_max_abs=float(abs(actual-qref).max()));rows.append(row);print(row,flush=True)
(root/'results.json').write_text(json.dumps(rows,indent=2))
