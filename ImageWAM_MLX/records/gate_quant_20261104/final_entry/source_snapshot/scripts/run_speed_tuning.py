import subprocess,json,sys
from pathlib import Path
root=Path('records/speed_tuning_20261103');root.mkdir(exist_ok=False)
cases=[('accuracy','accuracy',{}),('fast_vae',{},dict(vae='fast_bf16'))]
for mode in ('w1','w2','w4','w1g128','w2g128','w4g128','w2g32','nvfp4'):
 cases.append(('fast_'+mode,dict(action=mode),dict(vae='fast_bf16')))
for name,policy,options in cases:
 with (root/(name+'.log')).open('w') as f:
  result=subprocess.run([sys.executable,'scripts/sweep_quantized_speed.py','--policy',policy if isinstance(policy,str) else json.dumps(policy),'--options',json.dumps(options),'--output',str(root/name),'--iterations','30'],stdout=f,stderr=subprocess.STDOUT)
 print(name,result.returncode,flush=True)
