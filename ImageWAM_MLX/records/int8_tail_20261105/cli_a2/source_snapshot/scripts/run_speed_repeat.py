import subprocess,json,sys
from pathlib import Path
root=Path('records/speed_repeat_20261103');root.mkdir(exist_ok=False)
configs={'vae_only':{},'action_w1':{'action':'w1'},'action_w4':{'action':'w4'},'conditioning_w4':{'action':'w4','visual':'w4_conditioning'}}
order=list(configs)+list(reversed(configs))
for i,name in enumerate(order):
 with (root/(str(i)+'_'+name+'.log')).open('w') as f:
  result=subprocess.run([sys.executable,'scripts/sweep_quantized_speed.py','--policy',json.dumps(configs[name]),'--options',json.dumps(dict(vae='fast_bf16')),'--output',str(root/(str(i)+'_'+name)),'--iterations','40'],stdout=f,stderr=subprocess.STDOUT)
 print(i,name,result.returncode,flush=True)
