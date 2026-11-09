import subprocess,json,sys
from pathlib import Path
root=Path('records/speed_mixed_20261103');root.mkdir(exist_ok=False)
cases=[('vae_bf16',{},dict(vae='fast_bf16')),('vae_fp16',{},dict(vae='fast_fp16')),('qwen_fused',{},dict(qwen_fused=True)),('action_w1',dict(action='w1'),{}),('action_w2',dict(action='w2'),{}),('action_w4',dict(action='w4'),{}),('combined_w2',dict(action='w2'),dict(qwen_fused=True,vae='fast_bf16')),('combined_w1',dict(action='w1'),dict(qwen_fused=True,vae='fast_bf16')),('qwen_fused_w2',dict(qwen='w2',action='w2'),dict(qwen_fused=True,vae='fast_bf16')),('qwen_fused_nvfp4a',dict(qwen='nvfp4a',action='w2'),dict(qwen_fused=True,vae='fast_bf16'))]
for name,policy,options in cases:
 with (root/(name+'.log')).open('w') as f:
  result=subprocess.run([sys.executable,'scripts/sweep_quantized_speed.py','--policy',json.dumps(policy),'--options',json.dumps(options),'--output',str(root/name),'--iterations','20'],stdout=f,stderr=subprocess.STDOUT)
 print(name,result.returncode,flush=True)
