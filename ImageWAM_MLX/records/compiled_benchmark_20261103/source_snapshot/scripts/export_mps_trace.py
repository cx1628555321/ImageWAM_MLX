"""Independent reference generation in the original PyTorch environment."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser(); p.add_argument('--source',type=Path,required=True); p.add_argument('--reference',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
a=p.parse_args(); a.source=a.source.resolve(); a.output.mkdir(parents=True,exist_ok=False)
for path in ('src','third_party/flux2/src','scripts/m5pro'):sys.path.insert(0,str(a.source/path))
import torch
from imagewam.optimized_action_loader import create_imagewam_flux2_action_only
from optimizations import optimized
kwargs=json.loads((a.reference/'benchmark.json').read_text())['model_config']; kwargs.update(model_dtype=torch.bfloat16)
model=create_imagewam_flux2_action_only(**kwargs)
image=torch.from_numpy(np.load(a.reference/'model_input_bf16_as_fp32.npy')).to('mps',torch.bfloat16)
st=torch.from_numpy(np.load(a.reference/'proprio_normalized.npy'))
prompt=json.loads((a.reference/'input_manifest.json').read_text())['prompt']; trace={}
def arr(t):return t.detach().float().cpu().numpy()
with torch.no_grad(),optimized(model,'dynamic'):
    original=model._encode_flux2_prompts
    def text(prompts):
        result=original(prompts);trace['text']=arr(result[0]);return result
    model._encode_flux2_prompts=text
    original_image=model.vae.encode
    def encode(x):
        result=original_image(x);trace['latent']=arr(result).transpose(0,2,3,1);return result
    model.vae.encode=encode
    original_prefill=model.mot.prefill_flux2_video_cache
    def prefill(*args,**kwargs):
        result=original_prefill(*args,**kwargs)
        caches=result['double']+result['single']
        for i in (0,4,5,24):
            x=caches[i]['k'];trace[f'cache_k_{i}']=arr(x.reshape(1,x.shape[1],24,128).transpose(1,2))
        return result
    model.mot.prefill_flux2_video_cache=prefill
    old_step=model.infer_action_scheduler.step; step_index=[0]
    def step(*args):
        result=old_step(*args);trace[f'action_step_{step_index[0]}']=arr(result);step_index[0]+=1;return result
    model.infer_action_scheduler.step=step
    out=model.infer_action_flux2(prompt=prompt,input_image=image,proprio=st,action_horizon=16,num_inference_steps=10,seed=42,rand_device='cpu')
trace['image']=arr(image).transpose(0,2,3,1); trace['state']=arr(st.to(torch.bfloat16))[:,None]
np.savez(a.output/'trace.npz',**trace);np.save(a.output/'normalized.npy',out['action'].numpy())
print('MPS_TRACE_COMPLETE',a.output,flush=True)
