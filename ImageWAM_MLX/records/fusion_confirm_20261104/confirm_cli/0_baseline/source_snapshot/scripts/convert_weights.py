"""One-time CPU conversion. Never alters source models; refuses existing destination."""
import argparse, hashlib, json, time
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import save_file
p=argparse.ArgumentParser(); p.add_argument('--source',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=False)
manifest={'source':str(a.source.resolve()),'created':time.time(),'files':{},'dtype':'bfloat16','quantized':False}
def save(name,state):
    target=a.output/name
    save_file({k:v.to(torch.bfloat16).contiguous() for k,v in state.items() if v.is_floating_point()},str(target))
    digest=hashlib.sha256()
    with target.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):digest.update(b)
    manifest['files'][name]={'sha256':digest.hexdigest(),'bytes':target.stat().st_size,'tensors':{k:list(v.shape) for k,v in state.items() if v.is_floating_point()}}
    print(name,target.stat().st_size,flush=True)
payload=torch.load(a.source/'models/robotwin_flux2_4b/model.pt',map_location='cpu',weights_only=True,mmap=True)
for group,prefix in [('visual','mixtures.video.transformer.'),('action','mixtures.action.')]:
    state={k[len(prefix):]:v for k,v in payload['mot'].items() if k.startswith(prefix) and not k[len(prefix):].startswith('final_layer.')}
    save(group+'.safetensors',state)
save('proprio.safetensors',payload['proprio_encoder']); del payload,state
with safe_open(str(a.source/'models/flux2_klein_base_4b/ae.safetensors'),framework='pt') as f:
    save('vae.safetensors',{k:f.get_tensor(k) for k in f.keys() if not k.startswith('decoder.')})
q=a.source/'models/Qwen3-4B'
index=json.loads((q/'model.safetensors.index.json').read_text())['weight_map']
for i,shard in enumerate(sorted(set(index.values()))):
    with safe_open(str(q/shard),framework='pt') as f:
        state={}
        for k in f.keys():
            if k=='model.embed_tokens.weight' or (k.startswith('model.layers.') and int(k.split('.')[2])<27): state[k.removeprefix('model.')]=f.get_tensor(k)
        save(f'qwen-{i}.safetensors',state)
(a.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
print('COMPLETE',flush=True)
