"""Copy small tokenizer/stats and a reproducible example. No model re-downloads."""
import argparse,json,shutil,hashlib
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);a=p.parse_args();root=Path(__file__).resolve().parents[1]
assets=root/'assets';assets.mkdir(exist_ok=False)
q=assets/'models/Qwen3-4B';q.mkdir(parents=True)
for name in ('config.json','tokenizer.json','tokenizer_config.json','vocab.json','merges.txt'):
 shutil.copyfile(a.source/'models/Qwen3-4B'/name,q/name)
s=assets/'models/robotwin_flux2_4b';s.mkdir(parents=True);shutil.copyfile(a.source/'models/robotwin_flux2_4b/dataset_stats.json',s/'dataset_stats.json')
example=root/'examples/robotwin_ep0_frame0';example.mkdir(parents=True,exist_ok=False)
ref=a.source/'evaluate_results/m5pro_robotwin/20261103_dynamic_profile'
for name in ('inputs.npz','input_manifest.json','initial_noise_fp32.npy','reference_action.npy'):shutil.copyfile(ref/name,example/name)
(example/'prompt.txt').write_text(json.loads((ref/'input_manifest.json').read_text())['prompt'])
(root/'records/assets_manifest.json').write_text(json.dumps({str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for folder in (assets,example) for f in folder.rglob('*') if f.is_file()},indent=2))
shutil.copyfile(a.source/'LICENSE',root/'LICENSE.ImageWAM')
