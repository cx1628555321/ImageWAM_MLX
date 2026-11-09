"""Record action-only loading boundaries and compare saved regression outputs."""
import json, sys, math
from pathlib import Path
import numpy as np
import mlx.core as mx
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from imagewam_mlx import ImageWAM

out = Path('records/redundancy_audit_20261103')
manifest = json.loads(Path('weights/bf16/manifest.json').read_text())
model = ImageWAM('weights/bf16', 'assets', compact=False)
key = 'single_blocks.19.linear2.weight'
assert key not in model.visual.w
assert not any(k.startswith('decoder.') for k in model.vae.w)
assert not any(k.startswith('final_layer.') for k in model.visual.w)
assert not any(k.startswith('layers.27.') or k.startswith('lm_head.') for k in model.qwen.w)
files = sorted(Path('records/redundancy_validation_20261103').glob('*.npy'))
comparisons = []
for f in files:
    ref = Path('records/validation_v1') / f.name
    a, b = np.load(f), np.load(ref)
    comparisons.append({'file':f.name, 'bitwise_equal':bool(np.array_equal(a,b))})
assert len(files)==162 and all(r['bitwise_equal'] for r in comparisons)
result = {
    'status':'completed', 'comparisons':comparisons,
    'removed_runtime_tensor':key,
    'removed_logical_bytes':math.prod(manifest['files']['visual.safetensors']['tensors'][key])*2,
    'checkpoint_modified':False,
    'remaining_visual_full_projection':list(model.visual.w['single_blocks.19.linear1.weight'].shape),
    'remaining_vae_quant_conv':list(model.vae.w['encoder.quant_conv.weight'].shape),
    'note':'Runtime dictionary exclusion before mx.eval; physical disk reads were not measured. No speedup claim.'
}
(out/'audit.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k!='comparisons'},indent=2))
