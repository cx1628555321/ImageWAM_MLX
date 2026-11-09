"""Explicit experimental-runtime launcher; delegates all observation I/O to infer.py."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import runpy
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
p = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
p.add_argument('--variant', choices=['stock', 'runtime', 'native-w1', 'native-w4', 'action-bf16'], required=True)
p.add_argument('--profile', default='speed')
p.add_argument('--prefix-precision', default='all_int8')
p.add_argument('--visionzip', choices=['off', 'merge252', 'late_align32', 'prune252', 'prune216'], default='off',
               help='Experimental visual-prefix compression; native-w4 only, default off')
a, remaining = p.parse_known_args()
if a.profile != 'speed' or a.prefix_precision != 'all_int8':
    p.error('This launcher validates speed + all_int8. Use scripts/infer.py for other profiles.')
if a.visionzip != 'off' and a.variant != 'native-w4':
    p.error('--visionzip requires --variant native-w4')
os.environ.setdefault('MLX_MAX_OPS_PER_BUFFER', '256')
sys.path.insert(0, str(ROOT))
import mlx.core as mx
import imagewam_mlx
import imagewam_mlx.runtime as rt

version = importlib.metadata.version('mlx')
expected = '0.32.4' if a.variant == 'stock' else '0.32.5.dev20261106'
if version != expected:
    raise RuntimeError(f'{a.variant} requires tested MLX {expected}, found {version}')
if a.variant != 'stock':
    # Isolated, version-specific compatibility override used by validated workers.
    source = Path(rt.__file__).read_text()
    guard = "version('mlx')=='0.32.4'"
    if guard not in source:
        raise RuntimeError('Runtime guard changed; revalidate this experimental launcher')
    exec(compile(source.replace(guard, 'True'), rt.__file__, 'exec'), rt.__dict__)

class VersionedImageWAM(rt.ImageWAM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        start = time.perf_counter()
        inventory = []
        if a.variant.startswith('native-'):
            from imagewam_mlx.quantized import PackedRopeWeights
            from imagewam_mlx.workspace import workspace_step
            kind = a.variant.removeprefix('native-')
            weights = mx.load(str(Path(self.weights_path) / 'action.safetensors'))
            mx.eval(weights)
            self.action = PackedRopeWeights.from_weights(rt.RopeWeights(weights), kind)
            self._compiled_workspace_step = mx.compile(
                lambda c, x, t, d, pe: workspace_step(self, c, x, t, d, pe))
            self.quantization_info['action'] = self.action.quantization_info
            self.action_epilogue = False
            self.action_epilogue_reason = 'Native MLX packed-weight linear path'
            inventory = [dict(action_kind=kind, qq=False)]
        elif a.variant == 'action-bf16':
            from imagewam_mlx.workspace import workspace_step
            weights = mx.load(str(Path(self.weights_path) / 'action.safetensors'))
            mx.eval(weights)
            self.action = rt.RopeWeights(weights)
            self._compiled_workspace_step = mx.compile(
                lambda c, x, t, d, pe: workspace_step(self, c, x, t, d, pe))
            self.action_epilogue = False
            self.action_epilogue_reason = 'Original BF16 action weights; dense floating-point path'
            self.quantization_info['action'] = {'kind': 'bf16', 'source': 'original_weights'}
            inventory = [dict(action_kind='bf16', prefix_precision='all_int8',
                              weight_bytes=sum(v.nbytes for v in weights.values()),
                              explicit_dequantize_in_forward=False)]
        mx.synchronize()
        self.version_info = dict(variant=a.variant, mlx=version, mlx_core=mx.__file__,
                                 adaptation_s=time.perf_counter()-start, inventory=inventory)
        if a.visionzip != 'off':
            from imagewam_mlx.visionzip import install
            self.version_info['visionzip'] = install(self, a.visionzip)

imagewam_mlx.ImageWAM = VersionedImageWAM
sys.argv = [str(ROOT/'scripts/infer.py'), '--profile', 'speed', '--prefix-precision', 'all_int8', *remaining]
ns = runpy.run_path(str(ROOT/'scripts/infer.py'), run_name='__main__')
record = ns['a'].output/'run.json'
d = json.loads(record.read_text())
d['tested_version'] = ns['model'].version_info
record.write_text(json.dumps(d, indent=2))
