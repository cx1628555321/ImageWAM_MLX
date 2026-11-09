"""Frozen native-w4 + merge252 configuration, validated on 2026-11-09."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'versions/native-w4-zip252-20261109.json'


def main():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--inputs', type=Path, required=True)
    prompt = p.add_mutually_exclusive_group(required=True)
    prompt.add_argument('--instruction')
    prompt.add_argument('--prompt-file', type=Path)
    p.add_argument('--noise', type=Path)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--warmup', type=int, default=10)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.warmup < 0:
        p.error('--warmup must be nonnegative')
    manifest = json.loads(MANIFEST.read_text())
    for relative, expected in manifest['source_sha256'].items():
        if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'Frozen version source changed: {relative}; revalidate before updating manifest')
    # Resolve user paths before infer_version changes the working directory.
    args = ['--variant', 'native-w4', '--visionzip', 'merge252',
            '--steps', '10', '--horizon', '16', '--text-cache-size', '1',
            '--no-vae-asymmetric', '--inputs', str(a.inputs.resolve()),
            '--output', str(a.output.resolve()), '--warmup', str(a.warmup),
            '--seed', str(a.seed)]
    if a.prompt_file is not None:
        args += ['--prompt-file', str(a.prompt_file.resolve())]
    else:
        args += ['--instruction', a.instruction]
    if a.noise is not None:
        import numpy as np
        noise = np.load(a.noise, allow_pickle=False)
        if noise.shape != (1, 16, 14) or noise.dtype != np.float32 or not np.isfinite(noise).all():
            p.error('--noise must be finite float32 with shape [1,16,14]')
        args += ['--noise', str(a.noise.resolve())]
    os.environ['MLX_MAX_OPS_PER_BUFFER'] = '256'  # Operator count, not a memory limit.
    sys.argv = [str(ROOT / 'scripts/infer_version.py'), *args]
    runpy.run_path(sys.argv[0], run_name='__main__')
    # Use the resolved output path captured before the delegated launcher chdir.
    record = Path(args[args.index('--output') + 1]) / 'run.json'
    result = json.loads(record.read_text())
    result['frozen_version'] = {
        'id': manifest['id'],
        'manifest_sha256': hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        'source_hashes_verified': True,
        'allocator_policy': 'MLX defaults; no explicit cache/memory/wired limits',
    }
    record.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
