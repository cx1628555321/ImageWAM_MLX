"""Opt-in fused QKV experiment. Numerical equivalence is not guaranteed."""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from compile_experiments import Candidates
from qkv_experiment import FusedQwen
import os
os.environ['IMAGEWAM_MLX_OPTIMIZATION']='legacy'
import imagewam_mlx.model as mm


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--entry', choices=['infer', 'benchmark', 'validate_cases'], required=True)
    args, rest = parser.parse_known_args()
    original = mm.ImageWAM.__init__

    def init(self, *a, **kw):
        original(self, *a, **kw)
        self._compile_candidates = Candidates(self)
        self._compile_candidates.set('both_compile')
        self._fused_qwen = FusedQwen(self.qwen)
        self.qwen = mx.compile(self._fused_qwen.__call__)

    mm.ImageWAM.__init__ = init
    try:
        sys.argv = [str(ROOT / 'scripts' / (args.entry + '.py'))] + rest
        runpy.run_path(sys.argv[0], run_name='__main__')
        out = Path(rest[rest.index('--output') + 1])
        files = ['run_qkv_experimental.py', 'qkv_experiment.py', 'compile_experiments.py']
        (out / 'optimization.json').write_text(json.dumps({
            'mode': 'both_compile_qkv_experimental',
            'entry': args.entry,
            'policy': 'BF16; actual text length; no truncation or feature caching; extra rounding differences possible',
            'source_sha256': {f: hashlib.sha256((ROOT / 'scripts' / f).read_bytes()).hexdigest() for f in files}
        }, indent=2))
        if args.entry == 'validate_cases':
            rows = []
            for f in sorted(out.glob('*.npy')):
                ref = ROOT / 'records' / 'validation_v1' / f.name
                if ref.exists():
                    x, y = np.load(f), np.load(ref)
                    rows.append({'file': f.name, 'bitwise_equal': bool(np.array_equal(x, y)),
                                 'max_abs': float(np.max(np.abs(x.astype(np.float64) - y.astype(np.float64))))})
            assert sum(r['file'].endswith('_action.npy') for r in rows) == 51
            (out / 'qkv_vs_original_mlx.json').write_text(json.dumps(rows, indent=2))
    finally:
        mm.ImageWAM.__init__ = original


if __name__ == '__main__':
    main()
