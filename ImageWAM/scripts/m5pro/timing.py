"""MPS synchronized timing and atomic result serialization."""
import json
import time
from contextlib import contextmanager

def write(path, obj):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)

def stats(values):
    import numpy as np
    v = list(values)
    return dict(count=len(v), mean=float(np.mean(v)), median=float(np.median(v)), p95=float(np.percentile(v, 95)), std=float(np.std(v)), min=min(v), max=max(v))

class Recorder:

    def __init__(self):
        self.records = []
        self.phase = 'load'
        self.iteration = -1
        self.stack = []

    @contextmanager
    def span(self, name, category='module', **kwargs):
        import torch
        torch.mps.synchronize()
        parent = self.stack[-1] if self.stack else None
        self.stack.append(name)
        start = time.perf_counter()
        status = 'ok'
        try:
            yield
        except BaseException:
            status = 'failed'
            raise
        finally:
            torch.mps.synchronize()
            self.records.append(dict(name=name, category=category, parent=parent, phase=self.phase, iteration=self.iteration, wall_ms=(time.perf_counter() - start) * 1000, status=status))
            self.stack.pop()
