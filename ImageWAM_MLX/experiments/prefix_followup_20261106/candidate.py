"""Prefix-only experiments atop the validated native W1 action path."""
import ast
import re
from pathlib import Path
import mlx.core as mx
from imagewam_mlx import staged_visual as sv
from candidates import install as existing_install

def slim_quantizer():
    tree = ast.parse(Path('imagewam_mlx/staged_visual.py').read_text())
    node = next(n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'qx' for t in n.targets))
    expr = next(k.value for k in node.value.keywords if k.arg == 'source')
    def literal(n):
        if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Add):
            return literal(n.left) + literal(n.right)
        return ast.literal_eval(n)
    source = literal(expr)
    # The validated mm callers all specialize x_span=0; exponent loads are dead.
    source = '\n'.join(line for line in source.splitlines() if not re.search(r'\bx_exp\b', line))
    assert 'max_exp=0' in source and 'uint3 tid=' in source
    kernel = mx.fast.metal_kernel(name='prefix_no_unused_exponents', input_names=['x'],
        output_names=['x8', 'x_scale'], header=sv.header, source=source)
    def quantize(x):
        m, k = x.shape
        assert k % 64 == 0 and k // 64 <= 1024
        mp = (m+63)//64*64
        threads = ((k//64+31)//32)*32
        q, s = kernel(inputs=[x], template=[('T', x.dtype)], grid=(threads, mp, 1),
            threadgroup=(threads, 1, 1), output_shapes=[(mp,k),(m,)], output_dtypes=[mx.int8,mx.float32])
        # Static zeros are folded into Metal constant memory by mx.compile.
        # Alias the dynamic activation buffer instead: no new data or writes.
        # Consumers are guarded to x_span=0 and never read this argument.
        return q, s, q.view(mx.uint8)
    return quantize

def install(model, mode, raw=None, scales=None):
    inventory = existing_install(model, 'native_w1')
    if mode in ('fuse_bias', 'fuse_norm', 'fuse_both'):
        inventory += existing_install(model, mode)
    elif mode == 'slim_quant':
        old_mm = model.visual.mm
        def guarded_mm(**kwargs):
            if dict(kwargs['template'])['x_span'] != 0:
                raise ValueError('Exponent removal is only valid with x_span=0')
            return old_mm(**kwargs)
        model.visual.mm = guarded_mm
        sv.quantize = slim_quantizer()
        inventory.append({'omit_unused_exponents':True})
    elif mode != 'baseline':
        raise ValueError(mode)
    return inventory
