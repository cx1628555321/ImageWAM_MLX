"""Reversible torch.compile / TorchScript adapters for native GPU boundaries.

No eager fallback is permitted. TorchScript uses real registered model parameters
and a tensor-only trace interface, specialized to the example's shapes and static
arguments. A trace is not a proof about data-dependent branches: callers must
also compare independent real observations against the native eager model.
"""
from __future__ import annotations

from collections import Counter
from contextlib import ExitStack, contextmanager
from dataclasses import fields, is_dataclass
from functools import wraps
import time
import traceback
import warnings


BOUNDARIES = ("qwen", "ae", "visual", "action")


def dynamo_snapshot():
    from torch._dynamo.utils import counters
    return {str(group): {str(key): int(value) for key, value in values.items()}
            for group, values in counters.items()}


def dynamo_delta(before, after):
    output = {}
    for group in before.keys() | after.keys():
        a, b = before.get(group, {}), after.get(group, {})
        delta = {key: b.get(key, 0) - a.get(key, 0) for key in a.keys() | b.keys()}
        delta = {key: value for key, value in delta.items() if value}
        if delta:
            output[group] = delta
    return output


def _flatten_spec(item, tensors, torch):
    # Module-level recursion is deliberate. A local recursive function closes
    # over itself AND `tensors`, retaining every GPU leaf until cyclic GC runs.
    if torch.is_tensor(item):
        tensors.append(item)
        return ("tensor",)
    if is_dataclass(item) and not isinstance(item, type):
        names = tuple(field.name for field in fields(item))
        return ("dataclass", type(item), names, tuple(
            _flatten_spec(getattr(item, key), tensors, torch) for key in names))
    if isinstance(item, dict):
        keys = tuple(item)
        return ("dict", type(item), keys, tuple(
            _flatten_spec(item[key], tensors, torch) for key in keys))
    if isinstance(item, tuple):
        return ("tuple", type(item), tuple(_flatten_spec(part, tensors, torch) for part in item))
    if isinstance(item, list):
        return ("list", tuple(_flatten_spec(part, tensors, torch) for part in item))
    if item is None or isinstance(item, (str, bool, int, float, torch.dtype, torch.device)):
        return ("static", type(item), item)
    raise TypeError(f"Unsupported trace tree leaf: {type(item).__module__}.{type(item).__name__}")


def _flatten(value):
    """Separate tensors from a tensor-free spec, without per-call closure cycles."""
    import torch
    tensors = []
    return tensors, _flatten_spec(value, tensors, torch)


def _build_tree(iterator, spec):
    kind = spec[0]
    if kind == "tensor":
        return next(iterator)
    if kind == "static":
        return spec[2]
    if kind == "list":
        return [_build_tree(iterator, child) for child in spec[1]]
    if kind == "tuple":
        values = [_build_tree(iterator, child) for child in spec[2]]
        return spec[1](*values) if hasattr(spec[1], "_fields") else tuple(values)
    values = {key: _build_tree(iterator, child) for key, child in zip(spec[2], spec[3])}
    return spec[1](**values) if kind == "dataclass" else spec[1](values)


def _unflatten(tensors, specification):
    iterator = iter(tensors)
    result = _build_tree(iterator, specification)
    try:
        next(iterator)
    except StopIteration:
        return result
    raise ValueError("Extra tensors while reconstructing trace output")


def _signature(tensor):
    return (tuple(tensor.shape), str(tensor.dtype), str(tensor.device),
            tuple(tensor.stride()), bool(tensor.requires_grad))


def _synchronize(owner):
    import torch
    parameter = next(owner.parameters(), None)
    if parameter is not None and parameter.device.type == "cuda":
        torch.cuda.synchronize(parameter.device)


def _make_trace(owner, original, args, kwargs, information):
    import torch
    tensors, input_spec = _flatten((args, kwargs))
    if not tensors:
        raise ValueError("TorchScript boundary has no tensor inputs")
    input_signatures = tuple(_signature(tensor) for tensor in tensors)
    with torch.no_grad():
        reference = original(*args, **kwargs)
    reference_tensors, output_spec = _flatten(reference)
    if not reference_tensors:
        raise ValueError("TorchScript boundary has no tensor outputs")

    class TensorAdapter(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # Register the real owner: tracing a closure alone treats weights as
            # tensor constants and is both incorrect and excessively expensive.
            self.target = owner

        def forward(self, *flat_tensors):
            call_args, call_kwargs = _unflatten(flat_tensors, input_spec)
            result = original(*call_args, **call_kwargs)
            result_tensors, _ = _flatten(result)
            return tuple(result_tensors)

    adapter = TensorAdapter().eval()
    with torch.no_grad(), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", torch.jit.TracerWarning)
        traced = torch.jit.trace(adapter, tuple(tensors), strict=False, check_trace=False)
        # Keep parameters as parameters. Do not freeze/optimize_for_inference:
        # those transformations may inline or duplicate the 4B weights.
        traced_output = traced(*tensors)
    information["trace_warnings"] = list(dict.fromkeys(str(item.message) for item in caught))
    information["execution_engine"] = "torch.jit.ScriptModule"
    information["scriptmodule_type"] = type(traced).__name__
    parameter_pointers = {parameter.data_ptr() for parameter in owner.parameters()}
    traced_parameters = list(traced.parameters())
    if not traced_parameters or any(parameter.data_ptr() not in parameter_pointers for parameter in traced_parameters):
        raise RuntimeError("Trace did not retain the owner's existing registered parameters")
    information["registered_parameter_tensors"] = len(traced_parameters)
    information["parameters_share_original_storage"] = True
    information["trace_input_signatures"] = [list(signature) for signature in input_signatures]
    information["trace_output_tensor_count"] = len(reference_tensors)
    information["trace_scope"] = "Fixed input shapes, strides, dtype/device and static arguments; tensor values remain live inputs."
    if len(traced_output) != len(reference_tensors):
        raise RuntimeError("TorchScript output tensor count changed")
    with torch.no_grad():
        for index, (expected, actual) in enumerate(zip(reference_tensors, traced_output)):
            if not torch.allclose(expected, actual, atol=0.005, rtol=0.02, equal_nan=False):
                raise RuntimeError(f"TorchScript example output tensor {index} differs from native eager")
    information["example_output_matches_eager"] = True

    def call(*call_args, **call_kwargs):
        live_tensors, live_spec = _flatten((call_args, call_kwargs))
        if live_spec != input_spec or tuple(_signature(tensor) for tensor in live_tensors) != input_signatures:
            raise ValueError("TorchScript received a different input shape/stride/dtype/device/static argument; retrace explicitly")
        return _unflatten(traced(*live_tensors), output_spec)

    # Returning the traced first result avoids another unreported preparation
    # invocation. Subsequent calls always execute with their own tensor inputs.
    return call, _unflatten(traced_output, output_spec)


def _select(model):
    return {
        "qwen": (model.text_encoder.model, "forward"),
        "ae": (model.vae, "encode"),
        "visual": (model.mot, "prefill_flux2_video_cache"),
        "action": (model.mot, "forward_flux2_action_with_video_cache"),
    }


@contextmanager
def install(model, mode, metadata, *, boundaries=None):
    """Install lazy adapters; first invocations prepare outside timed samples.

    The runner sets metadata['phase'] before prepare/warmup/steady/profile. Calls
    per phase and preparation seconds are kept under metadata['boundaries'].
    Runner-side dynamo_snapshot() calls at phase boundaries detect graph changes
    without adding counter scans to every timed inference. Exceptions propagate;
    status 'unsupported'/'failed' never means an eager fallback was benchmarked.
    """
    import torch
    mode = str(mode)
    if mode not in ("compile", "torchscript"):
        raise ValueError(f"Unsupported execution mode: {mode}")
    selected = tuple(BOUNDARIES if boundaries is None else boundaries)
    if not selected or len(set(selected)) != len(selected) or any(name not in BOUNDARIES for name in selected):
        raise ValueError(f"Boundaries must be a nonempty unique subset of {BOUNDARIES}")
    if getattr(model, "training", True):
        raise ValueError("Call model.eval() before installing inference adapters")
    targets = _select(model)
    metadata.update(mode=mode, selected_boundaries=list(selected), boundaries={},
                    status="installed", fallback="none", torch_version=torch.__version__)
    # Preserve eager BF16/FP16 operator-boundary rounding when Inductor fuses
    # pointwise work. Verified present in the server's PyTorch 2.8.0+cu128;
    # this changes the arithmetic policy, not the comparison tolerances.
    options = {"triton.cudagraphs": False, "compile_threads": 1,
               "emulate_precision_casts": True}
    if mode == "compile":
        metadata["compile"] = dict(backend="inductor", fullgraph=True, dynamic=False, options=options,
                                   suppress_errors=False, error_on_recompile=True)
    metadata["dynamo_before_install"] = dynamo_snapshot()

    with ExitStack() as cleanup:
        if mode == "compile":
            import torch._dynamo.config as dynamo_config
            cleanup.enter_context(dynamo_config.patch(suppress_errors=False, error_on_recompile=True))
        else:
            metadata["torchscript"] = {
                "execution_engine": "torch.jit.ScriptModule",
                "optimized_execution": bool(torch._C._get_graph_executor_optimize()),
                "scope": "Keep the existing JIT executor optimization policy unchanged.",
                "tensor_lifetime_fix": "Module-level recursive tensor-tree helpers avoid per-call self-referential closures retaining GPU tensors until cyclic GC; no forced garbage collection.",
                "eager_fallback": False,
            }

        def add(name, owner, method):
            original = getattr(owner, method)
            had_own = method in vars(owner)
            old_own = vars(owner).get(method)
            information = {"status": "pending", "method": method, "calls_by_phase": {}, "prepare_seconds": None}
            metadata["boundaries"][name] = information
            prepared = None
            input_transform = None
            if mode == "torchscript" and name == "qwen":
                # Transformers 4.56.1's 2D -> causal-mask construction goes
                # through functorch/vmap, which this TorchScript tracer cannot
                # record. Reuse the native helper outside trace on EVERY call;
                # the resulting 4D mask remains a live tensor input to the JIT.
                from profile_3090_cudagraph import _qwen_mask_transform
                input_transform = _qwen_mask_transform(torch, owner, information)
                information["qwen_mask_adaptation"].update(
                    location="outside TorchScript trace, recomputed for every invocation",
                    reason="Native 2D causal-mask preparation uses functorch/vmap unsupported by this tracer; a native prepared 4D mask bypasses that path.",
                )
                information["input_transform_calls"] = 0
                information["input_transform_host_wall_ms"] = 0.0

            def transform(args, kwargs):
                if input_transform is None:
                    return args, kwargs
                transform_started = time.perf_counter()
                try:
                    return input_transform((args, kwargs))
                finally:
                    information["input_transform_calls"] += 1
                    information["input_transform_host_wall_ms"] += (time.perf_counter() - transform_started) * 1000

            @wraps(original)
            def execute(*args, **kwargs):
                nonlocal prepared
                phase = str(metadata.get("phase", "unspecified"))
                counts = information["calls_by_phase"]
                counts[phase] = counts.get(phase, 0) + 1
                if prepared is not None:
                    try:
                        args, kwargs = transform(args, kwargs)
                        return prepared(*args, **kwargs)
                    except BaseException as error:
                        information.update(status="failed", error_type=type(error).__name__, error=str(error), failed_phase=phase)
                        metadata["status"] = "failed"
                        raise
                _synchronize(owner)
                started = time.perf_counter()
                before = dynamo_snapshot()
                information.update(status="preparing", prepare_phase=phase)
                print(f"PREPARE_BEGIN {mode} {name}", flush=True)
                try:
                    if mode == "compile":
                        # Save/call the original bound method, never look it up
                        # through an attribute later replaced by timing hooks.
                        prepared = torch.compile(original, backend="inductor", fullgraph=True,
                                                 dynamic=False, options=options)
                        result = prepared(*args, **kwargs)
                    else:
                        args, kwargs = transform(args, kwargs)
                        prepared, result = _make_trace(owner, original, args, kwargs, information)
                    _synchronize(owner)
                    information["status"] = "ready"
                    print(f"PREPARE_READY {mode} {name} {time.perf_counter() - started:.3f}s", flush=True)
                    return result
                except BaseException as error:
                    prepared = None
                    information.update(status="unsupported" if mode == "torchscript" else "failed",
                                       error_type=type(error).__name__, error=str(error),
                                       traceback=traceback.format_exc(), failed_phase=phase)
                    metadata["status"] = information["status"]
                    raise
                finally:
                    information["prepare_seconds"] = time.perf_counter() - started
                    information["dynamo_prepare_delta"] = dynamo_delta(before, dynamo_snapshot())

            setattr(owner, method, execute)

            def restore():
                if had_own:
                    setattr(owner, method, old_own)
                else:
                    delattr(owner, method)
            cleanup.callback(restore)

        try:
            for name in selected:
                add(name, *targets[name])
            yield metadata
        finally:
            metadata["dynamo_after_run"] = dynamo_snapshot()
            metadata["dynamo_total_delta"] = dynamo_delta(metadata["dynamo_before_install"], metadata["dynamo_after_run"])
            metadata["prepare_seconds"] = sum(item["prepare_seconds"] or 0 for item in metadata["boundaries"].values())
            if metadata["status"] == "installed":
                metadata["status"] = "ready" if all(item["status"] == "ready" for item in metadata["boundaries"].values()) else "not_fully_prepared"
