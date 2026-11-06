"""Reversible CUDA Graph adapters for the fixed-shape native 3090 policy.

Graphs capture four existing eager call boundaries, not the full Python policy.
Every invocation copies the current input values into static buffers and copies
outputs out of the graph pool. Output copies preserve the original return tree
and prevent later replays/shared-pool graphs from overwriting visual K/V caches.
Those copies are deliberately included in benchmark latency.

Capture has no silent eager fallback. Use a preparation phase before measuring;
the first call in a phase named ``steady`` or ``profile`` may not create a graph.
This adapter is single-threaded and intended for the native eval/no_grad path.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import fields, is_dataclass
import functools
import math
import time
from typing import Any, Iterator


BOUNDARIES = {
    "qwen": "model.text_encoder.model.forward",
    "ae": "model.vae.encode",
    "visual": "model.mot.prefill_flux2_video_cache",
    "action": "model.mot.forward_flux2_action_with_video_cache",
}


def _type_name(value: Any) -> str:
    return f"{type(value).__module__}.{type(value).__qualname__}"


def _rebuild_mapping(original: Mapping, items: list[tuple[Any, Any]]) -> Mapping:
    if type(original) is dict:
        return dict(items)
    if type(original) is OrderedDict:
        return OrderedDict(items)
    if is_dataclass(original):
        # Transformers ModelOutput is both a dataclass and an OrderedDict. Its
        # constructor must receive named fields, not an iterable of pairs.
        return type(original)(**dict(items))
    try:
        return type(original)(items)
    except Exception as error:
        raise TypeError(f"Unsupported mapping return/input type {_type_name(original)}") from error


def _map_tensors(value: Any, tensor_fn, torch, *, path: str = "root", memo=None) -> Any:
    """Map leaves without losing namedtuple, dataclass or ModelOutput types."""
    if memo is None:
        memo = {}
    if torch.is_tensor(value):
        identity = id(value)
        if identity not in memo:
            memo[identity] = tensor_fn(value, path)
        return memo[identity]
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"Non-finite Python float at {path} is unsupported")
        return value
    if isinstance(value, Mapping):
        return _rebuild_mapping(value, [
            (key, _map_tensors(item, tensor_fn, torch, path=f"{path}[{key!r}]", memo=memo))
            for key, item in value.items()
        ])
    if isinstance(value, tuple):
        mapped = [_map_tensors(item, tensor_fn, torch, path=f"{path}[{index}]", memo=memo)
                  for index, item in enumerate(value)]
        return type(value)(*mapped) if hasattr(value, "_fields") else tuple(mapped)
    if type(value) is list:
        return [_map_tensors(item, tensor_fn, torch, path=f"{path}[{index}]", memo=memo)
                for index, item in enumerate(value)]
    if is_dataclass(value) and not isinstance(value, type):
        if any(not field.init for field in fields(value)):
            raise TypeError(f"Dataclass with init=False fields is unsupported at {path}")
        return type(value)(**{
            field.name: _map_tensors(getattr(value, field.name), tensor_fn, torch,
                                    path=f"{path}.{field.name}", memo=memo)
            for field in fields(value)
        })
    raise TypeError(f"Unsupported value at {path}: {_type_name(value)}")


def _signature(value: Any, torch, *, path: str = "root") -> Any:
    if torch.is_tensor(value):
        if value.layout != torch.strided:
            raise TypeError(f"Only strided tensors are supported at {path}, got {value.layout}")
        return {
            "kind": "tensor", "shape": list(value.shape), "stride": list(value.stride()),
            "dtype": str(value.dtype), "device": str(value.device),
            "layout": str(value.layout), "requires_grad": bool(value.requires_grad),
        }
    if value is None or type(value) in (bool, int, str):
        return {"kind": _type_name(value), "value": value}
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"Non-finite Python float at {path} is unsupported")
        return {"kind": "builtins.float", "value": value}
    if isinstance(value, Mapping):
        entries = []
        for key, item in value.items():
            if type(key) not in (str, int, bool):
                raise TypeError(f"Unsupported mapping key at {path}: {key!r}")
            entries.append([_type_name(key), key,
                            _signature(item, torch, path=f"{path}[{key!r}]")])
        return {"kind": "mapping", "type": _type_name(value), "entries": entries}
    if isinstance(value, (tuple, list)):
        return {"kind": "sequence", "type": _type_name(value),
                "items": [_signature(item, torch, path=f"{path}[{index}]")
                          for index, item in enumerate(value)]}
    if is_dataclass(value) and not isinstance(value, type):
        return {"kind": "dataclass", "type": _type_name(value),
                "fields": {field.name: _signature(getattr(value, field.name), torch,
                                                 path=f"{path}.{field.name}")
                           for field in fields(value)}}
    raise TypeError(f"Unsupported value at {path}: {_type_name(value)}")


def _copy_tree(target: Any, source: Any, torch) -> None:
    """The caller has checked the complete structure/shape/non-tensor signature."""
    if torch.is_tensor(target):
        target.copy_(source, non_blocking=True)
    elif isinstance(target, Mapping):
        for key in target:
            _copy_tree(target[key], source[key], torch)
    elif isinstance(target, (tuple, list)):
        for target_item, source_item in zip(target, source):
            _copy_tree(target_item, source_item, torch)
    elif is_dataclass(target) and not isinstance(target, type):
        for field in fields(target):
            _copy_tree(getattr(target, field.name), getattr(source, field.name), torch)


def _tensor_bytes(value: Any, torch) -> int:
    total = 0

    def count(tensor, path):
        nonlocal total
        total += tensor.numel() * tensor.element_size()
        return tensor

    _map_tensors(value, count, torch)
    return total


def _memory(torch, device, pool=None) -> dict:
    result = {
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "global_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "global_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }
    # Public memory_snapshot fields vary across PyTorch versions. A missing pool
    # ID means unavailable, never zero. Do not reset global allocator peak stats.
    if pool is not None:
        result["graph_pool_reserved_bytes"] = None
        result["graph_pool_allocated_bytes"] = None
        try:
            snapshot = torch.cuda.memory_snapshot()
            if any("segment_pool_id" in segment for segment in snapshot):
                segments = [segment for segment in snapshot
                            if tuple(segment.get("segment_pool_id", ())) == tuple(pool)
                            and segment.get("device") == device.index]
                result["graph_pool_reserved_bytes"] = sum(segment["total_size"] for segment in segments)
                result["graph_pool_allocated_bytes"] = sum(segment["allocated_size"] for segment in segments)
        except Exception as error:
            result["graph_pool_snapshot_error"] = f"{type(error).__name__}: {error}"
    return result


class _BoundaryGraph:
    def __init__(self, torch, original, name, metadata, record, device, stream, pool, warmup_steps,
                 input_transform=None):
        self.torch = torch
        self.original = original
        self.name = name
        self.metadata = metadata
        self.record = record
        self.device = device
        self.stream = stream
        self.pool = pool
        self.warmup_steps = warmup_steps
        self.input_transform = input_transform
        self.graph = None
        self.static_inputs = None
        self.static_outputs = None
        self.input_signature = None
        self.graph_input_signature = None
        self._busy = False

    def _clone(self, tensor, path):
        if tensor.device != self.device:
            raise ValueError(f"{self.name}: expected tensor on {self.device} at {path}; got {tensor.device}")
        if tensor.layout != self.torch.strided:
            raise TypeError(f"{self.name}: non-strided tensor at {path}")
        return tensor.detach().clone(memory_format=self.torch.preserve_format)

    def _prepare(self, inputs, phase_record, raw_signature):
        torch = self.torch
        torch.cuda.synchronize(self.device)
        total_start = time.perf_counter()
        self.record["memory_before_capture"] = _memory(torch, self.device, self.pool)
        self.input_signature = raw_signature
        self.graph_input_signature = _signature(inputs, torch)
        started = time.perf_counter()
        self.static_inputs = _map_tensors(inputs, self._clone, torch)
        torch.cuda.synchronize(self.device)
        self.record["static_input_setup_wall_ms"] = (time.perf_counter() - started) * 1000
        self.record["input_signature"] = self.input_signature
        self.record["graph_input_signature"] = self.graph_input_signature
        self.record["static_input_tensor_bytes"] = _tensor_bytes(self.static_inputs, torch)

        # Complete preceding native operations before warming on a side stream.
        self.stream.wait_stream(torch.cuda.current_stream(self.device))
        started = time.perf_counter()
        with torch.cuda.stream(self.stream):
            for _ in range(self.warmup_steps):
                warm_output = self.original(*self.static_inputs[0], **self.static_inputs[1])
                del warm_output
        self.stream.synchronize()
        self.record["warmup_wall_ms"] = (time.perf_counter() - started) * 1000
        self.record["warmup_native_calls"] = self.warmup_steps

        graph = torch.cuda.CUDAGraph()
        self.record["capture_attempts"] += 1
        phase_record["capture_attempts"] += 1
        started = time.perf_counter()
        with torch.cuda.graph(graph, pool=self.pool, stream=self.stream, capture_error_mode="global"):
            captured_outputs = self.original(*self.static_inputs[0], **self.static_inputs[1])
        self.stream.synchronize()
        self.record["capture_wall_ms"] = (time.perf_counter() - started) * 1000
        self.graph = graph
        self.static_outputs = captured_outputs
        self.record["captures"] += 1
        phase_record["captures"] += 1
        self.record["captured_output_tensor_bytes"] = _tensor_bytes(self.static_outputs, torch)
        self.record["memory_after_capture"] = _memory(torch, self.device, self.pool)
        self.record["prepare_total_wall_ms"] = (time.perf_counter() - total_start) * 1000
        self.record["status"] = "captured"
        torch.cuda.current_stream(self.device).wait_stream(self.stream)

    def __call__(self, *args, **kwargs):
        torch = self.torch
        if torch.is_grad_enabled():
            raise RuntimeError(f"{self.name}: CUDA Graph adapter requires the existing no_grad inference path")
        if self._busy:
            raise RuntimeError(f"{self.name}: concurrent/reentrant CUDA Graph use is unsupported")
        if self.record["status"] == "failed":
            raise RuntimeError(f"{self.name}: earlier CUDA Graph preparation/replay failed; no eager fallback")
        phase = str(self.metadata.get("phase", "unspecified"))
        phase_record = self.record["phases"].setdefault(
            phase, {"calls": 0, "capture_attempts": 0, "captures": 0, "replays": 0}
        )
        self.record["calls"] += 1
        phase_record["calls"] += 1
        self._busy = True
        try:
            inputs = (args, kwargs)
            first_call = self.graph is None
            raw_signature = _signature(inputs, torch)
            if not first_call and raw_signature != self.input_signature:
                raise ValueError(
                    f"{self.name}: CUDA Graph input structure/shape/stride/dtype/device or Python value changed; "
                    "this fixed-shape graph is invalid (no fallback or recapture)"
                )
            if self.input_transform is not None:
                transform_started = time.perf_counter()
                inputs = self.input_transform(inputs)
                self.record["input_transform_calls"] += 1
                self.record["input_transform_host_wall_ms"] += (time.perf_counter() - transform_started) * 1000
            if first_call:
                if phase in {"steady", "profile"}:
                    raise RuntimeError(f"{self.name}: missing preparation; refusing to capture during {phase}")
                self._prepare(inputs, phase_record, raw_signature)
            elif _signature(inputs, torch) != self.graph_input_signature:
                raise ValueError(
                    f"{self.name}: prepared CUDA Graph input signature changed; no fallback or recapture"
                )
            # Never reuse old values: this includes the complete fresh visual K/V
            # tree passed to every action iteration, even when its shape matches.
            _copy_tree(self.static_inputs, inputs, torch)
            started = time.perf_counter() if first_call else None
            self.graph.replay()
            self.record["replays"] += 1
            phase_record["replays"] += 1
            # Runs outside graph capture: these tensors belong to the ordinary
            # allocator and retain correct values after any subsequent replay.
            result = _map_tensors(self.static_outputs, self._clone, torch)
            if first_call:
                torch.cuda.synchronize(self.device)
                self.record["first_replay_and_output_clone_wall_ms"] = (time.perf_counter() - started) * 1000
                self.record["memory_after_first_replay_and_output_clone"] = _memory(torch, self.device, self.pool)
            return result
        except BaseException as error:
            self.record["status"] = "failed"
            self.record["error_type"] = type(error).__name__
            self.record["error"] = str(error)
            self.record["failure_phase"] = phase
            raise
        finally:
            self._busy = False


def _replace(stack: ExitStack, owner, attribute, replacement):
    owned = attribute in vars(owner)
    original = vars(owner).get(attribute)
    setattr(owner, attribute, replacement)

    def restore():
        if owned:
            setattr(owner, attribute, original)
        else:
            delattr(owner, attribute)

    stack.callback(restore)


def _qwen_mask_transform(torch, qwen, record):
    """Use Transformers' own helper outside capture for this no-cache call.

    transformers 4.56.1's 2D-mask preparation evaluates padding_mask.all() in a
    Python conditional. It synchronizes the device and cannot execute during
    CUDA Graph capture. Its already-prepared 4D-mask path bypasses that branch.
    The helper documents input_embeds as a shape/dtype carrier only, so an
    expanded view of one embedding-weight row avoids computing embeddings twice.
    """
    from transformers.masking_utils import create_causal_mask

    config = qwen.config
    if getattr(config, "_attn_implementation", None) != "sdpa":
        raise ValueError("Qwen CUDA Graph mask adaptation currently requires its native SDPA implementation")
    layer_types = getattr(config, "layer_types", None)
    if bool(getattr(qwen.model, "has_sliding_layers", False)) or (
        layer_types is not None and any(item != "full_attention" for item in layer_types)
    ):
        raise ValueError("Qwen CUDA Graph mask adaptation supports full causal attention, not sliding/hybrid attention")
    weight = qwen.get_input_embeddings().weight
    record["qwen_mask_adaptation"] = {
        "helper": "transformers.masking_utils.create_causal_mask",
        "location": "outside CUDA capture, recomputed for every invocation",
        "reason": "Native 2D SDPA mask preparation performs a CPU scalar check; a native prepared 4D mask bypasses it.",
        "constraints": "keyword input_ids + 2D attention_mask, use_cache=False, no past/custom position IDs/input embeddings, full causal attention",
        "input_embeds_carrier": "expanded view of one embedding-weight row; helper uses only shape/dtype/device",
        "unmasked_input": "If native helper returns None, call it with an identity AND mask to materialize the same causal pattern.",
        "included_in_end_to_end_timing": True,
    }

    def keep_all_positions(batch_index, head_index, query_index, key_index):
        # cache_position is arange(0, sequence_length), so this is an identity
        # intersection. Supplying it tells the native helper to materialize 4D.
        return query_index >= 0

    def prepare(inputs):
        args, kwargs = inputs
        if args:
            raise ValueError("Qwen CUDA Graph adaptation expects the native keyword-only call")
        if kwargs.get("use_cache") is not False:
            raise ValueError("Qwen CUDA Graph adaptation requires explicit use_cache=False")
        for key in ("past_key_values", "position_ids", "cache_position", "inputs_embeds"):
            if kwargs.get(key) is not None:
                raise ValueError(f"Qwen CUDA Graph adaptation does not support {key}")
        input_ids = kwargs.get("input_ids")
        padding_mask = kwargs.get("attention_mask")
        if not torch.is_tensor(input_ids) or not torch.is_tensor(padding_mask):
            raise TypeError("Qwen CUDA Graph adaptation requires tensor input_ids and attention_mask")
        if input_ids.ndim != 2 or padding_mask.shape != input_ids.shape:
            raise ValueError("Qwen CUDA Graph adaptation requires matching [batch,sequence] input IDs/mask")
        batch, length = input_ids.shape
        embeddings_carrier = weight[:1].unsqueeze(0).expand(batch, length, -1)
        cache_position = torch.arange(length, device=input_ids.device)
        helper_kwargs = dict(
            config=config, input_embeds=embeddings_carrier, attention_mask=padding_mask,
            cache_position=cache_position, past_key_values=None,
            position_ids=cache_position.unsqueeze(0),
        )
        mask = create_causal_mask(**helper_kwargs)
        if mask is None:
            mask = create_causal_mask(**helper_kwargs, and_mask_function=keep_all_positions)
        if not torch.is_tensor(mask) or mask.ndim != 4:
            raise TypeError("Native Qwen causal-mask helper did not return a 4D tensor")
        prepared = dict(kwargs)
        prepared["attention_mask"] = mask
        return args, prepared

    return prepare


@contextmanager
def install(model, metadata: dict, *, boundaries=None, warmup_steps: int = 3) -> Iterator[dict]:
    """Install lazy fixed-shape graphs, restoring all methods on exit.

    ``boundaries`` accepts qwen/ae/visual/action, all four by default. The runner
    updates ``metadata['phase']`` before preparation/warmup/steady/profile calls.
    Measurements and actual graph activation are stored in metadata['cuda_graph'].
    Input tensor values may vary; the complete nested non-tensor contract and all
    tensor shapes/strides/dtypes/devices must stay fixed for this installation.
    """
    import torch

    selected = tuple(BOUNDARIES if boundaries is None else boundaries)
    if not selected or len(set(selected)) != len(selected) or set(selected) - set(BOUNDARIES):
        raise ValueError(f"boundaries must be distinct members of {tuple(BOUNDARIES)}, got {selected}")
    if warmup_steps < 1:
        raise ValueError("warmup_steps must be positive")
    if getattr(model, "_profile_3090_cuda_graph", False):
        raise RuntimeError("CUDA Graph adapters are already installed on this model")
    if str(getattr(model, "stack", "")) != "flux2":
        raise ValueError("This CUDA Graph adapter supports only the native FLUX.2 policy")
    if model.training:
        raise ValueError("Call model.eval() before installing inference CUDA Graph adapters")
    device = torch.device(model.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA Graph inference requires the existing CUDA PyTorch environment")
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    targets = {
        "qwen": (model.text_encoder.model, "forward"),
        "ae": (model.vae, "encode"),
        "visual": (model.mot, "prefill_flux2_video_cache"),
        "action": (model.mot, "forward_flux2_action_with_video_cache"),
    }
    for name in selected:
        owner, _ = targets[name]
        if bool(getattr(owner, "training", False)):
            raise ValueError(f"{name}: target module must be in eval mode before graph capture")
    torch.cuda.synchronize(device)
    pool = torch.cuda.graph_pool_handle()
    stream = torch.cuda.Stream(device=device)
    report = {
        "implementation": "native_module_cuda_graphs",
        "requested_boundaries": list(selected), "effective_boundaries": [],
        "shared_pool": True, "pool_handle": list(pool),
        "memory_note": "Pool measurements are shared across boundaries, not additive per-graph sizes; allocator peaks are never reset by this adapter.",
        "warmup_calls_per_boundary": warmup_steps,
        "copies_dynamic_input_values_every_call": True,
        "output_trees_cloned_outside_capture": True,
        "no_eager_fallback": True, "no_steady_recapture": True,
        "signature_guard": "tensor shape/stride/dtype/device/requires_grad and complete Python structure/value",
        "memory_before_install": _memory(torch, device, pool),
        "boundaries": {},
    }
    metadata["cuda_graph"] = report
    adapters = []
    with ExitStack() as patches:
        _replace(patches, model, "_profile_3090_cuda_graph", True)
        for name in selected:
            owner, attribute = targets[name]
            original = getattr(owner, attribute)
            record = {
                "boundary": BOUNDARIES[name], "status": "not_executed",
                "calls": 0, "capture_attempts": 0, "captures": 0,
                "replays": 0, "phases": {},
                "input_transform_calls": 0, "input_transform_host_wall_ms": 0.0,
            }
            report["boundaries"][name] = record
            input_transform = _qwen_mask_transform(torch, owner, record) if name == "qwen" else None
            adapter = _BoundaryGraph(torch, original, name, metadata, record,
                                     device, stream, pool, warmup_steps, input_transform=input_transform)
            adapters.append(adapter)

            @functools.wraps(original)
            def wrapped(*args, _adapter=adapter, **kwargs):
                return _adapter(*args, **kwargs)

            _replace(patches, owner, attribute, wrapped)
        try:
            yield report
        finally:
            report["effective_boundaries"] = [
                name for name, record in report["boundaries"].items() if record["captures"] > 0
            ]
            report["all_requested_captured"] = set(report["effective_boundaries"]) == set(selected)
            report["steady_capture_count"] = sum(
                record["phases"].get("steady", {}).get("captures", 0)
                for record in report["boundaries"].values()
            )
            try:
                torch.cuda.synchronize(device)
                report["memory_before_uninstall"] = _memory(torch, device, pool)
            except Exception as error:
                report["uninstall_synchronize_error"] = f"{type(error).__name__}: {error}"
            # ExitStack restores original instance/class attribute lookup. Graphs
            # and static tensors are then released with this context; no allocator
            # cache is globally emptied and no model parameter is changed.
