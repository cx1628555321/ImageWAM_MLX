"""Optional timings for the 3090 benchmark; never changes model computation.

Load spans synchronize CUDA at their boundaries and report completed wall time.
Runtime spans enqueue CUDA events without synchronizing individual modules. Their
``wall_ms`` is host dispatch time, not completed GPU latency. Resolve events once
at the end of an iteration. All spans are inclusive: parents and children overlap
and must not be added together. CUDA event intervals are elapsed stream time,
which may include host launch gaps; they are not the sum of kernel durations.
"""

from __future__ import annotations

from contextlib import contextmanager
from functools import wraps
import inspect
import math
from pathlib import Path
import statistics
import time
from typing import Any, Iterable


def summarize(values: Iterable[float]) -> dict[str, float | int | None]:
    """Population stddev and linearly interpolated percentile, in input units."""
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return dict(count=0, mean=None, median=None, p95=None, min=None, max=None, std=None)
    if not all(math.isfinite(value) for value in ordered):
        raise ValueError("Timing samples must be finite")
    position = 0.95 * (len(ordered) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    p95 = ordered[low] + (ordered[high] - ordered[low]) * (position - low)
    return dict(
        count=len(ordered), mean=statistics.mean(ordered),
        median=statistics.median(ordered), p95=p95,
        min=ordered[0], max=ordered[-1], std=statistics.pstdev(ordered),
    )


class Recorder:
    def __init__(self, device: str = "cuda", enabled: bool = True):
        import torch

        self.torch = torch
        self.device = torch.device(device)
        self.enabled = bool(enabled)
        self.phase = "load"
        self.iteration: int | None = None
        self.records: list[dict[str, Any]] = []
        self._stack: list[dict[str, Any]] = []
        self._pending: list[tuple[dict[str, Any], Any, Any]] = []
        self.cuda_available = self.device.type == "cuda" and torch.cuda.is_available()

    def synchronize(self) -> None:
        if self.cuda_available:
            self.torch.cuda.synchronize(self.device)

    @contextmanager
    def span(
        self, name: str, category: str = "runtime", parent: str | None = None,
        synchronize: bool = False, cuda: bool = True, **metadata: Any,
    ):
        """Record one inclusive scope; load scopes always synchronize CUDA."""
        if not self.enabled:
            yield
            return
        synchronize = bool(synchronize or category == "load")
        if synchronize:
            self.synchronize()
        enclosing = self._stack[-1] if self._stack else None
        record = dict(
            id=len(self.records), phase=self.phase, iteration=self.iteration,
            name=str(name), category=str(category),
            parent=parent if parent is not None else (enclosing["name"] if enclosing else None),
            parent_id=enclosing["id"] if enclosing else None,
            inclusive=True, wall_ms=None, cuda_ms=None,
            wall_semantics="synchronized_elapsed" if synchronize else "host_dispatch_elapsed",
            overlap_note="Inclusive scope; do not sum with its descendants.",
            status="ok", metadata=metadata,
        )
        self.records.append(record)
        self._stack.append(record)
        start_event = end_event = None
        if self.cuda_available and cuda:
            start_event = self.torch.cuda.Event(enable_timing=True)
            end_event = self.torch.cuda.Event(enable_timing=True)
            start_event.record(self.torch.cuda.current_stream(self.device))
        started = time.perf_counter()
        try:
            yield record
        except BaseException as error:
            record["status"] = "error"
            record["error_type"] = type(error).__name__
            raise
        finally:
            if end_event is not None:
                end_event.record(self.torch.cuda.current_stream(self.device))
            if synchronize:
                self.synchronize()
            record["wall_ms"] = (time.perf_counter() - started) * 1000.0
            self._stack.pop()
            if start_event is not None:
                self._pending.append((record, start_event, end_event))
            if synchronize:
                self.resolve_pending(synchronize=False)

    def resolve_pending(self, synchronize: bool = True) -> None:
        """Call after each profiled iteration, before exporting JSON records."""
        if synchronize:
            self.synchronize()
        remaining = []
        for record, start, end in self._pending:
            if synchronize or end.query():
                record["cuda_ms"] = float(start.elapsed_time(end))
            else:
                remaining.append((record, start, end))
        self._pending = remaining

    def aggregate(
        self, records: Iterable[dict[str, Any]] | None = None, *,
        field: str | None = None,
        group_by: tuple[str, ...] = ("phase", "category", "name"),
    ) -> list[dict[str, Any]]:
        """Summarize individual calls; repeated denoise steps remain separate calls.

        With field=None (default), output nested wall_ms and cuda_ms summaries.
        With a selected field, output flattened summary statistics for that field.
        Only completed successful records are included. Resolve CUDA events first.
        """
        grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
        for record in self.records if records is None else records:
            if record.get("status") != "ok" or record.get("wall_ms") is None:
                continue
            key = tuple(record.get(column) for column in group_by)
            grouped.setdefault(key, []).append(record)
        output = []
        for key, samples in grouped.items():
            row = dict(zip(group_by, key))
            if field is not None:
                row.update(metric=field, **summarize(
                    item[field] for item in samples if item.get(field) is not None
                ))
            else:
                for metric in ("wall_ms", "cuda_ms"):
                    row[metric] = summarize(
                        item[metric] for item in samples if item.get(metric) is not None
                    )
                row["inclusive"] = True
                row["wall_semantics"] = samples[0]["wall_semantics"]
            output.append(row)
        return output


class _Patches:
    """Restore both owned and inherited attributes exactly, including on errors."""

    def __init__(self):
        self.undo: list[tuple[Any, str, bool, Any]] = []

    def set(self, owner: Any, name: str, replacement: Any) -> None:
        owned = name in vars(owner)
        original = vars(owner).get(name)
        self.undo.append((owner, name, owned, original))
        setattr(owner, name, replacement)

    def wrap(
        self, owner: Any, name: str, recorder: Recorder, label: str, *,
        selector=None, after=None, **metadata: Any,
    ) -> None:
        descriptor = inspect.getattr_static(owner, name)
        is_classmethod = isinstance(descriptor, classmethod)
        is_staticmethod = isinstance(descriptor, staticmethod)
        original = descriptor.__func__ if is_classmethod or is_staticmethod else getattr(owner, name)

        @wraps(original)
        def timed(*args, **kwargs):
            if selector is not None and not selector(*args, **kwargs):
                return original(*args, **kwargs)
            with recorder.span(label, category="load", **metadata):
                result = original(*args, **kwargs)
            if after is not None:
                after(args, kwargs, result)
            return result

        replacement = classmethod(timed) if is_classmethod else staticmethod(timed) if is_staticmethod else timed
        self.set(owner, name, replacement)

    def restore(self) -> None:
        for owner, name, owned, original in reversed(self.undo):
            if owned:
                setattr(owner, name, original)
            else:
                delattr(owner, name)
        self.undo.clear()


@contextmanager
def load_instrumentation(recorder: Recorder, modelpaths: dict[str, Any]):
    """Temporarily instrument only the native FLUX.2 ImageWAM loading route.

    modelpaths uses native model.model_paths keys: flux2_src, flux2, ae, qwen3;
    checkpoint identifies the policy .pt file. Alternate *_path keys are accepted.
    Constructors and aggregate loading scopes include their child operations.
    Disk reads include page-cache effects; this does not force a cold disk cache.
    No weight, device, dtype, loader argument, or return value is modified.
    """
    if not recorder.enabled:
        yield
        return
    import torch
    import safetensors.torch as safetensors_torch
    from imagewam.models.backbones.flux2_imports import ensure_flux2_importable

    source = modelpaths.get("flux2_src", modelpaths.get("flux2_src_path"))
    ensure_flux2_importable(source)
    from flux2.autoencoder import AutoEncoder
    from flux2.model import Flux2
    from imagewam.models.backbones.action_dit_flux2 import ActionDiTFlux2
    from imagewam.models.backbones import flux2_video_expert
    from imagewam.models.backbones.imagewam import ImageWAM
    from imagewam.models.backbones.mot import MoT
    from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen3ForCausalLM

    patches = _Patches()

    def canonical(value):
        if isinstance(value, (str, Path)):
            return str(Path(value).expanduser().resolve())
        return None

    visual_path = modelpaths.get("flux2", modelpaths.get("flux2_model_path"))
    ae_path = modelpaths.get("ae", modelpaths.get("ae_model_path"))
    checkpoint_path = modelpaths.get("checkpoint", modelpaths.get("checkpoint_path"))
    qwen_path = modelpaths.get("qwen3", modelpaths.get("qwen3_model_spec"))

    def match_file(expected):
        expected = canonical(expected)

        def matches(*args, **kwargs):
            value = args[0] if args else kwargs.get("filename", kwargs.get("f"))
            return expected is not None and canonical(value) == expected

        return matches

    # Patch the already-bound module alias separately: rebinding safetensors.torch
    # alone cannot intercept Flux2VideoExpert's module-level `load_sft` import.
    try:
        patches.wrap(ImageWAM, "from_flux2_klein_pretrained", recorder, "factory.total")
        patches.wrap(flux2_video_expert.Flux2VideoExpert, "from_pretrained", recorder,
                     "visual.total", path=str(visual_path))
        patches.wrap(Flux2, "__init__", recorder, "visual.initialize_meta")
        patches.wrap(flux2_video_expert, "load_sft", recorder, "visual.read_weights",
                     selector=match_file(visual_path), path=str(visual_path))
        patches.wrap(Flux2, "load_state_dict", recorder, "visual.apply_weights")
        patches.wrap(Flux2, "to", recorder, "visual.to_device_dtype")

        patches.wrap(ActionDiTFlux2, "from_pretrained", recorder, "action.total")
        patches.wrap(ActionDiTFlux2, "__init__", recorder, "action.initialize")
        patches.wrap(ActionDiTFlux2, "to", recorder, "action.to_device_dtype")
        patches.wrap(MoT, "__init__", recorder, "mot.assemble")

        patches.wrap(AutoEncoder, "__init__", recorder, "ae.initialize_meta")
        patches.wrap(safetensors_torch, "load_file", recorder, "ae.read_weights",
                     selector=match_file(ae_path), path=str(ae_path))
        patches.wrap(AutoEncoder, "load_state_dict", recorder, "ae.apply_weights")
        patches.wrap(AutoEncoder, "to", recorder, "ae.to_device_dtype")

        patches.wrap(AutoModelForCausalLM, "from_pretrained", recorder, "qwen.load_pretrained",
                     path=str(qwen_path))
        patches.wrap(Qwen3ForCausalLM, "to", recorder, "qwen.to_device_dtype")
        patches.wrap(AutoTokenizer, "from_pretrained", recorder, "qwen.load_tokenizer",
                     path=str(qwen_path))

        def capture_proprio(args, kwargs, result):
            proprio = getattr(args[0], "proprio_encoder", None)
            if proprio is not None:
                patches.wrap(proprio, "load_state_dict", recorder, "policy.apply_proprio_weights")

        patches.wrap(ImageWAM, "__init__", recorder, "imagewam.assemble", after=capture_proprio)
        patches.wrap(ImageWAM, "load_checkpoint", recorder, "policy.load_checkpoint_total",
                     path=str(checkpoint_path))
        patches.wrap(torch, "load", recorder, "policy.read_checkpoint",
                     selector=match_file(checkpoint_path), path=str(checkpoint_path))
        patches.wrap(MoT, "load_state_dict", recorder, "policy.apply_mot_weights")
        yield
    finally:
        patches.restore()
