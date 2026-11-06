"""Temporary instrumentation of the native FLUX.2 / LIBERO inference path.

No model implementation is copied or changed. All instance attributes and eval
functions are restored on exit, including when inference raises. The steady
latency benchmark must run *outside* this context: CUDA events and Python
wrappers themselves have a cost.

The recorder implements ``span(name, category, synchronize, cuda, **metadata)``.
Spans below never synchronize CUDA. Resolve their events once the full action
chunk has completed. Times are inclusive; do not add parent and child rows.
"""

from __future__ import annotations

import functools
import inspect
import sys
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from typing import Any, Iterator


MODULE_BOUNDARIES = {
    "policy.action_chunk": {
        "boundary": "eval_libero_single._predict_action_chunk",
        "includes": "Observation preparation, native model inference, CPU action conversion, action denormalization and gripper conversion.",
        "excludes": "Simulator step/render, observation acquisition, model loading, video writing.",
    },
    "observation.preprocess": {
        "boundary": "eval_libero_single._obs_to_model_input",
        "includes": "Two camera crop/resize/concatenation, tensor conversion, host-to-device image transfer, image and state normalization.",
    },
    "policy.model_forward": {
        "boundary": "ImageWAM.infer_action_flux2",
        "includes": "Text, proprio, AE, visual prefill, all action denoising steps, and final action device-to-host transfer.",
    },
    "text.encode": {
        "boundary": "ImageWAM._encode_flux2_prompts",
        "includes": "Chat template, tokenization, token host-to-device transfer, Qwen forward, selected hidden-state stack/reshape.",
    },
    "text.chat_template": {
        "boundary": "text_encoder.tokenizer.apply_chat_template",
        "includes": "CPU chat-template formatting; CUDA timing disabled for this row.",
    },
    "text.tokenize": {
        "boundary": "text_encoder.tokenizer.__call__",
        "includes": "CPU tokenization, truncation and padding; CUDA timing disabled for this row.",
    },
    "text.qwen_forward": {
        "boundary": "text_encoder.model.forward",
        "includes": "Native AutoModelForCausalLM forward with output_hidden_states=True and use_cache=False, including its actual LM-head work.",
    },
    "proprio.encode_and_pack": {
        "boundary": "ImageWAM._append_proprio_to_context_if_enabled",
        "includes": "State device transfer/projection and packing its token with the text context.",
    },
    "proprio.linear": {
        "boundary": "proprio_encoder.forward",
        "includes": "The state projection linear layer only.",
    },
    "image.encode_tokens": {
        "boundary": "ImageWAM._encode_flux2_image_tokens",
        "includes": "Image validation/device conversion, AE encode, latent packing and image position IDs.",
    },
    "image.ae_encode": {
        "boundary": "vae.encode",
        "includes": "Native FLUX.2 AutoEncoder encode, including its latent normalization/packing operations.",
    },
    "visual.prepare": {
        "boundary": "video_expert.pre_dit",
        "includes": "Image/text input projections, timestep/modulation preparation and positional embedding preparation.",
    },
    "visual.prefill": {
        "boundary": "MoT.prefill_flux2_video_cache",
        "includes": "All visual double/single blocks and construction of per-layer visual/text K/V caches, once per action chunk.",
    },
    "action.denoise_loop": {
        "boundary": "First _start_action_attention_capture_step entry through final infer_action_scheduler.step return",
        "includes": "All denoising iterations, timestep preparation, action input/output projections, cached-context transformer calls and scheduler updates.",
        "excludes": "Initial random action generation, schedule creation, visual prefill, attention-mask construction and final device-to-host action transfer.",
    },
    "action.denoise_step": {
        "boundary": "_start_action_attention_capture_step entry through the corresponding scheduler.step return",
        "includes": "One complete action iteration; step_index is zero-based in raw records.",
    },
    "action.prepare": {
        "boundary": "action_expert.pre_dit",
        "includes": "Action input projection, timestep/modulation embeddings and action position IDs.",
    },
    "action.transformer": {
        "boundary": "MoT.forward_flux2_action_with_video_cache",
        "includes": "All action double/single transformer blocks attending to cached visual/text K/V; called once per denoising step.",
    },
    "action.project": {
        "boundary": "action_expert.post_dit",
        "includes": "Action output projection.",
    },
    "action.scheduler": {
        "boundary": "infer_action_scheduler.step",
        "includes": "Flow-matching scheduler update of the action sample.",
    },
    "action.denormalize": {
        "boundary": "eval_libero_single._denormalize_action",
        "includes": "CPU float conversion, dataset-statistics inverse normalization and NumPy conversion; CUDA timing disabled for this row.",
        "excludes": "Subsequent gripper sign/range conversion, which remains in policy.action_chunk.",
    },
}


def _span(recorder: Any, name: str, *, cuda: bool = True, **metadata: Any):
    return recorder.span(
        name, category="module", synchronize=False, cuda=cuda, **metadata
    )


def _replace(stack: ExitStack, obj: Any, attribute: str, replacement: Any) -> None:
    """Restore the original descriptor lookup, not a saved bound-method shadow."""
    own_attributes = vars(obj)
    had_own_attribute = attribute in own_attributes
    old_value = own_attributes.get(attribute)
    setattr(obj, attribute, replacement)

    def restore() -> None:
        if had_own_attribute:
            setattr(obj, attribute, old_value)
        else:
            delattr(obj, attribute)

    stack.callback(restore)


def _wrap_method(
    stack: ExitStack,
    recorder: Any,
    obj: Any,
    attribute: str,
    span_name: str,
    *,
    cuda: bool = True,
) -> None:
    original = getattr(obj, attribute)

    @functools.wraps(original)
    def timed(*args: Any, **kwargs: Any):
        with _span(recorder, span_name, cuda=cuda):
            return original(*args, **kwargs)

    _replace(stack, obj, attribute, timed)


class _TokenizerProxy:
    """Instance-level tokenizer proxy: special __call__ lookup skips attributes."""

    def __init__(self, tokenizer: Any, recorder: Any):
        self._tokenizer = tokenizer
        self._recorder = recorder

    def __getattr__(self, name: str):
        return getattr(self._tokenizer, name)

    def __call__(self, *args: Any, **kwargs: Any):
        with _span(self._recorder, "text.tokenize", cuda=False):
            return self._tokenizer(*args, **kwargs)

    def apply_chat_template(self, *args: Any, **kwargs: Any):
        with _span(self._recorder, "text.chat_template", cuda=False):
            return self._tokenizer.apply_chat_template(*args, **kwargs)


@dataclass
class _LoopSpans:
    recorder: Any
    expected_steps: int
    loop_context: Any = None
    step_context: Any = None
    step_index: int | None = None
    completed_steps: int = 0

    def begin_step(self, step_index: int) -> None:
        if self.step_context is not None:
            raise RuntimeError("Previous action step did not finish at scheduler.step.")
        if step_index != self.completed_steps:
            raise RuntimeError(
                f"Unexpected action step index {step_index}; expected {self.completed_steps}."
            )
        if self.loop_context is None:
            self.loop_context = _span(
                self.recorder, "action.denoise_loop", num_steps=self.expected_steps
            )
            self.loop_context.__enter__()
        self.step_index = int(step_index)
        self.step_context = _span(
            self.recorder, "action.denoise_step", step_index=self.step_index
        )
        self.step_context.__enter__()

    def finish_step(self, exc_info=(None, None, None)) -> None:
        if self.step_context is None:
            return
        context, self.step_context = self.step_context, None
        context.__exit__(*exc_info)
        self.completed_steps += 1
        if exc_info[0] is not None or self.completed_steps == self.expected_steps:
            self.close(exc_info)

    def close(self, exc_info=(None, None, None)) -> None:
        if self.step_context is not None:
            context, self.step_context = self.step_context, None
            context.__exit__(*exc_info)
        if self.loop_context is not None:
            context, self.loop_context = self.loop_context, None
            context.__exit__(*exc_info)


@contextmanager
def module_instrumentation(recorder: Any, model: Any, eval_module: Any) -> Iterator[dict]:
    """Profile an already-loaded native model; yield machine-readable boundaries.

    This context is intentionally single-threaded and non-reentrant for a given
    model instance. Qwen must already be loaded; lazy loading belongs to the
    separately measured model-loading phase, not the runtime profile.
    """
    if str(getattr(model, "stack", "")) != "flux2":
        raise ValueError("3090 module instrumentation supports the native FLUX.2 path only.")
    text_encoder = getattr(model, "text_encoder", None)
    if text_encoder is None or not all(
        hasattr(text_encoder, name) for name in ("model", "tokenizer")
    ):
        raise ValueError("Load the Qwen text encoder/tokenizer before module instrumentation.")
    if getattr(model, "_profile_3090_instrumented", False):
        raise RuntimeError("This model already has 3090 module instrumentation installed.")

    with ExitStack() as patches:
        _replace(patches, model, "_profile_3090_instrumented", True)
        for obj, method, name, cuda in (
            (eval_module, "_predict_action_chunk", "policy.action_chunk", True),
            (eval_module, "_obs_to_model_input", "observation.preprocess", True),
            (eval_module, "_denormalize_action", "action.denormalize", False),
            (model, "_encode_flux2_prompts", "text.encode", True),
            (text_encoder.model, "forward", "text.qwen_forward", True),
            (model, "_append_proprio_to_context_if_enabled", "proprio.encode_and_pack", True),
            (model, "_encode_flux2_image_tokens", "image.encode_tokens", True),
            (model.vae, "encode", "image.ae_encode", True),
            (model.video_expert, "pre_dit", "visual.prepare", True),
            (model.mot, "prefill_flux2_video_cache", "visual.prefill", True),
            (model.action_expert, "pre_dit", "action.prepare", True),
            (model.mot, "forward_flux2_action_with_video_cache", "action.transformer", True),
            (model.action_expert, "post_dit", "action.project", True),
        ):
            _wrap_method(patches, recorder, obj, method, name, cuda=cuda)
        if model.proprio_encoder is not None:
            _wrap_method(
                patches, recorder, model.proprio_encoder, "forward", "proprio.linear"
            )
        _replace(
            patches, text_encoder, "tokenizer", _TokenizerProxy(text_encoder.tokenizer, recorder)
        )

        original_infer = model.infer_action_flux2
        signature = inspect.signature(original_infer)
        original_begin_step = model._start_action_attention_capture_step
        original_scheduler_step = model.infer_action_scheduler.step
        active_loop: list[_LoopSpans] = []

        @functools.wraps(original_infer)
        def timed_infer(*args: Any, **kwargs: Any):
            if active_loop:
                raise RuntimeError("Concurrent/reentrant inference is unsupported during module profiling.")
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            loop = _LoopSpans(recorder, int(bound.arguments["num_inference_steps"]))
            with _span(recorder, "policy.model_forward"):
                active_loop.append(loop)
                try:
                    result = original_infer(*args, **kwargs)
                    if loop.completed_steps != loop.expected_steps:
                        raise RuntimeError(
                            "Action loop instrumentation saw "
                            f"{loop.completed_steps} steps; expected {loop.expected_steps}."
                        )
                    return result
                finally:
                    loop.close(sys.exc_info())
                    active_loop.pop()

        @functools.wraps(original_begin_step)
        def timed_begin_step(step_idx: int, *args: Any, **kwargs: Any):
            if active_loop:
                active_loop[-1].begin_step(int(step_idx))
            return original_begin_step(step_idx, *args, **kwargs)

        @functools.wraps(original_scheduler_step)
        def timed_scheduler_step(*args: Any, **kwargs: Any):
            try:
                with _span(recorder, "action.scheduler"):
                    return original_scheduler_step(*args, **kwargs)
            finally:
                if active_loop:
                    active_loop[-1].finish_step(sys.exc_info())

        _replace(patches, model, "infer_action_flux2", timed_infer)
        _replace(patches, model, "_start_action_attention_capture_step", timed_begin_step)
        _replace(patches, model.infer_action_scheduler, "step", timed_scheduler_step)
        yield MODULE_BOUNDARIES
