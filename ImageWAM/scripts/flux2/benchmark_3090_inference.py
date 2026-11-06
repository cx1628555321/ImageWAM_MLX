#!/usr/bin/env python3
"""Load the native policy once, warm up, then benchmark real LIBERO observations.

Steady end-to-end measurements have no module hooks. A separate pass on the
same loaded model records inclusive module spans using CUDA events. Simulator
render/physics and file writing are outside the measured inference boundary.
"""
from __future__ import annotations

import argparse
import copy
from collections import defaultdict
from contextlib import ExitStack, contextmanager, nullcontext
import csv
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]


def save_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=ROOT.parent / "flux2_klein_4b/model.pt")
    parser.add_argument("--dataset-stats", type=Path, default=ROOT.parent / "flux2_klein_4b/dataset_stats.json")
    parser.add_argument("--flux2-src", type=Path, default=ROOT / "third_party/flux2")
    parser.add_argument("--visual-weights", type=Path, default=ROOT / "checkpoints/3090/flux2_imagewam_visual.safetensors")
    parser.add_argument("--ae-weights", type=Path, default=ROOT / "checkpoints/3090/ae.safetensors")
    parser.add_argument("--qwen-path", type=Path, default=ROOT / "checkpoints/3090/Qwen3-4B")
    parser.add_argument("--libero-dir", type=Path, default=ROOT / "third_party/LIBERO")
    parser.add_argument("--task-suite", default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--replan-steps", type=int, default=12)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--profile-iterations", type=int, default=10)
    parser.add_argument("--warmup-min", type=int, default=6)
    parser.add_argument("--warmup-max", type=int, default=20)
    parser.add_argument("--stability-window", type=int, default=3)
    parser.add_argument("--stability-tolerance", type=float, default=0.10)
    parser.add_argument("--execution-mode", choices=("eager", "inference_mode", "compile", "torchscript", "cudagraph"), default="eager")
    parser.add_argument("--execution-boundaries", default="qwen,ae,visual,action")
    parser.add_argument("--validate-execution", action="store_true", help="Compare nominal and second real observation against eager before timing")
    parser.add_argument("--reference-atol", type=float, default=0.005)
    parser.add_argument("--reference-rtol", type=float, default=0.02)
    parser.add_argument("--loader", choices=("native", "action_only"), default="native")
    parser.add_argument("--reference-dir", type=Path, help="Native Eager run used as independent reference for optimized loading")
    parser.add_argument("--audit-loading", action="store_true", help="Hash intermediate tensors during validation and audit parameter-owner calls")
    args = parser.parse_args()
    if args.loader == "action_only" and (args.execution_mode != "eager" or not args.reference_dir or not args.audit_loading):
        parser.error("action_only validation requires eager, --reference-dir and --audit-loading")
    if args.audit_loading:
        args.validate_execution = True
    for name in ("steps", "horizon", "replan_steps", "context_length", "iterations", "profile_iterations", "stability_window"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.warmup_min < 2 * args.stability_window or args.warmup_max < args.warmup_min:
        parser.error("warmup-max >= warmup-min >= 2 * stability-window is required")
    if not 0 < args.stability_tolerance < 1:
        parser.error("--stability-tolerance must be in (0, 1)")
    if args.replan_steps > args.horizon:
        parser.error("--replan-steps cannot exceed --horizon")
    if args.reference_atol < 0 or args.reference_rtol < 0:
        parser.error("Reference tolerances must be nonnegative")
    if not set(args.execution_boundaries.split(",")) <= {"qwen", "ae", "visual", "action"}:
        parser.error("Execution boundaries must be a comma-separated subset of qwen,ae,visual,action")
    return args


def stability(samples, window, tolerance):
    if len(samples) < 2 * window:
        return {"stable": False, "reason": "insufficient_windows"}
    previous, latest = samples[-2 * window:-window], samples[-window:]
    drift = abs(statistics.median(latest) - statistics.median(previous)) / statistics.median(previous)
    cv = statistics.pstdev(latest) / statistics.mean(latest)
    return {"stable": drift <= tolerance and cv <= tolerance,
            "median_relative_change": drift, "last_window_coefficient_of_variation": cv,
            "window": window, "tolerance": tolerance}


def command_output(command):
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


@contextmanager
def audit_inference_mode(torch, model, execution):
    """Check actual module entry context during validation only, never steady/profile."""
    audit = execution["inference_mode"]["module_context_audit"] = {}
    targets = (("qwen", model.text_encoder.model, "forward"),
               ("ae", model.vae, "encode"),
               ("visual", model.mot, "prefill_flux2_video_cache"),
               ("action", model.mot, "forward_flux2_action_with_video_cache"))
    with ExitStack() as stack:
        for name, owner, method in targets:
            own = method in vars(owner)
            previous = vars(owner).get(method)
            original = getattr(owner, method)

            def checked(*args, _name=name, _original=original, **kwargs):
                enabled = torch.is_inference_mode_enabled()
                grad = torch.is_grad_enabled()
                row = audit.setdefault(_name, {"calls": 0, "all_inference_mode_enabled": True,
                                                "all_grad_disabled": True})
                row["calls"] += 1
                row["all_inference_mode_enabled"] &= enabled
                row["all_grad_disabled"] &= not grad
                if not enabled or grad:
                    raise RuntimeError(f"inference_mode is not effective inside {_name}")
                return _original(*args, **kwargs)

            def restore(_owner=owner, _method=method, _own=own, _previous=previous):
                if _own:
                    setattr(_owner, _method, _previous)
                else:
                    delattr(_owner, _method)

            setattr(owner, method, checked)
            stack.callback(restore)
        yield
    if any(not audit.get(name, {}).get("calls") for name, _, _ in targets):
        raise RuntimeError("inference_mode validation did not reach every module")


def module_totals(records, completed_iterations, summarize):
    completed_iterations = set(completed_iterations)
    grouped = defaultdict(list)
    for record in records:
        if record["category"] == "module" and record["status"] == "ok" and record["iteration"] in completed_iterations:
            grouped[record["name"]].append(record)
    rows = []
    for name, calls in grouped.items():
        row = {"name": name, "calls": len(calls), "completed_inferences": len(completed_iterations), "calls_per_inference": len(calls) / len(completed_iterations),
               "inclusive": True, "note": "Repeated calls of this name are summed per inference; do not add parent and child names."}
        for metric in ("wall_ms", "cuda_ms"):
            per_iteration = defaultdict(float)
            for call in calls:
                if call[metric] is not None:
                    per_iteration[call["iteration"]] += call[metric]
            row[metric + "_per_inference"] = summarize(per_iteration.values())
            row[metric + "_per_call"] = summarize(call[metric] for call in calls if call[metric] is not None)
        rows.append(row)
    return rows


def write_reports(output, report, recorder, iterations, summarize):
    try:
        recorder.resolve_pending()
    except Exception as error:
        report["event_resolution_error"] = repr(error)
        report["status"] = "failed"
    report["records"] = recorder.records
    report["per_call_statistics"] = recorder.aggregate()
    completed = [row["iteration"] for row in iterations if row["phase"] == "profile"]
    report["modules"] = module_totals(recorder.records, completed, summarize)
    save_json(output / "benchmark.json", report)
    with (output / "iterations.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["phase", "iteration", "wall_ms"])
        writer.writeheader(); writer.writerows(iterations)
    with (output / "spans.csv").open("w", newline="") as stream:
        fields = ["id", "phase", "iteration", "category", "name", "parent_id", "parent", "wall_ms", "cuda_ms", "wall_semantics", "status"]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(recorder.records)
    with (output / "module_summary.csv").open("w", newline="") as stream:
        fields = ["name", "calls_per_inference", "cuda_ms_mean_per_inference", "cuda_ms_mean_per_call", "host_wall_ms_mean_per_inference"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in report["modules"]:
            writer.writerow({"name": row["name"], "calls_per_inference": row["calls_per_inference"],
                "cuda_ms_mean_per_inference": row["cuda_ms_per_inference"]["mean"],
                "cuda_ms_mean_per_call": row["cuda_ms_per_call"]["mean"],
                "host_wall_ms_mean_per_inference": row["wall_ms_per_inference"]["mean"]})
    lines = ["# ImageWAM 3090 性能统计", "", f"状态：{report['status']}。执行模式：{report.get('execution', {}).get('mode', 'eager')}。单位：ms。一次加载，同一真实观测重复推理。", "",
             "端到端范围：原始双相机观测和任务文本 → CPU 上可执行动作；含预处理、在线文本编码、模型和后处理，排除仿真 step/render、加载、预热和文件写入。", "",
             "## 加载", "", "同步 wall 时间，包含磁盘页缓存影响；父子行不能相加。", "", "| 阶段 | wall ms |", "|---|---:|"]
    for record in recorder.records:
        if record["category"] == "load":
            value = "—" if record["wall_ms"] is None else f"{record['wall_ms']:.3f}"
            lines.append(f"| {record['name']} | {value} |")
    if "warmup" in report:
        lines += ["", "## 预热与稳态", "", f"预热 {report['warmup']['iterations']} 次，实际阶段总耗时 {report['warmup']['total_wall_ms']:.3f} ms；稳定判据：{report['warmup']['stability']}。", "",
                  "| 测量 | 次数 | 平均 ms | 中位数 ms | p95 ms | 标准差 ms |", "|---|---:|---:|---:|---:|---:|"]
        for key, label in (("warmup", "预热单次"), ("steady", "无模块钩子端到端"), ("profile", "带模块钩子端到端，仅供核对")):
            if key in report:
                stats = report[key]["latency_ms"]
                lines.append(f"| {label} | {stats['count']} | {stats['mean']:.3f} | {stats['median']:.3f} | {stats['p95']:.3f} | {stats['std']:.3f} |")
        if "steady" in report:
            lines += ["", f"稳态样本稳定性检查：{report['steady']['stability']}。"]
    lines += ["", "## 模块", "", "CUDA event 是当前 stream 的经过时间，可含 CPU 提交间隙，不是纯 kernel 时间；host wall 未在模块边界同步。父子 inclusive 行不能相加。", "",
              "| 模块 | 每次推理调用数 | CUDA ms/整次推理 | CUDA ms/单次调用 | host wall ms/整次推理 |", "|---|---:|---:|---:|---:|"]
    fmt = lambda value: "—" if value is None else f"{value:.3f}"
    for row in report["modules"]:
        lines.append(f"| {row['name']} | {row['calls_per_inference']:g} | {fmt(row['cuda_ms_per_inference']['mean'])} | {fmt(row['cuda_ms_per_call']['mean'])} | {fmt(row['wall_ms_per_inference']['mean'])} |")
    if "execution" in report:
        execution = report["execution"]
        lines += ["", "## 执行方式与准备开销", "", "```json", json.dumps(execution, indent=2, ensure_ascii=False), "```"]
    lines += ["", "完整配置、模块边界、硬件、模型路径、稳定性与原始样本见 benchmark.json / resolved_config.yaml / iterations.csv / spans.csv。", ""]
    (output / "summary.md").write_text("\n".join(lines))


def main():
    started = time.perf_counter()
    args = parse_args()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "benchmark.json").exists():
        raise FileExistsError(f"Refusing to overwrite benchmark: {output}")
    for path in (ROOT, ROOT / "src", ROOT / "experiments/libero", args.flux2_src / "src", args.libero_dir):
        sys.path.insert(0, str(path.resolve()))
    os.environ.setdefault("LIBERO_CONFIG_PATH", str(ROOT / ".libero"))
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    import numpy as np
    import torch
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from experiments.libero import eval_libero_single as evaluation
    from profile_3090_timing import Recorder, load_instrumentation, summarize
    from profile_3090_modules import MODULE_BOUNDARIES, module_instrumentation
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires the existing CUDA PyTorch environment")
    paths = {"checkpoint": str(args.checkpoint.resolve()), "dataset_stats": str(args.dataset_stats.resolve()),
             "flux2_src": str(args.flux2_src.resolve()), "flux2": str(args.visual_weights.resolve()),
             "ae": str(args.ae_weights.resolve()), "qwen3": str(args.qwen_path.resolve())}
    for name, value in paths.items():
        if not Path(value).exists():
            raise FileNotFoundError(f"Missing {name}: {value}")
    overrides = ["task=libero_flux2_klein_4b_base_imagewam"]
    values = {"ckpt": paths["checkpoint"], "model.flux2_src_path": paths["flux2_src"],
        "model.flux2_model_path": paths["flux2"], "model.ae_model_path": paths["ae"],
        "model.qwen3_model_spec": paths["qwen3"], "model.load_text_encoder": True,
        "model.action_dit_pretrained_path": None, "model.proprio_dim": 8, "model.pack_proprio_after_text": True,
        "model.qwen_context_len": args.context_length, "data.train.dataset_dirs": [],
        "data.train.qwen_text_cache_dir": None, "data.train.qwen_context_len": args.context_length,
        "data.train.qwen_text_cache_format": "qwen3_flux2", "mixed_precision": "bf16", "wandb.enabled": False,
        "gpu_id": 0, "seed": args.seed, "EVALUATION.dataset_stats_path": paths["dataset_stats"],
        "EVALUATION.task_suite_name": args.task_suite, "EVALUATION.task_id": args.task_id,
        "EVALUATION.num_trials": 1, "EVALUATION.env_num": 1, "EVALUATION.device": "cuda:0",
        "EVALUATION.num_inference_steps": args.steps, "EVALUATION.action_horizon": args.horizon,
        "EVALUATION.replan_steps": args.replan_steps, "EVALUATION.visualize_future_video": False,
        "EVALUATION.output_dir": str(output)}
    overrides += [f"{key}={json.dumps(value)}" for key, value in values.items()]
    with initialize_config_dir(version_base="1.3", config_dir=str(ROOT / "configs")):
        cfg = compose(config_name="sim_libero_omnigen2", overrides=overrides)
    model_cfg = evaluation._resolve_model_cfg(cfg)
    if args.loader == "action_only":
        OmegaConf.set_struct(model_cfg, False)
        model_cfg._target_ = "imagewam.optimized_action_loader.create_imagewam_flux2_action_only"
        model_cfg.checkpoint_path = paths["checkpoint"]
    evaluation._save_resolved_eval_config(cfg, model_cfg, output / "resolved_config.yaml")
    evaluation.set_global_seed(args.seed, get_worker_init_fn=False)
    imports_ms = (time.perf_counter() - started) * 1000
    cuda_start = time.perf_counter()
    torch.cuda.init(); torch.cuda.synchronize()
    cuda_init_ms = (time.perf_counter() - cuda_start) * 1000
    recorder = Recorder("cuda:0")
    iterations = []
    report = {"schema_version": 2, "status": "running", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "model_paths": paths, "loading_count": 0, "boundary": "Raw LIBERO observation + task -> CPU denormalized action chunk; excludes simulator/render, loading, warmup, report writing",
        "input_policy": "Repeat the same real LIBERO initial observation; no text/image embedding cache and no environment actions executed during timing",
        "startup": {"imports_and_configuration_ms": imports_ms, "cuda_context_initialization_ms": cuda_init_ms},
        "environment": {"python": sys.version, "python_executable": sys.executable, "torch": torch.__version__,
            "torch_cuda": torch.version.cuda, "torch_file": torch.__file__, "gpu": torch.cuda.get_device_name(0),
            "gpu_before": command_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.used,pstate,clocks.sm,clocks.mem,power.limit", "--format=csv,noheader"]),
            "imagewam_commit": command_output(["git", "rev-parse", "HEAD"]),
            "flux2_commit": command_output(["git", "-C", paths["flux2_src"], "rev-parse", "HEAD"])},
        "module_boundaries": MODULE_BOUNDARIES,
        "execution": {"mode": args.execution_mode, "boundaries": args.execution_boundaries.split(","), "phase": "load",
            "precision": "native_bf16", "reference_atol": args.reference_atol, "reference_rtol": args.reference_rtol,
            "scope": "Selected GPU module boundaries; CPU preprocessing and Python control flow remain eager"}}
    save_json(output / "progress.json", report)
    env = None
    execution_stack = ExitStack()
    return_code = 1
    try:
        with (load_instrumentation(recorder, paths) if args.loader == "native" else nullcontext()):
            with recorder.span("model_load_total", category="load"):
                model = instantiate(model_cfg, model_dtype=torch.bfloat16, device="cuda:0",
                                    **({"recorder": recorder} if args.loader == "action_only" else {}))
                evaluation._load_model_checkpoint(model, paths["checkpoint"])
                model = model.to("cuda:0").eval()
        report["loader"] = {"name": args.loader, "metadata": getattr(model, "loading_metadata", None)}
        report["loader"]["resident_parameter_elements"] = sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in model.text_encoder.model.parameters())
        report["loader"]["resident_parameter_bytes"] = sum(p.numel()*p.element_size() for p in model.parameters()) + sum(p.numel()*p.element_size() for p in model.text_encoder.model.parameters())
        report["loading_count"] = 1
        save_json(output / "load_spans.json", recorder.records)
        print("MODEL_LOADED_ONCE", flush=True)
        setup_started = time.perf_counter()
        stats = evaluation.load_dataset_stats_from_json(paths["dataset_stats"])
        processor = instantiate(cfg.data.train.processor).eval()
        processor.set_normalizer_from_stats(stats)
        suite = evaluation.benchmark.get_benchmark_dict()[args.task_suite]()
        task = suite.get_task(args.task_id)
        initial = suite.get_task_init_states(args.task_id)[0]
        env, description = evaluation.get_libero_env(task, evaluation.LIBERO_ENV_RESOLUTION, args.seed)
        env.reset(); obs = env.set_init_state(initial)
        for _ in range(int(cfg.EVALUATION.num_steps_wait)):
            obs, _, _, _ = env.step(evaluation.get_libero_dummy_action())
        report["input"] = {"task_description": description, "initial_state_index": 0,
            "num_steps_wait": int(cfg.EVALUATION.num_steps_wait),
            "cameras": {key: {"shape": list(obs[key].shape), "sha256": hashlib.sha256(obs[key].tobytes()).hexdigest()}
                        for key in ("agentview_image", "robot0_eye_in_hand_image")}}
        height, width = map(int, cfg.data.train.video_size)
        report["simulator_and_processor_setup_ms"] = (time.perf_counter() - setup_started) * 1000
        reference = None
        execution = report["execution"]
        independent = None
        if args.reference_dir:
            independent = json.loads((args.reference_dir / "benchmark.json").read_text())
            if independent['status'] != 'complete' or independent.get('loader', {}).get('name') != 'native':
                raise ValueError('Reference must be a completed native-loader run')
            if independent['input'] != report['input'] or independent['model_paths'] != report['model_paths']:
                raise ValueError('Independent reference has different input or source paths')
            for key in ('steps','horizon','context_length','seed','task_suite','task_id'):
                if independent['configuration'][key] != report['configuration'][key]:
                    raise ValueError(f'Independent reference configuration mismatch: {key}')
            execution['independent_native_reference'] = str(args.reference_dir.resolve())
        if args.execution_mode == "inference_mode":
            execution["scope"] = "Full prediction call under torch.inference_mode(); native eager operators and Python orchestration, no compile/trace/capture"
            execution["boundaries"] = ["policy.action_chunk"]
            execution["inference_mode"] = {
                "enabled": True, "engine": "eager", "model_eval": not model.training,
                "scope": "Preprocessing through CPU denormalized actions, one context enter/exit per prediction",
                "excluded": "Model loading, simulator, eager reference generation",
                "compile_enabled": False, "torchscript_enabled": False, "cuda_graph_enabled": False,
                "calls_by_phase": {}, "reference_context_enabled": False}

        def predict_native(observation, prompt=None):
            result, _, _ = evaluation._predict_action_chunk(observation, description if prompt is None else prompt, model, processor, cfg,
                action_horizon=args.horizon, input_w=width, input_h=height, model_device="cuda:0")
            return result

        def predict(observation, prompt=None):
            if args.execution_mode != "inference_mode" or execution["phase"] == "eager_reference":
                if args.execution_mode == "inference_mode" and torch.is_inference_mode_enabled():
                    raise RuntimeError("Eager reference must be generated outside inference_mode")
                return predict_native(observation, prompt)
            with torch.inference_mode():
                if not torch.is_inference_mode_enabled() or torch.is_grad_enabled():
                    raise RuntimeError("torch.inference_mode context is not active")
                counts = execution["inference_mode"]["calls_by_phase"]
                counts[execution["phase"]] = counts.get(execution["phase"], 0) + 1
                return predict_native(observation, prompt)

        def compiler_snapshot():
            if args.execution_mode != "compile":
                return {}
            from profile_3090_compile import dynamo_snapshot
            return dynamo_snapshot()

        if args.validate_execution or args.execution_mode != "eager":
            execution["phase"] = "eager_reference"
            obs = copy.deepcopy(obs)
            reference_started = time.perf_counter()
            eager_a = (np.load(args.reference_dir / 'eager_reference_action.npy') if independent else predict(obs).copy())
            all_initial = suite.get_task_init_states(args.task_id)
            if len(all_initial) < 2:
                raise ValueError("Execution validation requires two real initial states")
            env.reset()
            other_obs = env.set_init_state(all_initial[1])
            for _ in range(int(cfg.EVALUATION.num_steps_wait)):
                other_obs, _, _, _ = env.step(evaluation.get_libero_dummy_action())
            other_obs = copy.deepcopy(other_obs)
            eager_b = (np.load(args.reference_dir / 'eager_second_action.npy') if independent else predict(other_obs).copy())
            alternate_prompt = description + " Carefully."
            eager_c = (np.load(args.reference_dir / 'eager_alternate_text_action.npy') if independent else predict(obs, alternate_prompt).copy())
            torch.cuda.synchronize()
            if not all(np.isfinite(value).all() for value in (eager_a, eager_b, eager_c)):
                raise ValueError("Non-finite eager reference action")
            execution["eager_reference_generation_ms"] = (time.perf_counter() - reference_started) * 1000
            execution["validation_input"] = {"initial_state_index": 1,
                "cameras": {key: hashlib.sha256(other_obs[key].tobytes()).hexdigest()
                            for key in ("agentview_image", "robot0_eye_in_hand_image")}}
            execution["eager_ab_max_difference"] = float(np.max(np.abs(eager_a - eager_b)))
            execution["alternate_prompt"] = alternate_prompt
            if independent and (execution['validation_input'] != independent['execution']['validation_input'] or alternate_prompt != independent['execution']['alternate_prompt']):
                raise ValueError('Independent reference validation inputs differ')
            execution["eager_ac_max_difference"] = float(np.max(np.abs(eager_a - eager_c)))
            if np.allclose(eager_a, eager_b, atol=args.reference_atol, rtol=args.reference_rtol):
                raise ValueError("Two-input validation is inconclusive: eager actions are too close under the validation tolerances")
            np.save(output / "eager_reference_action.npy", eager_a)
            np.save(output / "eager_second_action.npy", eager_b)
            np.save(output / "eager_alternate_text_action.npy", eager_c)
            execution["phase"] = "prepare"
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            boundaries = args.execution_boundaries.split(",")
            if args.execution_mode in {"compile", "torchscript"}:
                from profile_3090_compile import install
                execution_stack.enter_context(install(model, args.execution_mode, execution, boundaries=boundaries))
            elif args.execution_mode == "cudagraph":
                from profile_3090_cudagraph import install
                execution_stack.enter_context(install(model, execution, boundaries=boundaries))
            preparation_started = time.perf_counter()
            comparisons = []
            stage_records, coverage = {}, {}
            native_stages = json.loads((args.reference_dir / 'validation_intermediates.json').read_text()) if independent else None
            for label, observation, prompt, expected in (("A", obs, description, eager_a), ("B", other_obs, description, eager_b),
                                                         ("C_text", obs, alternate_prompt, eager_c), ("A_again", obs, description, eager_a)):
                begin = time.perf_counter()
                context_audit = (audit_inference_mode(torch, model, execution)
                                 if args.execution_mode == "inference_mode" and label == "A" else nullcontext())
                stage_context = nullcontext()
                if args.audit_loading:
                    from validate_3090_loading import capture
                    stage_records[label] = {}
                    stage_context = capture(model, stage_records[label], coverage if label == 'A' else None)
                with context_audit, stage_context:
                    actual = predict(observation, prompt).copy()
                np.save(output / f'actions_{label}.npy', actual)
                if args.audit_loading:
                    save_json(output / 'validation_intermediates.json', stage_records)
                    save_json(output / 'parameter_owner_coverage.json', coverage)
                    if native_stages is not None and stage_records[label] != native_stages[label]:
                        raise ValueError(f'Intermediate tensor exact hash mismatch for {label}')
                    if args.loader == 'action_only' and any(v['calls'] == 0 for v in coverage.values()):
                        raise ValueError('Optimized model still has parameter owners not reached in inference')
                torch.cuda.synchronize()
                if actual.shape != expected.shape or not np.isfinite(actual).all():
                    raise ValueError(f"Invalid execution output for {label}: shape={actual.shape}, finite={bool(np.isfinite(actual).all())}")
                difference = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
                item = {"input": label, "wall_ms": (time.perf_counter() - begin) * 1000,
                    "max_abs_difference": float(difference.max()), "mean_abs_difference": float(difference.mean()),
                    "rmse": float(np.sqrt(np.mean(difference ** 2))),
                    "allclose": bool(np.allclose(actual, expected, atol=args.reference_atol, rtol=args.reference_rtol)),
                    "gripper_equal": bool(np.array_equal(actual[:, -1], expected[:, -1]))}
                comparisons.append(item)
                execution["validation"] = comparisons
                save_json(output / "execution_progress.json", execution)
                print("EXECUTION_VALIDATION", json.dumps(item), flush=True)
                if not item["allclose"] or not item["gripper_equal"]:
                    raise ValueError(f"Execution validation against eager failed: {item}")
                if independent and not np.array_equal(actual, expected):
                    raise ValueError(f'Optimized loader action must be bitwise equal to native: {item}')
                if label == "A_again":
                    reference = actual.copy()
            execution["preparation_and_validation_ms"] = (time.perf_counter() - preparation_started) * 1000
            if args.audit_loading:
                execution['loading_validation'] = {'intermediate_hashes_equal_to_native': True if independent else None,
                    'all_loaded_parameter_owners_called': all(v['calls'] > 0 for v in coverage.values()),
                    'parameter_owner_count': len(coverage), 'labels': list(stage_records),
                    'note': 'Exact tensor hashes and parameter-owner forward hooks only during validation; not a per-scalar usage proof'}
            execution["preparation_max_allocated_bytes"] = torch.cuda.max_memory_allocated()
            execution["preparation_max_reserved_bytes"] = torch.cuda.max_memory_reserved()
            execution["compiler_after_prepare"] = compiler_snapshot()
            print("EXECUTION_READY", args.execution_mode, flush=True)
        def infer_once(phase, index):
            nonlocal reference
            execution["phase"] = phase
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            action = predict(obs)
            torch.cuda.synchronize()
            wall_ms = (time.perf_counter() - t0) * 1000
            assert action.shape == (args.horizon, 7), action.shape
            if not np.isfinite(action).all():
                raise ValueError("Non-finite action produced")
            if reference is None:
                reference = action.copy()
            difference = float(np.max(np.abs(action - reference)))
            report["maximum_action_difference"] = max(report.get("maximum_action_difference", 0), difference)
            if not np.allclose(action, reference, rtol=1e-5, atol=1e-5):
                raise ValueError(f"Repeated fixed-input action changed: max difference {difference}")
            iterations.append({"phase": phase, "iteration": index, "wall_ms": wall_ms})
            print(f"{phase} {index + 1}: {wall_ms:.3f} ms", flush=True)
            return wall_ms
        warmup = []
        warmup_started = time.perf_counter()
        for index in range(args.warmup_max):
            warmup.append(infer_once("warmup", index))
            state = stability(warmup, args.stability_window, args.stability_tolerance)
            if len(warmup) >= args.warmup_min and state["stable"]:
                break
        report["warmup"] = {"iterations": len(warmup), "total_wall_ms": (time.perf_counter() - warmup_started) * 1000,
            "first_call_ms": warmup[0], "latency_ms": summarize(warmup), "stability": state}
        print("WARMUP_STABILITY", json.dumps(state), flush=True)
        execution["compiler_before_steady"] = compiler_snapshot()
        torch.cuda.reset_peak_memory_stats()
        steady = [infer_once("steady", index) for index in range(args.iterations)]
        report["steady"] = {"latency_ms": summarize(steady), "stability": stability(steady, args.stability_window, args.stability_tolerance),
            "module_hooks_enabled": False, "max_allocated_bytes": torch.cuda.max_memory_allocated(),
            "max_reserved_bytes": torch.cuda.max_memory_reserved()}
        execution["compiler_after_steady"] = compiler_snapshot()
        profile = []
        recorder.phase = "profile"
        with module_instrumentation(recorder, model, evaluation):
            if args.execution_mode != "eager":
                execution["phase"] = "profile_prepare"
                recorder.enabled = False
                preparation_started = time.perf_counter()
                for _ in range(2):
                    checked = predict(obs)
                    if not np.allclose(checked, reference, rtol=1e-5, atol=1e-5):
                        raise ValueError("Profiling wrapper changed inference result")
                torch.cuda.synchronize()
                execution["profile_wrapper_preparation_ms"] = (time.perf_counter() - preparation_started) * 1000
                recorder.enabled = True
            execution["compiler_before_profile"] = compiler_snapshot()
            for index in range(args.profile_iterations):
                recorder.iteration = index
                profile.append(infer_once("profile", index))
                recorder.resolve_pending(synchronize=False)  # infer_once already synchronized the full iteration
            execution["compiler_after_profile"] = compiler_snapshot()
        report["profile"] = {"latency_ms": summarize(profile), "module_hooks_enabled": True,
            "mean_overhead_relative_to_steady": statistics.mean(profile) / statistics.mean(steady) - 1}
        np.save(output / "reference_action.npy", reference)
        report["status"] = ("complete_warmup_not_stable" if not state["stable"] else
                            "complete_steady_not_stable" if not report["steady"]["stability"]["stable"] else "complete")
        if args.execution_mode == "compile":
            from profile_3090_compile import dynamo_delta
            execution["compiler_steady_delta"] = dynamo_delta(execution["compiler_before_steady"], execution["compiler_after_steady"])
            execution["compiler_profile_delta"] = dynamo_delta(execution["compiler_before_profile"], execution["compiler_after_profile"])
            for key in ("compiler_steady_delta", "compiler_profile_delta"):
                if execution[key].get("stats", {}).get("unique_graphs", 0):
                    raise RuntimeError(f"Compilation occurred during measured interval: {key}={execution[key]}")
        report["total_elapsed_ms_before_report"] = (time.perf_counter() - started) * 1000
        return_code = 0 if report["status"] == "complete" else 2
    except BaseException as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        try:
            execution_stack.close()
            if env is not None:
                env.close()
        except Exception as cleanup_error:
            report["cleanup_error"] = repr(cleanup_error)
            if report["status"] != "failed":
                report["status"] = "failed_cleanup"
                return_code = 1
        finally:
            write_reports(output, report, recorder, iterations, summarize)
    print("STEADY_MS", json.dumps(report["steady"]["latency_ms"]), flush=True)
    print(f"REPORT {output / 'summary.md'}", flush=True)
    return 1 if report["status"].startswith("failed") else return_code


if __name__ == "__main__":
    raise SystemExit(main())
