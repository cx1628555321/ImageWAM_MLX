"""Independent execution/device checks, sensitivity probes and run provenance."""
import hashlib
import json
import tarfile
from pathlib import Path
import numpy as np
import torch
from robotwin_io import prepare_observation, denormalize_action
from timing import write


def compare(actual, expected):
    if actual.shape != expected.shape:
        raise ValueError(f"Shape mismatch: {actual.shape} vs {expected.shape}")
    diff = actual.astype(np.float64) - expected.astype(np.float64)
    return {"bitwise_equal": bool(np.array_equal(actual, expected)),
            "max_abs": float(np.abs(diff).max()), "mae": float(np.abs(diff).mean()),
            "rmse": float(np.sqrt(np.mean(diff ** 2)))}


def tensor_details(value):
    if isinstance(value, torch.Tensor):
        return [{"device": str(value.device), "dtype": str(value.dtype), "shape": list(value.shape)}]
    if isinstance(value, dict):
        return [t for v in value.values() for t in tensor_details(v)]
    if isinstance(value, (list, tuple)):
        return [t for v in value for t in tensor_details(v)]
    return []


def run_audit(model, images, state, proc, prompt, reference, out, reference_dir):
    """Runs after performance measurement; no hooks contaminate steady timings."""
    modules = {"qwen": model.text_encoder.model, "ae": model.vae,
               "visual": model.video_expert, "action": model.action_expert,
               "proprio": model.proprio_encoder}
    inventory = {}
    calls = {}
    handles = []
    for group, owner in modules.items():
        params = list(owner.parameters())
        inventory[group] = {"elements": sum(p.numel() for p in params),
                            "devices": sorted({str(p.device) for p in params}),
                            "dtypes": sorted({str(p.dtype) for p in params})}
        if any(p.device.type != "mps" for p in params):
            raise RuntimeError(f"Parameters outside MPS in {group}")
        for name, sub in owner.named_modules():
            if not list(sub.parameters(recurse=False)):
                continue
            key = group + "." + name
            def hook(module, args, kwargs, key=key):
                record = calls.setdefault(key, {"calls": 0, "first_inputs": tensor_details((args, kwargs))})
                record["calls"] += 1
            handles.append(sub.register_forward_pre_hook(hook, with_kwargs=True))

    def predict(im, st, text):
        image, proprio = prepare_observation(im, st, proc)
        with torch.no_grad():
            normalized = model.infer_action_flux2(
                prompt=text, input_image=image, proprio=proprio, action_horizon=16,
                num_inference_steps=10, seed=42, rand_device="cpu")["action"]
        actual = denormalize_action(normalized, proc)
        torch.mps.synchronize()
        if actual.shape != (16, 14) or not np.isfinite(actual).all():
            raise RuntimeError("Invalid audit output")
        return actual, normalized.numpy()

    try:
        actual, normalized = predict(images, state, prompt)
    finally:
        for handle in handles:
            handle.remove()
    result = {"parameter_inventory": inventory, "parameter_module_calls": calls,
              "same_input_with_hooks": compare(actual, reference)}
    if not result["same_input_with_hooks"]["bitwise_equal"]:
        raise RuntimeError("Device audit hooks changed action")
    if any(not any(k.startswith(group+".") for k in calls) for group in modules):
        raise RuntimeError("A model component was not executed")
    if reference_dir:
        reference_dir = Path(reference_dir).resolve()
        before = json.loads((reference_dir/"input_manifest.json").read_text())
        now = json.loads((out/"input_manifest.json").read_text())
        if before != now:
            raise ValueError("Reference input manifest differs")
        result["independent_reference"] = {
            "directory": str(reference_dir),
            "actions": compare(reference, np.load(reference_dir/"reference_action.npy")),
            "normalized": compare(normalized, np.load(reference_dir/"normalized_first_000.npy"))}
        if not result["independent_reference"]["actions"]["bitwise_equal"]:
            raise RuntimeError("Refactored action differs from historical baseline")

    sensitivity = {}
    for camera in images:
        changed = dict(images)
        changed[camera] = np.zeros_like(images[camera])
        action, _ = predict(changed, state, prompt)
        np.save(out/("audit_without_"+camera+".npy"), action)
        sensitivity[camera] = compare(action, reference)
    action, _ = predict(images, state + np.float32(0.1), prompt)
    np.save(out/"audit_changed_state.npy", action)
    sensitivity["state_plus_0.1"] = compare(action, reference)
    action, _ = predict(images, state, prompt + " Use the right arm instead.")
    np.save(out/"audit_changed_text.npy", action)
    sensitivity["changed_text"] = compare(action, reference)
    result["sensitivity"] = sensitivity
    result["sensitivity_note"] = "Synthetic input perturbations test dependency, not task success or accuracy."
    result["all_probes_change_output"] = all(v["max_abs"] > 0 for v in sensitivity.values())
    result["dataset_action_comparison"] = compare(reference, np.load(out/"dataset_action_chunk.npy"))
    write(out/"execution_audit.json", result)
    return result


def save_provenance(root, out, report):
    sources = list((root/"src").rglob("*.py")) + list((root/"third_party/flux2/src").rglob("*.py"))
    sources += list((root/"scripts/m5pro").glob("*.py"))
    sources += [root/"scripts/flux2/profile_3090_modules.py",
                root/"experiments/robotwin/imagewam_policy/deploy_policy.py",
                root/"configs/data/robotwin_omnigen2.yaml"]
    source_hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    with tarfile.open(out/"source_snapshot.tar.gz", "x:gz") as archive:
        for p in sources:
            archive.add(p, arcname=str(p.relative_to(root)))
    (out/"requirements-lock.txt").write_text(report["environment"]["packages"]+"\n")
    write(out/"source_manifest.json", source_hashes)
    # Hash immutable input/output artifacts only; benchmark.json is still being finalized.
    files = list(out.glob("*.npy")) + list(out.glob("*.npz")) + list(out.glob("*.png"))
    write(out/"artifact_sha256.json", {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
