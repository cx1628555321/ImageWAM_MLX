#!/usr/bin/env python3
"""Export the trained FLUX.2 visual transformer for the native inference loader.

An ImageWAM full checkpoint already contains these weights. Exporting its
canonical transformer subtree avoids downloading/loading the original base DiT;
the normal ImageWAM checkpoint loader still restores the full policy afterwards.
No weights are resized, cast, initialized, or otherwise changed by this script.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

import torch
from safetensors.torch import save_file

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

PREFIX = "mixtures.video.transformer."
ALIAS_PREFIXES = ("mixtures.video.double_blocks.", "mixtures.video.single_blocks.")


def _source_stat(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "device": stat.st_dev,
        "inode": stat.st_ino,
    }


def _refuse_existing(path: Path) -> None:
    # lexists also catches dangling symlinks rather than following them.
    if os.path.lexists(path):
        raise FileExistsError(f"Refusing to overwrite existing output: {path}")


def _new_temp(output: Path) -> Path:
    fd, name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    os.close(fd)
    return Path(name)


def _publish_pair(weights_temp: Path, weights_out: Path, metadata_temp: Path, metadata_out: Path) -> None:
    """Publish each complete file atomically, without replacing any existing path.

    Both temporary files live in the target directory. Linking the metadata first
    makes the weights filename the commit point; if the weights path races, only
    the metadata link created by this invocation is removed.
    """
    metadata_linked = False
    try:
        os.link(metadata_temp, metadata_out)
        metadata_linked = True
        os.link(weights_temp, weights_out)
    except BaseException:
        if metadata_linked:
            try:
                if os.path.samestat(metadata_temp.stat(), metadata_out.lstat()):
                    metadata_out.unlink()
            except FileNotFoundError:
                pass
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Complete ImageWAM model.pt checkpoint.")
    parser.add_argument("--output", required=True, help="New visual .safetensors file (never overwritten).")
    parser.add_argument("--flux2-src-path", default=os.environ.get("FLUX2_SRC"))
    parser.add_argument("--variant", default="klein-base-4b", choices=["klein-base-4b", "klein-base-9b"])
    args = parser.parse_args()
    if not args.flux2_src_path:
        parser.error("Set --flux2-src-path or FLUX2_SRC to the official FLUX.2 checkout.")

    checkpoint = Path(args.checkpoint).expanduser().resolve(strict=True)
    output = Path(args.output).expanduser().absolute()
    metadata_out = output.with_suffix(output.suffix + ".json")
    _refuse_existing(output)
    _refuse_existing(metadata_out)
    source_stat = _source_stat(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("mot"), dict):
        raise ValueError("Expected an ImageWAM checkpoint with a mot state dictionary.")
    checkpoint_format = payload.get("checkpoint_format", "full")
    if checkpoint_format not in {"full", "lora_merged"}:
        raise ValueError(f"Complete visual weights are required, got checkpoint_format={checkpoint_format!r}.")
    mot_state = payload["mot"]
    if not all(isinstance(key, str) for key in mot_state):
        raise ValueError("MoT state dictionary contains a non-string key.")

    visual = {}
    for key, tensor in mot_state.items():
        if not key.startswith(PREFIX):
            continue
        if not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu" or tensor.layout != torch.strided:
            raise ValueError(f"Expected a dense CPU tensor at {key}.")
        visual[key[len(PREFIX):]] = tensor
    if not visual:
        raise ValueError(f"No tensor keys begin with the exact prefix {PREFIX!r}.")

    from imagewam.models.backbones.flux2_imports import ensure_flux2_importable
    ensure_flux2_importable(args.flux2_src_path)
    from flux2.model import Flux2, Klein4BParams, Klein9BParams

    params = Klein4BParams() if args.variant == "klein-base-4b" else Klein9BParams()
    with torch.device("meta"):
        expected_model = Flux2(params)
    expected = expected_model.state_dict()
    missing = sorted(set(expected) - set(visual))
    unexpected = sorted(set(visual) - set(expected))
    wrong_shapes = {
        key: {"checkpoint": list(visual[key].shape), "expected": list(expected[key].shape)}
        for key in set(expected) & set(visual)
        if visual[key].shape != expected[key].shape
    }
    if missing or unexpected or wrong_shapes:
        raise ValueError(json.dumps({"missing_keys": missing, "unexpected_keys": unexpected,
                                     "wrong_shapes": wrong_shapes}, indent=2))
    # Let PyTorch also enforce the official model's complete loading contract.
    expected_model.load_state_dict(visual, strict=True, assign=True)

    # The wrapper registers double/single blocks twice. Only the exact canonical
    # subtree above is exported. If the canonical model itself shares a storage,
    # clone later entries for safetensors while preserving every tensor value.
    seen_storages: dict[tuple[str, int], str] = {}
    cloned_aliases = []
    for key, tensor in list(visual.items()):
        tensor = tensor.detach().contiguous()
        if tensor.numel():
            storage_id = (str(tensor.device), tensor.untyped_storage().data_ptr())
            if storage_id in seen_storages:
                cloned_aliases.append({"tensor": key, "shared_with": seen_storages[storage_id]})
                tensor = tensor.clone()
            else:
                seen_storages[storage_id] = key
        visual[key] = tensor

    manifest = {key: {"shape": list(value.shape), "dtype": str(value.dtype)}
                for key, value in sorted(visual.items())}
    tensor_bytes = sum(value.numel() * value.element_size() for value in visual.values())
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_checkpoint": str(checkpoint),
        "source_stat": source_stat,
        "source_checkpoint_format": checkpoint_format,
        "source_checkpoint_step": payload.get("step"),
        "source_checkpoint_dtype": str(payload.get("torch_dtype")),
        "source_mot_key_count": len(mot_state),
        "exported_prefix": PREFIX,
        "excluded_wrapper_alias_key_count": sum(key.startswith(ALIAS_PREFIXES) for key in mot_state),
        "variant": args.variant,
        "flux2_src_path": str(Path(args.flux2_src_path).expanduser().resolve()),
        "output": str(output),
        "tensor_count": len(visual),
        "tensor_elements": sum(value.numel() for value in visual.values()),
        "tensor_bytes": tensor_bytes,
        "dtype_tensor_counts": dict(Counter(str(value.dtype) for value in visual.values())),
        "cloned_canonical_storage_aliases": cloned_aliases,
        "strict_validation": {"missing_keys": [], "unexpected_keys": [], "wrong_shapes": {}},
        "tensor_manifest_sha256": hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
        "tensor_manifest": manifest,
    }
    if isinstance(metadata["source_checkpoint_step"], torch.Tensor):
        metadata["source_checkpoint_step"] = metadata["source_checkpoint_step"].item()

    output.parent.mkdir(parents=True, exist_ok=True)
    weights_temp = _new_temp(output)
    metadata_temp = _new_temp(metadata_out)
    try:
        save_file(visual, str(weights_temp), metadata={
            "format": "pt", "source": "ImageWAM full checkpoint visual transformer",
            "variant": args.variant, "checkpoint_step": str(metadata["source_checkpoint_step"]),
        })
        with weights_temp.open("rb") as stream:
            os.fsync(stream.fileno())
        if _source_stat(checkpoint) != source_stat:
            raise RuntimeError("Source checkpoint changed during export; refusing to publish output.")
        metadata["output_size_bytes"] = weights_temp.stat().st_size
        with metadata_temp.open("w", encoding="utf-8") as stream:
            json.dump(metadata, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        _publish_pair(weights_temp, output, metadata_temp, metadata_out)
    finally:
        weights_temp.unlink(missing_ok=True)
        metadata_temp.unlink(missing_ok=True)

    print(json.dumps({"output": str(output), "metadata": str(metadata_out),
                      "checkpoint_step": metadata["source_checkpoint_step"],
                      "tensor_count": len(visual), "tensor_bytes": tensor_bytes,
                      "output_size_bytes": metadata["output_size_bytes"],
                      "strict_validation": "passed"}, indent=2))


if __name__ == "__main__":
    main()
