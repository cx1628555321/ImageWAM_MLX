#!/usr/bin/env python3
"""Download and verify the public official FLUX.2 VAE inside ImageWAM."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

REPO_ID = "black-forest-labs/FLUX.2-klein-base-4B"
REVISION = "a3b4f4849157f664bdbc776fd7453c2783562f4d"
WEIGHTS = "vae/diffusion_pytorch_model.safetensors"
SOURCE_SHA256 = "ca70d2202afe6415bdbcb8793ba8cd99fd159cfe6192381504d6c4d3036e0f04"
SOURCE_BYTES = 168120878
CONFIG = "vae/config.json"
CONFIG_BLOB_SHA1 = "c3f38eb4c188a96a462519159356bd0fcc28cd14"


def save_json(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def inside_repo(root: Path, relative: str) -> Path:
    path = root / relative
    if not path.resolve().is_relative_to(root):
        raise ValueError(f"Path resolves outside the repository: {relative}")
    return path


def digest(path: Path, algorithm: str, prefix: bytes = b"") -> str:
    result = hashlib.new(algorithm, prefix)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--proxy", default="http://127.0.0.1:7897", help="Use an empty value for direct access")
    parser.add_argument("--verify-only", action="store_true", help="Verify existing local files without downloading")
    args = parser.parse_args()
    root = args.repo_root.resolve()
    if not (root / "pyproject.toml").is_file():
        parser.error("--repo-root must point to the ImageWAM repository")

    target = inside_repo(root, "checkpoints/3090/public_vae")
    cache = inside_repo(root, ".cache/huggingface")
    artifact_dir = inside_repo(root, "artifacts/3090_setup")
    report_path = inside_repo(root, "artifacts/3090_setup/ae_download.json")
    for path in (target, cache, artifact_dir):
        path.mkdir(parents=True, exist_ok=True)
    for relative in ("checkpoints/3090/public_vae/vae", "checkpoints/3090/public_vae/.cache",
                     ".cache/huggingface/hub", ".cache/huggingface/assets", ".cache/huggingface/xet"):
        inside_repo(root, relative)
    for name in (WEIGHTS, CONFIG):
        inside_repo(root, f"checkpoints/3090/public_vae/{name}")

    # Hugging Face reads these paths on import; anonymous access is explicit below.
    os.environ.update({
        "HF_HOME": str(cache),
        "HF_ENDPOINT": "https://huggingface.co",
        "HF_HUB_CACHE": str(cache / "hub"),
        "HF_ASSETS_CACHE": str(cache / "assets"),
        "HF_XET_CACHE": str(cache / "xet"),
        "HF_HUB_DISABLE_XET": "1",
        "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
        "HF_HUB_DOWNLOAD_TIMEOUT": "120",
        "HF_HUB_ETAG_TIMEOUT": "30",
    })
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        if args.proxy:
            os.environ[key] = args.proxy
        else:
            os.environ.pop(key, None)

    report = {
        "repo_id": REPO_ID, "revision": REVISION,
        "source_file": WEIGHTS, "source_sha256": SOURCE_SHA256,
        "source_bytes": SOURCE_BYTES, "target": str(target),
        "status": "running", "anonymous": True,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    save_json(report_path, report)
    try:
        if not args.verify_only:
            from huggingface_hub import hf_hub_download
            from huggingface_hub.utils import logging as hub_logging

            hub_logging.set_verbosity_error()
            for filename in (CONFIG, WEIGHTS):
                print(f"Downloading {REPO_ID}@{REVISION}/{filename}", flush=True)
                hf_hub_download(
                    repo_id=REPO_ID, revision=REVISION, filename=filename,
                    token=False, cache_dir=str(cache / "hub"), local_dir=str(target),
                )

        weights, config = target / WEIGHTS, target / CONFIG
        if not weights.is_file() or weights.stat().st_size != SOURCE_BYTES:
            raise ValueError("VAE checkpoint is missing or has the wrong size")
        actual_sha256 = digest(weights, "sha256")
        if actual_sha256 != SOURCE_SHA256:
            raise ValueError("VAE checkpoint SHA256 mismatch")
        if not config.is_file():
            raise ValueError("VAE config is missing")
        blob_prefix = f"blob {config.stat().st_size}\0".encode()
        if digest(config, "sha1", blob_prefix) != CONFIG_BLOB_SHA1:
            raise ValueError("VAE config Git blob SHA1 mismatch")
        report.update({
            "status": "complete", "verified_sha256": actual_sha256,
            "config_blob_sha1": CONFIG_BLOB_SHA1,
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
        save_json(report_path, report)
        print(f"COMPLETE: public VAE verified ({SOURCE_BYTES} bytes); report {report_path}", flush=True)
        return 0
    except Exception as exc:
        # Download exceptions can include signed CDN URLs; do not print their text.
        status = getattr(getattr(exc, "response", None), "status_code", None)
        error = type(exc).__name__ + (f" (HTTP {status})" if status is not None else "")
        report.update({"status": "failed", "error_type": error,
                       "failed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        save_json(report_path, report)
        print(f"FAILED: {error}; resumable files retained; report {report_path}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
