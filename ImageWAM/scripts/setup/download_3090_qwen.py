#!/usr/bin/env python3
"""Download and verify the pinned public Qwen3-4B checkpoint inside ImageWAM."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import fnmatch
import hashlib
import json
import os
import re
from pathlib import Path
import time

REPO_ID = "Qwen/Qwen3-4B"
REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
PATTERNS = (
    "config.json", "generation_config.json", "model*.safetensors",
    "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json",
    "special_tokens_map.json", "vocab.json", "merges.txt", "added_tokens.json",
    "chat_template.jinja",
)


def save_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def summarize_exception(exc: Exception) -> str:
    # Hub exceptions may contain signed CDN URLs. Keep logs free of those values.
    status = getattr(getattr(exc, "response", None), "status_code", None)
    return type(exc).__name__ + (f" (HTTP {status})" if status is not None else "")


def retry(function, description: str, attempts: int = 6):
    for attempt in range(1, attempts + 1):
        try:
            return function()
        except Exception as exc:
            print(f"{description}: attempt {attempt}/{attempts} failed: {summarize_exception(exc)}", flush=True)
            if attempt == attempts:
                raise
            time.sleep(min(5 * attempt, 30))


def make_session(proxy: str | None):
    import requests

    session = requests.Session()
    session.trust_env = False
    if proxy:
        session.proxies = {"http": proxy, "https": proxy}
    return session


def download_file(item, target: Path, proxy: str, transport: str) -> None:
    from huggingface_hub import hf_hub_url

    path = target / item.rfilename
    expected_size = item.size
    if not isinstance(expected_size, int) or expected_size < 1:
        raise ValueError(f"Missing size for {item.rfilename}")
    if path.is_file() and path.stat().st_size == expected_size:
        return
    partial_dir = target / ".cache/imagewam_download"
    partial_dir.mkdir(parents=True, exist_ok=True)
    partial = partial_dir / f"{item.rfilename}.partial"
    if partial.exists() and partial.stat().st_size > expected_size:
        raise ValueError(f"Oversized partial file: {item.rfilename}")
    if transport == "modelscope":
        url = f"https://www.modelscope.cn/models/{REPO_ID}/resolve/master/{item.rfilename}"
        session = make_session(None)
    else:
        url = hf_hub_url(REPO_ID, filename=item.rfilename, revision=REVISION)
        session = make_session(proxy)
    failures = 0
    block_size = 64 * 1024 * 1024
    while not partial.exists() or partial.stat().st_size < expected_size:
        start = partial.stat().st_size if partial.exists() else 0
        end = min(start + block_size, expected_size) - 1
        try:
            with session.get(url, headers={"Range": f"bytes={start}-{end}"}, stream=True, timeout=(10, 30)) as response:
                response.raise_for_status()
                if response.status_code == 206:
                    content_range = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
                    if content_range is None or int(content_range[1]) != start or int(content_range[3]) != expected_size:
                        raise ValueError("Invalid partial response range")
                    response_end = int(content_range[2])
                elif response.status_code == 200 and start == 0:
                    response_end = expected_size - 1
                else:
                    raise ValueError("Server ignored a resumed Range request")
                written = start
                with partial.open("ab") as stream:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if not chunk:
                            continue
                        if written + len(chunk) > response_end + 1 or written + len(chunk) > expected_size:
                            raise ValueError("Download exceeded the advertised file/range size")
                        stream.write(chunk)
                        stream.flush()
                        written += len(chunk)
                if written != response_end + 1:
                    raise ValueError("Incomplete response body")
                failures = 0
                print(f"Downloaded {item.rfilename}: {written}/{expected_size} bytes", flush=True)
        except Exception as exc:
            progressed = partial.exists() and partial.stat().st_size > start
            failures = 0 if progressed else failures + 1
            print(f"Retry {item.rfilename}: {summarize_exception(exc)}, consecutive stalls={failures}", flush=True)
            if failures >= 12:
                raise
            time.sleep(min(2 * max(failures, 1), 10))
    partial.replace(path)
    session.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--proxy", default="http://127.0.0.1:7897")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--transport", choices=("huggingface", "modelscope"), default="huggingface",
                        help="ModelScope is a transport fallback; every file must still match the pinned Hugging Face revision.")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        parser.error("--workers must be between 1 and 4")
    root = args.repo_root.resolve()
    if not (root / "pyproject.toml").is_file():
        parser.error("--repo-root must point to the ImageWAM repository")
    target = root / "checkpoints/3090/Qwen3-4B"
    cache = root / ".cache/huggingface"
    artifact_dir = root / "artifacts/3090_setup"
    for path in (target, cache, artifact_dir):
        path.mkdir(parents=True, exist_ok=True)

    # Set these before importing huggingface_hub, which reads constants at import.
    os.environ.update({
        "HF_HOME": str(cache),
        "HF_HUB_CACHE": str(cache / "hub"),
        "HF_HUB_DISABLE_XET": "1",
        "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
        "HF_HUB_DOWNLOAD_TIMEOUT": "120",
        "HF_HUB_ETAG_TIMEOUT": "30",
        "HTTP_PROXY": args.proxy,
        "HTTPS_PROXY": args.proxy,
        "http_proxy": args.proxy,
        "https_proxy": args.proxy,
        "NO_PROXY": "localhost,127.0.0.1",
        "no_proxy": "localhost,127.0.0.1",
    })
    import huggingface_hub
    from huggingface_hub import HfApi, configure_http_backend
    from huggingface_hub.utils import logging as hub_logging

    hub_logging.set_verbosity_error()
    configure_http_backend(backend_factory=lambda: make_session(args.proxy))
    report_path = artifact_dir / f"qwen_download_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}_{os.getpid()}.json"
    report = {
        "repo_id": REPO_ID, "revision": REVISION, "target": str(target),
        "huggingface_hub": huggingface_hub.__version__, "status": "running",
        "transport": args.transport,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "files": [],
    }
    save_json(report_path, report)
    print(f"Checkpoint: {REPO_ID}@{REVISION}", flush=True)
    print(f"Target: {target}\nVerification report: {report_path}", flush=True)
    try:
        info = retry(lambda: HfApi(token=False).model_info(REPO_ID, revision=REVISION, files_metadata=True), "Fetch pinned metadata")
        if info.sha != REVISION:
            raise ValueError("Hub returned a different revision")
        selected = [item for item in info.siblings if any(fnmatch.fnmatchcase(item.rfilename, pattern) for pattern in PATTERNS)]
        filenames = {item.rfilename for item in selected}
        required = {"config.json", "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json"}
        if not required.issubset(filenames):
            raise ValueError("Pinned repository metadata is missing expected checkpoint/tokenizer files")
        if any("/" in name or "\\" in name for name in filenames):
            raise ValueError("Unexpected non-root checkpoint file")
        report["resolved_revision"] = info.sha
        report["expected_bytes"] = sum(item.size or 0 for item in selected)
        report["expected_files"] = [
            {"name": item.rfilename, "bytes": item.size, "git_blob_id": item.blob_id,
             "lfs_sha256": (item.lfs.get("sha256") if isinstance(item.lfs, dict) else getattr(item.lfs, "sha256", None)) if item.lfs else None}
            for item in selected
        ]
        save_json(report_path, report)
        print(f"Selected {len(selected)} files, {report['expected_bytes']} bytes; transport={args.transport}; Xet disabled.", flush=True)
        if not args.verify_only:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                list(pool.map(lambda item: download_file(item, target, args.proxy, args.transport), selected))

        for item in sorted(selected, key=lambda item: item.rfilename):
            path = target / item.rfilename
            if not path.is_file() or path.stat().st_size != item.size:
                raise ValueError(f"Missing file or size mismatch: {item.rfilename}")
            sha256 = hashlib.sha256()
            git_sha1 = hashlib.sha1(f"blob {path.stat().st_size}\0".encode())
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    sha256.update(chunk)
                    git_sha1.update(chunk)
            lfs = item.lfs
            expected_sha256 = (lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)) if lfs else None
            if expected_sha256:
                if sha256.hexdigest() != expected_sha256:
                    raise ValueError(f"LFS SHA256 mismatch: {item.rfilename}")
                verification = "lfs_sha256"
            else:
                if git_sha1.hexdigest() != item.blob_id:
                    raise ValueError(f"Git blob SHA1 mismatch: {item.rfilename}")
                verification = "git_blob_sha1"
            report["files"].append({"name": item.rfilename, "bytes": item.size, "sha256": sha256.hexdigest(), "verified_by": verification})
            save_json(report_path, report)
            print(f"Verified {item.rfilename}: {item.size} bytes ({verification})", flush=True)

        index = json.loads((target / "model.safetensors.index.json").read_text())
        shard_names = set(index["weight_map"].values())
        if not shard_names or not shard_names.issubset(filenames):
            raise ValueError("Safetensors index references unavailable shards")
        report.update({"status": "complete", "shards": sorted(shard_names), "tensor_count": len(index["weight_map"]), "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        save_json(report_path, report)
        print(f"COMPLETE: {len(report['files'])} files verified; {len(shard_names)} shards; report {report_path}", flush=True)
        return 0
    except Exception as exc:
        report.update({"status": "failed", "error_type": summarize_exception(exc), "failed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        save_json(report_path, report)
        print(f"FAILED: {summarize_exception(exc)}; resumable files retained; report {report_path}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
