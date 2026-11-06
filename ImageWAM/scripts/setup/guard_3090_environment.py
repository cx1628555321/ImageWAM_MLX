#!/usr/bin/env python3
"""Protect an inherited CUDA/PyTorch stack during an inference-only uv install."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
import re
import sys
from pathlib import Path


def normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def protected(name: str) -> bool:
    name = normalize(name)
    return name in {"torch", "torchvision", "torchaudio", "triton"} or name.startswith("nvidia-")


def read_json(path: str) -> dict:
    return json.loads(Path(path).read_text())


def snapshot(output: str, repo: str) -> None:
    import torch

    packages = {}
    for dist in metadata.distributions():
        name = normalize(dist.metadata.get("Name", ""))
        if protected(name) and name not in packages:
            active = metadata.distribution(name)
            packages[name] = {
                "version": active.version,
                "site_packages": str(Path(active.locate_file("")).resolve()),
            }
    manifests = {}
    for name in ("pyproject.toml", "uv.lock"):
        path = Path(repo) / name
        manifests[name] = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
    payload = {
        "executable": sys.executable,
        "prefix": sys.prefix,
        "base_prefix": sys.base_prefix,
        "python_version": list(sys.version_info[:3]),
        "torch": {
            "version": torch.__version__,
            "cuda": torch.version.cuda,
            "file": str(Path(torch.__file__).resolve()),
        },
        "protected_packages": dict(sorted(packages.items())),
        "project_manifests": manifests,
    }
    Path(output).write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Recorded {output}: torch {torch.__version__}, CUDA {torch.version.cuda}, {torch.__file__}")


def compare(before_path: str, after_path: str, overlay: bool) -> None:
    before, after = read_json(before_path), read_json(after_path)
    errors = []
    for field in ("python_version", "torch", "protected_packages", "project_manifests"):
        if before[field] != after[field]:
            errors.append(f"{field} changed")
    if overlay:
        if after["prefix"] == after["base_prefix"]:
            errors.append("target interpreter is not a virtual environment")
        if Path(after["base_prefix"]).resolve() != Path(before["prefix"]).resolve():
            errors.append("virtual environment is not derived from BASE_PYTHON")
    elif before["prefix"] != after["prefix"]:
        errors.append("base interpreter prefix changed")
    if errors:
        raise SystemExit("Environment guard failed: " + "; ".join(errors))
    print(f"Environment guard passed: {after_path}")


def exclusions(before_path: str, output: str) -> None:
    names = set(read_json(before_path)["protected_packages"])
    names.update({"torch", "torchvision", "torchaudio", "triton"})
    # robosuite requests opencv-python; the explicit headless wheel supplies cv2.
    # Installing both distributions would give them overlapping files.
    names.add("opencv-python")
    Path(output).write_text("\n".join(sorted(names)) + "\n")
    print(f"Excluded {len(names)} inherited/alternative packages from uv resolution")


def check_lock(path: str) -> None:
    count = 0
    for line_no, raw in enumerate(Path(path).read_text().splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        # uv pip compile is intentionally used without hashes or index directives.
        match = re.fullmatch(r"([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s;]+)(?:\s*;\s*.+)?", line)
        if match is None:
            raise SystemExit(f"Unsupported/unpinned requirement at {path}:{line_no}: {line}")
        if protected(match.group(1)):
            raise SystemExit(f"Refusing installation of inherited GPU package: {match.group(1)}")
        count += 1
    if not count:
        raise SystemExit(f"Refusing empty dependency lock: {path}")
    print(f"Dependency guard passed: {count} pinned packages, no protected GPU packages")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    capture = commands.add_parser("snapshot")
    capture.add_argument("output")
    capture.add_argument("--repo", required=True)
    verify = commands.add_parser("compare")
    verify.add_argument("before")
    verify.add_argument("after")
    verify.add_argument("--overlay", action="store_true")
    exclude = commands.add_parser("exclusions")
    exclude.add_argument("before")
    exclude.add_argument("output")
    lock = commands.add_parser("check-lock")
    lock.add_argument("path")
    args = parser.parse_args()
    if args.command == "snapshot":
        snapshot(args.output, args.repo)
    elif args.command == "compare":
        compare(args.before, args.after, args.overlay)
    elif args.command == "exclusions":
        exclusions(args.before, args.output)
    else:
        check_lock(args.path)


if __name__ == "__main__":
    main()
