#!/usr/bin/env python3
"""Prepare repository-local LIBERO paths without importing its interactive setup."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--libero-dir", type=Path, default=REPO_ROOT / "third_party/LIBERO")
    parser.add_argument("--config-dir", type=Path, default=REPO_ROOT / ".libero")
    parser.add_argument("--datasets-dir", type=Path, default=REPO_ROOT / "data/libero")
    parser.add_argument("--patch-init-state-load", action="store_true",
                        help="Adapt only the repository-local LIBERO copy for trusted NumPy initial states.")
    args = parser.parse_args()
    libero_dir = args.libero_dir.expanduser().resolve(strict=True)
    benchmark_root = libero_dir / "libero/libero"
    init_file = benchmark_root / "__init__.py"
    if "LIBERO_CONFIG_PATH" not in init_file.read_text(encoding="utf-8"):
        raise ValueError(f"Unsupported LIBERO configuration interface: {init_file}")
    paths = {
        "benchmark_root": benchmark_root,
        "bddl_files": benchmark_root / "bddl_files",
        "init_states": benchmark_root / "init_files",
        "datasets": args.datasets_dir.expanduser().resolve(),
        "assets": benchmark_root / "assets",
    }
    for key, path in paths.items():
        if key != "datasets" and not path.is_dir():
            raise FileNotFoundError(f"Missing LIBERO {key}: {path}")
    paths["datasets"].mkdir(parents=True, exist_ok=True)
    expected = {key: str(path) for key, path in paths.items()}
    config_dir = args.config_dir.expanduser().resolve()
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "config.yaml"
    if os.path.lexists(config_path):
        import yaml
        actual = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if actual != expected:
            raise ValueError(f"Existing config differs; select a separate LIBERO_CONFIG_PATH instead of overwriting {config_path}")
    else:
        fd, temp_name = tempfile.mkstemp(prefix=".config-", suffix=".tmp", dir=config_dir)
        temp_path = Path(temp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                # JSON is valid YAML and keeps the five paths safely quoted.
                json.dump(expected, stream, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.link(temp_path, config_path)
        finally:
            temp_path.unlink(missing_ok=True)

    patched = False
    if args.patch_init_state_load:
        if not libero_dir.is_relative_to(REPO_ROOT):
            raise ValueError("Only a LIBERO copy inside this repository may be patched.")
        benchmark_file = benchmark_root / "benchmark/__init__.py"
        old = "init_states = torch.load(init_states_path)"
        new = "init_states = torch.load(init_states_path, weights_only=False)"
        content = benchmark_file.read_text(encoding="utf-8")
        if old in content:
            if content.count(old) != 1:
                raise ValueError(f"Ambiguous initial-state load expression: {benchmark_file}")
            updated = content.replace(old, new)
            fd, temp_name = tempfile.mkstemp(prefix=".benchmark-", suffix=".tmp", dir=benchmark_file.parent)
            temp_path = Path(temp_name)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as stream:
                    stream.write(updated)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.chmod(temp_path, benchmark_file.stat().st_mode & 0o777)
                os.replace(temp_path, benchmark_file)
            finally:
                temp_path.unlink(missing_ok=True)
            patched = True
        elif new not in content:
            raise ValueError(f"Unknown initial-state loading code: {benchmark_file}")
    print(json.dumps({"config": str(config_path), "init_state_load_patched": patched,
                      "paths": expected}, indent=2))


if __name__ == "__main__":
    main()
