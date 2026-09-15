from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

from .io_utils import ensure_run_dirs, write_json
from .paths import ACES_ROOT, DEFAULT_RUN_DIR, REPO_ROOT, SN2N_ROOT, VALID_ROOT


IMPORT_MODULES = ("torch", "numpy", "tifffile", "skimage", "scipy", "matplotlib", "pandas", "pywt", "einops")


def _git_info(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        return result
    inside = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "--is-inside-work-tree"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    result["inside_work_tree"] = inside.returncode == 0 and inside.stdout.strip() == "true"
    if not result["inside_work_tree"]:
        return result
    for key, args in {
        "commit": ["rev-parse", "HEAD"],
        "status_short": ["status", "--short"],
    }.items():
        try:
            proc = subprocess.run(
                ["git", "-C", str(path), *args],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            text = proc.stdout.strip() if proc.returncode == 0 else f"ERROR: {proc.stderr.strip()}"
            result[key] = text[:4000] + "\n...[truncated]" if len(text) > 4000 else text
        except Exception as exc:  # pragma: no cover - defensive environment capture
            result[key] = f"ERROR: {exc!r}"
    return result


def collect_environment() -> dict[str, Any]:
    output: dict[str, Any] = {
        "event": "aces_stage01_environment_check",
        "python": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "cwd": os.getcwd(),
        "repo_root": str(REPO_ROOT),
        "aces_root": str(ACES_ROOT),
        "paths": {
            "valid_root": str(VALID_ROOT),
            "sn2n_root": str(SN2N_ROOT),
        },
        "imports": {},
        "source_trees": {
            "repo": _git_info(REPO_ROOT),
            "valid": _git_info(VALID_ROOT),
            "sn2n": _git_info(SN2N_ROOT),
        },
    }

    for module_name in IMPORT_MODULES:
        try:
            module = importlib.import_module(module_name)
            output["imports"][module_name] = {
                "ok": True,
                "version": getattr(module, "__version__", "unknown"),
                "file": getattr(module, "__file__", None),
            }
        except Exception as exc:
            output["imports"][module_name] = {"ok": False, "error": repr(exc)}

    sys.path.insert(0, str(VALID_ROOT))
    sys.path.insert(0, str(SN2N_ROOT))
    for label, module_name in {
        "valid_sampling": "datasets.sampling",
        "sn2n_datagen": "SN2N.datagen",
    }.items():
        try:
            module = importlib.import_module(module_name)
            output["imports"][label] = {
                "ok": True,
                "file": getattr(module, "__file__", None),
            }
        except Exception as exc:
            output["imports"][label] = {"ok": False, "error": repr(exc)}

    try:
        import torch

        output["cuda"] = {
            "available": bool(torch.cuda.is_available()),
            "device_count": int(torch.cuda.device_count()),
            "current_device": int(torch.cuda.current_device()) if torch.cuda.is_available() else None,
            "devices": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
            "torch_version": torch.__version__,
        }
    except Exception as exc:
        output["cuda"] = {"available": False, "error": repr(exc)}

    output["all_required_imports_ok"] = all(item.get("ok", False) for item in output["imports"].values())
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ACES Stage 0 environment self-check.")
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR), help="Run directory for logs/environment.json")
    args = parser.parse_args(argv)

    dirs = ensure_run_dirs(args.run_dir)
    env = collect_environment()
    write_json(dirs["logs"] / "environment.json", env)
    print(json.dumps(env, indent=2, ensure_ascii=False))
    return 0 if env["all_required_imports_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
