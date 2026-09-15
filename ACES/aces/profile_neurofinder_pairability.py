from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile

from .io_utils import ensure_run_dirs, load_config, write_csv, write_json
from .neurofinder import load_roi_mask
from .profile_anisotropy import _distance_rows, _finite_rows


def _broadcast_masks(mask_yx: np.ndarray, frames: int) -> dict[str, np.ndarray]:
    foreground = np.broadcast_to(mask_yx.astype(bool), (frames, *mask_yx.shape))
    return {
        "all": np.ones((frames, *mask_yx.shape), dtype=bool),
        "foreground": foreground,
        "background": ~foreground,
    }


def _write_q_plot(rows: list[dict[str, Any]], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for axis in ("x", "y", "z"):
        selected = [
            row for row in rows
            if row["depth_bin"] == "global" and row["region"] == "foreground"
            and row["noise_source"] == "proxy" and row["axis"] == axis
        ]
        selected.sort(key=lambda row: int(row["distance_px"]))
        ax.plot([row["distance_px"] for row in selected], [row["Q"] for row in selected], marker="o", label=axis)
    ax.set_xlabel("distance (px for x/y, frames for z)")
    ax.set_ylabel("foreground proxy Q")
    ax.set_ylim(0, 1.02)
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def run(config_path: str | Path) -> dict[str, Any]:
    cfg = load_config(config_path)
    run_dir = Path(cfg["run_dir"])
    dirs = ensure_run_dirs(run_dir)

    stack = tifffile.imread(cfg["data"]["stack_path"]).astype(np.float32)
    if stack.ndim != 3 or min(stack.shape) < 2:
        raise ValueError(f"Expected neurofinder stack as [frame,y,x], got {stack.shape}")
    original_frames = int(stack.shape[0])
    max_frames = int(cfg["analysis"].get("max_frames", 0))
    if max_frames > 0 and stack.shape[0] > max_frames:
        indices = np.linspace(0, stack.shape[0] - 1, max_frames, dtype=np.int64)
        stack = np.ascontiguousarray(stack[indices])
    else:
        indices = np.arange(stack.shape[0], dtype=np.int64)
    roi_mask = load_roi_mask(cfg["data"]["regions_path"], tuple(stack.shape[1:]))
    masks = _broadcast_masks(roi_mask, stack.shape[0])

    analysis_cfg = {
        "seed": int(cfg.get("seed", 3407)),
        "physical_spacing_um": {"x": 1.0, "y": 1.0, "z": 1.0},
        "analysis": {
            "max_distance": int(cfg["analysis"].get("max_distance", 8)),
            "max_samples_per_stat": int(cfg["analysis"].get("max_samples_per_stat", 200_000)),
            "structure_sigma": float(cfg["analysis"].get("structure_sigma", 0.8)),
            "proxy_sigma": float(cfg["analysis"].get("proxy_sigma", 0.8)),
        },
    }
    repeats_view = stack[None, ...]
    rows = _distance_rows(repeats_view, stack, masks, analysis_cfg)
    _finite_rows(rows)
    write_csv(
        dirs["metrics"] / "pairability_curves.csv",
        rows,
        fieldnames=list(rows[0].keys()),
    )
    _write_q_plot(rows, dirs["figures"] / "q_curves_proxy_xyz.png")
    proxy_rows = [row for row in rows if row["noise_source"] == "proxy"]
    geometry = {}
    threshold = float(cfg["analysis"].get("q_threshold", 0.6))
    for axis in ("x", "y", "z"):
        values = {
            int(row["distance_px"]): float(row["Q"])
            for row in proxy_rows
            if row["depth_bin"] == "global" and row["axis"] == axis and row["region"] == "foreground"
        }
        valid = [distance for distance, value in values.items() if value >= threshold]
        geometry[axis] = max(valid) if valid else 1
    summary = {
        "event": "aces_neurofinder_proxy_pairability",
        "layout": "[frame,y,x]; z axis is frame/time, not optical depth.",
        "stack_shape_fyx": list(stack.shape),
        "original_frame_count": original_frames,
        "sampled_frame_indices": [int(indices[0]), int(indices[-1]), int(len(indices))],
        "roi_foreground_pixels": int(roi_mask.sum()),
        "noise_source_for_stage2": "proxy",
        "metric_for_stage2": "Q",
        "q_threshold": threshold,
        "derived_radii": geometry,
        "calibration": "No physical calibration found in local TIFF tags; distances are unitless pixels/frames.",
    }
    write_json(dirs["logs"] / "summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Profile neurofinder proxy pairability for ACES Stage 2.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    result = run(args.config)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
