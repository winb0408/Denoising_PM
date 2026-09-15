from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .io_utils import read_json
from .sampling_aniso import VALID_MODES


def _fmt(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except Exception:
        return "n/a"


def write_stage2_report(run_dir: str | Path) -> Path:
    root = Path(run_dir)
    summary_path = root / "logs" / "summary.json"
    summary = read_json(summary_path) if summary_path.exists() else {"modes": []}
    summary_modes = [mode_summary["mode"] for mode_summary in summary.get("modes", [])]
    discovered_modes = [mode for mode in VALID_MODES if (root / mode).is_dir()]
    modes = list(dict.fromkeys([*summary_modes, *discovered_modes]))
    rows = []
    for mode in modes:
        mode_summary_path = root / mode / "logs" / "summary.json"
        mode_summary = read_json(mode_summary_path)
        metrics = read_json(root / mode / "metrics" / "eval_metrics.json")
        sampler = read_json(root / mode / "logs" / "sampler_metadata.json")
        first_call = sampler["calls"][0] if sampler["calls"] else {}
        # 对齐评估仅下采样档产出；偏移档为 None，报告中显示 "-"（stride=1，等同全分辨率列）。
        aligned = mode_summary.get("eval_regions_aligned")
        rows.append({
            "mode": mode,
            "checkpoint": mode_summary["checkpoint"],
            "train_batches": mode_summary["train_batches"],
            "final_loss": mode_summary["final_loss"],
            "global_psnr": metrics["regions"]["all"]["psnr_db"],
            "foreground_psnr": metrics["regions"]["foreground"]["psnr_db"],
            "background_psnr": metrics["regions"]["background"]["psnr_db"],
            "sbr_pred": (
                metrics["regions"]["foreground"].get("pred_mean", float("nan"))
                / metrics["regions"]["background"].get("pred_mean", float("nan"))
            ),
            "background_pred_std": metrics["regions"]["background"].get("pred_std"),
            "foreground_voxels": metrics["regions"]["foreground"]["n_voxels"],
            "aligned_fg_psnr": aligned["regions"]["foreground"]["psnr_db"] if aligned else None,
            "aligned_stride": aligned["downsample_stride_zyx"] if aligned else None,
            "selected_patches": metrics["selected_patch_indices"],
            "offsets": first_call.get("offsets_zyx", []),
            "radii": first_call.get("radii_px"),
        })

    table = [
        "| mode | batches | final loss | PSNR all | PSNR fg | PSNR bg | SBR pred | bg pred std | PSNR fg (aligned) | align stride | fg voxels | eval patches | offsets zyx | radii px |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---|---|---|",
    ]
    for row in rows:
        aligned_psnr = _fmt(row["aligned_fg_psnr"]) if row["aligned_fg_psnr"] is not None else "-"
        aligned_stride = f"`{row['aligned_stride']}`" if row["aligned_stride"] is not None else "-"
        table.append(
            f"| {row['mode']} | {row['train_batches']} | {_fmt(row['final_loss'])} | {_fmt(row['global_psnr'])} | "
            f"{_fmt(row['foreground_psnr'])} | {_fmt(row['background_psnr'])} | {_fmt(row['sbr_pred'])} | "
            f"{_fmt(row['background_pred_std'])} | {aligned_psnr} | {aligned_stride} | {row['foreground_voxels']} | "
            f"`{row['selected_patches']}` | `{row['offsets']}` | `{row['radii']}` |"
        )
    report = f"""# ACES Stage 2 Sampling-Only Report

## Summary

- Run directory: `{root}`
- Modes: `{[row['mode'] for row in rows]}`
- This stage fixes VALID CRN + VALID loss and changes only the sampling operator.
- Metrics are patch-level mean9 proxy diagnostics and are split into all / foreground / background regions.
- `PSNR fg (aligned)` is an extra evaluation for downsampling modes (valid/sn2n_xy/iso3d): the model,
  target and mask are stride-downsampled to the resolution the model actually trained on, removing the
  train/test scale mismatch that the full-resolution `PSNR fg` column carries. Offset modes
  (empirical_block/ellipsoid) train at full resolution (stride=1) so their aligned value equals the
  full-resolution one and is shown as `-`. Both columns are kept so the two views can be compared.

## Results

{chr(10).join(table)}

## Artifacts

Each mode directory contains:

- `config.json`
- `checkpoints/<mode>_final.pth`
- `logs/train_losses.csv`
- `logs/sampler_metadata.json`
- `metrics/eval_metrics.json`
"""
    output = root / "report" / "stage02_sampling_report.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write ACES Stage 2 report.")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args(argv)
    output = write_stage2_report(args.run_dir)
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
