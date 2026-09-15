from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np

from .io_utils import ensure_run_dirs, read_json
from .paths import DEFAULT_RUN_DIR


def _read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _axis_table(rows: list[dict[str, Any]], source: str = "repeat", region: str = "all") -> str:
    lines = ["| axis | mean Q d1-d3 | Q(d=1) | Q(d=8) | mean N d1-d3 | mean S d1-d3 |", "|---|---:|---:|---:|---:|---:|"]
    for axis in ("x", "y", "z"):
        selected = [
            row
            for row in rows
            if row["axis"] == axis and row["noise_source"] == source and row["region"] == region
        ]
        d13 = [row for row in selected if int(row["distance_px"]) <= 3]
        q1 = next((_float(row["Q"]) for row in selected if int(row["distance_px"]) == 1), float("nan"))
        q8 = next((_float(row["Q"]) for row in selected if int(row["distance_px"]) == 8), float("nan"))
        lines.append(
            "| {axis} | {qmean:.4f} | {q1:.4f} | {q8:.4f} | {nmean:.4f} | {smean:.4f} |".format(
                axis=axis,
                qmean=float(np.nanmean([_float(row["Q"]) for row in d13])),
                q1=q1,
                q8=q8,
                nmean=float(np.nanmean([_float(row["N"]) for row in d13])),
                smean=float(np.nanmean([_float(row["S"]) for row in d13])),
            )
        )
    return "\n".join(lines)


def _corr_table(correlations: dict[str, Any]) -> str:
    lines = ["| axis/region | Pearson N | Spearman N | Pearson Q | Spearman Q |", "|---|---:|---:|---:|---:|"]
    for axis in ("x", "y", "z"):
        for region in ("all", "foreground", "background"):
            item = correlations.get(f"{axis}_{region}", {})
            lines.append(
                "| {key} | {pn} | {sn} | {pq} | {sq} |".format(
                    key=f"{axis}/{region}",
                    pn=_fmt_corr(item.get("pearson_N")),
                    sn=_fmt_corr(item.get("spearman_N")),
                    pq=_fmt_corr(item.get("pearson_Q")),
                    sq=_fmt_corr(item.get("spearman_Q")),
                )
            )
    return "\n".join(lines)


def _fmt_corr(item: Any) -> str:
    if not isinstance(item, dict) or item.get("r") is None:
        return "n/a"
    return f"{float(item['r']):.3f}"


def _imports_summary(env: dict[str, Any]) -> str:
    imports = env.get("imports", {})
    required = ("torch", "numpy", "tifffile", "skimage", "scipy", "matplotlib", "pandas", "pywt", "einops", "valid_sampling", "sn2n_datagen")
    lines = ["| module | status | version/file |", "|---|---|---|"]
    for name in required:
        item = imports.get(name, {})
        status = "ok" if item.get("ok") else "FAILED"
        version = item.get("version") or item.get("file") or item.get("error") or ""
        lines.append(f"| {name} | {status} | `{version}` |")
    return "\n".join(lines)


def write_report(run_dir: Path) -> Path:
    dirs = ensure_run_dirs(run_dir)
    rows = _read_rows(dirs["metrics"] / "pairability_curves.csv")
    correlations = read_json(dirs["metrics"] / "proxy_repeat_correlations.json")
    summary = read_json(dirs["logs"] / "profile_summary.json")
    manifest = read_json(dirs["logs"] / "data_manifest.json")
    env = read_json(dirs["logs"] / "environment.json")

    gate = "PASS" if summary.get("gate_anisotropy_observed") else "REVIEW"
    gate_metric = summary.get("gate_metric", "Q")
    gate_region = summary.get("gate_region", "foreground")
    gate_ratio = float(summary.get("gate_xy_to_z_ratio", float("nan")))
    all_ratio = float(summary.get("all_region_q_ratio_d1_d3", float("nan")))
    fg_s_ratio = float(summary.get("foreground_structure_xy_to_z_ratio", float("nan")))
    import_ok = "PASS" if env.get("all_required_imports_ok") else "FAIL"
    report = f"""# ACES Stage 0-1 Pairability Report

## Gate Summary

- Environment/import gate: **{import_ok}**
- Anisotropy gate (**{gate_region}** region, metric **{gate_metric}**): **{gate}**
- Gate XY/Z ratio d=1..3 (foreground {gate_metric}): **{gate_ratio:.4f}**
- Foreground structure (S) XY/Z ratio d=1..3: **{fg_s_ratio:.4f}**
- All-region Q XY/Z ratio d=1..3 (diluted, for reference only): **{all_ratio:.4f}**
- Repeat stack shape `[repeat,z,y,x]`: `{manifest.get("loaded_shape_repeat_z_y_x")}`
- Foreground fraction: **{float(manifest.get("foreground_fraction", 0.0)):.4f}**

The gate is judged on the **foreground** region: the background (~86% of voxels)
is nearly structure-free and dilutes the all-region ratio toward 1.0, so an
all-region gate produces a false negative. Structure continuity S is the
physically robust anisotropy signal (z structure decays fast under the ~4um
z-step); noise-independence N can be weak or reversed on z and must not be read
as "no anisotropy". If the gate is `REVIEW`, Stage 2 sampling-only training
should not start until the pairability premise is re-checked.

## Environment

{_imports_summary(env)}

CUDA devices: `{env.get("cuda", {}).get("devices")}`

## Global Pairability

Repeat-noise branch, **foreground** voxels (gate region):

{_axis_table(rows, source="repeat", region="foreground")}

Repeat-noise branch, all voxels (diluted by background, reference only):

{_axis_table(rows, source="repeat", region="all")}

Proxy-noise branch, foreground voxels:

{_axis_table(rows, source="proxy", region="foreground")}

## Proxy vs Repeat

{_corr_table(correlations)}

## Outputs

- Metrics: `metrics/pairability_curves.csv`
- Depth-conditioned metrics: `metrics/depth_conditioned_pairability.csv`
- Correlations: `metrics/proxy_repeat_correlations.json`
- Figures: `figures/q_curves_xyz.png`, `figures/proxy_vs_repeat.png`, `figures/depth_q_heatmaps.png`
"""
    output_path = dirs["report"] / "stage01_pairability_report.md"
    output_path.write_text(report, encoding="utf-8")
    return output_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write ACES Stage 0-1 report.")
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR), help="Run directory produced by profile_anisotropy.")
    args = parser.parse_args(argv)
    output = write_report(Path(args.run_dir).resolve())
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
