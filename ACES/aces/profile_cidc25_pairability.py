"""T4b 第一切片（Plan v2 §4 item 8 / v1 T4b）：CIDC25 有 GT 数据的 pairability 曲线。

对 AI4Life CIDC25 钙成像验证集（F0=干净 GT，F1/F2/F3=三个噪声级）计算逐轴
N_a(d)/S_a(d)/Q_a(d)：

- S_a(d)：干净 GT 平滑后的结构自相关（Pearson，shift d）——比 mean9 proxy 更真。
- N_a(d)：GT 噪声（noisy - clean）的去相关度 1-|autocorr(shift d)|——真实噪声，
  不是 proxy 高通近似。
- 同时输出 noise_source="proxy"（高通近似噪声）作对照，用于把 T3b 的
  "proxy-N 不可靠"结论延伸到第三个数据集。

轴语义：CIDC25 形状 [t=1500, y=490, x=490]，dim0 是时间轴。为保持与
qmask/geometry 的 z/y/x 约定兼容，CSV 中 axis 用 "z"/"y"/"x"，另附
axis_label 列（"t"/"y"/"x"）；summary.json 里记录该映射。

只做 pairability 测量，不做任何权重训练（符合 CIDC25 验证集使用条款）。
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
from scipy import ndimage

from .io_utils import ensure_run_dirs, write_csv, write_json
from .profile_anisotropy import (
    AXIS_TO_DIM,
    REGIONS,
    _autocorr_noise,
    _masked_flat_pair,
    _pearson_structure,
    _safe01,
    _shift_pair,
    build_region_masks,
)

AXIS_LABELS = {"z": "t", "y": "y", "x": "x"}


def _load_volume(path: str | Path, frame_crop: int | None) -> np.ndarray:
    data = tifffile.imread(path)
    if data.ndim != 3:
        raise ValueError(f"Expected 3D stack, got {data.shape} for {path}")
    if frame_crop:
        data = data[: min(frame_crop, data.shape[0])]
    return np.asarray(data, dtype=np.float32)


def profile_cidc25(
    clean_path: Path,
    noisy_paths: list[Path],
    run_dir: Path,
    max_distance: int = 8,
    max_samples: int = 500_000,
    structure_sigma: float = 0.8,
    proxy_sigma: float = 0.8,
    fg_percentile: float = 75.0,
    frame_crop: int | None = None,
    seed: int = 3407,
) -> dict[str, Any]:
    ensure_run_dirs(run_dir)
    clean = _load_volume(clean_path, frame_crop)
    structure = ndimage.gaussian_filter(clean, sigma=structure_sigma)
    masks = build_region_masks(clean, fg_percentile)
    rng = np.random.default_rng(seed)

    rows: list[dict[str, Any]] = []
    per_level_summary: dict[str, dict[str, Any]] = {}
    shape = clean.shape
    for noisy_path in noisy_paths:
        noisy = _load_volume(noisy_path, frame_crop)
        if noisy.shape != shape:
            raise ValueError(f"Shape mismatch {noisy_path}: {noisy.shape} vs clean {shape}")
        level = noisy_path.stem
        gt_noise = noisy - clean
        proxy_noise = noisy - ndimage.gaussian_filter(noisy, sigma=proxy_sigma)
        # 可测结构：实际管线只能从噪声采集本身平滑得到结构参考（3p/nf 的
        # structure_proxy=smooth(mean9) 同理）。clean 平滑版（S）是理想上限。
        structure_measurable = ndimage.gaussian_filter(noisy, sigma=structure_sigma)

        level_rows = 0
        for axis in ("z", "y", "x"):
            axis_len = clean.shape[AXIS_TO_DIM[axis]]
            for distance in range(1, min(max_distance, axis_len - 1) + 1):
                structure_pair = _shift_pair(structure, masks["all"], axis, distance)
                for region in REGIONS:
                    region_pair = _shift_pair(masks[region], masks[region], axis, distance)
                    pair_mask = region_pair.left_mask & region_pair.right_mask

                    s_left, s_right = _masked_flat_pair(
                        structure_pair.left, structure_pair.right, pair_mask, max_samples, rng
                    )
                    structure_corr = _pearson_structure(s_left, s_right)
                    s_value = _safe01(structure_corr)

                    meas_pair = _shift_pair(structure_measurable, masks["all"], axis, distance)
                    m_left, m_right = _masked_flat_pair(
                        meas_pair.left, meas_pair.right, pair_mask, max_samples, rng
                    )
                    meas_corr = _pearson_structure(m_left, m_right)

                    base = {
                        "axis": axis,
                        "axis_label": AXIS_LABELS[axis],
                        "distance_px": distance,
                        "region": region,
                        "S": s_value,
                        "structure_corr": structure_corr,
                        "S_measurable": _safe01(meas_corr),
                        "structure_corr_measurable": meas_corr,
                        "noise_level": level,
                        "sample_count_structure": int(s_left.size),
                    }
                    for source, noise_map in (("gt", gt_noise), ("proxy", proxy_noise)):
                        pair = _shift_pair(noise_map, np.broadcast_to(masks["all"], noise_map.shape), axis, distance)
                        pair_mask_n = np.broadcast_to(pair_mask, pair.left.shape)
                        n_left, n_right = _masked_flat_pair(pair.left, pair.right, pair_mask_n, max_samples, rng)
                        corr = _autocorr_noise(n_left, n_right)
                        n_value = _safe01(1.0 - abs(corr))
                        q_value = _safe01(n_value * s_value) if np.isfinite(n_value) and np.isfinite(s_value) else float("nan")
                        rows.append({
                            **base,
                            "noise_source": source,
                            "noise_corr": corr,
                            "N": n_value,
                            "Q": q_value,
                            "sample_count_noise": int(n_left.size),
                        })
                        level_rows += 1
        per_level_summary[level] = {
            "noisy_path": str(noisy_path),
            "rows": level_rows,
            "gt_noise_std": float(gt_noise.std()),
        }
        del noisy, gt_noise, proxy_noise

    csv_path = run_dir / "metrics" / "pairability_curves.csv"
    write_csv(csv_path, rows, fieldnames=list(rows[0].keys()))

    summary = {
        "event": "aces_cidc25_pairability_complete",
        "run_dir": str(run_dir),
        "clean_path": str(clean_path),
        "noisy_paths": [str(p) for p in noisy_paths],
        "shape_tyx": list(shape),
        "axis_semantics": {"z": "t (time, dim0)", "y": "y", "x": "x"},
        "max_distance": max_distance,
        "max_samples_per_stat": max_samples,
        "structure_sigma": structure_sigma,
        "proxy_sigma": proxy_sigma,
        "fg_percentile": fg_percentile,
        "frame_crop": frame_crop,
        "seed": seed,
        "rows_total": len(rows),
        "per_level": per_level_summary,
        "note": "T4b 第一切片：GT 噪声源 pairability；不训练任何权重。",
    }
    write_json(run_dir / "metrics" / "cidc25_pairability_summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CIDC25 GT pairability profile (T4b slice 1).")
    parser.add_argument("--clean", required=True, help="Path to clean GT stack (e.g. F0.tif)")
    parser.add_argument("--noisy", required=True, nargs="+", help="Paths to noisy stacks (F1/F2/F3)")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--max-distance", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=500_000)
    parser.add_argument("--structure-sigma", type=float, default=0.8)
    parser.add_argument("--proxy-sigma", type=float, default=0.8)
    parser.add_argument("--fg-percentile", type=float, default=75.0)
    parser.add_argument("--frame-crop", type=int, default=None, help="Crop the time axis to N frames for a quick pass")
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args(argv)

    summary = profile_cidc25(
        clean_path=Path(args.clean),
        noisy_paths=[Path(p) for p in args.noisy],
        run_dir=Path(args.run_dir),
        max_distance=args.max_distance,
        max_samples=args.max_samples,
        structure_sigma=args.structure_sigma,
        proxy_sigma=args.proxy_sigma,
        fg_percentile=args.fg_percentile,
        frame_crop=args.frame_crop,
        seed=args.seed,
    )
    print(f"[cidc25] rows={summary['rows_total']} -> {summary['run_dir']}/metrics/pairability_curves.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
