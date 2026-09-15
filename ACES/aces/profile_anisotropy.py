from __future__ import annotations

import argparse
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from scipy import ndimage, stats

from .env_check import collect_environment
from .io_utils import ensure_run_dirs, load_config, write_csv, write_json
from .paths import DEFAULT_RUN_DIR, MEAN9_PROXY, REPEATS_ROOT


AXIS_TO_DIM = {"z": 0, "y": 1, "x": 2}
REGIONS = ("all", "foreground", "background")
NOISE_SOURCES = ("proxy", "repeat")


@dataclass(frozen=True)
class PairArrays:
    left: np.ndarray
    right: np.ndarray
    left_mask: np.ndarray
    right_mask: np.ndarray


def _as_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def _slice_from_config(crop: dict[str, Any] | None, key: str, limit: int) -> slice:
    if not crop or key not in crop or crop[key] is None:
        return slice(None)
    value = crop[key]
    if isinstance(value, int):
        return slice(0, min(value, limit))
    if isinstance(value, (list, tuple)) and len(value) == 2:
        start = 0 if value[0] is None else int(value[0])
        stop = limit if value[1] is None else int(value[1])
        return slice(max(0, start), min(stop, limit))
    raise ValueError(f"Invalid crop entry for {key}: {value!r}")


def load_repeats(repeats_root: Path, crop: dict[str, Any] | None = None) -> np.ndarray:
    paths = sorted(repeats_root.glob("*.tif"))
    if len(paths) != 9:
        raise FileNotFoundError(f"Expected 9 repeat TIFFs under {repeats_root}, found {len(paths)}")
    first = tifffile.imread(paths[0])
    if first.ndim != 3:
        raise ValueError(f"Expected each repeat to be a 3D z/y/x volume, got {first.shape}")
    z_slice = _slice_from_config(crop, "z", first.shape[0])
    y_slice = _slice_from_config(crop, "y", first.shape[1])
    x_slice = _slice_from_config(crop, "x", first.shape[2])
    repeats = []
    for path in paths:
        volume = tifffile.imread(path)[z_slice, y_slice, x_slice]
        if volume.ndim != 3 or min(volume.shape) < 2:
            raise ValueError(f"3D volume collapsed after crop for {path}: {volume.shape}")
        repeats.append(np.asarray(volume, dtype=np.float32))
    stack = np.stack(repeats, axis=0)
    if stack.ndim != 4 or stack.shape[0] != 9 or min(stack.shape[1:]) < 2:
        raise ValueError(f"Expected [repeat,z,y,x] after loading, got {stack.shape}")
    return stack


def _normalize_unit(volume: np.ndarray) -> np.ndarray:
    data = np.asarray(volume, dtype=np.float32)
    finite = np.isfinite(data)
    if not finite.any():
        raise ValueError("Cannot normalize an all-nonfinite volume")
    lo, hi = np.percentile(data[finite], [1.0, 99.8])
    if hi <= lo:
        hi = float(data[finite].max())
        lo = float(data[finite].min())
    scale = max(float(hi - lo), 1e-6)
    return np.clip((data - float(lo)) / scale, 0.0, 1.0).astype(np.float32)


def build_region_masks(mean9: np.ndarray, percentile: float) -> dict[str, np.ndarray]:
    normalized = _normalize_unit(mean9)
    threshold = float(np.percentile(normalized, percentile))
    foreground = normalized > threshold
    if foreground.any():
        foreground = ndimage.binary_opening(foreground, structure=np.ones((1, 3, 3), dtype=bool))
        foreground = ndimage.binary_closing(foreground, structure=np.ones((1, 3, 3), dtype=bool))
    if foreground.sum() < max(128, foreground.size // 1000):
        foreground = normalized > float(np.percentile(normalized, max(50.0, percentile - 10.0)))
    foreground = np.asarray(foreground, dtype=bool)
    return {
        "all": np.ones_like(foreground, dtype=bool),
        "foreground": foreground,
        "background": ~foreground,
    }


def _shift_pair(array: np.ndarray, mask: np.ndarray, axis: str, distance: int) -> PairArrays:
    dim = AXIS_TO_DIM[axis] + (array.ndim - 3)
    if dim < 0 or dim >= array.ndim:
        raise ValueError(f"Cannot shift axis {axis!r} for array shape {array.shape}")
    left_slices = [slice(None)] * array.ndim
    right_slices = [slice(None)] * array.ndim
    left_slices[dim] = slice(0, -distance)
    right_slices[dim] = slice(distance, None)
    return PairArrays(
        left=array[tuple(left_slices)],
        right=array[tuple(right_slices)],
        left_mask=mask[tuple(left_slices)],
        right_mask=mask[tuple(right_slices)],
    )


def _subsample(values_a: np.ndarray, values_b: np.ndarray, max_samples: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    if values_a.size <= max_samples:
        return values_a, values_b
    indices = rng.choice(values_a.size, size=max_samples, replace=False)
    return values_a[indices], values_b[indices]


def _masked_flat_pair(
    left: np.ndarray,
    right: np.ndarray,
    pair_mask: np.ndarray,
    max_samples: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if pair_mask.shape != left.shape or pair_mask.shape != right.shape:
        raise ValueError(f"Mask/array shape mismatch: {pair_mask.shape}, {left.shape}, {right.shape}")
    valid_fraction = float(pair_mask.mean())
    if valid_fraction <= 0.0:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32)
    valid_estimate = int(pair_mask.size * valid_fraction)
    if valid_estimate <= max_samples * 3:
        valid = pair_mask & np.isfinite(left) & np.isfinite(right)
        if not valid.any():
            return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32)
        a = np.asarray(left[valid], dtype=np.float32)
        b = np.asarray(right[valid], dtype=np.float32)
        return _subsample(a, b, max_samples=max_samples, rng=rng)

    chosen_left: list[np.ndarray] = []
    chosen_right: list[np.ndarray] = []
    chosen_count = 0
    attempts = 0
    while chosen_count < max_samples and attempts < 24:
        attempts += 1
        need = max_samples - chosen_count
        draw_count = int(max(need * 2, math.ceil(need / max(valid_fraction, 1e-6) * 1.25)))
        draw_count = min(draw_count, 2_000_000)
        coords = tuple(rng.integers(0, dim, size=draw_count, dtype=np.int64) for dim in pair_mask.shape)
        keep = pair_mask[coords]
        if not keep.any():
            continue
        left_values = left[coords][keep]
        right_values = right[coords][keep]
        finite = np.isfinite(left_values) & np.isfinite(right_values)
        left_values = left_values[finite]
        right_values = right_values[finite]
        if left_values.size == 0:
            continue
        if left_values.size > need:
            left_values = left_values[:need]
            right_values = right_values[:need]
        chosen_left.append(np.asarray(left_values, dtype=np.float32))
        chosen_right.append(np.asarray(right_values, dtype=np.float32))
        chosen_count += int(left_values.size)
    if chosen_count == 0:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32)
    return np.concatenate(chosen_left)[:max_samples], np.concatenate(chosen_right)[:max_samples]


def _autocorr_noise(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 2:
        return 0.0
    a = left.astype(np.float64, copy=False)
    b = right.astype(np.float64, copy=False)
    a = a - float(a.mean())
    b = b - float(b.mean())
    denom = math.sqrt(float(np.mean(a * a)) * float(np.mean(b * b)))
    if denom <= 1e-12:
        return 0.0
    return float(np.mean(a * b) / denom)


def _pearson_structure(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 2:
        return 0.0
    a = left.astype(np.float64, copy=False)
    b = right.astype(np.float64, copy=False)
    a = a - float(a.mean())
    b = b - float(b.mean())
    denom = math.sqrt(float(np.sum(a * a)) * float(np.sum(b * b)))
    if denom <= 1e-12:
        return 0.0
    return float(np.sum(a * b) / denom)


def _safe01(value: float) -> float:
    if not np.isfinite(value):
        return float("nan")
    return float(np.clip(value, 0.0, 1.0))


def _distance_rows(
    repeats: np.ndarray,
    mean9: np.ndarray,
    masks: dict[str, np.ndarray],
    cfg: dict[str, Any],
    depth_bin: str | None = None,
) -> list[dict[str, Any]]:
    max_distance = int(cfg["analysis"].get("max_distance", 8))
    max_samples = int(cfg["analysis"].get("max_samples_per_stat", 500_000))
    seed = int(cfg.get("seed", 3407))
    spacing = cfg["physical_spacing_um"]
    smooth_sigma = float(cfg["analysis"].get("structure_sigma", 0.8))
    proxy_sigma = float(cfg["analysis"].get("proxy_sigma", 0.8))
    # NOTE: Python's built-in hash() is salted per-process (PYTHONHASHSEED), so
    # using it here would make the depth-bin RNG offset non-reproducible across
    # runs. Use a stable hash (blake2b) so results are bit-reproducible.
    if depth_bin is None:
        seed_offset = 0
    else:
        digest = hashlib.blake2b(depth_bin.encode("utf-8"), digest_size=8).digest()
        seed_offset = int.from_bytes(digest, "big") % 100_000
    rng = np.random.default_rng(seed + seed_offset)

    repeat_noise = repeats - repeats.mean(axis=0, keepdims=True)
    proxy_source = repeats[0]
    proxy_smooth = ndimage.gaussian_filter(proxy_source, sigma=proxy_sigma)
    proxy_noise = proxy_source - proxy_smooth
    structure_proxy = ndimage.gaussian_filter(mean9, sigma=smooth_sigma)

    rows: list[dict[str, Any]] = []
    for axis in ("x", "y", "z"):
        axis_len = mean9.shape[AXIS_TO_DIM[axis]]
        for distance in range(1, min(max_distance, axis_len - 1) + 1):
            structure_pair = _shift_pair(structure_proxy, masks["all"], axis, distance)
            for region in REGIONS:
                region_pair = _shift_pair(masks[region], masks[region], axis, distance)
                pair_mask = region_pair.left_mask & region_pair.right_mask
                s_left, s_right = _masked_flat_pair(
                    structure_pair.left,
                    structure_pair.right,
                    pair_mask,
                    max_samples=max_samples,
                    rng=rng,
                )
                structure_corr = _pearson_structure(s_left, s_right)
                s_value = _safe01(structure_corr)

                proxy_pair = _shift_pair(proxy_noise, masks["all"], axis, distance)
                p_left, p_right = _masked_flat_pair(proxy_pair.left, proxy_pair.right, pair_mask, max_samples, rng)
                proxy_corr = _autocorr_noise(p_left, p_right)
                proxy_n = _safe01(1.0 - abs(proxy_corr))

                repeat_pair = _shift_pair(repeat_noise, np.broadcast_to(masks["all"], repeat_noise.shape), axis, distance)
                repeat_mask = np.broadcast_to(pair_mask, repeat_pair.left.shape)
                r_left, r_right = _masked_flat_pair(repeat_pair.left, repeat_pair.right, repeat_mask, max_samples, rng)
                repeat_corr = _autocorr_noise(r_left, r_right)
                repeat_n = _safe01(1.0 - abs(repeat_corr))

                base = {
                    "depth_bin": depth_bin or "global",
                    "axis": axis,
                    "distance_px": distance,
                    "distance_um": float(distance * spacing[axis]),
                    "region": region,
                    "S": s_value,
                    "structure_corr": structure_corr,
                    "proxy_noise_corr": proxy_corr,
                    "repeat_noise_corr": repeat_corr,
                    "sample_count_structure": int(s_left.size),
                    "sample_count_proxy": int(p_left.size),
                    "sample_count_repeat": int(r_left.size),
                }
                for source, n_value in (("proxy", proxy_n), ("repeat", repeat_n)):
                    q_value = _safe01(n_value * s_value) if np.isfinite(n_value) and np.isfinite(s_value) else float("nan")
                    rows.append({**base, "noise_source": source, "N": n_value, "Q": q_value})
    return rows


def _depth_rows(repeats: np.ndarray, mean9: np.ndarray, masks: dict[str, np.ndarray], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    bins = int(cfg["analysis"].get("depth_bins", 5))
    if bins <= 0:
        return []
    z_count = mean9.shape[0]
    edges = np.linspace(0, z_count, bins + 1, dtype=int)
    rows: list[dict[str, Any]] = []
    for index in range(bins):
        start, stop = int(edges[index]), int(edges[index + 1])
        if stop - start < 2:
            continue
        label = f"z{start:03d}_{stop - 1:03d}"
        local_repeats = repeats[:, start:stop]
        local_mean = mean9[start:stop]
        local_masks = {name: mask[start:stop] for name, mask in masks.items()}
        rows.extend(_distance_rows(local_repeats, local_mean, local_masks, cfg, depth_bin=label))
    return rows


def _finite_rows(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        for key in ("N", "S", "Q"):
            value = row[key]
            if not np.isfinite(value):
                raise FloatingPointError(f"Non-finite {key} in row: {row}")
            if value < -1e-6 or value > 1.0 + 1e-6:
                raise ValueError(f"{key} outside [0,1] in row: {row}")


def compute_correlations(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {"event": "aces_proxy_repeat_correlation"}
    for axis in ("x", "y", "z"):
        for region in REGIONS:
            proxy = [
                row
                for row in rows
                if row["depth_bin"] == "global"
                and row["axis"] == axis
                and row["region"] == region
                and row["noise_source"] == "proxy"
            ]
            repeat = [
                row
                for row in rows
                if row["depth_bin"] == "global"
                and row["axis"] == axis
                and row["region"] == region
                and row["noise_source"] == "repeat"
            ]
            proxy_by_d = {int(row["distance_px"]): row for row in proxy}
            repeat_by_d = {int(row["distance_px"]): row for row in repeat}
            distances = sorted(set(proxy_by_d) & set(repeat_by_d))
            n_proxy = np.asarray([proxy_by_d[d]["N"] for d in distances], dtype=np.float64)
            n_repeat = np.asarray([repeat_by_d[d]["N"] for d in distances], dtype=np.float64)
            q_proxy = np.asarray([proxy_by_d[d]["Q"] for d in distances], dtype=np.float64)
            q_repeat = np.asarray([repeat_by_d[d]["Q"] for d in distances], dtype=np.float64)
            key = f"{axis}_{region}"
            output[key] = {
                "distance_px": distances,
                "pearson_N": _corr_value(stats.pearsonr, n_proxy, n_repeat),
                "spearman_N": _corr_value(stats.spearmanr, n_proxy, n_repeat),
                "pearson_Q": _corr_value(stats.pearsonr, q_proxy, q_repeat),
                "spearman_Q": _corr_value(stats.spearmanr, q_proxy, q_repeat),
            }
    return output


def _corr_value(func: Any, left: np.ndarray, right: np.ndarray) -> dict[str, float | None]:
    valid = np.isfinite(left) & np.isfinite(right)
    if int(valid.sum()) < 3 or float(np.std(left[valid])) <= 1e-12 or float(np.std(right[valid])) <= 1e-12:
        return {"r": None, "p": None}
    result = func(left[valid], right[valid])
    return {"r": float(result.statistic), "p": float(result.pvalue)}


def plot_q_curves(rows: list[dict[str, Any]], output_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    colors = {"x": "#1f77b4", "y": "#2ca02c", "z": "#d62728"}
    for axis_index, source in enumerate(NOISE_SOURCES + ("both",)):
        ax = axes[axis_index]
        for axis in ("x", "y", "z"):
            if source == "both":
                source_rows = [
                    row for row in rows
                    if row["depth_bin"] == "global" and row["region"] == "all" and row["axis"] == axis
                ]
                by_distance: dict[int, list[float]] = {}
                for row in source_rows:
                    by_distance.setdefault(int(row["distance_px"]), []).append(float(row["Q"]))
                distances = sorted(by_distance)
                q_values = [float(np.mean(by_distance[d])) for d in distances]
                label = axis
            else:
                source_rows = [
                    row for row in rows
                    if row["depth_bin"] == "global"
                    and row["region"] == "all"
                    and row["axis"] == axis
                    and row["noise_source"] == source
                ]
                distances = [int(row["distance_px"]) for row in source_rows]
                q_values = [float(row["Q"]) for row in source_rows]
                label = axis
            ax.plot(distances, q_values, marker="o", label=label, color=colors[axis])
        ax.set_title(f"Q curves ({source})")
        ax.set_xlabel("distance (px)")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("Q=N*S")
    axes[-1].legend(loc="best")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_proxy_vs_repeat(rows: list[dict[str, Any]], output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.8))
    for metric, ax in zip(("N", "Q"), axes):
        for axis in ("x", "y", "z"):
            proxy_rows = {
                int(row["distance_px"]): row
                for row in rows
                if row["depth_bin"] == "global"
                and row["region"] == "all"
                and row["axis"] == axis
                and row["noise_source"] == "proxy"
            }
            repeat_rows = {
                int(row["distance_px"]): row
                for row in rows
                if row["depth_bin"] == "global"
                and row["region"] == "all"
                and row["axis"] == axis
                and row["noise_source"] == "repeat"
            }
            distances = sorted(set(proxy_rows) & set(repeat_rows))
            ax.scatter(
                [proxy_rows[d][metric] for d in distances],
                [repeat_rows[d][metric] for d in distances],
                label=axis,
                s=36,
            )
        ax.plot([0, 1], [0, 1], color="black", linewidth=1, alpha=0.35)
        ax.set_xlabel(f"proxy {metric}")
        ax.set_ylabel(f"repeat {metric}")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.25)
    axes[-1].legend(loc="best")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_depth_heatmaps(rows: list[dict[str, Any]], output_path: Path) -> None:
    depth_bins = sorted({row["depth_bin"] for row in rows if row["depth_bin"] != "global"})
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for ax, axis in zip(axes, ("x", "y", "z")):
        axis_rows = [
            row
            for row in rows
            if row["depth_bin"] != "global"
            and row["axis"] == axis
            and row["region"] == "all"
            and row["noise_source"] == "repeat"
        ]
        distances = sorted({int(row["distance_px"]) for row in axis_rows})
        matrix = np.full((len(depth_bins), len(distances)), np.nan, dtype=np.float32)
        bin_to_i = {name: i for i, name in enumerate(depth_bins)}
        dist_to_i = {distance: i for i, distance in enumerate(distances)}
        for row in axis_rows:
            matrix[bin_to_i[row["depth_bin"]], dist_to_i[int(row["distance_px"])]] = float(row["Q"])
        image = ax.imshow(matrix, aspect="auto", vmin=0, vmax=1, cmap="viridis")
        ax.set_title(f"{axis} repeat Q")
        ax.set_xlabel("distance (px)")
        ax.set_xticks(range(len(distances)), distances)
        ax.set_yticks(range(len(depth_bins)), depth_bins)
    axes[0].set_ylabel("depth bin")
    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.8, label="Q")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _region_ratio(rows: list[dict[str, Any]], region: str, metric: str) -> tuple[dict[str, float | None], float]:
    """Compute per-axis mean over d=1..3 for a region/metric and the XY/Z ratio.

    metric in {"Q","S","N"}. Uses the repeat (gold-standard) noise branch and the
    global (non-depth-binned) rows only.
    """
    selected = [
        row
        for row in rows
        if row["depth_bin"] == "global" and row["region"] == region and row["noise_source"] == "repeat"
    ]
    means: dict[str, float | None] = {}
    for axis in ("x", "y", "z"):
        values = [float(row[metric]) for row in selected if row["axis"] == axis and int(row["distance_px"]) <= 3]
        means[axis] = float(np.mean(values)) if values else None
    xy = [value for key, value in means.items() if key in {"x", "y"} and value is not None]
    xy_mean = float(np.mean(xy)) if xy else 0.0
    z_mean = means["z"]
    ratio = xy_mean / max(float(z_mean or 0.0), 1e-6)
    return means, ratio


def _write_summary(rows: list[dict[str, Any]], cfg: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    # NOTE: the anisotropy gate must be judged on the FOREGROUND region.
    # The background (~86% of voxels here) is nearly structure-free, so its S is
    # uniformly high across axes and dilutes the all-region ratio toward 1.0 (a
    # false negative). This mirrors the R1 discipline in the technical plan
    # ("global metrics get diluted by background; report foreground separately").
    gate_metric = str(cfg["analysis"].get("gate_metric", "Q"))
    gate_region = str(cfg["analysis"].get("gate_region", "foreground"))
    gate_threshold = float(cfg["analysis"].get("gate_xy_z_ratio", 1.2))

    ratios: dict[str, Any] = {}
    for region in REGIONS:
        for metric in ("Q", "S", "N"):
            means, ratio = _region_ratio(rows, region, metric)
            ratios[f"{region}_{metric}"] = {"axis_mean_d1_d3": means, "xy_to_z_ratio": ratio}

    gate_ratio = ratios[f"{gate_region}_{gate_metric}"]["xy_to_z_ratio"]
    # Structure anisotropy is the physically robust signal (z structure decays
    # fast due to the ~4um z-step); noise-independence anisotropy may be weak or
    # even reversed, so we also surface the foreground S ratio for auditability.
    fg_s_ratio = ratios["foreground_S"]["xy_to_z_ratio"]
    summary = {
        "event": "aces_stage01_profile_complete",
        "run_dir": str(run_dir),
        "config": cfg,
        "gate_metric": gate_metric,
        "gate_region": gate_region,
        "gate_threshold_xy_z_ratio": gate_threshold,
        # Back-compat field: all-region Q ratio (the previous, diluted gate value).
        "all_region_q_ratio_d1_d3": ratios["all_Q"]["xy_to_z_ratio"],
        "global_repeat_q_mean_d1_d3": ratios["all_Q"]["axis_mean_d1_d3"],
        "foreground_repeat_q_mean_d1_d3": ratios["foreground_Q"]["axis_mean_d1_d3"],
        "foreground_repeat_s_mean_d1_d3": ratios["foreground_S"]["axis_mean_d1_d3"],
        "region_metric_ratios": ratios,
        "gate_xy_to_z_ratio": gate_ratio,
        "foreground_structure_xy_to_z_ratio": fg_s_ratio,
        "gate_anisotropy_observed": bool(gate_ratio >= gate_threshold),
    }
    write_json(run_dir / "logs" / "profile_summary.json", summary)
    return summary


def run_profile(config_path: Path) -> dict[str, Any]:
    cfg = load_config(config_path)
    run_dir = _as_path(cfg.get("run_dir", DEFAULT_RUN_DIR))
    dirs = ensure_run_dirs(run_dir)

    repeats_root = _as_path(cfg["data"].get("repeats_root", REPEATS_ROOT))
    crop = cfg["data"].get("crop")
    repeats = load_repeats(repeats_root, crop=crop)
    mean9 = repeats.mean(axis=0, dtype=np.float32)
    if repeats.shape[1] < 2 or repeats.shape[2] < 2 or repeats.shape[3] < 2:
        raise ValueError(f"3D fail-fast: invalid repeat stack shape {repeats.shape}")

    mask_percentile = float(cfg["analysis"].get("foreground_percentile", 75.0))
    masks = build_region_masks(mean9, percentile=mask_percentile)
    rows = _distance_rows(repeats, mean9, masks, cfg)
    rows.extend(_depth_rows(repeats, mean9, masks, cfg))
    _finite_rows(rows)

    fieldnames = [
        "depth_bin",
        "axis",
        "distance_px",
        "distance_um",
        "region",
        "noise_source",
        "N",
        "S",
        "Q",
        "structure_corr",
        "proxy_noise_corr",
        "repeat_noise_corr",
        "sample_count_structure",
        "sample_count_proxy",
        "sample_count_repeat",
    ]
    write_csv(dirs["metrics"] / "pairability_curves.csv", [row for row in rows if row["depth_bin"] == "global"], fieldnames)
    write_csv(dirs["metrics"] / "depth_conditioned_pairability.csv", [row for row in rows if row["depth_bin"] != "global"], fieldnames)
    correlations = compute_correlations(rows)
    write_json(dirs["metrics"] / "proxy_repeat_correlations.json", correlations)

    plot_q_curves(rows, dirs["figures"] / "q_curves_xyz.png")
    plot_proxy_vs_repeat(rows, dirs["figures"] / "proxy_vs_repeat.png")
    plot_depth_heatmaps(rows, dirs["figures"] / "depth_q_heatmaps.png")

    env_path = dirs["logs"] / "environment.json"
    if not env_path.exists():
        write_json(env_path, collect_environment())

    manifest = {
        "event": "aces_stage01_data_manifest",
        "repeats_root": str(repeats_root),
        "repeat_count": int(repeats.shape[0]),
        "loaded_shape_repeat_z_y_x": list(repeats.shape),
        "dtype_after_load": str(repeats.dtype),
        "crop": crop,
        "mean9_proxy_reference": str(MEAN9_PROXY),
        "foreground_voxels": int(masks["foreground"].sum()),
        "background_voxels": int(masks["background"].sum()),
        "foreground_fraction": float(masks["foreground"].mean()),
    }
    write_json(dirs["logs"] / "data_manifest.json", manifest)
    summary = _write_summary(rows, cfg, run_dir)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Profile ACES anisotropic pairability Q_a(d).")
    parser.add_argument("--config", required=True, help="YAML/JSON config path.")
    args = parser.parse_args(argv)
    summary = run_profile(Path(args.config))
    print(f"Wrote ACES Stage 1 profile to {summary['run_dir']}")
    print(
        f"Gate region={summary['gate_region']} metric={summary['gate_metric']} "
        f"XY/Z ratio d1-d3: {summary['gate_xy_to_z_ratio']:.4f} "
        f"(all-region Q ratio {summary['all_region_q_ratio_d1_d3']:.4f}) "
        f"observed={summary['gate_anisotropy_observed']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
