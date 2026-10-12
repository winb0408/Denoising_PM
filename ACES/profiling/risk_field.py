"""Pair-Risk Field CLI（修正方案 §8-§11 / Experiment B — risk field）。

用法（仓库根目录，PYTHONPATH=ACES）::

    PYTHONPATH=ACES python -m profiling.risk_field \
        --stack <repeat_stack.tif> --run-dir <run_dir> [--crop-z 60 ...]

流程：
  1. 读入 repeat stack tif（[N,H,W] 帧序列按布局重排为 [K,Z,Y,X]，或直接 4D）；
  2. 构建 offset 候选库（axis + diagonal + family pairwise 的并集）；
  3. 对每个 offset 计算 R_N（repeat 分支）、R_S、R_I 及其分量；
  4. 输出 ``<run_dir>/metrics/risk_field.csv``、``risk_field.json``
     （逐 offset worst/best 摘要）与 ``risk_families.csv``
     （family 层面 mean/max 聚合）；
  5. 打印紧凑的 top/bottom 表。

诊断组合分（方案 §11，仅作 feasibility / diagnostic，不是最终 utility）::

    R_base = 0.35·R_N + 0.40·R_S + 0.25·R_I
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import tifffile

from aces.io_utils import write_csv, write_json

from .candidates import (
    CandidateFamily,
    OffsetZyx,
    default_family_library,
    default_offset_library,
)
from .risk import NoiseRiskEstimator, ShortcutRiskEstimator, StructureRiskEstimator

# R_base 诊断权重（方案 §11 推荐值；Phase 1 需做 sensitivity analysis）
LAMBDA_N, LAMBDA_S, LAMBDA_I = 0.35, 0.40, 0.25

RISK_METRICS = ("rn", "rs", "ri", "r_base")

CSV_FIELDNAMES = [
    "offset_voxel_z", "offset_voxel_y", "offset_voxel_x",
    "offset_um_z", "offset_um_y", "offset_um_x",
    "rn", "rn_corr", "rnv", "rn_star", "rn_samples",
    "rs", "rs_s1", "rs_ssim", "rs_s3", "rs_lf_corr",
    "ri", "ri_nmse", "ri_corr_y",
    "r_base",
]

FAMILY_FIELDNAMES = [
    "family_id", "n_pairs",
    "rn_mean", "rn_max", "rn_star_mean", "rn_star_max",
    "rs_mean", "rs_max",
    "ri_mean", "ri_max",
    "r_base_mean", "r_base_max",
    "worst_pair_offset", "worst_pair_r_base",
]

# 3P VALID 数据的默认 spacing（zyx，um）。z=4.0 为布局推导标定值，
# xy=0.8096639067 来自 TIFF Info 的 XResolution/YResolution（见 stage01 配置注释）。
DEFAULT_SPACING_ZYX = (4.0, 0.8096639067, 0.8096639067)


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------

def load_repeat_stack(
    path: str | Path,
    layout: str = "depth_repeat",
    repeats: int = 9,
    depth: int | None = None,
    crop_z: int | None = None,
    crop_y: int | None = None,
    crop_x: int | None = None,
) -> np.ndarray:
    """读入 repeat stack tif 并重排为 [K,Z,Y,X] float32。

    layout:
      - ``depth_repeat``：帧序为 depth-major（frame = z*K + k，即 275 组
        每组 9 个 repeat——3P_neuron_raw.tif 的审计结论），reshape 为
        [D,K,H,W] 后 transpose 到 [K,Z,Y,X]；
      - ``repeat_depth``：帧序为 repeat-major（frame = k*D + z）。
    4D 输入（已是 [K,Z,Y,X]）直接使用。
    """
    raw = tifffile.imread(str(path))
    if raw.ndim == 4:
        stack = np.asarray(raw, dtype=np.float32)
    elif raw.ndim == 3:
        frames, height, width = raw.shape
        if repeats < 2:
            raise ValueError(f"repeats 必须 >= 2，收到 {repeats}")
        if depth is None:
            if frames % repeats != 0:
                raise ValueError(
                    f"帧数 {frames} 不能被 repeats={repeats} 整除，请显式给 --depth"
                )
            depth = frames // repeats
        if depth * repeats != frames:
            raise ValueError(f"帧数 {frames} != depth({depth})*repeats({repeats})")
        if layout == "depth_repeat":
            stack = raw.reshape(depth, repeats, height, width).transpose(1, 0, 2, 3)
        elif layout == "repeat_depth":
            stack = raw.reshape(repeats, depth, height, width)
        else:
            raise ValueError(f"未知 layout: {layout!r}（应为 depth_repeat / repeat_depth）")
        stack = np.ascontiguousarray(stack, dtype=np.float32)
    else:
        raise ValueError(f"期望 3D 帧序列或 4D stack，收到 shape {raw.shape}")
    # 裁剪（取前 n 个平面/行/列，保持冒烟快速）
    if crop_z is not None:
        stack = stack[:, :crop_z]
    if crop_y is not None:
        stack = stack[:, :, :crop_y]
    if crop_x is not None:
        stack = stack[:, :, :, :crop_x]
    if min(stack.shape[1:]) < 4:
        raise ValueError(f"裁剪后空间维度过小: {stack.shape}")
    return np.ascontiguousarray(stack, dtype=np.float32)


# ---------------------------------------------------------------------------
# 风险场计算
# ---------------------------------------------------------------------------

def build_offset_set() -> list[OffsetZyx]:
    """默认 offset 库 ∪ 全部 family pairwise offset（去重）。"""
    offsets: list[OffsetZyx] = [cand.offset_zyx for cand in default_offset_library()]
    for family in default_family_library():
        offsets.extend(family.pairwise_offsets())
    unique: list[OffsetZyx] = []
    for offset in offsets:
        if offset not in unique:
            unique.append(offset)
    return unique


def compute_offset_table(
    stack: np.ndarray,
    offsets: list[OffsetZyx],
    spacing_zyx: tuple[float, float, float],
    stride: int = 1,
    structure_sigma: float = 0.8,
    tau_i: float = 1.0,
    alpha_n: float = 0.5,
    var_sigma: float = 2.0,
) -> list[dict[str, Any]]:
    """对每个 offset 计算 R_N / R_S / R_I 全套分量，返回 CSV 行列表。

    R_N 走 repeat 分支（K 个残差）；R_S 与 R_I 在 Ȳ（repeat 均值）上计算，
    以压低单次采样的噪声对结构/shortcut 统计的干扰。
    """
    mean_volume = stack.mean(axis=0, dtype=np.float32)
    noise_est = NoiseRiskEstimator(stride=stride, var_sigma=var_sigma).fit(stack)
    struct_est = StructureRiskEstimator(sigma=structure_sigma, stride=stride).fit(mean_volume)
    shortcut_est = ShortcutRiskEstimator(tau_i=tau_i, stride=stride).fit(mean_volume)

    rows: list[dict[str, Any]] = []
    for offset in offsets:
        rn_parts = noise_est.risk(offset, alpha=alpha_n)
        rs_parts = struct_est.risk(offset)
        ri_parts = shortcut_est.risk(offset)
        r_base = float(
            LAMBDA_N * rn_parts["rn"] + LAMBDA_S * rs_parts["rs"] + LAMBDA_I * ri_parts["ri"]
        )
        row: dict[str, Any] = {
            "offset_voxel_z": int(offset[0]),
            "offset_voxel_y": int(offset[1]),
            "offset_voxel_x": int(offset[2]),
            "offset_um_z": float(offset[0] * spacing_zyx[0]),
            "offset_um_y": float(offset[1] * spacing_zyx[1]),
            "offset_um_x": float(offset[2] * spacing_zyx[2]),
            **rn_parts,
            **rs_parts,
            **ri_parts,
            "r_base": r_base,
        }
        rows.append(row)
    return rows


def family_summary(
    family: CandidateFamily,
    per_offset: dict[OffsetZyx, dict[str, Any]],
) -> dict[str, Any]:
    """把 family 的全部 pairwise offset 风险聚合为 mean / max（方案 §6）。

    ``per_offset`` 必须覆盖该 family 的每个 Δ_ij（由调用方保证，缺失时抛
    KeyError——宁可失败也不能静默用 0 填充）。
    """
    deltas = family.pairwise_offsets()
    missing = [delta for delta in deltas if delta not in per_offset]
    if missing:
        raise KeyError(f"family {family.family_id} 缺少 pairwise offset 风险: {missing}")
    summary: dict[str, Any] = {
        "family_id": family.family_id,
        "n_pairs": len(deltas),
    }
    for metric in ("rn", "rn_star", "rs", "ri", "r_base"):
        values = [float(per_offset[delta][metric]) for delta in deltas]
        summary[f"{metric}_mean"] = float(np.mean(values))
        summary[f"{metric}_max"] = float(np.max(values))
    # worst-case pair（按 R_base）
    worst_delta = max(deltas, key=lambda delta: float(per_offset[delta]["r_base"]))
    summary["worst_pair_offset"] = "/".join(str(v) for v in worst_delta)
    summary["worst_pair_r_base"] = float(per_offset[worst_delta]["r_base"])
    return summary


# ---------------------------------------------------------------------------
# 摘要与打印
# ---------------------------------------------------------------------------

def _top_bottom(
    rows: list[dict[str, Any]], metric: str, top_k: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """按 metric 排序，返回 (worst top-k, best bottom-k)。"""
    ordered = sorted(
        rows,
        key=lambda row: (float(row[metric]), row["offset_voxel_z"], row["offset_voxel_y"], row["offset_voxel_x"]),
        reverse=True,
    )
    return ordered[:top_k], ordered[-top_k:][::-1]


def _fmt_offset(row: dict[str, Any]) -> str:
    return f"({row['offset_voxel_z']:+d},{row['offset_voxel_y']:+d},{row['offset_voxel_x']:+d})"


def _print_table(rows: list[dict[str, Any]], metric: str, top_k: int) -> None:
    worst, best = _top_bottom(rows, metric, top_k)
    print(f"== {metric} worst-{len(worst)} ==")
    for row in worst:
        print(f"  {_fmt_offset(row)}  {metric}={float(row[metric]):.4g}")
    print(f"== {metric} best-{len(best)} ==")
    for row in best:
        print(f"  {_fmt_offset(row)}  {metric}={float(row[metric]):.4g}")


def build_json_summary(
    rows: list[dict[str, Any]],
    families: list[dict[str, Any]],
    config: dict[str, Any],
    top_k: int,
) -> dict[str, Any]:
    """risk_field.json 摘要：逐指标 worst/best offset + family 聚合。"""
    summary: dict[str, Any] = {
        "event": "aces_m1_risk_field",
        "config": config,
        "n_offsets": len(rows),
        "n_families": len(families),
        "lambda": {"rn": LAMBDA_N, "rs": LAMBDA_S, "ri": LAMBDA_I},
    }
    for metric in RISK_METRICS:
        worst, best = _top_bottom(rows, metric, top_k)
        summary[f"{metric}_worst"] = [
            {"offset_zyx": [row["offset_voxel_z"], row["offset_voxel_y"], row["offset_voxel_x"]],
             "offset_um_zyx": [row["offset_um_z"], row["offset_um_y"], row["offset_um_x"]],
             "value": float(row[metric])}
            for row in worst
        ]
        summary[f"{metric}_best"] = [
            {"offset_zyx": [row["offset_voxel_z"], row["offset_voxel_y"], row["offset_voxel_x"]],
             "offset_um_zyx": [row["offset_um_z"], row["offset_um_y"], row["offset_um_x"]],
             "value": float(row[metric])}
            for row in best
        ]
    summary["families"] = families
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = Path(args.run_dir).expanduser().resolve()
    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    stack = load_repeat_stack(
        args.stack,
        layout=args.layout,
        repeats=args.repeats,
        depth=args.depth,
        crop_z=args.crop_z,
        crop_y=args.crop_y,
        crop_x=args.crop_x,
    )
    spacing_zyx = (args.spacing_z, args.spacing_y, args.spacing_x)
    offsets = build_offset_set()
    rows = compute_offset_table(
        stack,
        offsets,
        spacing_zyx=spacing_zyx,
        stride=args.stride,
        structure_sigma=args.structure_sigma,
        tau_i=args.tau_i,
        alpha_n=args.alpha_n,
        var_sigma=args.var_sigma,
    )
    # 确保所有值有限（与 profile_anisotropy 的 fail-fast 纪律一致）
    for row in rows:
        for key in ("rn", "rs", "ri", "r_base"):
            if not np.isfinite(float(row[key])):
                raise FloatingPointError(f"非有限 {key} 于 offset 行: {row}")

    per_offset: dict[OffsetZyx, dict[str, Any]] = {
        (row["offset_voxel_z"], row["offset_voxel_y"], row["offset_voxel_x"]): row
        for row in rows
    }
    families = [
        family_summary(family, per_offset)
        for family in default_family_library()
    ]

    write_csv(metrics_dir / "risk_field.csv", rows, CSV_FIELDNAMES)
    write_csv(metrics_dir / "risk_families.csv", families, FAMILY_FIELDNAMES)
    config = {
        "stack": str(Path(args.stack).expanduser().resolve()),
        "stack_shape_kzyx": list(stack.shape),
        "layout": args.layout,
        "spacing_um_zyx": list(spacing_zyx),
        "stride": args.stride,
        "structure_sigma": args.structure_sigma,
        "tau_i": args.tau_i,
        "alpha_n": args.alpha_n,
        "var_sigma": args.var_sigma,
        "crop_zyx": [args.crop_z, args.crop_y, args.crop_x],
    }
    summary = build_json_summary(rows, families, config, top_k=args.top_k)
    write_json(metrics_dir / "risk_field.json", summary)

    print(f"risk field: {len(rows)} offsets, {len(families)} families -> {metrics_dir}")
    print(f"stack [K,Z,Y,X]={stack.shape}, spacing(um, zyx)={spacing_zyx}")
    for metric in RISK_METRICS:
        _print_table(rows, metric, args.top_k)
    print("== families by worst-case r_base (top-5) ==")
    for fam in sorted(families, key=lambda f: f["r_base_max"], reverse=True)[:5]:
        print(
            f"  {fam['family_id']:<16s} r_base_mean={fam['r_base_mean']:.4f} "
            f"r_base_max={fam['r_base_max']:.4f} worst_pair={fam['worst_pair_offset']}"
        )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="计算 pair-risk field（R_N/R_S/R_I per offset + family 聚合）。"
    )
    parser.add_argument("--stack", required=True, help="repeat stack tif 路径（3D 帧序列或 4D [K,Z,Y,X]）。")
    parser.add_argument("--run-dir", required=True, help="输出 run 目录（写 metrics/risk_field.*）。")
    parser.add_argument("--layout", default="depth_repeat", choices=("depth_repeat", "repeat_depth"),
                        help="3D 帧序列布局：depth_repeat=frame=z*K+k（默认，3P 审计结论）；repeat_depth=frame=k*D+z。")
    parser.add_argument("--repeats", type=int, default=9, help="repeat 数 K（默认 9）。")
    parser.add_argument("--depth", type=int, default=None, help="z 深度数 D（默认自动推断）。")
    parser.add_argument("--crop-z", type=int, default=None, help="z 向裁剪（取前 n 个平面）。")
    parser.add_argument("--crop-y", type=int, default=None, help="y 向裁剪。")
    parser.add_argument("--crop-x", type=int, default=None, help="x 向裁剪。")
    parser.add_argument("--spacing-z", type=float, default=DEFAULT_SPACING_ZYX[0], help="z spacing um/px。")
    parser.add_argument("--spacing-y", type=float, default=DEFAULT_SPACING_ZYX[1], help="y spacing um/px。")
    parser.add_argument("--spacing-x", type=float, default=DEFAULT_SPACING_ZYX[2], help="x spacing um/px。")
    parser.add_argument("--stride", type=int, default=1, help="体素步长子采样（控制内存/加速）。")
    parser.add_argument("--structure-sigma", type=float, default=0.8, help="S_LF 高斯 sigma。")
    parser.add_argument("--tau-i", type=float, default=1.0,
                        help="R_I 的 τ_I（默认 1.0：固定 0.05 在噪声主导数据上饱和失效，M1 冒烟实测）。")
    parser.add_argument("--alpha-n", type=float, default=0.5, help="R_N* 的 α_N。")
    parser.add_argument("--var-sigma", type=float, default=2.0,
                        help="R_NV 方差图平滑 sigma（0 关闭；小 K 下必须平滑，否则 R_NV 饱和）。")
    parser.add_argument("--top-k", type=int, default=5, help="摘要 top/bottom 数量。")
    args = parser.parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
