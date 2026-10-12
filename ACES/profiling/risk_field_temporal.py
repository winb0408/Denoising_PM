"""时间轴 Pair-Risk Field CLI（CIDC25 / Neurofinder 2D 时间序列扩展，M1）。

现有 :mod:`profiling.risk_field` 面向 repeat stack [K,Z,Y,X]（同一 volume 的
K 次重复采集），以空间 offset 为自变量。本模块把自变量换成**时间轴**：
给定 2D 时间序列 [T,Y,X]，对每个时间偏移 Δt 计算 R_N / R_S / R_I：

1. Noise Risk R_N（时间噪声相关）::

       ρ_N(Δt) = corr(N_t(r), N_{t+Δt}(r))   # 对 (t, r) 池化
       R_N(Δt) = |ρ_N(Δt)|

   噪声来源两档（输出列 ``noise_source``）：
   - ``gt``：给 --gt 时 N_t = Y_t − F0_t（真噪声，逐帧 GT 残差）；
   - ``proxy``：无 GT 时 N_t = Y_t − mean_t(Y)（时间均值残差代理）。
     **注意**：该代理会把慢变信号混入"噪声"（帧间慢变化的信号成分
     在相邻帧上高度相关 → ρ_N 虚高），解读 proxy 的 R_N 时必须记住
     这一点——它给出的是"时间高通残差"的相关性上界，不是纯噪声。

2. Structure Risk R_S（结构时间相关，约定与 StructureRiskEstimator 一致）::

       S_t = G_σ(结构源)_t          # 逐帧 2D 平滑（不做时间向平滑：
                                   # 结构的时间变化正是待测量）
       R_S1 = 1 − corr(S_t, S_{t+Δt})
       R_S3 = 1 − mean(corr(∂_y S_t, ∂_y S_{t+Δt}), corr(∂_x S_t, ∂_x S_{t+Δt}))
       R_S  = 0.7·R̃_S1 + 0.3·R̃_S3   # 省略 local SSIM 后的重新归一化权重

   结构源：GT 时用 F0；无 GT 时用 Y 本身（逐帧平滑压噪声，但仍有
   噪声/慢信号泄漏——与 proxy R_N 同样的 caveat）。

3. Identity / Shortcut Risk R_I：直接复用 :class:`ShortcutRiskEstimator`。
   把 [T,Y,X] 当作 [Z=T,Y,X] 体数据，时间偏移 Δt 映射到 z 轴 offset
   (Δt,0,0)（``shifted_pair`` 的 valid-crop 语义对时间轴完全成立，
   见 tests/test_risk_temporal.py 的对照验证）::

       nMSE(Δt) = E[(Y_t − Y_{t+Δt})²] / Var(Y)
       R_I(Δt) = exp(−nMSE/τ_I) · max(ρ_Y(Δt), 0)

R_base 诊断组合分与 risk_field.py 相同（0.35/0.40/0.25，方案 §11）。

用法（仓库根目录，PYTHONPATH=ACES）::

    PYTHONPATH=ACES python -m profiling.risk_field_temporal \
        --stack <noisy.tif> [--gt <clean.tif>] --run-dir <run_dir> \
        [--dt-list 1,2,4,8] [--stride 4] [--frame-crop 600] \
        [--noise-level-label F1]

输出：``<run_dir>/metrics/risk_field_temporal.csv``（每个 Δt 一行）与
``risk_field_temporal.json``（含全部分量的数值明细）。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
from scipy import ndimage

from aces.io_utils import write_csv, write_json

from .risk import (
    OffsetZyx,
    ShortcutRiskEstimator,
    StructureRiskEstimator,
    _clip01,
    _pearson,
    shifted_pair,
)
from .risk_field import LAMBDA_I, LAMBDA_N, LAMBDA_S

# 默认 Δt 列表：覆盖 rgrid 实验已知的效用峰值区（F1 d*=16 / F2 d*=8 / F3 d*=64）
DEFAULT_DT_LIST = (1, 2, 3, 4, 6, 8, 16, 32, 64)

CSV_FIELDNAMES = [
    "dt",
    "rn",
    "rn_components",
    "rs",
    "rs_components",
    "ri",
    "ri_components",
    "r_base",
    "noise_source",
]

# R_S 权重复用 StructureRiskEstimator 的类常量（省略 SSIM 后重新归一化）
_W_S1 = StructureRiskEstimator.W_S1  # 0.7
_W_S3 = StructureRiskEstimator.W_S3  # 0.3


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------

def load_temporal_stack(
    path: str | Path,
    frame_crop: int | None = None,
) -> np.ndarray:
    """读入 2D 时间序列 tif 并返回 [T,Y,X] float32。

    ``frame_crop`` 只取前 N 帧（长序列快速冒烟/裁剪用）。3D 以外的
    布局（如 4D）直接报错——本模块不做 repeat-stack 布局推断。
    """
    raw = tifffile.imread(str(path))
    if raw.ndim != 3:
        raise ValueError(f"期望 3D 时间序列 [T,Y,X]，收到 shape {raw.shape}: {path}")
    if frame_crop is not None:
        if frame_crop < 2:
            raise ValueError(f"frame_crop 必须 >= 2，收到 {frame_crop}")
        raw = raw[:frame_crop]
    if min(raw.shape[1:]) < 4:
        raise ValueError(f"空间维度过小: {raw.shape}")
    return np.ascontiguousarray(raw, dtype=np.float32)


# ---------------------------------------------------------------------------
# 时间轴风险计算
# ---------------------------------------------------------------------------

def compute_noise_volume(
    noisy: np.ndarray,
    gt: np.ndarray | None,
) -> tuple[np.ndarray, str]:
    """构造噪声体 N 与其来源标签。

    - 有 GT：N = Y − F0（真噪声），noise_source="gt"；
    - 无 GT：N = Y − mean_t(Y)（时间均值残差代理），noise_source="proxy"。
      代理的慢信号混叠 caveat 见模块 docstring。
    """
    if gt is None:
        mean_frame = noisy.mean(axis=0, dtype=np.float32)
        noise = noisy - mean_frame[None, ...]
        source = "proxy"
    else:
        if gt.shape != noisy.shape:
            raise ValueError(f"GT shape {gt.shape} 与噪声序列 {noisy.shape} 不一致")
        noise = noisy - gt
        source = "gt"
    return np.ascontiguousarray(noise, dtype=np.float32), source


def compute_structure_volume(source: np.ndarray, sigma: float) -> np.ndarray:
    """逐帧 2D 高斯平滑的结构代理 S（sigma=(0,σ,σ)，不做时间向平滑）。"""
    smoothed = ndimage.gaussian_filter(source, sigma=(0.0, float(sigma), float(sigma)))
    return np.ascontiguousarray(smoothed, dtype=np.float32)


def _temporal_offset(dt: int) -> OffsetZyx:
    """Δt → zyx offset：时间轴映射到 z 轴（[T,Y,X] 视作 [Z,Y,X]）。"""
    return (int(dt), 0, 0)


def compute_temporal_table(
    noisy: np.ndarray,
    gt: np.ndarray | None,
    dt_list: list[int],
    stride: int = 1,
    structure_sigma: float = 0.8,
    tau_i: float = 1.0,
) -> tuple[list[dict[str, Any]], str]:
    """对每个 Δt 计算 R_N / R_S / R_I 全套分量，返回 (CSV 行列表, noise_source)。

    行字段（数值型，CSV 额外拼装 components 字符串列）：
    dt / rn, rn_corr, rn_samples / rs, rs_s1, rs_s3, rs_lf_corr /
    ri, ri_nmse, ri_corr_y / r_base / noise_source。
    """
    data = np.asarray(noisy, dtype=np.float32)
    if data.ndim != 3:
        raise ValueError(f"时间序列应为 [T,Y,X] 3D，收到 shape {data.shape}")
    if gt is not None and np.asarray(gt).shape != data.shape:
        raise ValueError(f"GT shape {np.asarray(gt).shape} 与噪声序列 {data.shape} 不一致")

    noise, noise_source = compute_noise_volume(data, gt)
    # 结构代理：GT 优先；无 GT 时用逐帧平滑的 Y（caveat 见 docstring）
    struct = compute_structure_volume(data if gt is None else gt, structure_sigma)
    grads_yx = (np.gradient(struct, axis=1), np.gradient(struct, axis=2))
    # R_I 直接复用 ShortcutRiskEstimator（[T,Y,X] 当 [Z,Y,X]，Δt 映射 z 轴）
    shortcut_est = ShortcutRiskEstimator(tau_i=tau_i, stride=stride).fit(data)

    rows: list[dict[str, Any]] = []
    for dt in dt_list:
        if dt < 0 or dt >= data.shape[0] - 1:
            raise ValueError(f"Δt={dt} 超出时间范围 [0, {data.shape[0] - 2}]")
        offset = _temporal_offset(dt)

        # --- R_N：噪声时间相关（对 (t,r) 池化的 Pearson） ---
        noise_pair = shifted_pair(noise, offset, stride=stride)
        if noise_pair is None:
            raise ValueError(f"Δt={dt} 相对时间序列过大，无有效帧对")
        rho_n = _pearson(noise_pair[0], noise_pair[1])
        rn = _clip01(abs(rho_n))

        # --- R_S：结构相关（0.7/0.3，同 StructureRiskEstimator 约定） ---
        lf_pair = shifted_pair(struct, offset, stride=stride)
        if lf_pair is None:
            raise ValueError(f"Δt={dt} 相对时间序列过大，无有效帧对")
        rho_lf = _pearson(lf_pair[0], lf_pair[1])
        rs_s1 = _clip01(1.0 - rho_lf)
        grad_corrs = [
            _pearson(*shifted_pair(grad, offset, stride=stride))  # type: ignore[misc]
            for grad in grads_yx
        ]
        rs_s3 = _clip01(1.0 - float(np.mean(grad_corrs)))
        rs = _clip01(_W_S1 * rs_s1 + _W_S3 * rs_s3)

        # --- R_I：复用 ShortcutRiskEstimator ---
        ri_parts = shortcut_est.risk(offset)

        r_base = float(
            LAMBDA_N * rn + LAMBDA_S * rs + LAMBDA_I * ri_parts["ri"]
        )
        rows.append({
            "dt": int(dt),
            "rn": rn,
            "rn_corr": float(rho_n),
            "rn_samples": int(noise_pair[0].size),
            "rs": rs,
            "rs_s1": rs_s1,
            "rs_s3": rs_s3,
            "rs_lf_corr": float(rho_lf),
            "ri": ri_parts["ri"],
            "ri_nmse": ri_parts["ri_nmse"],
            "ri_corr_y": ri_parts["ri_corr_y"],
            "r_base": r_base,
            "noise_source": noise_source,
        })
    return rows, noise_source


# ---------------------------------------------------------------------------
# 输出
# ---------------------------------------------------------------------------

def to_csv_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """把数值行转成规格列（dt + 主指标 + components 字符串 + noise_source）。"""
    csv_rows: list[dict[str, Any]] = []
    for row in rows:
        csv_rows.append({
            "dt": row["dt"],
            "rn": row["rn"],
            "rn_components": (
                f"rn_corr={row['rn_corr']:+.4f};rn_samples={row['rn_samples']}"
            ),
            "rs": row["rs"],
            "rs_components": (
                f"rs_s1={row['rs_s1']:.4f};rs_s3={row['rs_s3']:.4f};"
                f"rs_lf_corr={row['rs_lf_corr']:+.4f}"
            ),
            "ri": row["ri"],
            "ri_components": (
                f"ri_nmse={row['ri_nmse']:.4f};ri_corr_y={row['ri_corr_y']:+.4f}"
            ),
            "r_base": row["r_base"],
            "noise_source": row["noise_source"],
        })
    return csv_rows


def build_json_summary(
    rows: list[dict[str, Any]],
    noise_source: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    """risk_field_temporal.json 摘要：config + 逐 Δt 全分量数值明细。"""
    return {
        "event": "aces_m1_risk_field_temporal",
        "config": config,
        "noise_source": noise_source,
        "lambda": {"rn": LAMBDA_N, "rs": LAMBDA_S, "ri": LAMBDA_I},
        "n_dt": len(rows),
        "rows": rows,
    }


def _print_table(rows: list[dict[str, Any]], noise_source: str) -> None:
    print(f"== temporal risk field (noise_source={noise_source}) ==")
    print(f"  {'dt':>4s}  {'R_N':>8s}  {'R_S':>8s}  {'R_I':>8s}  {'R_base':>8s}")
    for row in rows:
        print(
            f"  {row['dt']:>4d}  {float(row['rn']):8.4f}  {float(row['rs']):8.4f} "
            f"  {float(row['ri']):8.4f}  {float(row['r_base']):8.4f}"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = Path(args.run_dir).expanduser().resolve()
    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    stack = load_temporal_stack(args.stack, frame_crop=args.frame_crop)
    gt = (
        load_temporal_stack(args.gt, frame_crop=args.frame_crop)
        if args.gt is not None else None
    )
    dt_list = [int(tok) for tok in str(args.dt_list).split(",") if tok.strip()]
    if not dt_list:
        raise ValueError(f"--dt-list 为空: {args.dt_list!r}")

    rows, noise_source = compute_temporal_table(
        stack,
        gt,
        dt_list,
        stride=args.stride,
        structure_sigma=args.structure_sigma,
        tau_i=args.tau_i,
    )
    # 与 risk_field.py 一致的 fail-fast：主指标必须有限
    for row in rows:
        for key in ("rn", "rs", "ri", "r_base"):
            if not np.isfinite(float(row[key])):
                raise FloatingPointError(f"非有限 {key} 于 dt 行: {row}")

    write_csv(metrics_dir / "risk_field_temporal.csv", to_csv_rows(rows), CSV_FIELDNAMES)
    config = {
        "stack": str(Path(args.stack).expanduser().resolve()),
        "gt": str(Path(args.gt).expanduser().resolve()) if args.gt is not None else None,
        "noise_level_label": args.noise_level_label,
        "stack_shape_tyx": list(stack.shape),
        "dt_list": dt_list,
        "stride": args.stride,
        "structure_sigma": args.structure_sigma,
        "tau_i": args.tau_i,
        "frame_crop": args.frame_crop,
    }
    summary = build_json_summary(rows, noise_source, config)
    write_json(metrics_dir / "risk_field_temporal.json", summary)

    print(
        f"temporal risk field: {len(rows)} dts, noise_source={noise_source} "
        f"-> {metrics_dir}"
    )
    print(f"stack [T,Y,X]={stack.shape}, stride={args.stride}, label={args.noise_level_label}")
    _print_table(rows, noise_source)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="计算时间轴 pair-risk field（R_N/R_S/R_I vs Δt，CIDC25/Neurofinder 用）。"
    )
    parser.add_argument("--stack", required=True, help="噪声时间序列 tif 路径 [T,Y,X]。")
    parser.add_argument("--gt", default=None, help="可选 GT tif（同形状 [T,Y,X]）；给出则 noise_source=gt。")
    parser.add_argument("--run-dir", required=True, help="输出 run 目录（写 metrics/risk_field_temporal.*）。")
    parser.add_argument("--dt-list", default=",".join(str(d) for d in DEFAULT_DT_LIST),
                        help="逗号分隔的时间偏移列表（默认 1,2,3,4,6,8,16,32,64）。")
    parser.add_argument("--stride", type=int, default=1, help="体素步长子采样（控制内存/加速）。")
    parser.add_argument("--frame-crop", type=int, default=None, help="只取前 N 帧。")
    parser.add_argument("--tau-i", type=float, default=1.0, help="R_I 的 τ_I（默认 1.0，同 risk_field）。")
    parser.add_argument("--structure-sigma", type=float, default=0.8, help="结构代理逐帧高斯 sigma。")
    parser.add_argument("--noise-level-label", default="", help="输出标签（如 F1/F2/F3，写入 config）。")
    args = parser.parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
