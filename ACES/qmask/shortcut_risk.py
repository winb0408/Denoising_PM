"""T4b A1（Plan v2 §7 P-A）：shortcut-risk 离线分析——r(d)/nMSE(d) 能否解释实测 PSNR(d)。

背景（H3 证伪，小结 #12）：τ_s/Q 门控对最优时间配对距离反预测（Spearman≈−1）。
替代假说：小 d 时 target≈input，恒等/平均捷径主导，模型偷懒；大 d 惩罚捷径。
可从纯噪声数据直接测量的量（无需 GT、无需训练）：

- r(d)   = Pearson corr(F_t, F_{t+d})   —— pair 相似度，高 ⇒ 捷径可行
- nMSE(d) = MSE(F_t, F_{t+d}) / Var(F_t) —— 归一化配对距离，低 ⇒ 捷径可行

验证内容：
1. r(d)/nMSE(d) 与实测 PSNR(d)（rgrid distance_summary）的 Spearman（对比 Q_t 的 ≈−1）；
2. r(d) 曲线形状是否解释 PSNR(d) "陡升 d1→d8 后平台"；
3. 判据演示：d_pred(τ) = 最小 d 使 r(d) ≤ τ，与实测平台起点/argmax 对比。

数据：Validation F1/F2/F3（各噪声档）、F0（结构参照，r 应随结构漂移衰减）、
Training 4 栈（部署时只有训练数据可用的口径）。仅离线统计，不更新权重。

用法：
    python -m qmask.shortcut_risk --config <t4b cfg.yaml>            # 正式
    python -m qmask.shortcut_risk --config <t4b cfg.yaml> --smoke    # 冒烟（/tmp）
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from aces.io_utils import load_config
from qmask.t4b_rgrid import (
    EVAL_STACKS,
    GT_STACK,
    TRAIN_STACKS,
    _open_stack,
    eval_stack_stats,
)

TILE = 128
Z_STEP = 2  # 配对采样 z 步长（i.i.d. 噪声下足够密）


def _stack_curves(
    stack: np.ndarray,
    distances: list[int],
    n_tiles: int,
    seed: int,
) -> dict[str, list[float]]:
    """对一个栈采样 n_tiles 个 128×128 tile，d 扫描，返回 r(d)/nMSE(d)（tile 池化）。"""
    rng = np.random.default_rng(seed)
    _, y_len, x_len = stack.shape
    tiles = [
        (int(rng.integers(0, y_len - TILE)), int(rng.integers(0, x_len - TILE)))
        for _ in range(n_tiles)
    ]
    t_len = stack.shape[0]
    # 池化矩：n, Σa, Σb, Σa², Σb², Σab, Σ(a-b)²
    mom = {d: np.zeros(7, dtype=np.float64) for d in distances}
    for y0, x0 in tiles:
        a_full = np.asarray(stack[:, y0 : y0 + TILE, x0 : x0 + TILE], dtype=np.float32)
        for d in distances:
            if d >= t_len:
                continue
            a = a_full[:-d:Z_STEP].ravel().astype(np.float64)
            b = a_full[d::Z_STEP].ravel().astype(np.float64)
            m = mom[d]
            m[0] += a.size
            m[1] += a.sum(); m[2] += b.sum()
            m[3] += (a * a).sum(); m[4] += (b * b).sum()
            m[5] += (a * b).sum()
            m[6] += ((a - b) ** 2).sum()
        del a_full
    out_r, out_nmse = [], []
    for d in distances:
        n, s1a, s1b, s2a, s2b, sab, ssd = mom[d]
        if n == 0:
            out_r.append(float("nan")); out_nmse.append(float("nan")); continue
        cov = n * sab - s1a * s1b
        var_a = max(n * s2a - s1a * s1a, 1e-9)
        var_b = max(n * s2b - s1b * s1b, 1e-9)
        r = float(cov / np.sqrt(var_a * var_b))
        mean_a = s1a / n
        var_total = max(s2a / n - mean_a * mean_a, 1e-9)
        nmse = float((ssd / n) / var_total)
        out_r.append(r)
        out_nmse.append(nmse)
    return {"r": out_r, "nmse": out_nmse}


def _spearman(x: list[float], y: list[float]) -> float | None:
    from scipy.stats import spearmanr

    pairs = [(a, b) for a, b in zip(x, y) if np.isfinite(a) and np.isfinite(b)]
    if len(pairs) < 3:
        return None
    rho, _ = spearmanr([p[0] for p in pairs], [p[1] for p in pairs])
    return float(rho)


def measured_psnr(run_dir: Path, level: str, weight_set: str, distances: list[int]) -> list[float]:
    """从 rgrid 各 d 的 eval_{weight_set}_{level}.json 取 fg PSNR(d)。"""
    vals = []
    for d in distances:
        path = run_dir / f"d{d}" / "metrics" / f"eval_{weight_set}_{level}.json"
        if not path.exists():
            vals.append(float("nan"))
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        vals.append(float(data["regions"]["foreground"]["psnr_db"]))
    return vals


def run_analysis(cfg: dict[str, Any], smoke: bool = False) -> dict[str, Any]:
    run_dir = Path(cfg["run_dir"])
    data_root = Path(cfg["data_root"])
    distances = list(cfg.get("analysis", {}).get("distances", [1, 2, 3, 4, 6, 8, 16, 32, 64]))
    weight_set = str(cfg.get("analysis", {}).get("weight_set", "best"))
    n_tiles = 2 if smoke else 6
    seed = int(cfg.get("seed", 3407))

    stacks: dict[str, Path] = {
        **{f"train:{name.split('/')[-1]}": data_root / name for name in TRAIN_STACKS},
        "F0(structure-ref)": data_root / "Validation" / GT_STACK,
    }
    curves: dict[str, dict[str, list[float]]] = {}
    for tag, path in stacks.items():
        print(f"[shortcut] {tag} ...", flush=True)
        curves[tag] = _stack_curves(_open_stack(path), distances, n_tiles, seed)
    for level in EVAL_STACKS:
        print(f"[shortcut] Validation/{level} ...", flush=True)
        mean_lvl, std_lvl = eval_stack_stats(cfg, level)
        del mean_lvl, std_lvl  # r/nMSE 对仿射不变，直接用原始计数
        curves[level] = _stack_curves(
            _open_stack(data_root / "Validation" / EVAL_STACKS[level]), distances, n_tiles, seed
        )

    # 与实测 PSNR(d) 的关联
    levels = ["F1", "F2", "F3"]
    association: dict[str, Any] = {}
    pooled_r, pooled_nmse, pooled_psnr = [], [], []
    for level in levels:
        psnr = measured_psnr(run_dir, level, weight_set, distances)
        pooled_r += curves[level]["r"]
        pooled_nmse += curves[level]["nmse"]
        pooled_psnr += psnr
        association[level] = {
            "measured_fg_psnr_db": psnr,
            "spearman_r_vs_psnr": _spearman(curves[level]["r"], psnr),
            "spearman_nmse_vs_psnr": _spearman(curves[level]["nmse"], psnr),
        }
    association["pooled"] = {
        "spearman_r_vs_psnr": _spearman(pooled_r, pooled_psnr),
        "spearman_nmse_vs_psnr": _spearman(pooled_nmse, pooled_psnr),
    }

    # 判据演示：d_pred(τ) = 最小 d 使 r(d) ≤ τ；对比实测平台起点（PSNR ≥ max−1dB）
    plateau_onset: dict[str, int | None] = {}
    for level in levels:
        psnr = association[level]["measured_fg_psnr_db"]
        finite = [p for p in psnr if np.isfinite(p)]
        if not finite:
            plateau_onset[level] = None
            continue
        thr = max(finite) - 1.0
        plateau_onset[level] = next(
            (d for d, p in zip(distances, psnr) if np.isfinite(p) and p >= thr), None
        )

    result = {
        "event": "t4b_shortcut_risk_analysis",
        "distances": distances,
        "n_tiles": n_tiles,
        "z_step": Z_STEP,
        "tile": TILE,
        "weight_set": weight_set,
        "curves": curves,
        "association": association,
        "plateau_onset_db1": plateau_onset,
        "d_pred_by_tau": {
            f"r<={tau}": {
                level: next(
                    (d for d, r in zip(distances, curves[level]["r"]) if np.isfinite(r) and r <= tau),
                    None,
                )
                for level in levels
            }
            for tau in (0.3, 0.5, 0.7)
        },
    }
    out_dir = Path("/tmp/shortcut_risk_smoke") if smoke else run_dir / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "shortcut_risk_analysis.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )

    # 摘要打印
    print("\n=== shortcut-risk 分析摘要 ===")
    header = "stack            " + " ".join(f"d={d:<6g}" for d in distances)
    print(header)
    for tag, c in curves.items():
        print(f"{tag:16s} r   " + " ".join(f"{v:7.4f}" for v in c["r"]))
        print(f"{'':16s} nMSE" + " ".join(f"{v:7.3f}" for v in c["nmse"]))
    for level in levels:
        a = association[level]
        print(
            f"{level}: Spearman(r,PSNR)={a['spearman_r_vs_psnr']} "
            f"nMSE={a['spearman_nmse_vs_psnr']} 平台起点={plateau_onset[level]} "
            f"PSNR(d)={[round(p, 2) for p in a['measured_fg_psnr_db']]}"
        )
    print(f"pooled: Spearman(r,PSNR)={association['pooled']['spearman_r_vs_psnr']}")
    print(f"d_pred_by_tau: {json.dumps(result['d_pred_by_tau'])}")
    print(f"输出: {out_dir / 'shortcut_risk_analysis.json'}")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="T4b A1: shortcut-risk offline analysis")
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    run_analysis(cfg, smoke=args.smoke)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
