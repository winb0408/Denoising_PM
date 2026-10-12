"""Exp D（adaptive-hard）解析版：Oracle Gap 分析（零新训练）。

数据全部来自已有 M2 runs：
- 3P 空间：10 family × 3 seed 20ep oracle（families/*.json）+ probe（micro_probe.csv）
- CIDC25 时间轴：9 dt rgrid 20ep 单 seed oracle + 时间轴 probe（micro_probe_temporal.csv）

协议：
- 3P：权重在 dev seed 3407 oracle 上拟合（避免全 seed 循环），选择臂的 gap 在
  held-out seed 3408/3409 均值上评估；oracle-best 以 3-seed 均值取 argmax（上界）。
- CIDC25：regime 内拟合（单 seed，保留项），gap 直接对 rgrid oracle。

臂（每 regime 一个选择规则）：
- oracle-best   : argmax U_oracle（上界）
- adaptive-hard : argmax U_v2(w_regime)（ours）
- fixed-VALID   : valid_corners / dt=1（最近邻 = 相似度缺省）
- fixed-QMask   : qmask_xy / dt=1
- fixed-worst   : 库内最差（下界参考）

用法：
    PYTHONPATH=ACES python -m calibration.exp_d_gap \
        --runs-root /data3/wjb/Denoising_PM/ACES/runs --out <json>
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import pandas as pd

FEATURES = ["loss_neighbor_mean", "cross_view_consistency"]


def load_3p(runs_root: pathlib.Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    seeds = {}
    for s in (3407, 3408, 3409):
        d = {}
        for jf in sorted((runs_root / f"m2_oracle_3p_seed{s}" / "families").glob("*.json")):
            if jf.name == "summary.json":
                continue
            c = json.loads(jf.read_text(encoding="utf-8"))
            d[c["family_id"]] = c["eval"]["u_oracle_gain"]
        seeds[s] = d
    oracle = pd.DataFrame(seeds)  # index=family, cols=seeds
    oracle["mean"] = oracle.mean(axis=1)
    oracle["std"] = oracle.std(axis=1, ddof=1)
    probe = pd.read_csv(runs_root / "m2_micro_probe_3p_seed3407" / "micro_probe.csv").set_index("family_id")
    return oracle, probe


def load_cidc(runs_root: pathlib.Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """9 dt × 3 seed（3407/3408/3409）eval_final_F1 psnr_all → mean/std/逐 seed 列。"""
    seeds = {}
    for s in (3407, 3408, 3409):
        d = {}
        for dt in (1, 2, 3, 4, 6, 8, 16, 32, 64):
            m = json.loads(
                (runs_root / f"stage01_cidc25_rgrid_seed{s}/d{dt}/metrics/eval_final_F1.json").read_text()
            )
            d[dt] = m["regions"]["all"]["psnr_db"]
        seeds[s] = d
    oracle = pd.DataFrame(seeds)
    oracle["mean"] = oracle.mean(axis=1)
    oracle["std"] = oracle.std(axis=1, ddof=1)
    probe = pd.read_csv(runs_root / "m2_micro_probe_cidc25_seed3407" / "micro_probe_temporal.csv").set_index("dt")
    return oracle, probe


def fit_weights(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    w, *_ = np.linalg.lstsq(x, y, rcond=None)
    return w


def regime_gaps(oracle: pd.DataFrame, probe: pd.DataFrame, u_v2: pd.Series,
                fixed: dict[str, float], label: str, heldout_cols: list | None = None) -> dict:
    """各臂 gap = U_oracle(oracle-best) − U_oracle(臂)。heldout_cols 给定时只在其上评估。

    oracle 需含 "mean" 列（3P 由 load_3p 生成；CIDC25 单 seed 时先补 mean=u_oracle）。
    """
    if "mean" not in oracle.columns:
        oracle = oracle.assign(mean=oracle["u_oracle"])
    idx = oracle.index.tolist()
    best = oracle["mean"].idxmax()
    best_u = oracle.loc[best, "mean"]
    arms = {"oracle-best": best}
    arms.update({f"adaptive-hard(v2)": u_v2.idxmax()})
    arms.update(fixed)

    eval_cols = heldout_cols if heldout_cols is not None else ["mean"]
    out = {"regime": label, "oracle_best": {"arm": best, "u": float(best_u)}}
    for name, arm in arms.items():
        row = oracle.loc[arm]
        per_eval = {}
        for c in eval_cols:
            if c == "mean":
                per_eval[c] = float(best_u - row["mean"])
            else:
                # held-out 单 seed：oracle-best 也用同列评估（同 seed 公平）
                per_eval[c] = float(oracle.loc[best, c] - row[c])
        if heldout_cols is not None and len(heldout_cols) > 1:
            ho_mean = float(np.mean([oracle.loc[best, c] - row[c] for c in heldout_cols]))
            out[name] = {"arm": arm, "gap_heldout_mean": ho_mean, "gap_per_seed": per_eval}
        else:
            out[name] = {"arm": arm, "gap": list(per_eval.values())[0]}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    # /data3 2026-10-08 拔盘迁移至 /ssd1：默认自动探测（ssd1 优先，data3 回退）
    _default_root = "/ssd1/wjb/Denoising_PM/ACES/runs"
    if not pathlib.Path(_default_root).exists():
        _default_root = "/data3/wjb/Denoising_PM/ACES/runs"
    ap.add_argument("--runs-root", default=_default_root)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    root = pathlib.Path(args.runs_root)

    results = {}

    # ---- 3P 空间 ----
    oracle3, probe3 = load_3p(root)
    fams = oracle3.index.tolist()
    x = probe3.loc[fams, FEATURES].to_numpy(float)
    # dev = seed3407 oracle；held-out = 3408/3409
    w_dev = fit_weights(x, oracle3.loc[fams, 3407].to_numpy(float))
    u_v2_3p = pd.Series(x @ w_dev, index=fams)
    fixed3 = {
        "fixed-VALID": "valid_corners",
        "fixed-QMask": "qmask_xy",
        "fixed-worst": oracle3["mean"].idxmin(),
    }
    results["3p"] = regime_gaps(oracle3, probe3, u_v2_3p, fixed3, "3P spatial (20ep, eval on held-out seeds 3408/3409)", heldout_cols=[3408, 3409])
    results["3p"]["weights_dev_s3407"] = dict(zip(FEATURES, [float(v) for v in w_dev]))

    # ---- CIDC25 时间轴（3 seed；权重在 3-seed 均值上拟合，gap 逐 seed 评估）----
    oraclec, probec = load_cidc(root)
    dts = oraclec.index.tolist()
    xc = probec.loc[dts, FEATURES].to_numpy(float)
    wc = fit_weights(xc, oraclec.loc[dts, "mean"].to_numpy(float))
    u_v2_c = pd.Series(xc @ wc, index=dts)
    fixedc = {
        "fixed-nearest(dt=1)": 1,
        "fixed-worst": int(oraclec["mean"].idxmin()),
    }
    results["cidc25"] = regime_gaps(oraclec, probec, u_v2_c, fixedc, "CIDC25 temporal (20ep × 3 seed, weights fit on mean, per-seed gaps)", heldout_cols=[3407, 3408, 3409])
    results["cidc25"]["weights_inregime_3seed"] = dict(zip(FEATURES, [float(v) for v in wc]))

    # ---- 汇总打印 ----
    for reg, r in results.items():
        print(f"\n=== {r['regime']} ===")
        print(f"oracle-best: {r['oracle_best']['arm']} (U={r['oracle_best']['u']:.3f})")
        for name in ("adaptive-hard(v2)", "fixed-VALID", "fixed-QMask", "fixed-nearest(dt=1)", "fixed-worst"):
            if name not in r:
                continue
            a = r[name]
            if "gap_heldout_mean" in a:
                per = ", ".join(f"s{s}={g:+.3f}" for s, g in a["gap_per_seed"].items())
                print(f"  {name:<24} {a['arm']:<16} gap(held-out)={a['gap_heldout_mean']:+.3f} dB  ({per})")
            else:
                print(f"  {name:<24} {a['arm']:<16} gap={a['gap']:+.3f} dB")

    def _default(o):  # numpy 标量转 JSON
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        raise TypeError(o)

    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(results, indent=1, ensure_ascii=False, default=_default))
        print(f"\n→ {args.out}")


if __name__ == "__main__":
    main()
