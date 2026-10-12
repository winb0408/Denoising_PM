"""Gate P2→P3 终判: 10 家族 × 3 seed oracle 均值上重拟合 U_probe v2。

用法（仓库根目录）:
  PYTHONPATH=ACES python -m calibration.v2_judgment \
    --seed-dirs /data3/.../m2_oracle_3p_seed3407,/data3/.../m2_oracle_3p_seed3408,/data3/.../m2_oracle_3p_seed3409 \
    --probe-csv /data3/.../m2_micro_probe_3p_seed3407/micro_probe.csv \
    --out-dir /data3/.../m2_oracle_3p_seed3407

输出:
  - u_oracle_3seed_full.json : 逐家族 mean/std/seeds
  - u_probe_v2_3seed.json   : v2 权重(在 3-seed 均值 oracle 上拟合) + full/LOO Spearman
  - stdout 判定表
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

FEATURES = ["loss_neighbor_mean", "cross_view_consistency"]


def load_oracle(seed_dirs: list[Path]) -> pd.DataFrame:
    """优先从 families/*.json 合并（resume 场景 oracle.csv 可能只含本次 run 的家族）。"""
    rows = []
    for d in seed_dirs:
        fam_dir = d / "families"
        if fam_dir.is_dir() and any(fam_dir.glob("*.json")):
            for jf in sorted(fam_dir.glob("*.json")):
                if jf.name == "summary.json":
                    continue
                cached = json.loads(jf.read_text(encoding="utf-8"))
                rows.append(
                    {
                        "family_id": cached["family_id"],
                        "u_oracle_gain": cached["eval"]["u_oracle_gain"],
                        "seed_dir": d.name,
                    }
                )
        else:
            csv = d / "oracle.csv"
            if not csv.exists():
                raise FileNotFoundError(csv)
            df = pd.read_csv(csv)
            df["seed_dir"] = d.name
            rows.append(df[["family_id", "u_oracle_gain", "seed_dir"]])
    long = pd.DataFrame(rows)
    return long


def fit_v2(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float, float, list[float]]:
    """2 特征无截距线性回归 + Spearman(full) + Spearman(LOO)。"""
    n = len(y)
    w, *_ = np.linalg.lstsq(x, y, rcond=None)
    rho_full = spearmanr(x @ w, y).statistic
    pred_loo = np.empty(n)
    for i in range(n):
        m = np.ones(n, dtype=bool)
        m[i] = False
        wi, *_ = np.linalg.lstsq(x[m], y[m], rcond=None)
        pred_loo[i] = x[i] @ wi
    rho_loo = spearmanr(pred_loo, y).statistic
    return w, rho_full, float(rho_loo), [float(v) for v in pred_loo]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-dirs", required=True, help="逗号分隔的 3 个 oracle run 目录")
    ap.add_argument("--probe-csv", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    seed_dirs = [Path(p) for p in args.seed_dirs.split(",")]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    long = load_oracle(seed_dirs)

    agg = long.groupby("family_id")["u_oracle_gain"].agg(["mean", "std"])
    per_seed = {
        fid: {
            "mean": float(g["u_oracle_gain"].mean()),
            "std": float(g["u_oracle_gain"].std(ddof=1)) if len(g) > 1 else 0.0,
            "seeds": [float(v) for v in g["u_oracle_gain"]],
        }
        for fid, g in long.groupby("family_id")
    }
    out_dir.joinpath("u_oracle_3seed_full.json").write_text(
        json.dumps(per_seed, indent=1, sort_keys=True)
    )

    probe = pd.read_csv(args.probe_csv).set_index("family_id")
    fams = agg.index.tolist()
    missing = [f for f in fams if f not in probe.index]
    if missing:
        raise SystemExit(f"probe.csv 缺少家族: {missing}")

    x = probe.loc[fams, FEATURES].to_numpy(dtype=float)
    y = agg.loc[fams, "mean"].to_numpy(dtype=float)
    w, rho_full, rho_loo, loo_list = fit_v2(x, y)

    result = {
        "features": FEATURES,
        "weights": [float(v) for v in w],
        "spearman_full": float(rho_full),
        "spearman_loo": float(rho_loo),
        "u_probe_v2_loo_pred": {f: float(v) for f, v in zip(fams, loo_list)},
        "n_families": len(fams),
        "n_seeds": len(seed_dirs),
        "u_oracle_mean": {f: float(v) for f, v in zip(fams, y)},
        "u_probe_v2": {f: float(v) for f, v in zip(fams, x @ w)},
        "probe_features": {
            f: {feat: float(probe.loc[f, feat]) for feat in FEATURES} for f in fams
        },
    }
    out_dir.joinpath("u_probe_v2_3seed.json").write_text(
        json.dumps(result, indent=1, sort_keys=True)
    )

    order = np.argsort(-y)
    print(f"\n=== Gate P2→P3 终判 ({len(fams)} 家族 × {len(seed_dirs)} seed) ===")
    print(f"{'family':<16}{'U_oracle(mean±std)':<22}{'U_v2':<8}")
    for i in order:
        f = fams[i]
        print(f"{f:<16}{y[i]:>7.3f} ± {per_seed[f]['std']:.3f}      {x[i] @ w:>7.3f}")
    print(f"\nweights: {dict(zip(FEATURES, np.round(w, 4)))}")
    print(f"Spearman(full) = {rho_full:+.4f}")
    print(f"Spearman(LOO)  = {rho_loo:+.4f}")


if __name__ == "__main__":
    main()
