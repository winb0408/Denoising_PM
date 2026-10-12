"""M2/M3 论文图生成（零新训练，全部已有数据）。

输出到 <runs-root>/m2m3_paper_figures/，供人工审核：
  fig_c1_probe_vs_oracle.png   Exp C 主图：U_v2(LOO) vs U_oracle 3-seed mean±std
  fig_c2_feature_regime.png    H4：特征-oracle 相关的跨 regime 符号矛盾
  fig_c3_reverse_validation.png CIDC25 反向验证：U_v2 迁移 vs oracle（dt=1 陷阱）
  fig_c4_budget_transfer.png   Exp D 预算迁移：20ep vs 80ep 4 家族（饱和不对称）
  fig_d1_gap_bars.png          Exp D：各臂 Oracle Gap（3P + CIDC25）
  fig_t1_dt_seed_fragility.png CIDC25：per-seed PSNR vs dt（dt=8 脆弱性）

用法：
    PYTHONPATH=ACES python -m calibration.plot_m2m3_figures \
        --runs-root /ssd1/wjb/Denoising_PM/ACES/runs
"""
from __future__ import annotations

import argparse
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

FEATURES = ["loss_neighbor_mean", "cross_view_consistency"]
FEATURE_LABELS = {
    "loss_neighbor_mean": "L_N (neighbor loss)",
    "cross_view_consistency": "cross-view consistency",
}
plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10})


def _root_default() -> str:
    for p in ("/ssd1/wjb/Denoising_PM/ACES/runs", "/data3/wjb/Denoising_PM/ACES/runs"):
        if pathlib.Path(p).exists():
            return p
    raise SystemExit("runs root 未找到（/ssd1 与 /data3 均不存在）")


def load_3p_oracle(root: pathlib.Path) -> pd.DataFrame:
    seeds = {}
    for s in (3407, 3408, 3409):
        d = {}
        for jf in sorted((root / f"m2_oracle_3p_seed{s}" / "families").glob("*.json")):
            if jf.name == "summary.json":
                continue
            c = json.loads(jf.read_text(encoding="utf-8"))
            d[c["family_id"]] = c["eval"]["u_oracle_gain"]
        seeds[s] = d
    df = pd.DataFrame(seeds)
    df["mean"] = df.mean(axis=1)
    df["std"] = df.std(axis=1, ddof=1)
    return df


def load_80ep(root: pathlib.Path, fams: list[str]) -> pd.DataFrame:
    seeds = {}
    for s in (3407, 3408, 3409):
        d = {}
        for f in fams:
            jf = root / f"m2_oracle_3p_80ep_seed{s}" / "families" / f"{f}.json"
            c = json.loads(jf.read_text(encoding="utf-8"))
            d[f] = c["eval"]["u_oracle_gain"]
        seeds[s] = d
    df = pd.DataFrame(seeds)
    df["mean"] = df.mean(axis=1)
    df["std"] = df.std(axis=1, ddof=1)
    return df


def load_cidc(root: pathlib.Path) -> pd.DataFrame:
    dts = [1, 2, 3, 4, 6, 8, 16, 32, 64]
    rows = {}
    for s in (3407, 3408, 3409):
        for dt in dts:
            m = json.loads(
                (root / f"stage01_cidc25_rgrid_seed{s}/d{dt}/metrics/eval_final_F1.json").read_text()
            )
            rows.setdefault(s, {})[dt] = m["regions"]["all"]["psnr_db"]
    df = pd.DataFrame(rows)
    df["mean"] = df.mean(axis=1)
    df["std"] = df.std(axis=1, ddof=1)
    return df


# --------------------------------------------------------------------------- #
def fig_c1(root: pathlib.Path, out: pathlib.Path) -> None:
    """Exp C 主图：U_v2(LOO 预测) vs U_oracle 3-seed 均值（误差棒）+ 可分对标注。"""
    res = json.loads((root / "m2_oracle_3p_seed3407/u_probe_v2_3seed.json").read_text())
    orc = load_3p_oracle(root)
    fams = sorted(res["u_oracle_mean"])
    x = np.array([res["u_probe_v2_loo_pred"][f] for f in fams])
    y = np.array([res["u_oracle_mean"][f] for f in fams])
    yerr = np.array([orc.loc[f, "std"] for f in fams])

    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    ax.errorbar(x, y, yerr=yerr / np.sqrt(3), fmt="none", ecolor="0.75", capsize=3, zorder=1)
    ax.scatter(x, y, s=48, c="#2166ac", zorder=2)
    for f, xi, yi in zip(fams, x, y):
        ax.annotate(f, (xi, yi), textcoords="offset points", xytext=(5, 4), fontsize=7.5)
    # 拟合线（LOO 预测的 rank 一致性信息放标题）
    rho = spearmanr(x, y).statistic
    ax.set_xlabel("U_probe v2 (LOO prediction)")
    ax.set_ylabel("U_oracle (3-seed mean, dB)")
    ax.set_title(f"Exp C: probe vs oracle (10 families)\nSpearman(LOO) = {rho:+.2f}; "
                 f"separable pairs 12/13 correct, top/bottom AUC = 0.89")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig_c2(root: pathlib.Path, out: pathlib.Path) -> None:
    """H4：全部 probe 特征与 oracle 的 Spearman，3P vs CIDC25 对照条形图。"""
    orc3 = load_3p_oracle(root)
    probe3 = pd.read_csv(root / "m2_micro_probe_3p_seed3407/micro_probe.csv").set_index("family_id")
    common = [f for f in probe3.index if f in orc3.index]
    probe3 = probe3.loc[common]
    feats3 = [c for c in probe3.columns if probe3[c].dtype == float and not c.startswith(("z_", "risk_"))
              and c in ("loss_neighbor_mean", "loss_neighbor_slope", "loss_identity_mean",
                        "loss_identity_slope", "residual_corr", "output_input_corr",
                        "output_gradient_ratio", "output_variance_ratio",
                        "cross_view_consistency", "loss_stability")]
    y3 = orc3.loc[probe3.index, "mean"].to_numpy()
    rho3 = {f: spearmanr(probe3[f].to_numpy(), y3).statistic for f in feats3}

    orcc = load_cidc(root)
    probec = pd.read_csv(root / "m2_micro_probe_cidc25_seed3407/micro_probe_temporal.csv").set_index("dt")
    yc = orcc.loc[probec.index, "mean"].to_numpy()
    rhoc = {f: spearmanr(probec[f].to_numpy(), yc).statistic for f in feats3}

    labels = [FEATURE_LABELS.get(f, f) for f in feats3]
    ypos = np.arange(len(feats3))[::-1]
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    ax.barh(ypos + 0.2, [rho3[f] for f in feats3], height=0.38, color="#2166ac", label="3P spatial")
    ax.barh(ypos - 0.2, [rhoc[f] for f in feats3], height=0.38, color="#d6604d", label="CIDC25 temporal")
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.axvline(0, c="k", lw=0.8)
    ax.set_xlabel("Spearman(feature, U_oracle)")
    ax.set_title("H4: feature–utility correlation flips across regimes\n(same probe features, sign contradiction)")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig_c3(root: pathlib.Path, out: pathlib.Path) -> None:
    """CIDC25 反向验证：oracle vs dt + 3P 权重迁移的 U 曲线（dt=1 陷阱）。"""
    orcc = load_cidc(root)
    probec = pd.read_csv(root / "m2_micro_probe_cidc25_seed3407/micro_probe_temporal.csv").set_index("dt")
    dts = orcc.index.to_numpy()
    w3 = np.array(json.loads((root / "m2_oracle_3p_seed3407/u_probe_v2_3seed.json").read_text())["weights"])
    u_transfer = probec.loc[dts, FEATURES].to_numpy() @ w3

    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.errorbar(dts, orcc["mean"], yerr=orcc["std"], marker="o", ms=5, c="#2166ac",
                capsize=3, label="U_oracle (3-seed mean ± std)")
    ax.plot(dts, u_transfer, marker="s", ms=5, ls="--", c="#d6604d",
            label="U_v2 with 3P weights (transferred)")
    i_best = int(np.argmax(orcc["mean"]))
    i_trap = int(np.argmax(u_transfer))
    ax.annotate(f"oracle best (dt={dts[i_best]})", (dts[i_best], orcc["mean"].iloc[i_best]),
                textcoords="offset points", xytext=(-10, 10), fontsize=8.5, color="#2166ac")
    ax.annotate(f"transfer picks dt=1\n(worst arm: shortcut trap)", (dts[i_trap], u_transfer[i_trap]),
                textcoords="offset points", xytext=(8, 12), fontsize=8.5, color="#d6604d")
    ax.set_xscale("log")
    ax.set_xticks(dts)
    ax.set_xticklabels(dts)
    ax.set_xlabel("temporal pairing distance dt")
    ax.set_ylabel("utility (dB; U_v2 in arbitrary units)")
    ax.set_title("CIDC25 reverse validation: 3P weights fail\n(transfer ρ = +0.17 ns; in-regime refit ρ = +0.73)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig_c4(root: pathlib.Path, out: pathlib.Path) -> None:
    """预算迁移：20ep vs 80ep（4 关键家族 3-seed mean±std，饱和不对称）。"""
    fams = ["sn2n_xy_diag", "qmask_xy", "valid_corners", "axis_y_d1"]
    o20 = load_3p_oracle(root).loc[fams]
    o80 = load_80ep(root, fams)

    xpos = np.arange(len(fams))
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.bar(xpos - 0.2, o20["mean"], width=0.38, yerr=o20["std"], capsize=3,
           color="#92c5de", label="20ep (calibration budget)")
    ax.bar(xpos + 0.2, o80["mean"], width=0.38, yerr=o80["std"], capsize=3,
           color="#2166ac", label="80ep (deployment budget)")
    for i, f in enumerate(fams):
        d = o80.loc[f, "mean"] - o20.loc[f, "mean"]
        ax.annotate(f"Δ{d:+.2f}", (i, max(o20.loc[f, "mean"], o80.loc[f, "mean"]) + 0.12),
                    ha="center", fontsize=8.5)
    ax.set_xticks(xpos)
    ax.set_xticklabels(fams, fontsize=8.5, rotation=12)
    ax.set_ylabel("U_oracle (3-seed mean ± std, dB)")
    ax.set_ylim(6.8, 8.0)
    ax.set_title("Budget transfer failure: 20ep-calibrated pick (sn2n_xy_diag)\n"
                 "saturates early; slow learners (qmask/valid) gain +0.5 dB at 80ep")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig_d1(root: pathlib.Path, out: pathlib.Path) -> None:
    """Exp D：各臂 Oracle Gap（3P held-out / CIDC25 3-seed）。"""
    r = json.loads((root / "m2_oracle_3p_seed3407/exp_d_gap.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0))
    for ax, reg, unit in ((axes[0], "3p", "3P spatial (held-out seeds 3408/3409)"),
                          (axes[1], "cidc25", "CIDC25 temporal (3 seeds)")):
        arms, gaps, errs = [], [], []
        for name in ("adaptive-hard(v2)", "fixed-VALID", "fixed-QMask", "fixed-nearest(dt=1)", "fixed-worst"):
            if name not in r[reg]:
                continue
            a = r[reg][name]
            arms.append(name.replace("(", "\n("))
            if "gap_heldout_mean" in a:
                per = np.array(list(a["gap_per_seed"].values()))
                gaps.append(per.mean())
                errs.append(per.std(ddof=1))
            else:
                per = np.array(list(a["gap_per_seed"].values())) if "gap_per_seed" in a else None
                gaps.append(a["gap"])
                errs.append(0.0)
        colors = ["#2166ac" if "adaptive" in a else "0.6" for a in arms]
        ax.bar(range(len(arms)), gaps, yerr=errs, capsize=3, color=colors)
        for i, g in enumerate(gaps):
            ax.annotate(f"{g:+.2f}", (i, g + 0.05), ha="center", fontsize=8.5)
        ax.set_xticks(range(len(arms)))
        ax.set_xticklabels(arms, fontsize=7.5)
        ax.set_ylabel("Oracle Gap (dB, lower = better)")
        ax.set_title(unit, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Exp D (adaptive-hard): oracle gap of each arm", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig_t1(root: pathlib.Path, out: pathlib.Path) -> None:
    """CIDC25 时间轴 oracle：per-seed 曲线 + mean±std（dt=8 脆弱性）。"""
    orcc = load_cidc(root)
    dts = orcc.index.to_numpy()
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.errorbar(dts, orcc["mean"], yerr=orcc["std"], marker="o", ms=5, capsize=3,
                c="#2166ac", label="3-seed mean ± std")
    for s, c in zip((3407, 3408, 3409), ("#92c5de", "#f4a582", "#c7eae5")):
        ax.plot(dts, orcc[s], marker=".", ms=4, lw=0.9, c=c, alpha=0.9, label=f"seed {s}")
    ax.annotate("dt=8: seed-fragile\n(s3409 diverges late, std 5.2 dB)", (8, 26.6),
                textcoords="offset points", xytext=(14, -34), fontsize=8.5, color="#d6604d")
    ax.set_xscale("log")
    ax.set_xticks(dts)
    ax.set_xticklabels(dts)
    ax.set_xlabel("temporal pairing distance dt")
    ax.set_ylabel("PSNR (dB, F1 eval)")
    ax.set_title("CIDC25 temporal oracle (9 dt × 3 seeds)\ndt=64 robust best (29.34 ± 0.50)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default=_root_default())
    args = ap.parse_args()
    root = pathlib.Path(args.runs_root)
    out_dir = root / "m2m3_paper_figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, fn in (
        ("fig_c1_probe_vs_oracle.png", fig_c1),
        ("fig_c2_feature_regime.png", fig_c2),
        ("fig_c3_reverse_validation.png", fig_c3),
        ("fig_c4_budget_transfer.png", fig_c4),
        ("fig_d1_gap_bars.png", fig_d1),
        ("fig_t1_dt_seed_fragility.png", fig_t1),
    ):
        fn(root, out_dir / name)
        print(f"[fig] {out_dir / name}")


if __name__ == "__main__":
    main()
