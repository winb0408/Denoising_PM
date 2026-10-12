"""green1300 真实数据 deep-signal 分析：深层结构/荧光信号突出 + 噪声抑制定量与定性。

配合 `calibration/oracle.py` 的 `save_checkpoint=True` 产物
（`runs/m2_oracle_green1300_seed3407/families/{family}_final.pth`）使用：

- 全体积推理：非重叠 tile（z 起点 [0,64,128,177]，y/x 起点 [0,128]）拼回 [Z,Y,X]；
  归一化用 checkpoint 自带 pool_stats，与训练/评估同口径；
- 定量（vs AVG-25 伪真值 + 无参考）：
  * PSNR(z 带)：输入帧 / 各 family 输出 vs AVG-25（psnr_db 与 oracle 同实现）；
  * SBR(z) / CNR(z)：逐 plane（top10% 亮 vs bottom50% 背景），输入/AVG/各 family 曲线；
  * 背景高频噪声：背景体素 Laplacian std（无参考噪声抑制 proxy）；
- 定性：深层区（z≥150）3 行（不同 z0 各取 AVG-25 最亮 tile）×
  （输入帧 | AVG-25 | 各 family）turbo 伪彩面板 + review 切块（≤1200px）。

输出目录 `<run-dir>/deepsignal/`：metrics.json、curves.csv、
fig_deep_curves.png（≤1200px 可直读）、fig_deep_qualitative.png（大图禁直读）、
review/deep_row_{1,2,3}.png。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from calibration.oracle import psnr_db  # noqa: E402
from calibration.render_qualitative_3p import (  # noqa: E402
    CMAP,
    ZOOM_PCTL,
    load_family_model,
    render_panel,
)

Z_PATCH, Y_PATCH, X_PATCH = 64, 128, 128
INPUT_FRAME = 0
# 深层定性行：z0 固定三档（覆盖 z 152–241），每档取 AVG-25 最亮 tile
DEEP_Z0_GRID = (152, 170, 177)
BANDS = [("shallow", 0, 80), ("mid", 80, 160), ("deep", 160, 241)]


def tile_starts(length: int, tile: int) -> list[int]:
    starts = list(range(0, length - tile + 1, tile))
    if starts[-1] + tile < length:
        starts.append(length - tile)
    return starts


def full_inference(net, mean: float, std: float, raw: np.ndarray, device) -> np.ndarray:
    """raw [Z,Y,X] → 去噪 [Z,Y,X]（非重叠 tile，边界 tile 右对齐）。"""
    z_len, y_len, x_len = raw.shape
    out = np.zeros_like(raw, dtype=np.float32)
    for z0 in tile_starts(z_len, Z_PATCH):
        for y0 in tile_starts(y_len, Y_PATCH):
            for x0 in tile_starts(x_len, X_PATCH):
                block = raw[z0 : z0 + Z_PATCH, y0 : y0 + Y_PATCH, x0 : x0 + X_PATCH]
                norm = (block.astype(np.float32) - mean) / std
                with torch.no_grad():
                    t = torch.from_numpy(norm)[None, None].to(device)
                    o = net(t)[0, 0].cpu().numpy().astype(np.float32) * std + mean
                out[z0 : z0 + Z_PATCH, y0 : y0 + Y_PATCH, x0 : x0 + X_PATCH] = o
    return out


def plane_sbr_cnr(img: np.ndarray) -> tuple[float, float]:
    """单 plane：SBR=top10% 均值 / bottom50% 均值；CNR=(μ_sig−μ_bg)/σ_bg。"""
    p = img.astype(np.float64)
    hi_thr = np.percentile(p, 90.0)
    lo_thr = np.percentile(p, 50.0)
    sig = p[p >= hi_thr]
    bg = p[p <= lo_thr]
    mu_sig, mu_bg = float(sig.mean()), float(bg.mean())
    sigma_bg = float(bg.std())
    sbr = mu_sig / max(mu_bg, 1e-6)
    cnr = (mu_sig - mu_bg) / max(sigma_bg, 1e-6)
    return sbr, cnr


def bg_lap_std(img: np.ndarray) -> float:
    """背景（bottom50%）体素 Laplacian std——无参考噪声强度 proxy。"""
    lap = (
        4.0 * img
        - np.roll(img, 1, 0) - np.roll(img, -1, 0)
        - np.roll(img, 1, 1) - np.roll(img, -1, 1)
    ).astype(np.float64)
    thr = np.percentile(img, 50.0)
    bg = lap[img <= thr]
    return float(bg.std())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="repeat stack tif [K,Z,Y,X]")
    parser.add_argument("--run-dir", required=True, help="oracle save_checkpoint run 目录")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    run_dir = Path(args.run_dir)
    out_dir = run_dir / "deepsignal"
    review_dir = out_dir / "review"
    review_dir.mkdir(parents=True, exist_ok=True)

    volume = np.asarray(tifffile.imread(args.data))
    if volume.ndim != 4:
        raise SystemExit(f"需要 repeat stack [K,Z,Y,X], got {volume.shape}")
    ref = volume.astype(np.float64).mean(axis=0).astype(np.float32)  # AVG-25
    raw = np.asarray(volume[INPUT_FRAME], dtype=np.float32)

    fam_ckpts = sorted((run_dir / "families").glob("*_final.pth"))
    families = [p.name[: -len("_final.pth")] for p in fam_ckpts]
    print(f"[deep] families: {families}", flush=True)

    outputs: dict[str, np.ndarray] = {}
    u_gains: dict[str, float] = {}
    for fam in families:
        net, mean, std, gain = load_family_model(run_dir, fam, device)
        outputs[fam] = full_inference(net, mean, std, raw, device)
        u_gains[fam] = gain
        print(f"[deep] {fam}: 推理完成 U_oracle={gain:+.2f}dB", flush=True)

    # ---------------- 定量 ----------------
    imgs: dict[str, np.ndarray] = {"input": raw, "avg25": ref, **outputs}
    z_len = raw.shape[0]

    metrics: dict = {"bands": [b[0] for b in BANDS], "psnr_band": {}, "sbr_band": {}, "cnr_band": {}, "bg_lap_band": {}}
    curves_rows = []
    for name, img in imgs.items():
        # 逐 plane 曲线
        for z in range(z_len):
            sbr, cnr = plane_sbr_cnr(img[z])
            curves_rows.append({"name": name, "z": z, "sbr": sbr, "cnr": cnr})
        # 带 agg + PSNR
        for bname, lo, hi in BANDS:
            band = img[lo:hi]
            ref_band = ref[lo:hi]
            if name != "avg25":
                metrics["psnr_band"].setdefault(name, {})[bname] = psnr_db(band, ref_band)
            sbrs, cnrs = [], []
            for z in range(lo, hi):
                s, c = plane_sbr_cnr(img[z])
                sbrs.append(s)
                cnrs.append(c)
            metrics["sbr_band"].setdefault(name, {})[bname] = float(np.mean(sbrs))
            metrics["cnr_band"].setdefault(name, {})[bname] = float(np.mean(cnrs))
            metrics["bg_lap_band"].setdefault(name, {})[bname] = bg_lap_std(band)
    metrics["u_oracle_gain"] = u_gains

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    with open(out_dir / "curves.csv", "w", encoding="utf-8") as f:
        f.write("name,z,sbr,cnr\n")
        for r in curves_rows:
            f.write(f'{r["name"]},{r["z"]},{r["sbr"]:.4f},{r["cnr"]:.4f}\n')

    # ---------------- 曲线图（≤1200px 直读） ----------------
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6), dpi=100)
    names = ["input", "avg25"] + families
    colors = plt.cm.tab10(np.linspace(0, 1, len(names)))
    for name, color in zip(names, colors):
        zs = [r["z"] for r in curves_rows if r["name"] == name]
        for ax, key in zip(axes, ["sbr", "cnr"]):
            vals = [r[key] for r in curves_rows if r["name"] == name]
            ax.plot(zs, vals, label=name, color=color, lw=0.9,
                    linestyle="--" if name == "input" else "-")
    axes[0].set_title("SBR(z)", fontsize=9)
    axes[1].set_title("CNR(z)", fontsize=9)
    # 第三格：带 PSNR 条形
    fam_names = ["input"] + families
    band_x = np.arange(len(BANDS))
    width = 0.8 / len(fam_names)
    for i, name in enumerate(fam_names):
        vals = [metrics["psnr_band"][name][b[0]] for b in BANDS]
        axes[2].bar(band_x + (i - len(fam_names) / 2) * width, vals, width,
                    label=name, color=colors[i if name == "input" else names.index(name)])
    axes[2].set_xticks(band_x)
    axes[2].set_xticklabels([b[0] for b in BANDS], fontsize=8)
    axes[2].set_title("PSNR vs AVG-25 (band)", fontsize=9)
    for ax in axes:
        ax.legend(fontsize=5, ncol=2)
    fig.suptitle("green1300 deep-signal: SBR/CNR curves + band PSNR", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_deep_curves.png")
    plt.close(fig)

    # ---------------- 深层定性面板 ----------------
    chosen = []
    for z0 in DEEP_Z0_GRID:
        best, best_key = None, -1.0
        for y0 in range(0, raw.shape[1] - Y_PATCH + 1, Y_PATCH):
            for x0 in range(0, raw.shape[2] - X_PATCH + 1, X_PATCH):
                key = float(ref[z0 : z0 + Z_PATCH, y0 : y0 + Y_PATCH, x0 : x0 + X_PATCH].max())
                if key > best_key:
                    best, best_key = (z0, y0, x0), key
        chosen.append(best)

    ncols = 2 + len(families)
    all_rows = []
    for r, (z0, y0, x0) in enumerate(chosen):
        z_mid = Z_PATCH // 2
        panels = [(raw[z0 + z_mid, y0 : y0 + Y_PATCH, x0 : x0 + X_PATCH], f"input (z={z0 + z_mid})"),
                  (ref[z0 + z_mid, y0 : y0 + Y_PATCH, x0 : x0 + X_PATCH], "AVG-25 pseudo-GT")]
        for fam in families:
            o = outputs[fam][z0 + z_mid, y0 : y0 + Y_PATCH, x0 : x0 + X_PATCH]
            panels.append((o, f"{fam}\nU={u_gains[fam]:+.2f}dB"))
        all_rows.append(panels)

    fig, axes = plt.subplots(3, ncols, figsize=(1.6 * ncols, 1.6 * 3), dpi=200)
    for r, panels in enumerate(all_rows):
        lo = min(float(np.percentile(p, ZOOM_PCTL[0])) for p, _ in panels)
        hi = max(float(np.percentile(p, ZOOM_PCTL[1])) for p, _ in panels)
        for c, (img, title) in enumerate(panels):
            render_panel(axes[r, c], img, title, lo, hi)
    fig.suptitle("green1300 deep region (z≥152): input | AVG-25 | families", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_deep_qualitative.png")
    plt.close(fig)

    tile_dpi = int(1150 / (1.5 * ncols))
    for r, panels in enumerate(all_rows):
        fig, ax = plt.subplots(1, ncols, figsize=(1.5 * ncols, 1.7), dpi=tile_dpi)
        lo = min(float(np.percentile(p, ZOOM_PCTL[0])) for p, _ in panels)
        hi = max(float(np.percentile(p, ZOOM_PCTL[1])) for p, _ in panels)
        for c, (img, title) in enumerate(panels):
            render_panel(ax[c], img, title, lo, hi)
        fig.tight_layout()
        fig.savefig(review_dir / f"deep_row_{r + 1}.png")
        plt.close(fig)

    print(f"[deep] 完成：{out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
