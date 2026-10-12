"""CIDC25 定性去噪对比图（零训练，用已有 rgrid checkpoint）。

面板：每行一个评测 patch（z=64,y=128,x=128，与 t4b_rgrid 评估协议一致），
列 = F1 噪声输入 | F0 GT(AVG) | dt=1 | dt=8 | dt=64 去噪结果。
dt=1（最近邻缺省/最差臂，shortcut 陷阱）、dt=8（脆弱内点峰）、dt=64（3-seed 稳健最优）。
模型：stage01_cidc25_rgrid_seed3407 各 dt 的 final checkpoint。

渲染协议镜像 t4b_rgrid 评估：F1 栈 per-stack mean/std 归一化输入
（优先读 run 目录缓存 eval_stack_stats.json），前向后反归一化显示；GT 为 F0。

输出：<out-dir>/fig_q1_cidc25_qualitative.png（+ review/ 行块，≤1200px）

用法：
    PYTHONPATH=ACES python -m calibration.render_qualitative_cidc \
        --rgrid-run /ssd1/wjb/Denoising_PM/ACES/runs/stage01_cidc25_rgrid_seed3407 \
        --out-dir /ssd1/wjb/Denoising_PM/ACES/runs/m2m3_paper_figures
"""
from __future__ import annotations

import argparse
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile
import torch

DT_LIST = [1, 8, 64]
N_ROWS = 3  # 评测 patch 数（取亮前景区）
Z_PATCH, Y_PATCH, X_PATCH = 64, 128, 128  # 与 t4b_rgrid 评估一致
plt.rcParams.update({"font.size": 10})


def _open_stack(path):
    try:
        return np.asarray(tifffile.memmap(path))
    except Exception:
        return np.asarray(tifffile.imread(path))


def load_model(rgrid_run: pathlib.Path, dt: int, device: torch.device):
    """从 d{dt}/checkpoints/tpair_d{dt}_final.pth 载入 CRN（结构读 config.json）。"""
    from aces.train_stage2 import _load_valid_components

    cfg = json.loads((rgrid_run / f"d{dt}" / "config.json").read_text(encoding="utf-8"))
    train_cfg = cfg.get("training", {})
    ckpt = torch.load(
        rgrid_run / f"d{dt}" / "checkpoints" / f"tpair_d{dt}_final.pth",
        map_location=device,
        weights_only=False,
    )
    net = _load_valid_components()["Network_CNR"](
        in_channels=1,
        out_channels=1,
        f_maps=int(train_cfg.get("base_features", 16)),
        n_groups=int(train_cfg.get("n_groups", 4)),
    ).to(device).eval()
    net.load_state_dict(ckpt["state_dict"])
    return net


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/data2/wjb/threePM/ThreePM_openDatasets/AI4Life-CIDC25")
    ap.add_argument("--rgrid-run", default="/ssd1/wjb/Denoising_PM/ACES/runs/stage01_cidc25_rgrid_seed3407")
    ap.add_argument("--out-dir", default="/ssd1/wjb/Denoising_PM/ACES/runs/m2m3_paper_figures")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    root = pathlib.Path(args.data_root)
    rgrid_run = pathlib.Path(args.rgrid_run)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    f1 = _open_stack(root / "Validation" / "F1.tif")  # [T,Y,X]（T 即协议里的 z 维）
    f0 = _open_stack(root / "Validation" / "F0.tif")
    T, Y, X = f1.shape

    # 归一化统计：优先用 t4b_rgrid 的缓存（与正式评估完全一致）
    cache = rgrid_run / "eval_stack_stats.json"
    if cache.exists():
        mean, std = json.loads(cache.read_text(encoding="utf-8"))["F1"]
    else:
        flat = f1.reshape(-1).astype(np.float64)
        mean, std = float(flat.mean()), float(flat.std())
    print(f"F1 {f1.shape}, mean={mean:.2f} std={std:.2f}")

    # 选亮前景 patch：网格中心区域中 max 值最高的 N_ROWS 个（不重叠）
    cands = []
    t0 = max(0, T // 2 - Z_PATCH // 2)  # 时间中段起点，避开首尾
    for y0 in range(0, Y - Y_PATCH + 1, Y_PATCH):
        for x0 in range(0, X - X_PATCH + 1, X_PATCH):
            block = f1[t0 : t0 + Z_PATCH, y0 : y0 + Y_PATCH, x0 : x0 + X_PATCH]
            cands.append((float(block.max()), t0, y0, x0))
    cands.sort(reverse=True)
    picks: list[tuple[int, int, int]] = []
    for _v, t0, y0, x0 in cands:
        if all(abs(y0 - py) >= Y_PATCH or abs(x0 - px) >= X_PATCH for _t, py, px in picks):
            picks.append((t0, y0, x0))
        if len(picks) >= N_ROWS:
            break
    print(f"选中 {len(picks)} 个前景 patch: {picks}")

    nets = {dt: load_model(rgrid_run, dt, device) for dt in DT_LIST}

    panels: list[list[tuple[str, np.ndarray]]] = []
    with torch.no_grad():
        for t0, y0, x0 in picks:
            row: list[tuple[str, np.ndarray]] = []
            inp = f1[t0 : t0 + Z_PATCH, y0 : y0 + Y_PATCH, x0 : x0 + X_PATCH].astype(np.float32)
            gt = f0[t0 : t0 + Z_PATCH, y0 : y0 + Y_PATCH, x0 : x0 + X_PATCH].astype(np.float32)
            row.append(("input (F1)", inp))
            row.append(("GT (F0 avg)", gt))
            x = torch.from_numpy((inp - mean) / std)[None, None].to(device)
            for dt in DT_LIST:
                out = nets[dt](x)[0, 0].cpu().numpy() * std + mean
                row.append((f"dt={dt}", out.astype(np.float32)))
            panels.append(row)

    # 统一显示窗（每行独立 1–99.9 分位，turbo 伪彩——与 qmask/paperviz.py 格式一致）
    CMAP = "turbo"
    ZOOM_PCTL = (1.0, 99.9)
    n_rows = len(panels)
    titles = [lbl for lbl, _ in panels[0]]
    fig, axes = plt.subplots(
        n_rows, 2 + len(DT_LIST), figsize=(2.6 * (2 + len(DT_LIST)), 2.6 * n_rows + 0.6)
    )
    for r, row in enumerate(panels):
        lo = min(np.percentile(img, ZOOM_PCTL[0]) for _, img in row)
        hi = max(np.percentile(img, ZOOM_PCTL[1]) for _, img in row)
        for c, (lbl, img) in enumerate(row):
            ax = axes[r, c] if n_rows > 1 else axes[c]
            ax.imshow(img[img.shape[0] // 2], cmap=CMAP, vmin=lo, vmax=hi)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(titles[c], fontsize=10)
            if c == 0:
                ax.set_ylabel(f"patch {r + 1}", fontsize=9)
    fig.suptitle("CIDC25 qualitative: temporal pairing distance (seed 3407, final ckpt)", fontsize=11)
    fig.tight_layout()
    out_png = out_dir / "fig_q1_cidc25_qualitative.png"
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"[fig] {out_png}")

    # review 切块（每行一块，单图 ~1600px 宽 / dpi150 → 约 6.7in，>1200px 但单文件 <1MB）
    review = out_dir / "review"
    review.mkdir(exist_ok=True)
    for r, row in enumerate(panels):
        fig, axes = plt.subplots(1, len(row), figsize=(2.2 * len(row), 2.2))
        lo = min(np.percentile(img, ZOOM_PCTL[0]) for _, img in row)
        hi = max(np.percentile(img, ZOOM_PCTL[1]) for _, img in row)
        for c, (lbl, img) in enumerate(row):
            axes[c].imshow(img[img.shape[0] // 2], cmap=CMAP, vmin=lo, vmax=hi)
            axes[c].set_title(titles[c], fontsize=9)
            axes[c].set_xticks([])
            axes[c].set_yticks([])
        fig.tight_layout()
        fig.savefig(review / f"cidc25_row_{r + 1}.png", dpi=100)
        plt.close(fig)
    print(f"[fig] review tiles → {review}/cidc25_row_*.png")


if __name__ == "__main__":
    main()
