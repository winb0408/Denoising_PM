"""3P 定性（视觉）对比面板：4 family pilot 去噪结果 vs AVG-9 伪真值。

配合 `calibration/oracle.py` 的 `save_checkpoint=True` 重训练产物
（`runs/m2_qual_3p_seed3407/families/{family}_final.pth`）使用：

- 归一化：每个 checkpoint 自带 pool_stats（训练池全局 mean/std），推理同口径；
- 伪真值：9 帧 repeat 在同一 patch 区域的均值（AVG-9），与 oracle 评估一致；
- patch：64×128×128，z 中切片显示，逐行 1–99.5 百分位显示窗；
- 面板：输入（frame 0 raw）| AVG-9 伪 GT | qmask_xy | valid_corners |
  sn2n_xy_diag | axis_z_d2（各列标注该 checkpoint 的 U_oracle 增益）。

主图 fig_q2_3p_qualitative.png（>1200px，禁止直接 Read）；
review 切块 review/3p_row_{1,2,3}.png（≤1200px，人工审阅用）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import tifffile

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from aces.train_stage2 import _load_valid_components  # noqa: E402

FAMILIES = ["qmask_xy", "valid_corners", "sn2n_xy_diag", "axis_z_d2"]
Z_PATCH, Y_PATCH, X_PATCH = 64, 128, 128
INPUT_FRAME = 0  # 展示用的 repeat 帧序号（输入列）


def load_family_model(run_dir: Path, family: str, device: torch.device):
    ckpt_path = run_dir / "families" / f"{family}_final.pth"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    net = _load_valid_components()["Network_CNR"](
        in_channels=1,
        out_channels=1,
        f_maps=int(cfg["base_features"]),
        n_groups=int(cfg["n_groups"]),
    ).to(device).eval()
    net.load_state_dict(ckpt["state_dict"])
    stats = ckpt["pool_stats"]
    return net, float(stats["mean"]), float(stats["std"]), float(ckpt.get("u_oracle_gain", float("nan")))


def select_rows(volume: np.ndarray, n_rows: int) -> list[tuple[int, int, int]]:
    """每个 z 三分带各选一个 AVG-9 最亮 patch（非重叠 128 网格）。"""
    _k, z_len, y_len, x_len = volume.shape
    ref_full = volume.astype(np.float64).mean(axis=0)
    band_edges = [0, z_len // 3, 2 * z_len // 3, z_len]
    chosen: list[tuple[int, int, int]] = []
    for b in range(3):
        z_lo, z_hi = band_edges[b], band_edges[b + 1]
        z0 = max(0, min(z_hi - Z_PATCH, (z_lo + z_hi - Z_PATCH) // 2))
        best, best_key = None, -1.0
        for y0 in range(0, y_len - Y_PATCH + 1, Y_PATCH):
            for x0 in range(0, x_len - X_PATCH + 1, X_PATCH):
                block = ref_full[z0 : z0 + Z_PATCH, y0 : y0 + Y_PATCH, x0 : x0 + X_PATCH]
                key = float(block.max())
                if key > best_key:
                    best, best_key = (z0, y0, x0), key
        chosen.append(best)
        assert best is not None
    return chosen[:n_rows]


CMAP = "turbo"  # 与 qmask/paperviz.py 格式一致
ZOOM_PCTL = (1.0, 99.9)  # paperviz ZOOM_PCTL


def render_panel(ax, img: np.ndarray, title: str, vmin: float, vmax: float) -> None:
    ax.imshow(img, cmap=CMAP, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=7)
    ax.axis("off")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        default="/ssd1/wjb/Denoising_PM/ACES/data/3P_neuron_repeat_stack.tif",
    )
    parser.add_argument(
        "--run-dir",
        default="/ssd1/wjb/Denoising_PM/ACES/runs/m2_qual_3p_seed3407",
        help="save_checkpoint=True 的 oracle 重训练 run 目录",
    )
    parser.add_argument(
        "--out-dir",
        default="/ssd1/wjb/Denoising_PM/ACES/runs/m2m3_paper_figures",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    volume = np.asarray(tifffile.imread(args.data))
    if volume.ndim != 4:
        raise SystemExit(f"需要 repeat stack [K,Z,Y,X], got {volume.shape}")

    nets = {}
    for fam in FAMILIES:
        ckpt = Path(args.run_dir) / "families" / f"{fam}_final.pth"
        if not ckpt.exists():
            raise SystemExit(f"缺 checkpoint: {ckpt}（训练是否完成？）")
        nets[fam] = load_family_model(Path(args.run_dir), fam, device)

    rows = select_rows(volume, n_rows=3)
    out_dir = Path(args.out_dir)
    review_dir = out_dir / "review"
    review_dir.mkdir(parents=True, exist_ok=True)

    ncols = 2 + len(FAMILIES)
    fig, axes = plt.subplots(3, ncols, figsize=(2.6 * ncols, 2.6 * 3), dpi=200)
    band_names = ["shallow", "mid", "deep"]
    all_row_panels: list[list[tuple[np.ndarray, str]]] = []
    row_windows: list[tuple[float, float]] = []
    for r, (z0, y0, x0) in enumerate(rows):
        region = volume[
            :,
            z0 : z0 + Z_PATCH,
            y0 : y0 + Y_PATCH,
            x0 : x0 + X_PATCH,
        ]
        ref = region.astype(np.float64).mean(axis=0).astype(np.float32)
        raw = np.asarray(region[INPUT_FRAME], dtype=np.float32)
        z_mid = Z_PATCH // 2
        panels: list[tuple[np.ndarray, str]] = [
            (raw[z_mid], f"input (frame {INPUT_FRAME})"),
            (ref[z_mid], "AVG-9 pseudo-GT"),
        ]
        for fam in FAMILIES:
            net, mean, std, gain = nets[fam]
            norm = (raw - mean) / std
            with torch.no_grad():
                t = torch.from_numpy(norm.astype(np.float32))[None, None].to(device)
                out = net(t)[0, 0].cpu().numpy().astype(np.float32) * std + mean
            panels.append((out[z_mid], f"{fam}\nU={gain:+.2f}dB"))
        lo = min(float(np.percentile(p, ZOOM_PCTL[0])) for p, _ in panels)
        hi = max(float(np.percentile(p, ZOOM_PCTL[1])) for p, _ in panels)
        all_row_panels.append(panels)
        row_windows.append((lo, hi))
        for c, (img, title) in enumerate(panels):
            render_panel(axes[r, c], img, title, lo, hi)
        axes[r, 0].set_ylabel(f"row {r+1} ({band_names[r]})\nz0={z0},y0={y0},x0={x0}", fontsize=6)

    fig.suptitle("3P qualitative: 4 family pilots (20ep, seed 3407) vs AVG-9 pseudo-GT", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig_path = out_dir / "fig_q2_3p_qualitative.png"
    fig.savefig(fig_path)
    plt.close(fig)

    # review 切块：每行一张，≤1200px（ncols=6 → figsize 宽 1.95/panel @ dpi=100）
    for r, (panels, (lo, hi)) in enumerate(zip(all_row_panels, row_windows)):
        fig_row, axes_row = plt.subplots(1, ncols, figsize=(1.95 * ncols, 2.2), dpi=100)
        for c, (img, title) in enumerate(panels):
            render_panel(axes_row[c], img, title, lo, hi)
        fig_row.tight_layout()
        tile_path = review_dir / f"3p_row_{r + 1}.png"
        fig_row.savefig(tile_path)
        plt.close(fig_row)
        print(f"[render_3p] row {r + 1} ({band_names[r]} z0={rows[r][0]}) → {tile_path}")

    print(f"[render_3p] 主图 → {fig_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
