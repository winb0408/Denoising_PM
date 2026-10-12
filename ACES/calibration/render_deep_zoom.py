"""green1300 深层区高分辨定性放大（配合 analyze_green_deepsignal 使用）。

`analyze_green_deepsignal.py` 的 review 切块把 9 个面板压到 1150 px 宽
（≈127 px/panel），细结构/平滑程度难以判读；本脚本对**单个深层 tile**
做 2 行 × 3 列、每面板 ~400 px 的放大渲染，用于判定某个 family 的
PSNR 增益是否来自过平滑（丢结构）而非真去噪。

输入：`calibration/oracle.py --save-checkpoint` 产物（families/*_final.pth）
输出（默认 <run-dir>/deepsignal/zoom/）：
  zoom_tile_z<z0>.png    整 tile（128×128）放大：input | AVG-25 | 3 个 family
  zoom_detail_z<z0>.png  以 AVG-25 最亮像素为中心裁 64×64 的细节放大
两张图都固定 ≤1200 px 宽，可直接 Read。

用法：
  cd /data2/wjb/Denoising_PM
  PYTHONPATH=ACES CUDA_VISIBLE_DEVICES=0 python -m calibration.render_deep_zoom \
    --data /ssd1/wjb/Denoising_PM/ACES/data/green1300_repeat_stack.tif \
    --run-dir /ssd1/wjb/Denoising_PM/ACES/runs/m2_oracle_green1300_seed3407 \
    --z0 177 --families qmask_yz sn2n_xy_diag qmask_xy valid_corners
"""

from __future__ import annotations

import argparse
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

from calibration.analyze_green_deepsignal import (  # noqa: E402
    INPUT_FRAME,
    X_PATCH,
    Y_PATCH,
    Z_PATCH,
    full_inference,
)
from calibration.render_qualitative_3p import (  # noqa: E402
    CMAP,
    ZOOM_PCTL,
    load_family_model,
    render_panel,
)

DETAIL = 64  # 细节裁剪边长


def pick_tile(ref: np.ndarray, z0: int) -> tuple[int, int]:
    """深层 z0 上取 AVG-25 最亮的 (y0, x0)（与 analyze_green_deepsignal 同逻辑）。"""
    best, best_key = (0, 0), -1.0
    for y0 in range(0, ref.shape[1] - Y_PATCH + 1, Y_PATCH):
        for x0 in range(0, ref.shape[2] - X_PATCH + 1, X_PATCH):
            key = float(ref[z0 : z0 + Z_PATCH, y0 : y0 + Y_PATCH, x0 : x0 + X_PATCH].max())
            if key > best_key:
                best, best_key = (y0, x0), key
    return best


def render_grid(panels, out_path: Path, suptitle: str) -> None:
    """2×3 面板，每面板 ~400 px，整图 1200×800。"""
    ncols = 3
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.0, 2.0 * nrows), dpi=200)
    axes = np.atleast_2d(axes).ravel()
    lo = min(float(np.percentile(p, ZOOM_PCTL[0])) for p, _ in panels)
    hi = max(float(np.percentile(p, ZOOM_PCTL[1])) for p, _ in panels)
    for ax, (img, title) in zip(axes, panels):
        render_panel(ax, img, title, lo, hi)
    for ax in axes[len(panels) :]:
        ax.axis("off")
    fig.suptitle(suptitle, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[zoom] wrote {out_path}", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="repeat stack tif [K,Z,Y,X]")
    parser.add_argument("--run-dir", required=True, help="oracle save_checkpoint run 目录")
    parser.add_argument("--z0", type=int, default=177, help="深层 tile 的 z 起点")
    parser.add_argument(
        "--families",
        nargs="+",
        default=["qmask_yz", "sn2n_xy_diag", "qmask_xy", "valid_corners"],
    )
    parser.add_argument("--out-dir", default=None, help="默认 <run-dir>/deepsignal/zoom")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "deepsignal" / "zoom"
    out_dir.mkdir(parents=True, exist_ok=True)

    volume = np.asarray(tifffile.imread(args.data))
    if volume.ndim != 4:
        raise SystemExit(f"需要 repeat stack [K,Z,Y,X], got {volume.shape}")
    ref = volume.astype(np.float64).mean(axis=0).astype(np.float32)
    raw = np.asarray(volume[INPUT_FRAME], dtype=np.float32)

    z0 = args.z0
    y0, x0 = pick_tile(ref, z0)
    z_mid = Z_PATCH // 2
    print(f"[zoom] tile z0={z0} y0={y0} x0={x0} mid-plane z={z0 + z_mid}", flush=True)

    sl = (slice(z0 + z_mid, z0 + z_mid + 1), slice(y0, y0 + Y_PATCH), slice(x0, x0 + X_PATCH))
    tile_panels = [
        (raw[sl][0], f"input (frame {INPUT_FRAME}, z={z0 + z_mid})"),
        (ref[sl][0], "AVG-25 pseudo-GT"),
    ]
    gain_txt: dict[str, str] = {}
    for fam in args.families:
        net, mean, std, gain = load_family_model(run_dir, fam, device)
        out = full_inference(net, mean, std, raw, device)
        gain_txt[fam] = f"{fam}\nU={gain:+.2f}dB"
        tile_panels.append((out[sl][0], gain_txt[fam]))

    render_grid(
        tile_panels,
        out_dir / f"zoom_tile_z{z0}.png",
        f"green1300 deep tile (z={z0 + z_mid}) zoom: input | AVG-25 | families",
    )

    # 细节：以 AVG-25 最亮像素为中心裁 DETAIL×DETAIL
    plane = ref[sl][0]
    cy, cx = np.unravel_index(int(np.argmax(plane)), plane.shape)
    hy = max(0, min(Y_PATCH - DETAIL, cy - DETAIL // 2))
    hx = max(0, min(X_PATCH - DETAIL, cx - DETAIL // 2))
    det = (slice(hy, hy + DETAIL), slice(hx, hx + DETAIL))
    detail_panels = [(p[det], t.replace("\n", " ")) for p, t in tile_panels]
    render_grid(
        detail_panels,
        out_dir / f"zoom_detail_z{z0}.png",
        f"green1300 deep detail {DETAIL}x{DETAIL} @ (y={y0 + hy}, x={x0 + hx})",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
