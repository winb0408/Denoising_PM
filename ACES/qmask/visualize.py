"""QMask 结果可视化：几何曲线/参与掩码 + 去噪定性图。

去噪图必须先标准化输入、推理后反标准化（与训练/评估一致），防“全黑图”回归。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile
import torch

from aces.paths import VALID_ROOT
from .geometry import AXIS_ORDER


def _robust_lims(img: np.ndarray, lo: float = 0.5, hi: float = 99.5) -> tuple[float, float]:
    finite = img[np.isfinite(img)]
    if finite.size == 0:
        return (0.0, 1.0)
    vmin, vmax = np.percentile(finite, [lo, hi])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin, vmax = float(finite.min()), float(finite.max())
        if vmax <= vmin:
            vmax = vmin + 1e-3
    return float(vmin), float(vmax)


def _load_model(mode_dir: Path, epoch_type: str, device: torch.device) -> torch.nn.Module:
    cfg = json.loads((mode_dir / "config.json").read_text(encoding="utf-8"))
    train_cfg = cfg.get("training", cfg)
    sys.path.insert(0, str(VALID_ROOT))
    from models.network import Network_CNR

    model = Network_CNR(
        in_channels=1,
        out_channels=1,
        f_maps=int(train_cfg["base_features"]),
        n_groups=int(train_cfg["n_groups"]),
    ).to(device)
    ckpt = mode_dir / "checkpoints" / f"{mode_dir.name}_{epoch_type}.pth"
    if not ckpt.exists():
        raise FileNotFoundError(f"checkpoint missing: {ckpt}")
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state["state_dict"])
    model.eval()
    return model


def _first_tif(folder: str | Path) -> Path:
    folder = Path(folder)
    files = sorted(folder.glob("*.tif")) + sorted(folder.glob("*.tiff"))
    if not files:
        raise FileNotFoundError(f"no tif in {folder}")
    return files[0]


def render_geometry_figure(geometry, out_path: str | Path) -> None:
    """绘制 N/S/Q(d) 曲线与阈值，标注各轴参与判定。"""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    csv_path = Path(geometry.source) if geometry.source else None
    series_by_axis: dict[str, dict[str, list[tuple[int, float]]]] = {}
    if csv_path and csv_path.exists():
        import csv
        with csv_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("depth_bin") != "global" or row.get("region") != geometry.region or row.get("noise_source") != geometry.noise_source:
                    continue
                axis, d = row.get("axis"), row.get("distance_px")
                if axis not in AXIS_ORDER or d is None:
                    continue
                series_by_axis.setdefault(axis, {}).setdefault("N", []).append((int(float(d)), float(row["N"])))
                series_by_axis[axis].setdefault("S", []).append((int(float(d)), float(row["S"])))
                series_by_axis[axis].setdefault("Q", []).append((int(float(d)), float(row["Q"])))

    colors = {"z": "#d62728", "y": "#2ca02c", "x": "#1f77b4"}
    for col, series_name in enumerate(("N", "S", "Q")):
        ax = axes[col]
        for axis in AXIS_ORDER:
            pts = sorted(series_by_axis.get(axis, {}).get(series_name, []))
            if not pts:
                continue
            d, v = zip(*pts)
            ax.plot(d, v, marker="o", label=f"{axis}{' (on)' if geometry.axes[axis].participate else ' (off)'}", color=colors[axis])
        if series_name == "N":
            ax.axhline(geometry.tau_n, color="k", ls="--", lw=1, label=f"tau_n={geometry.tau_n}")
        elif series_name == "S":
            ax.axhline(geometry.tau_s, color="k", ls="--", lw=1, label=f"tau_s={geometry.tau_s}")
        else:
            ax.axhline(geometry.tau_q, color="k", ls="--", lw=1, label=f"tau_q={geometry.tau_q}")
        ax.legend(fontsize=7)
        ax.set_title(f"{series_name}(d)")
        ax.set_xlabel("distance px")
        ax.grid(alpha=0.3)

    fig.suptitle(f"QMask geometry  region={geometry.region} noise={geometry.noise_source}\nparticipate={geometry.effective_participate()}", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def render_qualitative(run_dir: str | Path, modes: list[str] | None = None) -> Path:
    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    modes = modes or [m for m in ("valid", "sn2n_xy", "iso3d", "qmask") if (run_dir / m).exists()]

    stack_path = _first_tif(cfg["training"]["train_folder"])
    stack = tifffile.imread(stack_path).astype(np.float32)
    stat_mean = float(stack.mean())
    stat_std = float(stack.std())
    target = tifffile.imread(cfg["evaluation"].get("mean9_proxy")).astype(np.float32)

    z_start = stack.shape[0] // 4
    z_end = z_start + 32
    y_start = stack.shape[1] // 4
    y_end = y_start + 256
    x_start = stack.shape[2] // 4
    x_end = x_start + 256
    noisy = stack[z_start:z_end, y_start:y_end, x_start:x_end]
    mid_frame = noisy.shape[0] // 2
    if target.ndim == 2:
        target_crop = target[y_start:y_end, x_start:x_end]
    else:
        target_crop = target[z_start + mid_frame, y_start:y_end, x_start:x_end]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    results: dict[str, dict[str, np.ndarray]] = {}
    for mode in modes:
        results[mode] = {}
        for epoch_type in ("best", "final"):
            try:
                model = _load_model(run_dir / mode, epoch_type, device)
            except FileNotFoundError:
                continue
            with torch.no_grad():
                inp = torch.from_numpy((noisy - stat_mean) / stat_std).float().unsqueeze(0).unsqueeze(0).to(device)
                out = model(inp).squeeze().cpu().numpy()
            results[mode][epoch_type] = out.astype(np.float32) * stat_std + stat_mean
            del model
            torch.cuda.empty_cache()

    # 数值自检：以动态范围判定“塌缩到近零”的旧全黑图 bug。
    target_span = max(float(np.ptp(target_crop[np.isfinite(target_crop)])), 1e-6)
    for mode, per_epoch in results.items():
        for epoch_type, out in per_epoch.items():
            span = float(np.ptp(out[mid_frame]))
            span_ratio = span / target_span
            median_ratio = float(np.median(out[mid_frame])) / (float(np.median(target_crop)) + 1e-6)
            if span_ratio < 0.05:
                raise RuntimeError(
                    f"{mode}/{epoch_type} output dynamic range collapsed "
                    f"(ptp {span:.1f} vs target {target_span:.1f}, ratio {span_ratio:.4f}); "
                    "normalization/de-normalization likely missing or model collapsed"
                )
            print(f"  [check] {mode}/{epoch_type} ptp_ratio={span_ratio:.3f} median_ratio={median_ratio:.3f}")

    n_modes = len(modes)
    fig, axes = plt.subplots(n_modes, 4, figsize=(16, 4 * n_modes), squeeze=False)
    for row, mode in enumerate(modes):
        ax = axes[row]
        vmin, vmax = _robust_lims(noisy[mid_frame])
        ax[0].imshow(noisy[mid_frame], cmap="magma", vmin=vmin, vmax=vmax)
        ax[0].set_title(f"{mode}\ninput frame {mid_frame}\n[{vmin:.0f},{vmax:.0f}]", fontsize=8)
        ax[0].axis("off")
        for col, epoch_type in enumerate(("best", "final"), start=1):
            if epoch_type in results[mode]:
                vmin, vmax = _robust_lims(results[mode][epoch_type][mid_frame])
                ax[col].imshow(results[mode][epoch_type][mid_frame], cmap="magma", vmin=vmin, vmax=vmax)
                ax[col].set_title(f"{epoch_type}\n[{vmin:.0f},{vmax:.0f}]", fontsize=8)
            else:
                ax[col].text(0.5, 0.5, "n/a", ha="center", va="center")
            ax[col].axis("off")
        vmin, vmax = _robust_lims(target_crop)
        ax[3].imshow(target_crop, cmap="magma", vmin=vmin, vmax=vmax)
        ax[3].set_title(f"target/mean9\n[{vmin:.0f},{vmax:.0f}]", fontsize=8)
        ax[3].axis("off")
    fig.tight_layout()
    out_path = run_dir / "figures" / "qualitative_comparison.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")

    # 固化流程：大图一落盘就生成 review 切块，后续定性检查只允许读切块，
    # 避免多模态 API 因原图超限（>1200px / >3.5MB）而卡死。
    from .preview import make_review_previews

    make_review_previews(run_dir)
    print(f"saved {run_dir / 'figures' / 'review' / 'manifest.json'}")
    return out_path


def main(argv: list[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Render QMask geometry + qualitative figures for a run.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--modes", default=None, help="comma-separated modes, default all found")
    args = parser.parse_args(argv)
    run_dir = Path(args.run_dir)
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    geo_cfg = cfg["geometry"]
    from .geometry import geometry_from_stage1_run
    geo = geometry_from_stage1_run(
        geo_cfg["stage1_run_dir"],
        region=geo_cfg.get("region", "foreground"),
        noise_source=geo_cfg.get("noise_source", "repeat"),
        tau_n=float(geo_cfg.get("tau_n", 0.95)),
        tau_s=float(geo_cfg.get("tau_s", 0.85)),
        tau_q=float(geo_cfg.get("tau_q", 0.60)),
    )
    out = run_dir / "figures" / "qmask_geometry_nsq.png"
    render_geometry_figure(geo, out)
    print(f"saved {out}")
    modes = [m.strip() for m in args.modes.split(",")] if args.modes else None
    render_qualitative(run_dir, modes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
