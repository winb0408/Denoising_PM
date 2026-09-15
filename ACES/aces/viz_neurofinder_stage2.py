from __future__ import annotations

import argparse
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

from .evaluate_stage2 import _mask_patch, _target_patch
from .io_utils import write_json
from .paths import VALID_ROOT


MODES = ["valid", "sn2n_xy", "iso3d", "empirical_block", "ellipsoid"]


def _load_valid_dataset(train_cfg: dict[str, Any]):
    sys.path.insert(0, str(VALID_ROOT))
    from datasets.dataset_fs import ReadDatasets

    return ReadDatasets(
        dataPath=train_cfg["train_folder"],
        mode="train",
        dataType="3D",
        dataExtension="tif",
        z_patch=int(train_cfg["z_patch"]),
        w_patch=int(train_cfg["w_patch"]),
        h_patch=int(train_cfg["h_patch"]),
        z_overlap=float(train_cfg["z_overlap"]),
        w_overlap=float(train_cfg["w_overlap"]),
        h_overlap=float(train_cfg["h_overlap"]),
        patch_num=int(train_cfg["patch_num"]),
        dataNum=int(train_cfg.get("train_frame_num", 10000)),
    )


def _load_model(path: Path, train_cfg: dict[str, Any], device: torch.device) -> torch.nn.Module:
    sys.path.insert(0, str(VALID_ROOT))
    from models.network import Network_CNR

    model = Network_CNR(
        in_channels=1,
        out_channels=1,
        f_maps=int(train_cfg["base_features"]),
        n_groups=int(train_cfg["n_groups"]),
    ).to(device)
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def _region_stats(volume: np.ndarray, roi_yx: np.ndarray) -> dict[str, float]:
    roi = np.broadcast_to(roi_yx.astype(bool), volume.shape)
    bg = ~roi
    fg_mean = float(volume[roi].mean()) if roi.any() else float("nan")
    bg_mean = float(volume[bg].mean()) if bg.any() else float("nan")
    return {
        "fg_mean": fg_mean,
        "bg_mean": bg_mean,
        "sbr": fg_mean / bg_mean if bg_mean else float("nan"),
        "bg_std": float(volume[bg].std()) if bg.any() else float("nan"),
    }


def _best_frame(noisy: np.ndarray, roi_yx: np.ndarray) -> int:
    roi = roi_yx.astype(bool)
    bg = ~roi
    if not roi.any() or not bg.any():
        return noisy.shape[0] // 2
    scores = [float(noisy[z][roi].mean() - noisy[z][bg].mean()) for z in range(noisy.shape[0])]
    return int(np.argmax(scores))


def _imshow_with_roi(ax, image: np.ndarray, roi_yx: np.ndarray, title: str, vmin: float, vmax: float) -> None:
    ax.imshow(image, cmap="magma", vmin=vmin, vmax=vmax)
    if roi_yx.any():
        ax.contour(roi_yx.astype(float), levels=[0.5], colors="#45f3ff", linewidths=0.45)
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def _plot_patch(
    output: Path,
    *,
    patch_index: int,
    origin_zyx: tuple[int, int, int],
    noisy: np.ndarray,
    target_yx: np.ndarray,
    roi_yx: np.ndarray,
    predictions: dict[str, np.ndarray],
    per_patch_stats: dict[str, dict[str, float]],
) -> None:
    frame = _best_frame(noisy, roi_yx)
    columns = ["input"] + MODES + ["temporal mean"]
    frame_images = [noisy[frame]] + [predictions[mode][frame] for mode in MODES] + [target_yx]
    mean_images = [noisy.mean(axis=0)] + [predictions[mode].mean(axis=0) for mode in MODES] + [target_yx]
    display_pool = np.concatenate([image.reshape(-1) for image in frame_images + mean_images])
    vmin, vmax = np.percentile(display_pool[np.isfinite(display_pool)], [1.0, 99.5])

    fig, axes = plt.subplots(2, len(columns), figsize=(2.45 * len(columns), 5.2))
    for col, name in enumerate(columns):
        if name in predictions:
            stats = per_patch_stats[name]
            suffix = f"\nSBR {stats['sbr']:.3f} bgσ {stats['bg_std']:.1f}"
        elif name == "input":
            stats = per_patch_stats["input"]
            suffix = f"\nSBR {stats['sbr']:.3f} bgσ {stats['bg_std']:.1f}"
        else:
            stats = per_patch_stats["target"]
            suffix = f"\nSBR {stats['sbr']:.3f}"
        _imshow_with_roi(axes[0, col], frame_images[col], roi_yx, f"{name}{suffix}", float(vmin), float(vmax))
        _imshow_with_roi(axes[1, col], mean_images[col], roi_yx, name, float(vmin), float(vmax))
    axes[0, 0].set_ylabel(f"frame {frame}", fontsize=8)
    axes[1, 0].set_ylabel("time mean", fontsize=8)
    fig.suptitle(f"neurofinder.00.00 patch idx={patch_index} origin(frame,y,x)={origin_zyx}", fontsize=10)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def _plot_metric_bars(output: Path, metrics: dict[str, dict[str, float]]) -> None:
    modes = list(metrics)
    x = np.arange(len(modes))
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.3))
    specs = [
        ("foreground_psnr", "PSNR fg", "dB"),
        ("sbr", "SBR pred", "ratio"),
        ("bg_std", "background std", "intensity"),
    ]
    for ax, (key, title, ylabel) in zip(axes, specs):
        values = [metrics[mode][key] for mode in modes]
        ax.bar(x, values, color=["#8da0cb", "#66c2a5", "#fc8d62", "#e78ac3", "#a6d854"])
        ax.set_xticks(x, modes, rotation=35, ha="right")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def run(run_dir: str | Path, out_dir: str | Path | None, n_patches: int, checkpoint: str) -> dict[str, Any]:
    root = Path(run_dir)
    output_root = Path(out_dir) if out_dir else root / "visualization"
    figures = output_root / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    cfg = json.loads((root / "valid" / "config.json").read_text(encoding="utf-8"))
    train_cfg = cfg["training"]
    eval_cfg = cfg["evaluation"]
    target = tifffile.imread(eval_cfg["mean9_proxy"]).astype(np.float32)
    roi_mask = tifffile.imread(eval_cfg["foreground_mask"]).astype(bool)
    dataset = _load_valid_dataset(train_cfg)
    selected = json.loads((root / "valid" / "metrics" / "eval_metrics.json").read_text(encoding="utf-8"))[
        "selected_patch_indices"
    ][:n_patches]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    models = {
        mode: _load_model(root / mode / "checkpoints" / f"{mode}_{checkpoint}.pth", train_cfg, device)
        for mode in MODES
    }

    aggregate: dict[str, list[dict[str, float]]] = {"input": [], "target": [], **{mode: [] for mode in MODES}}
    patch_rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for rank, patch_index in enumerate(selected):
            patch, image_idx, z_idx, y_idx, x_idx = dataset[int(patch_index)]
            input_np = patch.squeeze(0).numpy().astype(np.float32)
            noisy = input_np * float(dataset.image_stds[0]) + float(dataset.image_means[0])
            origin = (int(z_idx), int(y_idx), int(x_idx))
            target_patch = _target_patch(target, *origin, tuple(noisy.shape))[0]
            roi_patch = _mask_patch(roi_mask, *origin, tuple(noisy.shape))[0].astype(bool)

            predictions: dict[str, np.ndarray] = {}
            for mode, model in models.items():
                tensor = torch.from_numpy(np.ascontiguousarray(input_np)).unsqueeze(0).unsqueeze(0).to(device)
                pred = model(tensor).squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32)
                predictions[mode] = pred * float(dataset.image_stds[0]) + float(dataset.image_means[0])

            per_patch_stats = {
                "input": _region_stats(noisy, roi_patch),
                "target": _region_stats(np.broadcast_to(target_patch, noisy.shape), roi_patch),
            }
            per_patch_stats.update({mode: _region_stats(predictions[mode], roi_patch) for mode in MODES})
            for key, stats in per_patch_stats.items():
                aggregate[key].append(stats)
            patch_rows.append({"rank": rank, "patch_index": int(patch_index), "origin_zyx": list(origin), "stats": per_patch_stats})
            _plot_patch(
                figures / f"patch{rank}_idx{int(patch_index)}.png",
                patch_index=int(patch_index),
                origin_zyx=origin,
                noisy=noisy,
                target_yx=target_patch,
                roi_yx=roi_patch,
                predictions=predictions,
                per_patch_stats=per_patch_stats,
            )

    report_metrics: dict[str, dict[str, float]] = {}
    for mode in MODES:
        eval_metrics = json.loads((root / mode / "metrics" / "eval_metrics.json").read_text(encoding="utf-8"))
        report_metrics[mode] = {
            "foreground_psnr": float(eval_metrics["regions"]["foreground"]["psnr_db"]),
            "sbr": float(np.nanmean([row["sbr"] for row in aggregate[mode]])),
            "bg_std": float(np.nanmean([row["bg_std"] for row in aggregate[mode]])),
        }
    _plot_metric_bars(figures / "metric_bars.png", report_metrics)

    result = {
        "event": "aces_neurofinder_stage2_visualization",
        "run_dir": str(root),
        "checkpoint": checkpoint,
        "patch_indices": [int(index) for index in selected],
        "figures": [str(path) for path in sorted(figures.glob("*.png"))],
        "metrics": report_metrics,
        "patch_rows": patch_rows,
    }
    write_json(output_root / "visualization_metrics.json", result)
    _write_report(output_root / "visualization_report.md", result)
    return result


def _write_report(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# ACES Neurofinder Stage 2 Visualization",
        "",
        f"- run: `{result['run_dir']}`",
        f"- checkpoint: `{result['checkpoint']}`",
        f"- patch indices: `{result['patch_indices']}`",
        "",
        "| mode | PSNR fg | SBR pred | bg pred std |",
        "|---|---:|---:|---:|",
    ]
    for mode in MODES:
        metric = result["metrics"][mode]
        lines.append(
            f"| {mode} | {metric['foreground_psnr']:.4f} | {metric['sbr']:.4f} | {metric['bg_std']:.4f} |"
        )
    lines.extend(["", "## Figures", ""])
    for figure in result["figures"]:
        rel = Path(figure).name
        lines.append(f"- [{rel}](figures/{rel})")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Visualize ACES Stage 2 five-mode outputs on neurofinder.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--n-patches", type=int, default=4)
    parser.add_argument("--checkpoint", default="final", choices=["final", "best"])
    args = parser.parse_args(argv)
    result = run(args.run_dir, args.out_dir, args.n_patches, args.checkpoint)
    print(json.dumps({"out": str(Path(args.out_dir) if args.out_dir else Path(args.run_dir) / "visualization"), "figures": result["figures"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
