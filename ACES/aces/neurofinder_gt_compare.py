from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import numpy as np
import tifffile
import torch

from .io_utils import write_json
from .paths import VALID_ROOT


MODES = ["valid", "sn2n_xy", "iso3d", "empirical_block", "ellipsoid"]
MULTIMETHOD_ROOT = Path("Results/neurofinder_00_00_multimethod_comparison_20260822")


def _load_model(checkpoint: Path, train_cfg: dict[str, Any], device: torch.device) -> torch.nn.Module:
    sys.path.insert(0, str(VALID_ROOT))
    from models.network import Network_CNR

    model = Network_CNR(
        in_channels=1,
        out_channels=1,
        f_maps=int(train_cfg["base_features"]),
        n_groups=int(train_cfg["n_groups"]),
    ).to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["state_dict"])
    model.eval()
    return model


def _normalize(image: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip((image.astype(np.float32) - lo) / (hi - lo + 1e-6), 0, 1)


def infer_stacks(
    run_dir: str | Path,
    output_root: str | Path,
    *,
    checkpoint: str = "final",
    tile_zyx: tuple[int, int, int] = (32, 128, 128),
    gpu: str = "0",
) -> dict[str, Any]:
    run_root = Path(run_dir)
    out_root = Path(output_root)
    cfg = json.loads((run_root / "valid" / "config.json").read_text(encoding="utf-8"))
    train_cfg = cfg["training"]
    stack_path = Path(train_cfg["train_folder"]) / "neurofinder_00_00_stack.tif"
    stack = tifffile.memmap(stack_path)
    shape = tuple(int(v) for v in stack.shape)
    mean = float(np.asarray(stack[:: max(1, shape[0] // 128)], dtype=np.float32).mean())
    std = float(np.asarray(stack[:: max(1, shape[0] // 128)], dtype=np.float32).std())
    std = max(std, 1e-6)

    device = torch.device("cuda:0" if torch.cuda.is_available() and gpu != "cpu" else "cpu")
    outputs: dict[str, str] = {}
    started = time.perf_counter()
    for mode in MODES:
        model = _load_model(run_root / mode / "checkpoints" / f"{mode}_{checkpoint}.pth", train_cfg, device)
        mode_out = out_root / "outputs" / mode / "neurofinder.00.00_stack.tif"
        mode_out.parent.mkdir(parents=True, exist_ok=True)
        writer = tifffile.TiffWriter(mode_out, bigtiff=True)
        tz, ty, tx = tile_zyx
        try:
            for z0 in range(0, shape[0], tz):
                z1 = min(z0 + tz, shape[0])
                planes = np.zeros((z1 - z0, shape[1], shape[2]), dtype=np.float32)
                for y0 in range(0, shape[1], ty):
                    for x0 in range(0, shape[2], tx):
                        y1 = min(y0 + ty, shape[1])
                        x1 = min(x0 + tx, shape[2])
                        patch = np.asarray(stack[z0:z1, y0:y1, x0:x1], dtype=np.float32)
                        pad = [(0, tz - patch.shape[0]), (0, ty - patch.shape[1]), (0, tx - patch.shape[2])]
                        if any(width for _, width in pad):
                            patch = np.pad(patch, pad, mode="edge")
                        norm = (patch - mean) / std
                        tensor = torch.from_numpy(np.ascontiguousarray(norm)).unsqueeze(0).unsqueeze(0).to(device)
                        with torch.no_grad():
                            pred = model(tensor).squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32)
                        pred = pred[: z1 - z0, : y1 - y0, : x1 - x0] * std + mean
                        planes[:, y0:y1, x0:x1] = pred
                writer.write(np.clip(planes, 0, 65535).astype(np.uint16), contiguous=True)
                print(f"[{mode}] wrote frames {z0}:{z1}", flush=True)
        finally:
            writer.close()
        outputs[mode] = str(mode_out)
    summary = {
        "event": "aces_neurofinder_full_stack_inference_complete",
        "run_dir": str(run_root),
        "output_root": str(out_root),
        "source_stack": str(stack_path),
        "source_shape": list(shape),
        "checkpoint": checkpoint,
        "tile_zyx": list(tile_zyx),
        "normalization": {"mean": mean, "std": std, "source": "sampled global stack moments"},
        "outputs": outputs,
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(out_root / "logs" / "inference_summary.json", summary)
    return summary


def _pixel_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    pred_b = pred > 0
    gt_b = gt > 0
    tp = int(np.logical_and(pred_b, gt_b).sum())
    tn = int(np.logical_and(~pred_b, ~gt_b).sum())
    fp = int(np.logical_and(pred_b, ~gt_b).sum())
    fn = int(np.logical_and(~pred_b, gt_b).sum())
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    return {
        "tp_px": tp,
        "tn_px": tn,
        "fp_px": fp,
        "fn_px": fn,
        "accuracy": (tp + tn) / (tp + tn + fp + fn + 1e-12),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall + 1e-12),
    }


def _instance_metrics(pred: np.ndarray, gt: np.ndarray, iou_threshold: float) -> dict[str, Any]:
    pred_ids = [int(v) for v in np.unique(pred) if v != 0]
    gt_ids = [int(v) for v in np.unique(gt) if v != 0]
    pred_areas = {pid: int((pred == pid).sum()) for pid in pred_ids}
    pairs = []
    for gid in gt_ids:
        g = gt == gid
        overlap_ids, counts = np.unique(pred[g], return_counts=True)
        g_area = int(g.sum())
        for pid, inter in zip(overlap_ids, counts):
            if pid == 0:
                continue
            iou = float(inter / (g_area + pred_areas[int(pid)] - inter + 1e-12))
            if iou >= iou_threshold:
                pairs.append((iou, gid, int(pid)))
    pairs.sort(reverse=True)
    used_g, used_p, matches = set(), set(), []
    for iou, gid, pid in pairs:
        if gid in used_g or pid in used_p:
            continue
        used_g.add(gid)
        used_p.add(pid)
        matches.append({"gt_id": gid, "pred_id": pid, "iou": iou})
    tp = len(matches)
    fp = len(pred_ids) - tp
    fn = len(gt_ids) - tp
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    return {
        "iou_threshold": iou_threshold,
        "gt_instances": len(gt_ids),
        "pred_instances": len(pred_ids),
        "tp_instances": tp,
        "fp_instances": fp,
        "fn_instances": fn,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall + 1e-12),
        "mean_matched_iou": float(np.mean([m["iou"] for m in matches])) if matches else 0.0,
        "matches": matches,
    }


def _segment_cellpose(image: np.ndarray, gpu: bool) -> tuple[np.ndarray, dict[str, Any]]:
    from cellpose import models

    model = models.CellposeModel(gpu=gpu, pretrained_model="cpsam_v2")
    masks, flows, styles = model.eval(image.astype(np.float32), channels=[0, 0], diameter=20, min_size=15)
    return masks.astype(np.int32), {
        "segmenter": "cellpose",
        "pretrained_model": "cpsam_v2",
        "diameter": 20,
        "channels": [0, 0],
        "projection": "temporal maximum intensity projection",
    }


def _stack_mip(path: Path) -> np.ndarray:
    stack = tifffile.memmap(path)
    if stack.ndim == 3:
        return np.asarray(stack).max(axis=0)
    if stack.ndim == 4:
        return np.asarray(stack).max(axis=(0, 1))
    raise ValueError(f"Expected 3D or blocked 4D TIFF stack at {path}, got shape {stack.shape}")


def _select_rois(gt: np.ndarray, size: int = 84, limit: int = 4) -> list[tuple[int, int, int, int, int]]:
    selected_path = MULTIMETHOD_ROOT / "figures" / "paper_style" / "selected_rois.json"
    if selected_path.exists():
        raw = json.loads(selected_path.read_text(encoding="utf-8"))
        return [(int(r["gt_id"]), int(r["y"]), int(r["x"]), int(r["h"]), int(r["w"])) for r in raw[:limit]]
    from skimage.measure import regionprops

    props = sorted(regionprops(gt), key=lambda p: p.area, reverse=True)
    rois = []
    taken: list[tuple[int, int]] = []
    height, width = gt.shape
    for prop in props:
        cy, cx = map(int, prop.centroid)
        if any(abs(cy - y) < size and abs(cx - x) < size for y, x in taken):
            continue
        y0 = min(max(cy - size // 2, 0), height - size)
        x0 = min(max(cx - size // 2, 0), width - size)
        rois.append((int(prop.label), y0, x0, size, size))
        taken.append((cy, cx))
        if len(rois) >= limit:
            break
    return rois


def _make_figures(
    output_root: Path,
    methods: list[str],
    mips: dict[str, np.ndarray],
    labels: dict[str, np.ndarray],
    gt: np.ndarray,
    rows: list[dict[str, Any]],
) -> dict[str, str]:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from skimage import segmentation

    fig_dir = output_root / "figures" / "paper_style"
    fig_dir.mkdir(parents=True, exist_ok=True)
    gt_mask = gt > 0
    rois = _select_rois(gt, size=84, limit=4)
    combined = np.concatenate([mips[m].ravel()[::32] for m in methods])
    shared_lo, shared_hi = np.percentile(combined, [1.0, 99.85])
    cmap = plt.get_cmap("turbo")

    fig, axes = plt.subplots(2, len(methods), figsize=(3.0 * len(methods), 6.0), dpi=180)
    if len(methods) == 1:
        axes = axes[:, None]
    for col, method in enumerate(methods):
        image = mips[method]
        axes[0, col].imshow(_normalize(image, shared_lo, shared_hi), cmap="turbo")
        axes[0, col].set_title(method, fontsize=8)
        axes[0, col].axis("off")
        if col == 0:
            for _, y, x, h, w in rois:
                axes[0, col].add_patch(Rectangle((x, y), w, h, fill=False, edgecolor="white", linewidth=0.8))
        pred_b = labels[method] > 0
        base = cmap(_normalize(image, *np.percentile(image, [1.0, 99.85])))[..., :3]
        masked = np.zeros_like(base)
        display_mask = np.logical_or(gt_mask, pred_b)
        masked[display_mask] = base[display_mask]
        gt_bd = segmentation.find_boundaries(gt_mask, mode="outer")
        pred_bd = segmentation.find_boundaries(pred_b, mode="outer")
        masked[gt_bd] = np.array([1.0, 0.1, 0.1])
        masked[pred_bd] = np.array([0.0, 1.0, 0.25])
        axes[1, col].imshow(masked)
        inst03 = next(r for r in rows if r["method"] == method)["instance_iou_0_3"]
        axes[1, col].set_title(f"GT red / pred green, F1={inst03['f1']:.3f}", fontsize=8)
        axes[1, col].axis("off")
    fig.suptitle("ACES Neurofinder 00.00 sampling modes: denoising MIP and Cellpose segmentation", fontsize=11)
    fig.tight_layout()
    overview = fig_dir / "aces_neurofinder_sampling_pseudocolor_segmentation.png"
    fig.savefig(overview, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(len(rois), len(methods), figsize=(2.1 * len(methods), 2.2 * len(rois)), dpi=200)
    if len(rois) == 1:
        axes = axes[None, :]
    if len(methods) == 1:
        axes = axes[:, None]
    for row, (label_id, y, x, h, w) in enumerate(rois):
        for col, method in enumerate(methods):
            crop = mips[method][y : y + h, x : x + w]
            lo, hi = np.percentile(crop, [1.0, 99.9])
            axes[row, col].imshow(_normalize(crop, lo, hi), cmap="turbo")
            gt_bd = segmentation.find_boundaries(gt[y : y + h, x : x + w] == label_id, mode="outer")
            pred_bd = segmentation.find_boundaries(labels[method][y : y + h, x : x + w] > 0, mode="outer")
            axes[row, col].contour(gt_bd, levels=[0.5], colors="red", linewidths=0.55)
            axes[row, col].contour(pred_bd, levels=[0.5], colors="lime", linewidths=0.45)
            if row == 0:
                axes[row, col].set_title(method, fontsize=7)
            if col == 0:
                axes[row, col].set_ylabel(f"ROI {row + 1}", fontsize=7)
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
    fig.suptitle("ACES sampling modes ROI zooms: pseudocolor MIP with GT/pred contours", fontsize=10)
    fig.tight_layout()
    roi_fig = fig_dir / "aces_neurofinder_sampling_single_neuron_roi_zooms.png"
    fig.savefig(roi_fig, bbox_inches="tight")
    plt.close(fig)

    write_json(fig_dir / "selected_rois.json", [{"gt_id": gid, "y": y, "x": x, "h": h, "w": w} for gid, y, x, h, w in rois])
    return {"overview": str(overview), "roi_zooms": str(roi_fig), "selected_rois": str(fig_dir / "selected_rois.json")}


def evaluate(output_root: str | Path, *, gpu: str = "0", include_raw: bool = True) -> dict[str, Any]:
    output_root = Path(output_root)
    gt = tifffile.imread(MULTIMETHOD_ROOT / "data" / "gt_instance_labels.tif").astype(np.int32)
    paths: dict[str, Path] = {}
    if include_raw:
        paths["raw"] = MULTIMETHOD_ROOT / "data" / "neurofinder.00.00_stack.tif"
    for mode in MODES:
        path = output_root / "outputs" / mode / "neurofinder.00.00_stack.tif"
        if path.exists():
            paths[mode] = path
    pred_dir = output_root / "predictions" / "segmentation"
    pred_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    mips: dict[str, np.ndarray] = {}
    labels_by_method: dict[str, np.ndarray] = {}
    segmenter_meta: dict[str, Any] | None = None
    for method, path in paths.items():
        mip = _stack_mip(path)
        mips[method] = mip
        labels, segmenter_meta = _segment_cellpose(mip, gpu=(gpu != "cpu"))
        labels_by_method[method] = labels
        tifffile.imwrite(pred_dir / f"{method}_labels.tif", labels.astype(np.uint16), photometric="minisblack")
        rows.append(
            {
                "method": method,
                "path": str(path),
                "mip_mean": float(mip.mean()),
                "mip_std": float(mip.std()),
                "mip_p99": float(np.percentile(mip, 99)),
                "pixel": _pixel_metrics(labels, gt),
                "instance_iou_0_3": _instance_metrics(labels, gt, 0.3),
                "instance_iou_0_5": _instance_metrics(labels, gt, 0.5),
            }
        )

    metrics_dir = output_root / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    csv_path = metrics_dir / "segmentation_metrics_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "method",
                "pixel_accuracy",
                "pixel_precision",
                "pixel_recall",
                "pixel_f1",
                "inst03_precision",
                "inst03_recall",
                "inst03_f1",
                "inst03_tp",
                "inst03_fp",
                "inst03_fn",
                "inst05_precision",
                "inst05_recall",
                "inst05_f1",
                "pred_instances",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "method": row["method"],
                    "pixel_accuracy": row["pixel"]["accuracy"],
                    "pixel_precision": row["pixel"]["precision"],
                    "pixel_recall": row["pixel"]["recall"],
                    "pixel_f1": row["pixel"]["f1"],
                    "inst03_precision": row["instance_iou_0_3"]["precision"],
                    "inst03_recall": row["instance_iou_0_3"]["recall"],
                    "inst03_f1": row["instance_iou_0_3"]["f1"],
                    "inst03_tp": row["instance_iou_0_3"]["tp_instances"],
                    "inst03_fp": row["instance_iou_0_3"]["fp_instances"],
                    "inst03_fn": row["instance_iou_0_3"]["fn_instances"],
                    "inst05_precision": row["instance_iou_0_5"]["precision"],
                    "inst05_recall": row["instance_iou_0_5"]["recall"],
                    "inst05_f1": row["instance_iou_0_5"]["f1"],
                    "pred_instances": row["instance_iou_0_3"]["pred_instances"],
                }
            )
    figures = _make_figures(output_root, list(paths), mips, labels_by_method, gt, rows)
    summary = {
        "event": "aces_neurofinder_gt_comparison_complete",
        "output_root": str(output_root),
        "method_stack_paths": {k: str(v) for k, v in paths.items()},
        "segmenter": segmenter_meta,
        "metrics": rows,
        "csv": str(csv_path),
        "figures": figures,
        "reference_metrics": _read_reference_metrics(),
        "reference_protocol": str(MULTIMETHOD_ROOT / "scripts" / "run_neurofinder_multimethod.py"),
    }
    write_json(metrics_dir / "segmentation_metrics.json", summary)
    _write_report(output_root, summary)
    return summary


def _read_reference_metrics() -> list[dict[str, str]]:
    csv_path = MULTIMETHOD_ROOT / "metrics" / "segmentation_metrics_summary.csv"
    if not csv_path.exists():
        return []
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_report(output_root: Path, summary: dict[str, Any]) -> None:
    report = output_root / "report" / "aces_neurofinder_gt_sampling_report.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# ACES Neurofinder 00.00 GT Sampling-Mode Comparison",
        "",
        "## Scope",
        "",
        "This report evaluates ACES Stage 2 five sampling modes using the same Neurofinder GT protocol as the previous multi-method comparison: temporal MIP -> Cellpose cpsam_v2 -> pixel and instance IoU matching against Neurofinder ROI labels.",
        "",
        "## Figures",
        "",
    ]
    for name, figure in summary["figures"].items():
        if figure.endswith(".png"):
            rel = Path("..") / Path(figure).relative_to(output_root)
            lines.append(f"![{name}]({rel.as_posix()})")
            lines.append("")
    lines.extend(
        [
            "## Metrics",
            "",
            "| Method | Pixel Acc | Pixel Recall | Pixel F1 | Inst F1 IoU0.3 | Inst Recall IoU0.3 | Inst F1 IoU0.5 | Pred Inst |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["metrics"]:
        lines.append(
            f"| {row['method']} | {row['pixel']['accuracy']:.4f} | {row['pixel']['recall']:.4f} | "
            f"{row['pixel']['f1']:.4f} | {row['instance_iou_0_3']['f1']:.4f} | "
            f"{row['instance_iou_0_3']['recall']:.4f} | {row['instance_iou_0_5']['f1']:.4f} | "
            f"{row['instance_iou_0_3']['pred_instances']} |"
        )
    reference_rows = summary.get("reference_metrics", [])
    if reference_rows:
        lines.extend(
            [
                "",
                "## Previous Multimethod Reference",
                "",
                "The table below is copied from the previous Neurofinder 00.00 multimethod comparison CSV and uses the same GT protocol. It is included only as context; the ACES rows above come from the current Stage 2 fastpilot checkpoints.",
                "",
                "| Method | Pixel Recall | Pixel F1 | Inst F1 IoU0.3 | Inst Recall IoU0.3 | Inst F1 IoU0.5 | Pred Inst |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in reference_rows:
            lines.append(
                f"| {row['method']} | {float(row['pixel_recall']):.4f} | {float(row['pixel_f1']):.4f} | "
                f"{float(row['inst03_f1']):.4f} | {float(row['inst03_recall']):.4f} | "
                f"{float(row['inst05_f1']):.4f} | {int(float(row['pred_instances']))} |"
            )
    lines.extend(["", "Full outputs:", f"- CSV: `{summary['csv']}`", f"- JSON: `{output_root / 'metrics' / 'segmentation_metrics.json'}`"])
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ACES Neurofinder full-stack inference and GT comparison.")
    parser.add_argument("command", choices=("infer", "evaluate", "all"))
    parser.add_argument("--run-dir", default="ACES/runs/stage02_neurofinder_00_00_fastpilot_seed3407")
    parser.add_argument("--output-root", default="ACES/runs/stage02_neurofinder_00_00_fastpilot_seed3407/gt_comparison")
    parser.add_argument("--checkpoint", default="final", choices=("final", "best"))
    parser.add_argument("--gpu", default="0")
    args = parser.parse_args(argv)
    if args.command == "infer":
        result = infer_stacks(args.run_dir, args.output_root, checkpoint=args.checkpoint, gpu=args.gpu)
    elif args.command == "evaluate":
        result = evaluate(args.output_root, gpu=args.gpu)
    else:
        infer_stacks(args.run_dir, args.output_root, checkpoint=args.checkpoint, gpu=args.gpu)
        result = evaluate(args.output_root, gpu=args.gpu)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
