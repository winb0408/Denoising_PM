"""QMask 复用 ACES 评估组件，补一项深度分箱前景 PSNR（对 pilot 门控用）。"""
from __future__ import annotations

import numpy as np
import torch
from pathlib import Path
from typing import Any

from aces.evaluate_stage2 import (
    _load_target_and_masks,
    _mask_patch,
    _region_metrics,
    _stride_slice,
    _target_patch,
)


def depth_binned_foreground_psnr(
    *,
    model: torch.nn.Module,
    dataset: Any,
    device: torch.device,
    mean9_path: str | Path,
    foreground_percentile: float = 75.0,
    foreground_mask_path: str | Path | None = None,
    bins: int = 5,
    stride: tuple[int, int, int] = (1, 1, 1),
    patch_cap: int = 2000,
) -> dict[str, Any]:
    """遍历 image 0 的全部 patch，按 z 起点均分成 bins，汇总前景 PSNR。

    该指标反映“不同深度/时间带上的前景重建质量”，补充单个数码的 all/fg/bg PSNR。
    """
    target, masks = _load_target_and_masks(mean9_path, foreground_percentile, foreground_mask_path)
    indexed_positions = getattr(dataset, "indices", None)
    if indexed_positions is None:
        indexed_positions = []
        for item_index in range(len(dataset)):
            _, image_idx, z_idx, y_idx, x_idx = dataset[item_index]
            indexed_positions.append((int(image_idx), int(z_idx), int(y_idx), int(x_idx)))

    patch_shape = (int(dataset.z_patch), int(dataset.w_patch), int(dataset.h_patch))
    model.eval()
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for item_index, (image_idx, z_idx, y_idx, x_idx) in enumerate(indexed_positions):
            if int(image_idx) != 0:
                continue
            if len(rows) >= patch_cap:
                break
            patch = dataset[item_index][0]
            input_np = patch.squeeze(0).numpy() if patch.dim() == 4 else patch.numpy()
            z0, y0, x0 = int(z_idx), int(y_idx), int(x_idx)
            target_full = _target_patch(target, z0, y0, x0, tuple(input_np.shape))
            mask_full = _mask_patch(masks["foreground"], z0, y0, x0, tuple(input_np.shape))
            if stride != (1, 1, 1):
                input_np = _stride_slice(input_np, stride)
                target_full = _stride_slice(target_full, stride)
                mask_full = _stride_slice(mask_full, stride)
            input_tensor = torch.from_numpy(np.ascontiguousarray(input_np)).unsqueeze(0).unsqueeze(0).to(device)
            output = model(input_tensor).squeeze(0).squeeze(0).detach().cpu()
            output_denorm = output.numpy().astype(np.float32) * float(dataset.image_stds[0]) + float(dataset.image_means[0])
            target_patch = target_full[: output_denorm.shape[0], : output_denorm.shape[1], : output_denorm.shape[2]]
            mask_patch = mask_full[: output_denorm.shape[0], : output_denorm.shape[1], : output_denorm.shape[2]]
            fg = _region_metrics(output_denorm, target_patch, mask_patch)
            allm = _region_metrics(output_denorm, target_patch, np.ones_like(mask_patch, dtype=bool))
            rows.append({
                "patch_index": item_index,
                "z_idx": z0,
                "foreground_psnr_db": fg["psnr_db"],
                "all_psnr_db": allm["psnr_db"],
                "pred_mean": fg["pred_mean"],
                "target_mean": fg["target_mean"],
                "n_fg": fg["n"],
            })

    if not rows:
        return {"bins": [], "overall_foreground_psnr_db": float("nan"), "patch_count": 0}

    z_values = [row["z_idx"] for row in rows]
    z_min, z_max = min(z_values), max(z_values)
    span = max(z_max - z_min, 1)
    bin_rows: dict[int, list[dict[str, Any]]] = {b: [] for b in range(bins)}
    for row in rows:
        b = min(bins - 1, int((row["z_idx"] - z_min) * bins / span))
        bin_rows[b].append(row)

    summary_bins = []
    for b in range(bins):
        entries = bin_rows[b]
        if not entries:
            summary_bins.append({"bin": b, "z_range": None, "foreground_psnr_db": float("nan"), "patch_count": 0})
            continue
        zs = [e["z_idx"] for e in entries]
        psnrs = [e["foreground_psnr_db"] for e in entries if e["foreground_psnr_db"] == e["foreground_psnr_db"]]
        summary_bins.append({
            "bin": b,
            "z_range": [min(zs), max(zs)],
            "foreground_psnr_db": float(np.mean(psnrs)) if psnrs else float("nan"),
            "patch_count": len(entries),
        })

    psnrs_all = [e["foreground_psnr_db"] for e in rows if e["foreground_psnr_db"] == e["foreground_psnr_db"]]
    return {
        "bins": summary_bins,
        "overall_foreground_psnr_db": float(np.mean(psnrs_all)) if psnrs_all else float("nan"),
        "patch_count": len(rows),
        "z_range_global": [z_min, z_max],
        "bins_count": bins,
        "stride": list(stride),
    }
