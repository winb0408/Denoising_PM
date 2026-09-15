"""E1-3P：SN2N 原仓库预测在 E2 同款 patch 协议下的统一评估（Plan v2 §3.3 外部表）。

E2（eval_valid_original.py）用 18 个前景丰富 patch 评估原 VALID；本脚本把
SN2N 原仓库的 stack-level 预测切到完全相同的 patch 网格与选点，产出外部表
可直接并排的两行。

差异点（必须记录）：SN2N 自监督输出无绝对尺度，预测先做全局线性尺度对齐
（stack-level 最小二乘拟合到 mean9 proxy），再进入与 E2 相同的 patch 指标；
VALID 行无此步（网络直接输出目标尺度）。

用法（仓库根目录）：
    PYTHONPATH=ACES python -m aces.eval_sn2n_original_3p
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import tifffile

from .paths import VALID_ROOT

PRED = Path("/data3/wjb/Denoising_PM/Results/sn2n_original_3p/predictions")
STACK_EVAL = Path("/data3/wjb/Denoising_PM/Results/sn2n_original_3p/unified_eval.json")
OUTPUT = Path("/data3/wjb/Denoising_PM/Results/sn2n_original_3p/patch_eval_metrics.json")
# 与 E2 完全一致的评估协议
REFERENCE_CONFIG = Path("/data2/wjb/Denoising_PM/ACES/runs/stage02_valid_40ep_seed3407/valid/config.json")


def main() -> int:
    from .evaluate_stage2 import _load_target_and_masks, _mask_patch, _region_metrics

    ref = json.loads(REFERENCE_CONFIG.read_text())
    train_cfg = ref["training"]
    eval_cfg = ref["evaluation"]
    max_patches = int(eval_cfg.get("patch_count", 18))

    sys.path.insert(0, str(VALID_ROOT))
    from datasets.dataset_fs import ReadDatasets

    dataset = ReadDatasets(
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

    mean9, masks = _load_target_and_masks(
        eval_cfg["mean9_proxy"],
        float(eval_cfg.get("foreground_percentile", 75.0)),
        eval_cfg.get("foreground_mask"),
    )

    indexed_positions = getattr(dataset, "indices", None)
    if indexed_positions is None:
        indexed_positions = [tuple(int(v) for v in dataset[item_index][1:]) for item_index in range(len(dataset))]
    else:
        # dataset.indices 条目为 (image_idx, z, w, h) 四元组
        indexed_positions = [tuple(int(v) for v in entry) for entry in indexed_positions]
    patch_shape = (int(dataset.z_patch), int(dataset.w_patch), int(dataset.h_patch))
    candidate_scores = []
    for item_index, (image_idx, z_idx, y_idx, x_idx) in enumerate(indexed_positions):
        if int(image_idx) != 0:
            continue
        fg_count = int(_mask_patch(masks["foreground"], int(z_idx), int(y_idx), int(x_idx), patch_shape).sum())
        candidate_scores.append((fg_count, item_index))
    selected = [item for _, item in sorted(candidate_scores, reverse=True)[:max_patches]]

    pred_path = sorted(PRED.glob("*.tif"))[0]
    pred_raw = tifffile.imread(pred_path).astype(np.float32)
    stack_eval = json.loads(STACK_EVAL.read_text())
    a1 = float(stack_eval["scale_fit"]["a"])
    a0 = float(stack_eval["scale_fit"]["b"])
    pred_aligned = a1 * pred_raw + a0

    rows = []
    for item_index in selected:
        image_idx, z0, y0, x0 = indexed_positions[item_index]
        pred_patch = pred_aligned[z0 : z0 + patch_shape[0], y0 : y0 + patch_shape[1], x0 : x0 + patch_shape[2]]
        target = _mask_patch(mean9, z0, y0, x0, patch_shape) * 1.0
        target = mean9[z0 : z0 + patch_shape[0], y0 : y0 + patch_shape[1], x0 : x0 + patch_shape[2]]
        if pred_patch.shape != target.shape:
            continue
        for region, mask in masks.items():
            m = _mask_patch(mask, z0, y0, x0, pred_patch.shape)
            rows.append({"patch_index": item_index, "region": region, **_region_metrics(pred_patch, target, m)})

    summary = {
        "event": "e1_sn2n_original_patch_eval",
        "prediction": str(pred_path),
        "scale_alignment": {"a": a1, "b": a0, "note": "global linear fit at stack level; SN2N output has no absolute scale"},
        "reference_protocol": str(REFERENCE_CONFIG),
        "patch_count": max_patches,
        "selected_patch_indices": selected,
        "selection": "foreground_rich",
        "rows": rows,
    }
    regions = {}
    for region in ("all", "foreground", "background"):
        sub = [r for r in rows if r["region"] == region]
        regions[region] = {
            "psnr_db_mean": float(np.mean([r["psnr_db"] for r in sub])),
            "n_patches": len(sub),
        }
    summary["regions"] = regions
    OUTPUT.write_text(json.dumps(summary, indent=2))
    for region, st in regions.items():
        print(f"{region:12s} patch-mean psnr={st['psnr_db_mean']:.2f} dB ({st['n_patches']} patches)")
    print("saved", OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
