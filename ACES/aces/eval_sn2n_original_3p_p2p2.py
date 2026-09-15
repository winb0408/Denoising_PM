"""E1-3P-p2p2:E1-3P 的 P2Pmode=2 单变量重跑,用与 E2/E1-3P 完全相同的 patch 协议评估。

复用 aces/eval_sn2n_original_3p.py 的协议(ReadDatasets 同网格 + top-18 fg patch +
_region_metrics),仅路径换到 p2p2 run;尺度对齐在脚本内 stack 级 lstsq 完成
(不依赖 unified_eval.json)。

用法(仓库根目录): PYTHONPATH=ACES python -m aces.eval_sn2n_original_3p_p2p2
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import tifffile

PRED = Path("/data3/wjb/Denoising_PM/Results/sn2n_original_3p_p2p2/predictions")
OUTPUT = Path("/data3/wjb/Denoising_PM/Results/sn2n_original_3p_p2p2/patch_eval_metrics.json")
MEAN9 = Path("/data2/wjb/Denoising_PM/Methods/FAST-main/reproduction/valid_comparison/"
             "experiments/valid_paper_3p_neuron/original_workflow_paper_params_repeat01_seed3407/"
             "data/gt_proxy/3P_neuron_mean9_gt_proxy_275z.tif")
REFERENCE_CONFIG = Path("/data2/wjb/Denoising_PM/ACES/runs/stage02_valid_40ep_seed3407/valid/config.json")


def main() -> int:
    from .evaluate_stage2 import _load_target_and_masks, _mask_patch, _region_metrics

    ref = json.loads(REFERENCE_CONFIG.read_text())
    train_cfg = ref["training"]
    eval_cfg = ref["evaluation"]
    max_patches = int(eval_cfg.get("patch_count", 18))

    sys.path.insert(0, str(Path("/data2/wjb/Denoising_PM/Methods/VALID-v1.0/FDU-donglab-VALID-34f5637")))
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
        str(eval_cfg["mean9_proxy"]),
        float(eval_cfg.get("foreground_percentile", 75.0)),
        eval_cfg.get("foreground_mask"),
    )

    indexed_positions = getattr(dataset, "indices", None)
    if indexed_positions is None:
        indexed_positions = [tuple(int(v) for v in dataset[item_index][1:]) for item_index in range(len(dataset))]
    else:
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
    z = min(pred_raw.shape[0], mean9.shape[0]); y = min(pred_raw.shape[1], mean9.shape[1]); x = min(pred_raw.shape[2], mean9.shape[2])
    pred_raw = pred_raw[:z, :y, :x]
    A = np.stack([pred_raw.ravel(), np.ones(pred_raw.size)], axis=1)
    coef, *_ = np.linalg.lstsq(A, mean9[:z, :y, :x].ravel(), rcond=None)
    a1, a0 = float(coef[0]), float(coef[1])
    print(f"scale fit: {a1:.4f}*pred + {a0:.2f}")
    pred_aligned = a1 * pred_raw + a0

    rows = []
    for item_index in selected:
        image_idx, z0, y0, x0 = indexed_positions[item_index]
        z0, y0, x0 = int(z0), int(y0), int(x0)
        pred_patch = pred_aligned[z0:z0 + patch_shape[0], y0:y0 + patch_shape[1], x0:x0 + patch_shape[2]]
        target = mean9[z0:z0 + patch_shape[0], y0:y0 + patch_shape[1], x0:x0 + patch_shape[2]]
        if pred_patch.shape != target.shape:
            continue
        for region, mask in masks.items():
            m = _mask_patch(mask, z0, y0, x0, pred_patch.shape)
            rows.append({"patch_index": item_index, "region": region, **_region_metrics(pred_patch, target, m)})

    regions = {}
    for region in ("all", "foreground", "background"):
        sub = [r for r in rows if r["region"] == region]
        regions[region] = {"psnr_db_mean": float(np.mean([r["psnr_db"] for r in sub])), "n_patches": len(sub)}
    summary = {
        "event": "e1_sn2n_original_3p_p2p2_patch_eval",
        "prediction": str(pred_path),
        "scale_alignment": {"a": a1, "b": a0, "note": "global linear fit at stack level; SN2N output has no absolute scale"},
        "reference_protocol": str(REFERENCE_CONFIG),
        "patch_count": max_patches,
        "selected_patch_indices": selected,
        "selection": "foreground_rich",
        "regions": regions,
        "rows": rows,
    }
    OUTPUT.write_text(json.dumps(summary, indent=2))
    for region, st in regions.items():
        print(f"{region:12s} patch-mean psnr={st['psnr_db_mean']:.2f} dB ({st['n_patches']} patches)")
    print("saved", OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
