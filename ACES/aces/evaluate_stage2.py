from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
import torch

from .io_utils import write_json
from .paths import MEAN9_PROXY, VALID_ROOT
from .profile_anisotropy import build_region_masks


def _ssim_3d(prediction: np.ndarray, target: np.ndarray, data_range: float) -> float:
    """窗口 SSIM（3D 体 patch，win=7）。窗口大于体积维度时缩小到最大奇数窗。"""
    from skimage.metrics import structural_similarity

    win = 7
    for size in prediction.shape:
        if size < win:
            win = size if size % 2 == 1 else size - 1
    if win < 3:
        return float("nan")
    value = structural_similarity(
        prediction.astype(np.float64, copy=False),
        target.astype(np.float64, copy=False),
        win_size=win,
        data_range=max(data_range, 1e-6),
    )
    return float(value)


def _region_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    bg_mask: np.ndarray | None = None,
) -> dict[str, float]:
    valid = mask & np.isfinite(prediction) & np.isfinite(target)
    if not valid.any():
        return {
            "mse": float("nan"),
            "mae": float("nan"),
            "psnr_db": float("nan"),
            "ssim": float("nan"),
            "snr_db": float("nan"),
            "sbr_db": float("nan"),
            "pred_mean": float("nan"),
            "target_mean": float("nan"),
            "pred_std": float("nan"),
            "target_std": float("nan"),
            "n": 0,
        }
    pred = prediction[valid].astype(np.float64, copy=False)
    ref = target[valid].astype(np.float64, copy=False)
    diff = pred - ref
    mse = float(np.mean(diff * diff))
    mae = float(np.mean(np.abs(diff)))
    data_range = float(np.percentile(ref, 99.9) - np.percentile(ref, 0.1))
    psnr = math.inf if mse == 0 else float(10.0 * math.log10((max(data_range, 1e-6) ** 2) / mse))
    # SSIM 是全 patch 的空间结构指标：在完整 patch 上算一次（与 region 无关），
    # data_range 固定用全 patch 分位（保证三个 region 行数值一致）；
    # SNR/SBR 只对含前景的 region 有定义。
    full_ref = target[np.isfinite(target)].astype(np.float64, copy=False)
    full_range = float(np.percentile(full_ref, 99.9) - np.percentile(full_ref, 0.1)) if full_ref.size else float("nan")
    ssim = _ssim_3d(prediction, target, full_range)
    snr_db = float("nan")
    sbr_db = float("nan")
    if bg_mask is not None:
        bg_valid = bg_mask & np.isfinite(prediction)
        if bg_valid.any():
            bg_pred = prediction[bg_valid].astype(np.float64, copy=False)
            bg_std = float(bg_pred.std())
            bg_mean = float(bg_pred.mean())
            fg_mean = float(pred.mean())
            snr_db = math.inf if bg_std == 0 else float(20.0 * math.log10(max(fg_mean, 1e-6) / max(bg_std, 1e-6)))
            sbr_db = math.inf if bg_mean <= 0 else float(20.0 * math.log10(max(fg_mean, 1e-6) / max(bg_mean, 1e-6)))
    return {
        "mse": mse,
        "mae": mae,
        "psnr_db": psnr,
        "ssim": ssim,
        "snr_db": snr_db,
        "sbr_db": sbr_db,
        "pred_mean": float(pred.mean()),
        "target_mean": float(ref.mean()),
        "pred_std": float(pred.std()),
        "target_std": float(ref.std()),
        "n": int(valid.sum()),
    }


def _stride_slice(array: np.ndarray, stride: tuple[int, int, int]) -> np.ndarray:
    """按 (sz, sy, sx) 对 3D 体积做跨步下采样。stride=(1,1,1) 时原样返回。"""
    sz, sy, sx = stride
    return array[::sz, ::sy, ::sx]


def _pad_to_shape(array: np.ndarray, shape: tuple[int, ...], *, constant: float | bool) -> np.ndarray:
    slices = tuple(slice(0, min(size, array.shape[index])) for index, size in enumerate(shape))
    cropped = array[slices]
    pad_width = [(0, size - cropped.shape[index]) for index, size in enumerate(shape)]
    if any(width for _, width in pad_width):
        cropped = np.pad(cropped, pad_width, mode="constant", constant_values=constant)
    return cropped


def _target_patch(target: np.ndarray, z0: int, y0: int, x0: int, shape_zyx: tuple[int, int, int]) -> np.ndarray:
    if target.ndim == 2:
        patch_2d = target[y0 : y0 + shape_zyx[1], x0 : x0 + shape_zyx[2]]
        patch_2d = _pad_to_shape(patch_2d, shape_zyx[1:], constant=0.0)
        return np.broadcast_to(patch_2d, shape_zyx).astype(np.float32, copy=False)
    patch = target[z0 : z0 + shape_zyx[0], y0 : y0 + shape_zyx[1], x0 : x0 + shape_zyx[2]]
    return _pad_to_shape(patch, shape_zyx, constant=0.0)


def _mask_patch(mask: np.ndarray, z0: int, y0: int, x0: int, shape_zyx: tuple[int, int, int]) -> np.ndarray:
    if mask.ndim == 2:
        patch_2d = mask[y0 : y0 + shape_zyx[1], x0 : x0 + shape_zyx[2]]
        patch_2d = _pad_to_shape(patch_2d, shape_zyx[1:], constant=False)
        return np.broadcast_to(patch_2d, shape_zyx)
    patch = mask[z0 : z0 + shape_zyx[0], y0 : y0 + shape_zyx[1], x0 : x0 + shape_zyx[2]]
    return _pad_to_shape(patch, shape_zyx, constant=False)


def _load_target_and_masks(
    mean9_path: str | Path,
    foreground_percentile: float,
    foreground_mask_path: str | Path | None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    target = tifffile.imread(mean9_path).astype(np.float32)
    if foreground_mask_path:
        foreground = tifffile.imread(foreground_mask_path).astype(bool)
        masks = {
            "all": np.ones_like(foreground, dtype=bool),
            "foreground": foreground,
            "background": ~foreground,
        }
    else:
        masks = build_region_masks(target, percentile=foreground_percentile)
    return target, masks


def evaluate_model_on_patches(
    *,
    model: torch.nn.Module,
    dataset: Any,
    device: torch.device,
    output_dir: Path,
    mean9_path: str | Path = MEAN9_PROXY,
    max_patches: int = 2,
    foreground_percentile: float = 75.0,
    foreground_mask_path: str | Path | None = None,
    downsample_stride: tuple[int, int, int] | None = None,
    metrics_filename: str = "eval_metrics.json",
) -> dict[str, Any]:
    # downsample_stride 非空时启用"对齐评估"：对输入 patch 与 mean9 目标/掩码施加
    # 相同的跨步下采样，使下采样类采样档（valid/sn2n_xy/iso3d）在评估时看到的分辨率
    # 与其训练时一致，消除 train/test 尺度错配这一混淆因素。stride=(1,1,1) 等价于原始全分辨率评估。
    stride = downsample_stride
    align_active = stride is not None and tuple(stride) != (1, 1, 1)
    mean9, masks = _load_target_and_masks(mean9_path, foreground_percentile, foreground_mask_path)
    candidate_scores: list[tuple[int, int]] = []
    indexed_positions = getattr(dataset, "indices", None)
    if indexed_positions is None:
        indexed_positions = []
        for item_index in range(len(dataset)):
            _, image_idx, z_idx, y_idx, x_idx = dataset[item_index]
            indexed_positions.append((int(image_idx), int(z_idx), int(y_idx), int(x_idx)))
    for item_index, (image_idx, z_idx, y_idx, x_idx) in enumerate(indexed_positions):
        if int(image_idx) != 0:
            continue
        z0, y0, x0 = int(z_idx), int(y_idx), int(x_idx)
        patch_shape = (int(dataset.z_patch), int(dataset.w_patch), int(dataset.h_patch))
        foreground_count = int(_mask_patch(masks["foreground"], z0, y0, x0, patch_shape).sum())
        candidate_scores.append((foreground_count, item_index))
    selected_indices = [item for _, item in sorted(candidate_scores, reverse=True)[:max_patches]]
    model.eval()
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for item_index in selected_indices:
            patch, image_idx, z_idx, y_idx, x_idx = dataset[item_index]
            if int(image_idx) != 0:
                continue
            input_np = patch.squeeze(0).numpy() if patch.dim() == 4 else patch.numpy()
            z0, y0, x0 = int(z_idx), int(y_idx), int(x_idx)
            # 目标 patch（全分辨率），随后按需与输入同步下采样，保证 pred/target 网格一致。
            target_full = _target_patch(mean9, z0, y0, x0, tuple(input_np.shape))
            mask_full = {
                name: _mask_patch(mask, z0, y0, x0, tuple(input_np.shape))
                for name, mask in masks.items()
            }
            if align_active:
                input_np = _stride_slice(input_np, stride)
                target_full = _stride_slice(target_full, stride)
                mask_full = {name: _stride_slice(m, stride) for name, m in mask_full.items()}
            input_tensor = torch.from_numpy(np.ascontiguousarray(input_np)).unsqueeze(0).unsqueeze(0).to(device)
            output = model(input_tensor).squeeze(0).squeeze(0).detach().cpu()
            output_denorm = output.numpy().astype(np.float32) * float(dataset.image_stds[0]) + float(dataset.image_means[0])
            target = target_full[: output_denorm.shape[0], : output_denorm.shape[1], : output_denorm.shape[2]]
            if target.shape != output_denorm.shape:
                continue
            patch_masks = {
                name: m[: output_denorm.shape[0], : output_denorm.shape[1], : output_denorm.shape[2]]
                for name, m in mask_full.items()
            }
            for region, mask in patch_masks.items():
                # SNR/SBR 需要背景参考：给 all/foreground 行传背景掩码（background 行自身即背景）。
                bg_mask = patch_masks.get("background") if region in ("all", "foreground") else None
                rows.append({
                    "patch_index": item_index,
                    "region": region,
                    **_region_metrics(output_denorm, target, mask, bg_mask=bg_mask),
                })

    summary: dict[str, Any] = {
        "event": "aces_stage2_patch_evaluation",
        "patch_count": max_patches,
        "selected_patch_indices": selected_indices,
        "selection": "foreground_rich",
        "eval_mode": "downsample_aligned" if align_active else "full_resolution",
        "downsample_stride_zyx": list(stride) if align_active else [1, 1, 1],
        "regions": {},
    }
    for region in ("all", "foreground", "background"):
        selected = [row for row in rows if row["region"] == region and row["n"] > 0]
        summary["regions"][region] = {}
        for key in ("mse", "mae", "psnr_db", "ssim", "snr_db", "sbr_db",
                    "pred_mean", "target_mean", "pred_std", "target_std"):
            vals = [float(row[key]) for row in selected if np.isfinite(row[key])]
            summary["regions"][region][key] = float(np.mean(vals)) if vals else float("nan")
        summary["regions"][region]["n_voxels"] = int(sum(int(row["n"]) for row in selected))
    summary["per_patch"] = rows
    write_json(output_dir / "metrics" / metrics_filename, summary)
    return summary


def evaluate_checkpoint(
    *,
    checkpoint_path: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    from .io_utils import load_config

    cfg = load_config(config_path)
    sys.path.insert(0, str(VALID_ROOT))
    from datasets.dataset_fs import ReadDatasets
    from models.network import Network_CNR

    train_cfg = cfg["training"]
    eval_cfg = cfg["evaluation"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = ReadDatasets(
        dataPath=train_cfg["train_folder"],
        mode="test",
        dataType="3D",
        dataExtension="tif",
        z_patch=int(train_cfg["z_patch"]),
        w_patch=int(train_cfg["w_patch"]),
        h_patch=int(train_cfg["h_patch"]),
        z_overlap=float(train_cfg["z_overlap"]),
        w_overlap=float(train_cfg["w_overlap"]),
        h_overlap=float(train_cfg["h_overlap"]),
        patch_num=int(eval_cfg.get("patch_count", 2)),
        dataNum=int(train_cfg.get("train_frame_num", 10000)),
    )
    model = Network_CNR(
        in_channels=1,
        out_channels=1,
        f_maps=int(train_cfg["base_features"]),
        n_groups=int(train_cfg["n_groups"]),
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    return evaluate_model_on_patches(
        model=model,
        dataset=dataset,
        device=device,
        output_dir=Path(output_dir),
        mean9_path=eval_cfg.get("mean9_proxy", str(MEAN9_PROXY)),
        max_patches=int(eval_cfg.get("patch_count", 2)),
        foreground_percentile=float(eval_cfg.get("foreground_percentile", 75.0)),
        foreground_mask_path=eval_cfg.get("foreground_mask"),
    )


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate an ACES Stage 2 checkpoint on mean9 proxy patches.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    result = evaluate_checkpoint(checkpoint_path=args.checkpoint, config_path=args.config, output_dir=args.output_dir)
    print(json.dumps(result["regions"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
