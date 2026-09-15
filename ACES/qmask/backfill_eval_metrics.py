"""T4b C2（Plan v2 §7 P-C）：历史 run 补列 SSIM/SNR/SBR——仅重评估，不重训。

C1 扩展了 `_region_metrics`（SSIM/SNR/SBR），本脚本用已存 checkpoint 重跑
`evaluate_model_on_patches`，原地覆写 eval_metrics(.aligned).json（旧键全部保留，
新增 ssim/snr_db/sbr_db 列）。已含 ssim 的文件自动跳过（幂等，可断点续跑）。

覆盖范围：
- qmask80ep ×3 seed（`runs/stage02_nf_qmask80ep_seed*/qmask/`）
- 矩阵 4 臂 ×3 seed（`runs/stage02_nf_matrix4_seed*/<arm>/`）
- rgrid d1–d64（`runs/stage01_cidc25_rgrid_seed3407/d*/`，best + final × F1/F2/F3）

用法：
    PYTHONPATH=ACES python -m qmask.backfill_eval_metrics --family qmask_nf
    PYTHONPATH=ACES python -m qmask.backfill_eval_metrics --family rgrid
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from aces.io_utils import load_config
from aces.train_stage2 import _load_valid_components

QMASK80EP_ROOT = Path("/data3/wjb/Denoising_PM/ACES/runs")
RGRID_RUN = Path("/data3/wjb/Denoising_PM/ACES/runs/stage01_cidc25_rgrid_seed3407")
MATRIX_ROOT = QMASK80EP_ROOT


def _has_ssim(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    regions = data.get("regions", {})
    return bool(regions) and all("ssim" in m for m in regions.values())


def _eval_mode_dir(mode_dir: Path, mode: str, device: torch.device, valid: dict[str, Any]) -> None:
    """final ckpt → 覆写 eval_metrics(.aligned).json（保数字、加新列，官方口径不变）；
    best ckpt → 另存 eval_best_metrics(.aligned).json（best/final 差值保留分析用）。"""
    from aces.evaluate_stage2 import evaluate_model_on_patches

    cfg = load_config(str(mode_dir / "config.json"))
    train_cfg = cfg["training"]
    eval_cfg = cfg["evaluation"]

    def build_dataset():
        return valid["ReadDatasets"](
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

    def load_model(weight: str):
        ckpt_path = mode_dir / "checkpoints" / f"{mode}_{weight}.pth"
        if not ckpt_path.exists():
            return None
        model = valid["Network_CNR"](
            in_channels=1,
            out_channels=1,
            f_maps=int(train_cfg["base_features"]),
            n_groups=int(train_cfg["n_groups"]),
        ).to(device)
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck["state_dict"])
        model.eval()
        return model

    aligned_stride: tuple[int, int, int] | None = None
    old_aligned = mode_dir / "metrics" / "eval_metrics_aligned.json"
    if old_aligned.exists():
        aligned_stride = tuple(json.loads(old_aligned.read_text(encoding="utf-8")).get("downsample_stride_zyx", [1, 1, 1]))

    for weight, full_name, aligned_name in (
        ("final", "eval_metrics.json", "eval_metrics_aligned.json"),
        ("best", "eval_best_metrics.json", "eval_best_metrics_aligned.json"),
    ):
        model = load_model(weight)
        if model is None:
            print(f"[skip] {mode_dir} 无 {weight} ckpt", flush=True)
            continue
        dataset = build_dataset()
        full_path = mode_dir / "metrics" / full_name
        if not _has_ssim(full_path):
            evaluate_model_on_patches(
                model=model,
                dataset=dataset,
                device=device,
                output_dir=mode_dir,
                mean9_path=eval_cfg.get("mean9_proxy"),
                max_patches=int(eval_cfg.get("patch_count", 2)),
                foreground_percentile=float(eval_cfg.get("foreground_percentile", 75.0)),
                foreground_mask_path=eval_cfg.get("foreground_mask"),
                metrics_filename=full_name,
            )
            print(f"[done] {full_path}", flush=True)
        else:
            print(f"[skip] {full_path} 已含 ssim", flush=True)
        if aligned_stride and tuple(aligned_stride) != (1, 1, 1):
            aligned_path = mode_dir / "metrics" / aligned_name
            if not _has_ssim(aligned_path):
                evaluate_model_on_patches(
                    model=model,
                    dataset=build_dataset(),
                    device=device,
                    output_dir=mode_dir,
                    mean9_path=eval_cfg.get("mean9_proxy"),
                    max_patches=int(eval_cfg.get("patch_count", 2)),
                    foreground_percentile=float(eval_cfg.get("foreground_percentile", 75.0)),
                    foreground_mask_path=eval_cfg.get("foreground_mask"),
                    downsample_stride=aligned_stride,
                    metrics_filename=aligned_name,
                )
                print(f"[done] {aligned_path} (stride={aligned_stride})", flush=True)
            else:
                print(f"[skip] {aligned_path} 已含 ssim", flush=True)
        del model
        torch.cuda.empty_cache()


def _eval_rgrid_d(d_dir: Path, distance: int, device: torch.device, valid: dict[str, Any]) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from qmask.t4b_rgrid import (
        EVAL_STACKS,
        GT_STACK,
        SingleStackEvalDataset,
        eval_stack_stats,
    )

    cfg = load_config(str(d_dir / "config.json"))
    train_cfg = cfg["training"]
    eval_cfg = cfg["evaluation"]
    stack_shape = None
    from aces.evaluate_stage2 import evaluate_model_on_patches

    for weight_tag, ck_name in (
        ("best", f"tpair_d{distance}_best.pth"),
        ("final", f"tpair_d{distance}_final.pth"),
    ):
        ckpt_path = d_dir / "checkpoints" / ck_name
        if not ckpt_path.exists():
            print(f"[skip] {ckpt_path}", flush=True)
            continue
        model = valid["Network_CNR"](
            in_channels=1,
            out_channels=1,
            f_maps=int(train_cfg["base_features"]),
            n_groups=int(train_cfg["n_groups"]),
        ).to(device)
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck["state_dict"])
        model.eval()
        for level in EVAL_STACKS:
            out_name = f"eval_{weight_tag}_{level}.json"
            if _has_ssim(d_dir / "metrics" / out_name):
                print(f"[skip] {d_dir / 'metrics' / out_name}", flush=True)
                continue
            mean_lvl, std_lvl = eval_stack_stats(cfg, level)
            ds = SingleStackEvalDataset(
                Path(cfg["data_root"]) / "Validation" / EVAL_STACKS[level],
                mean_lvl,
                std_lvl,
                z_patch=int(train_cfg["z_patch"]),
                w_patch=int(train_cfg["w_patch"]),
                h_patch=int(train_cfg["h_patch"]),
            )
            evaluate_model_on_patches(
                model=model,
                dataset=ds,
                device=device,
                output_dir=d_dir,
                mean9_path=Path(cfg["data_root"]) / "Validation" / GT_STACK,
                max_patches=int(eval_cfg.get("patch_count", 18)),
                foreground_percentile=float(eval_cfg.get("foreground_percentile", 75.0)),
                metrics_filename=out_name,
            )
            print(f"[done] {d_dir / 'metrics' / out_name}", flush=True)
            if stack_shape is None:
                stack_shape = ds.stack.shape
        del model
        torch.cuda.empty_cache()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="C2: backfill SSIM/SNR/SBR into historical eval JSONs")
    parser.add_argument("--family", required=True, choices=("qmask_nf", "matrix_nf", "rgrid"))
    parser.add_argument("--gpu", type=int, default=1)
    args = parser.parse_args(argv)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    valid = _load_valid_components()

    if args.family in ("qmask_nf", "matrix_nf"):
        roots = (
            sorted(QMASK80EP_ROOT.glob("stage02_nf_qmask80ep_seed*"))
            if args.family == "qmask_nf"
            else sorted(MATRIX_ROOT.glob("stage02_nf_matrix4_seed*"))
        )
        for run_root in roots:
            mode_dirs = (
                [run_root / "qmask"]
                if args.family == "qmask_nf"
                else [d for d in sorted(run_root.iterdir()) if (d / "config.json").exists()]
            )
            for mode_dir in mode_dirs:
                mode = mode_dir.name
                if (mode_dir / "config.json").exists():
                    print(f"=== {mode_dir} ({mode})", flush=True)
                    _eval_mode_dir(mode_dir, mode, device, valid)
    else:
        for d_dir in sorted(RGRID_RUN.glob("d*")):
            if not (d_dir / "config.json").exists():
                continue
            distance = int(d_dir.name[1:])
            print(f"=== {d_dir} (d={distance})", flush=True)
            _eval_rgrid_d(d_dir, distance, device, valid)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
