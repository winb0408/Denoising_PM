"""QMask Stage2 sampling-only 训练 runner（独立，不修改 ACES 原代码）。

控制档 valid/sn2n_xy/iso3d 只读调用 aces.sampling_aniso.sample_four_views；
qmask 调用本包 sampling.qmask_sampler；模型/损失/评估复用 VALID 与 aces。
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from aces.evaluate_stage2 import evaluate_model_on_patches
from aces.io_utils import load_config, write_json
from aces.pairability import PairabilityGeometry  # noqa: F401  (type reference kept for compatibility)
from aces.sampling_aniso import metadata_to_dict, sample_four_views
from aces.train_stage2 import (
    _background_variance,
    _load_valid_components,
    _set_seed,
    _write_eval_curve,
    _write_loss_csv,
)

from .evaluate import depth_binned_foreground_psnr
from .geometry import geometry_from_stage1_run, preview_text, write_geometry
from .sampling import qmask_sampler

MODEL_MODES = ("valid", "sn2n_xy", "iso3d", "qmask")


def _sample_views(mode, input_tensor, device_geometry, seed, row_mode="fixed"):
    if mode == "qmask":
        return qmask_sampler(input_tensor, device_geometry, seed=seed, row_mode=row_mode)
    return sample_four_views(input_tensor, mode, geometry=None, seed=seed)


def _grad3d_l1(pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    """3D 前向差分梯度 L1（B2 锐度对齐项）。"""
    dz = (pred[:, :, 1:] - pred[:, :, :-1]) - (tgt[:, :, 1:] - tgt[:, :, :-1])
    dy = (pred[:, :, :, 1:] - pred[:, :, :, :-1]) - (tgt[:, :, :, 1:] - tgt[:, :, :, :-1])
    dx = (pred[..., 1:] - pred[..., :-1]) - (tgt[..., 1:] - tgt[..., :-1])
    return (dz.abs().mean() + dy.abs().mean() + dx.abs().mean()) / 3.0


def _fg_weighted_l2(pred: torch.Tensor, tgt: torch.Tensor, percentile: float, alpha: float) -> torch.Tensor:
    """前景加权 L2（B3）：fg = tgt 超过其 percentile 分位，fg 权重 1+alpha、bg 权重 1。"""
    with torch.no_grad():
        flat = tgt.flatten(1)
        thr = torch.quantile(flat, percentile / 100.0, dim=1).view(-1, 1, 1, 1, 1)
        fg = (tgt > thr).float()
    w = 1.0 + alpha * fg
    return (w * (pred - tgt) ** 2).mean()


def run_mode(cfg: dict[str, Any], mode: str, geometry=None) -> dict[str, Any]:
    if mode not in MODEL_MODES:
        raise ValueError(f"Unknown mode {mode!r}; expected {MODEL_MODES}")
    seed = int(cfg.get("seed", 3407))
    # Q-3 消融开关：sampling.row_mode="mosaic" 时 qmask 用逐位置随机 Tetris 行
    # （VALID-like 马赛克视图）；默认 "fixed" 保持共位相位分裂。
    row_mode = str(cfg.get("sampling", {}).get("row_mode", "fixed"))
    _set_seed(seed)
    run_root = Path(cfg["run_dir"]).resolve()
    mode_dir = run_root / mode
    for rel in ("logs", "metrics", "checkpoints"):
        (mode_dir / rel).mkdir(parents=True, exist_ok=True)
    write_json(mode_dir / "config.json", cfg)

    _now = time.perf_counter()
    valid = _load_valid_components()
    train_cfg = cfg["training"]
    dataset = valid["ReadDatasets"](
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
    loader = DataLoader(
        dataset,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(train_cfg["num_workers"]),
        pin_memory=True,
        collate_fn=valid["custom_collate_fn"],
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = valid["Network_CNR"](
        in_channels=1,
        out_channels=1,
        f_maps=int(train_cfg["base_features"]),
        n_groups=int(train_cfg["n_groups"]),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(train_cfg["lr"]), betas=(0.9, 0.999))
    l2 = torch.nn.MSELoss()
    hessian_loss = valid["HessianConstraintLoss3D"]()
    dwt3d = valid["DWT_3D"](wavename="haar")

    total_epochs = int(train_cfg["epochs"])
    eval_every = int(train_cfg.get("eval_every_epochs", 0))
    bg_reg_weight = float(train_cfg.get("bg_reg_weight", 0.0))
    grad_loss_weight = float(train_cfg.get("gradient_loss_weight", 0.0))  # B2
    fg_loss_weight = float(train_cfg.get("fg_loss_weight", 0.0))  # B3
    fg_percentile = float(train_cfg.get("fg_percentile", 75.0))  # B3
    bg_percentile = float(train_cfg.get("bg_percentile", 25.0))
    best_psnr = float("-inf")
    best_epoch = -1
    loss_rows: list[dict[str, Any]] = []
    metadata_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []

    def mid_eval(epoch_idx: int) -> None:
        nonlocal best_psnr, best_epoch
        model.eval()
        summary = evaluate_model_on_patches(
            model=model,
            dataset=dataset,
            device=device,
            output_dir=mode_dir,
            mean9_path=cfg["evaluation"].get("mean9_proxy"),
            max_patches=int(cfg["evaluation"].get("patch_count", 1)),
            foreground_percentile=float(cfg["evaluation"].get("foreground_percentile", 75.0)),
            foreground_mask_path=cfg["evaluation"].get("foreground_mask"),
        )
        fg_psnr = float(summary["regions"]["foreground"]["psnr_db"])
        eval_rows.append({
            "epoch": epoch_idx,
            "batch": len(loss_rows),
            "fg_psnr_db": fg_psnr,
            "all_psnr_db": float(summary["regions"]["all"]["psnr_db"]),
        })
        if fg_psnr > best_psnr:
            best_psnr = fg_psnr
            best_epoch = epoch_idx
            torch.save(
                {"state_dict": model.state_dict(), "mode": mode, "epoch": epoch_idx, "fg_psnr_db": fg_psnr},
                mode_dir / "checkpoints" / f"{mode}_best.pth",
            )
        torch.save(
            {"state_dict": model.state_dict(), "mode": mode, "epoch": epoch_idx, "fg_psnr_db": fg_psnr},
            mode_dir / "checkpoints" / f"{mode}_epoch{epoch_idx:04d}.pth",
        )
        _write_eval_curve(mode_dir / "logs" / "eval_curve.csv", eval_rows)
        model.train()

    model.train()
    for epoch in range(total_epochs):
        print(f"[{mode}] epoch {epoch + 1}/{total_epochs} start", flush=True)
        for batch_idx, data in enumerate(loader):
            input_tensor = data[0].to(device).squeeze(1)
            views, metadata = _sample_views(
                mode, input_tensor, geometry, seed=seed * 100_000 + epoch * 1_000 + batch_idx, row_mode=row_mode
            )
            metadata_rows.append(metadata_to_dict(metadata))
            noisy_sub1, noisy_sub2, noisy_sub3, noisy_sub4 = [view.unsqueeze(1) for view in views]
            noisy_output_1 = model(noisy_sub1)
            noisy_output_2 = model(noisy_sub2)
            lll1, llh1, lhl1, _, hll1, _, _, _ = dwt3d(noisy_output_1)
            lll2, llh2, lhl2, _, hll2, _, _, _ = dwt3d(noisy_output_2)
            # B3（P-B）：fg_loss_weight>0 时 loss2neighbor 的前景体素加权
            # （fg = target 视图超过 fg_percentile 分位），把容量向前景倾斜提 SBR。
            if fg_loss_weight > 0:
                loss2neighbor = 0.5 * _fg_weighted_l2(
                    noisy_output_1, noisy_sub3, fg_percentile, fg_loss_weight
                ) + 0.5 * _fg_weighted_l2(
                    noisy_output_2, noisy_sub4, fg_percentile, fg_loss_weight
                )
            else:
                loss2neighbor = 0.5 * l2(noisy_output_1, noisy_sub3) + 0.5 * l2(noisy_output_2, noisy_sub4)
            loss_idt = l2(noisy_output_1, noisy_output_2)
            # B2（P-B）：gradient_loss_weight>0 时加 3D 前向差分梯度 L1 对齐项（锐度约束）。
            loss_grad = 0.5 * _grad3d_l1(noisy_output_1, noisy_sub3) + 0.5 * _grad3d_l1(noisy_output_2, noisy_sub4)
            loss_reg = hessian_loss(torch.cat([lll1, llh1, lhl1, hll1, lll2, llh2, lhl2, hll2], dim=1))
            loss_bg = 0.5 * (
                _background_variance(noisy_output_1, noisy_sub3, bg_percentile)
                + _background_variance(noisy_output_2, noisy_sub4, bg_percentile)
            )
            total_loss = (
                loss2neighbor
                + loss_idt
                + float(train_cfg["weight_reg"]) * loss_reg
                + bg_reg_weight * loss_bg
                + grad_loss_weight * loss_grad
            )
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(train_cfg["clip_gradients"]))
            optimizer.step()
            loss_rows.append({
                "epoch": epoch + 1,
                "batch": batch_idx + 1,
                "total_loss": float(total_loss.detach().cpu()),
                "loss2neighbor": float(loss2neighbor.detach().cpu()),
                "loss_idt": float(loss_idt.detach().cpu()),
                "loss_reg": float(loss_reg.detach().cpu()),
                "loss_bg": float(loss_bg.detach().cpu()),
                "loss_grad": float(loss_grad.detach().cpu()),
            })
            if int(train_cfg.get("max_batches_per_epoch", -1)) > 0 and batch_idx + 1 >= int(train_cfg["max_batches_per_epoch"]):
                break
        if eval_every > 0 and ((epoch + 1) % eval_every == 0 or (epoch + 1) == total_epochs):
            print(f"[{mode}] epoch {epoch + 1}/{total_epochs} eval start", flush=True)
            mid_eval(epoch + 1)
            print(f"[{mode}] epoch {epoch + 1}/{total_epochs} eval done", flush=True)

    if eval_rows:
        _write_eval_curve(mode_dir / "logs" / "eval_curve.csv", eval_rows)

    checkpoint_path = mode_dir / "checkpoints" / f"{mode}_final.pth"
    torch.save({"state_dict": model.state_dict(), "mode": mode, "config": cfg, "loss_rows": loss_rows}, checkpoint_path)
    _write_loss_csv(mode_dir / "logs" / "train_losses.csv", loss_rows)
    write_json(mode_dir / "logs" / "sampler_metadata.json", {"event": "qmask_sampler_metadata", "calls": metadata_rows})

    eval_summary = evaluate_model_on_patches(
        model=model,
        dataset=dataset,
        device=device,
        output_dir=mode_dir,
        mean9_path=cfg["evaluation"].get("mean9_proxy"),
        max_patches=int(cfg["evaluation"].get("patch_count", 1)),
        foreground_percentile=float(cfg["evaluation"].get("foreground_percentile", 75.0)),
        foreground_mask_path=cfg["evaluation"].get("foreground_mask"),
        metrics_filename="eval_metrics.json",
    )

    probe_views, _ = _sample_views(
        mode,
        torch.zeros(1, int(train_cfg["z_patch"]), int(train_cfg["w_patch"]), int(train_cfg["h_patch"]), device=device),
        geometry,
        seed=seed,
        row_mode=row_mode,
    )
    in_shape = (int(train_cfg["z_patch"]), int(train_cfg["w_patch"]), int(train_cfg["h_patch"]))
    out_shape = tuple(int(s) for s in probe_views[0].shape[1:])
    stride = tuple(max(1, round(i / o)) for i, o in zip(in_shape, out_shape))
    aligned = None
    if stride != (1, 1, 1):
        aligned = evaluate_model_on_patches(
            model=model,
            dataset=dataset,
            device=device,
            output_dir=mode_dir,
            mean9_path=cfg["evaluation"].get("mean9_proxy"),
            max_patches=int(cfg["evaluation"].get("patch_count", 1)),
            foreground_percentile=float(cfg["evaluation"].get("foreground_percentile", 75.0)),
            foreground_mask_path=cfg["evaluation"].get("foreground_mask"),
            downsample_stride=stride,
            metrics_filename="eval_metrics_aligned.json",
        )

    depth = depth_binned_foreground_psnr(
        model=model,
        dataset=dataset,
        device=device,
        mean9_path=cfg["evaluation"].get("mean9_proxy"),
        foreground_percentile=float(cfg["evaluation"].get("foreground_percentile", 75.0)),
        foreground_mask_path=cfg["evaluation"].get("foreground_mask"),
        bins=int(cfg["evaluation"].get("depth_bins", 5)),
        stride=stride,
        patch_cap=int(cfg["evaluation"].get("depth_patch_cap", 2000)),
    )
    write_json(mode_dir / "metrics" / "depth_binned_metrics.json", depth)

    summary = {
        "event": "qmask_stage2_mode_complete",
        "mode": mode,
        "mode_dir": str(mode_dir),
        "checkpoint": str(checkpoint_path),
        "train_batches": len(loss_rows),
        "elapsed_seconds": time.perf_counter() - _now,
        "final_loss": loss_rows[-1]["total_loss"] if loss_rows else None,
        "eval_regions": eval_summary["regions"],
        "eval_regions_aligned": aligned["regions"] if aligned else None,
        "aligned_stride": list(stride),
        "depth_binned": depth,
        "best_epoch": best_epoch,
        "best_fg_psnr_db": best_psnr if best_epoch > 0 else None,
        "eval_curve": eval_rows,
    }
    write_json(mode_dir / "logs" / "summary.json", summary)
    return summary


def run_stage2(config_path: str | Path, modes: list[str] | None = None) -> dict[str, Any]:
    cfg = load_config(config_path)
    selected_modes = modes or list(cfg["sampling"]["modes"])
    run_root = Path(cfg["run_dir"]).resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    write_json(run_root / "config.json", cfg)

    geometry = None
    if "qmask" in selected_modes:
        geo_cfg = cfg["geometry"]
        geometry = geometry_from_stage1_run(
            geo_cfg["stage1_run_dir"],
            region=geo_cfg.get("region", "foreground"),
            noise_source=geo_cfg.get("noise_source", "repeat"),
            tau_n=float(geo_cfg.get("tau_n", 0.95)),
            tau_s=float(geo_cfg.get("tau_s", 0.85)),
            tau_q=float(geo_cfg.get("tau_q", 0.60)),
            # Q-2：proxy-only 数据可设 n_decision=false 降级 N 决策（默认 true 保持原行为，
            # 切换前须按 Plan v2 §4-3 先做验证 pilot）。
            use_n_decision=bool(geo_cfg.get("n_decision", True)),
        )
        print(preview_text(geometry))
        write_geometry(run_root / "qmask_geometry.json", geometry)

    summaries = [run_mode(cfg, mode, geometry=geometry) for mode in selected_modes]
    result = {"event": "qmask_stage2_complete", "run_dir": str(run_root), "modes": summaries}
    write_json(run_root / "logs" / "summary.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train QMask Stage2 sampling-only modes.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--modes", default="all", help="Comma-separated modes or 'all'.")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    modes = list(cfg["sampling"]["modes"]) if args.modes == "all" else [m.strip() for m in args.modes.split(",") if m.strip()]
    result = run_stage2(args.config, modes=modes)
    print(json.dumps({"run_dir": result["run_dir"], "modes": result["modes"]}, indent=2, default=str)[:2000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
