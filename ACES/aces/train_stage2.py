from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .evaluate_stage2 import evaluate_model_on_patches
from .io_utils import load_config, write_json
from .pairability import geometry_from_stage1_run
from .paths import DEFAULT_RUN_DIR, MEAN9_PROXY, VALID_ROOT
from .sampling_aniso import VALID_MODES, metadata_to_dict, sample_four_views


def _load_valid_components() -> dict[str, Any]:
    sys.path.insert(0, str(VALID_ROOT))
    from datasets.dataset_fs import ReadDatasets, custom_collate_fn
    from datasets.sampling import DWT_3D
    from models.network import HessianConstraintLoss3D, Network_CNR

    return {
        "ReadDatasets": ReadDatasets,
        "custom_collate_fn": custom_collate_fn,
        "DWT_3D": DWT_3D,
        "HessianConstraintLoss3D": HessianConstraintLoss3D,
        "Network_CNR": Network_CNR,
    }


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def _background_variance(pred: torch.Tensor, target: torch.Tensor, percentile: float) -> torch.Tensor:
    """背景体素上的输出方差惩罚（自监督，无需 GT）。

    背景由 target（邻居目标视图）逐样本低于给定分位数的体素定义；
    在该掩码内计算 pred 的方差（相对其背景均值的均方偏差），
    以直接压低背景残留噪声，而不惩罚背景绝对电平。

    Args:
        pred: 网络输出 [N,1,Z,Y,X]。
        target: 对应邻居目标视图，同形状，用于界定背景掩码。
        percentile: 背景分位阈值(0-100)，如 25 表示取 target 最低 25% 体素为背景。

    Returns:
        标量 tensor：各样本背景方差的均值；无有效背景体素时返回 0。
    """
    if percentile <= 0:
        return pred.new_zeros(())
    losses = []
    for i in range(pred.shape[0]):
        t = target[i].reshape(-1)
        p = pred[i].reshape(-1)
        thr = torch.quantile(t, percentile / 100.0)
        mask = t <= thr
        if int(mask.sum()) < 2:
            continue
        bg = p[mask]
        losses.append(((bg - bg.mean()) ** 2).mean())
    if not losses:
        return pred.new_zeros(())
    return torch.stack(losses).mean()


def _write_loss_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "batch", "total_loss", "loss2neighbor", "loss_idt", "loss_reg", "loss_bg", "loss_grad"])
        writer.writeheader()
        writer.writerows(rows)


def _write_eval_curve(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "batch", "fg_psnr_db", "all_psnr_db"])
        writer.writeheader()
        writer.writerows(rows)


def _latest_sampler_metadata(path: Path, metadata_rows: list[dict[str, Any]]) -> None:
    write_json(path, {"event": "aces_stage2_sampler_metadata", "calls": metadata_rows})


def _ensure_stage2_mode_dirs(mode_dir: Path) -> None:
    for rel in ("logs", "metrics", "checkpoints"):
        (mode_dir / rel).mkdir(parents=True, exist_ok=True)


def run_mode(cfg: dict[str, Any], mode: str) -> dict[str, Any]:
    if mode not in VALID_MODES:
        raise ValueError(f"Unknown mode {mode!r}; expected {VALID_MODES}")
    seed = int(cfg.get("seed", 3407))
    _set_seed(seed)
    run_root = Path(cfg["run_dir"]).resolve()
    mode_dir = run_root / mode
    _ensure_stage2_mode_dirs(mode_dir)
    write_json(mode_dir / "config.json", cfg)

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
    geometry = None
    if mode in {"empirical_block", "ellipsoid"}:
        sampling_cfg = cfg["sampling"]
        geometry = geometry_from_stage1_run(
            sampling_cfg["stage1_run_dir"],
            threshold=float(sampling_cfg["q_threshold"]),
            region=str(sampling_cfg.get("region", "foreground")),
            noise_source=str(sampling_cfg.get("noise_source", "repeat")),
            metric=str(sampling_cfg.get("metric", "Q")),
        )

    loss_rows: list[dict[str, Any]] = []
    metadata_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    total_epochs = int(train_cfg["epochs"])
    # eval_every_epochs=0 关闭中途评估；>0 时每该轮数评估一次并存 checkpoint，
    # 记录验证曲线以判断收敛并挑选最优 epoch（数据仅 18 patch，需防过拟合）。
    eval_every = int(train_cfg.get("eval_every_epochs", 0))
    best_psnr = float("-inf")
    best_epoch = -1
    # 背景正则超参：默认 0 → 完全向后兼容（老 config 无此字段时等价原损失）。
    bg_reg_weight = float(train_cfg.get("bg_reg_weight", 0.0))
    bg_percentile = float(train_cfg.get("bg_percentile", 25.0))

    def _mid_eval(epoch_idx: int) -> None:
        nonlocal best_psnr, best_epoch
        model.eval()
        summary = evaluate_model_on_patches(
            model=model,
            dataset=dataset,
            device=device,
            output_dir=mode_dir,
            mean9_path=cfg["evaluation"].get("mean9_proxy", str(MEAN9_PROXY)),
            max_patches=int(cfg["evaluation"].get("patch_count", 2)),
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
        # 即时落盘验证曲线，便于长训练途中查看收敛，而非等训练结束后一次性写。
        _write_eval_curve(mode_dir / "logs" / "eval_curve.csv", eval_rows)
        model.train()

    model.train()
    for epoch in range(total_epochs):
        print(f"[{mode}] epoch {epoch + 1}/{total_epochs} start", flush=True)
        for batch_idx, data in enumerate(loader):
            input_tensor = data[0].to(device).squeeze(1)
            views, metadata = sample_four_views(
                input_tensor,
                mode,
                geometry=geometry,
                seed=seed * 100_000 + epoch * 1_000 + batch_idx,
            )
            metadata_rows.append(metadata_to_dict(metadata))
            noisy_sub1, noisy_sub2, noisy_sub3, noisy_sub4 = [view.unsqueeze(1) for view in views]
            noisy_output_1 = model(noisy_sub1)
            noisy_output_2 = model(noisy_sub2)
            lll1, llh1, lhl1, _, hll1, _, _, _ = dwt3d(noisy_output_1)
            lll2, llh2, lhl2, _, hll2, _, _, _ = dwt3d(noisy_output_2)
            loss2neighbor = 0.5 * l2(noisy_output_1, noisy_sub3) + 0.5 * l2(noisy_output_2, noisy_sub4)
            loss_idt = l2(noisy_output_1, noisy_output_2)
            loss_reg = hessian_loss(torch.cat([lll1, llh1, lhl1, hll1, lll2, llh2, lhl2, hll2], dim=1))
            # 背景正则（默认关闭：bg_reg_weight=0 时 total_loss 完全等价原损失）。
            # 在“背景体素”上惩罚输出方差以压低背景残留噪声 std_bg_pred，
            # 不惩罚背景绝对电平以避免误伤弱信号。背景由目标视图低于 bg_percentile
            # 分位数的体素自监督定义（无需 GT），对两路输出分别计算后取均值。
            loss_bg = 0.5 * (
                _background_variance(noisy_output_1, noisy_sub3, bg_percentile)
                + _background_variance(noisy_output_2, noisy_sub4, bg_percentile)
            )
            total_loss = (
                loss2neighbor
                + loss_idt
                + float(train_cfg["weight_reg"]) * loss_reg
                + bg_reg_weight * loss_bg
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
            })
            if batch_idx == 0 or (batch_idx + 1) % int(train_cfg.get("log_every_batches", 10)) == 0:
                print(
                    f"[{mode}] epoch {epoch + 1}/{total_epochs} batch {batch_idx + 1} "
                    f"loss={float(total_loss.detach().cpu()):.6f}",
                    flush=True,
                )
            if int(train_cfg.get("max_batches_per_epoch", -1)) > 0 and batch_idx + 1 >= int(train_cfg["max_batches_per_epoch"]):
                break
        # 周期性评估：记录前景 PSNR 曲线、存 best/周期 checkpoint（长训练必需）
        if eval_every > 0 and ((epoch + 1) % eval_every == 0 or (epoch + 1) == total_epochs):
            print(f"[{mode}] epoch {epoch + 1}/{total_epochs} eval start", flush=True)
            _mid_eval(epoch + 1)
            print(f"[{mode}] epoch {epoch + 1}/{total_epochs} eval done", flush=True)

    if eval_rows:
        _write_eval_curve(mode_dir / "logs" / "eval_curve.csv", eval_rows)

    checkpoint_path = mode_dir / "checkpoints" / f"{mode}_final.pth"
    torch.save({"state_dict": model.state_dict(), "mode": mode, "config": cfg, "loss_rows": loss_rows}, checkpoint_path)
    _write_loss_csv(mode_dir / "logs" / "train_losses.csv", loss_rows)
    _latest_sampler_metadata(mode_dir / "logs" / "sampler_metadata.json", metadata_rows)
    # 主评估：全分辨率（所有档一致喂全分辨率 patch），保留为当前版本对比基线。
    eval_summary = evaluate_model_on_patches(
        model=model,
        dataset=dataset,
        device=device,
        output_dir=mode_dir,
        mean9_path=cfg["evaluation"].get("mean9_proxy", str(MEAN9_PROXY)),
        max_patches=int(cfg["evaluation"].get("patch_count", 2)),
        foreground_percentile=float(cfg["evaluation"].get("foreground_percentile", 75.0)),
        foreground_mask_path=cfg["evaluation"].get("foreground_mask"),
        metrics_filename="eval_metrics.json",
    )
    # 额外的"对齐评估"：下采样类采样档（valid/sn2n_xy/iso3d）训练时看到的是低分辨率视图，
    # 全分辨率评估会引入 train/test 尺度错配，对偏移类档（empirical_block/ellipsoid）不公平地不利。
    # 这里对下采样档按其采样步长做同步下采样后再评估，消除该混淆因素；两套结果都保留以便对照。
    # stride 从采样器实际输入/输出 shape 比推导（偏移档比值为 1 → stride=(1,1,1)，对齐结果等同全分辨率）。
    probe_views, _probe_meta = sample_four_views(
        torch.zeros(1, int(train_cfg["z_patch"]), int(train_cfg["w_patch"]), int(train_cfg["h_patch"]), device=device),
        mode,
        geometry=geometry,
        seed=seed,
    )
    in_shape = (int(train_cfg["z_patch"]), int(train_cfg["w_patch"]), int(train_cfg["h_patch"]))
    out_shape = tuple(int(s) for s in probe_views[0].shape[1:])
    stride = tuple(max(1, round(i / o)) for i, o in zip(in_shape, out_shape))
    aligned_summary = None
    if stride != (1, 1, 1):
        aligned = evaluate_model_on_patches(
            model=model,
            dataset=dataset,
            device=device,
            output_dir=mode_dir,
            mean9_path=cfg["evaluation"].get("mean9_proxy", str(MEAN9_PROXY)),
            max_patches=int(cfg["evaluation"].get("patch_count", 2)),
            foreground_percentile=float(cfg["evaluation"].get("foreground_percentile", 75.0)),
            foreground_mask_path=cfg["evaluation"].get("foreground_mask"),
            downsample_stride=stride,
            metrics_filename="eval_metrics_aligned.json",
        )
        aligned_summary = {"downsample_stride_zyx": list(stride), "regions": aligned["regions"]}

    summary = {
        "event": "aces_stage2_mode_complete",
        "mode": mode,
        "mode_dir": str(mode_dir),
        "checkpoint": str(checkpoint_path),
        "train_batches": len(loss_rows),
        "elapsed_seconds": time.perf_counter() - started,
        "final_loss": loss_rows[-1]["total_loss"] if loss_rows else None,
        "eval_regions": eval_summary["regions"],
        "eval_regions_aligned": aligned_summary,
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
    (run_root / "logs").mkdir(parents=True, exist_ok=True)
    write_json(run_root / "config.json", cfg)
    summaries = [run_mode(cfg, mode) for mode in selected_modes]
    result = {"event": "aces_stage2_sampling_only_complete", "run_dir": str(run_root), "modes": summaries}
    write_json(run_root / "logs" / "summary.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train ACES Stage 2 sampling-only modes.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--modes", default="all", help="Comma-separated modes or 'all'.")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    modes = list(cfg["sampling"]["modes"]) if args.modes == "all" else [item.strip() for item in args.modes.split(",") if item.strip()]
    result = run_stage2(args.config, modes=modes)
    print(json.dumps({"run_dir": result["run_dir"], "modes": [item["mode"] for item in result["modes"]]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
