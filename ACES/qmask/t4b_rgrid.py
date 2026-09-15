"""T4b 切片 2（Plan v2 §4 item 8/10b）：r_a（时间配对距离）网格训练变体 →
GT 实测最优 vs Q 预测的 Spearman 闭环（H3 完整闭环）。

设计（使用条款约束：AI4Life CIDC25 Validation 集只做评估/模型选择，
权重更新只能用 Training 集 A1/B1/C2/D2）：

- 训练变体 = 时间配对距离 d ∈ {1,2,3,4,6,8}：
    input  = 帧 t   .. t+63   的 y/x 对角相位双视图 v1=(0,0) v2=(1,1)
    target = 帧 t+d .. t+d+63 的同相位双视图       v3=(0,0) v4=(1,1)
  即 input/target 内容差 = 恰好 d 帧时间漂移（噪声独立），y/x 相位分裂仅用于
  产生 loss_idt 的第二视图。损失与 qmask/train.py 完全一致
  （loss2neighbor + loss_idt + Hessian reg），d 网格内部唯一变量是 d。
- 评估：每个 d 模型在 Validation F1/F2/F3 上做 patch 评估（与矩阵同规格：
  18 patch、前景 75 分位），target = 真值 F0（model selection 用途，符合条款）。
  实测最优 d* = argmax_d fg PSNR。
- 闭环（--analyze）：Q_t(d)（第一切片 stage01_cidc25_gt_pairability 曲线，
  region=foreground, τ_s=0.85）与实测 PSNR(d) 的 Spearman；Q 预测
  d_max = max{d : S_measurable(d) >= τ_s} 与实测 argmax 的一致性。

用法：
    python -m qmask.t4b_rgrid --config <cfg.yaml> --distance 3
    python -m qmask.t4b_rgrid --config <cfg.yaml> --analyze
    python -m qmask.t4b_rgrid --config <cfg.yaml> --distance 1 --smoke
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
import torch
from torch.utils.data import DataLoader, Dataset

from aces.evaluate_stage2 import evaluate_model_on_patches
from aces.io_utils import load_config, write_json
from aces.train_stage2 import (
    _background_variance,
    _load_valid_components,
    _set_seed,
    _write_loss_csv,
)

TRAIN_STACKS = ("Training/A1.tif", "Training/B1.tif", "Training/C2.tif", "Training/D2.tif")
EVAL_STACKS = {"F1": "F1.tif", "F2": "F2.tif", "F3": "F3.tif"}
GT_STACK = "F0.tif"
STATS_FILENAME = "train_stack_stats.json"


# --------------------------------------------------------------------------- #
# 数据集
# --------------------------------------------------------------------------- #
def _open_stack(path: str | Path) -> np.ndarray:
    """memmap 优先（多进程共享 page cache），失败则整读。"""
    try:
        return tifffile.memmap(path)
    except Exception:
        return tifffile.imread(path)


def _stack_stats(stack: np.ndarray) -> tuple[float, float]:
    total = 0.0
    total_sq = 0.0
    n = 0
    for z0 in range(0, stack.shape[0], 100):
        block = np.asarray(stack[z0 : z0 + 100], dtype=np.float64)
        total += block.sum()
        total_sq += (block**2).sum()
        n += block.size
    mean = total / n
    std = math.sqrt(max(total_sq / n - mean**2, 0.0))
    return float(mean), float(std)


def compute_train_stack_stats(data_root: str | Path, cache_path: str | Path) -> dict[str, list[float]]:
    """4 个 Training 栈的 per-stack mean/std（缓存为 json，供 6 个并行 run 复用）。"""
    cache_path = Path(cache_path)
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    stats: dict[str, list[float]] = {}
    for name in TRAIN_STACKS:
        stack = _open_stack(Path(data_root) / name)
        mean, std = _stack_stats(stack)
        stats[name] = [mean, std]
        del stack
        print(f"[stats] {name}: mean={mean:.4f} std={std:.4f}", flush=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


class PairDistanceDataset(Dataset):
    """时间配对距离 d 训练集：返回 (patch_t, patch_t+d)，均按所属栈 mean/std 归一化。"""

    def __init__(
        self,
        data_root: str | Path,
        distance: int,
        stats: dict[str, list[float]],
        z_patch: int = 64,
        w_patch: int = 128,
        h_patch: int = 128,
        overlap: float = 0.1,
    ):
        self.distance = int(distance)
        self.z_patch, self.w_patch, self.h_patch = z_patch, w_patch, h_patch
        step = lambda size, patch: max(1, int(round(patch * (1.0 - overlap))))  # noqa: E731
        self.stacks: list[np.ndarray] = []
        self.means: list[float] = []
        self.stds: list[float] = []
        positions: list[tuple[int, int, int, int]] = []
        for img_idx, name in enumerate(TRAIN_STACKS):
            stack = _open_stack(Path(data_root) / name)
            t_len, y_len, x_len = stack.shape
            mean, std = stats[name]
            self.stacks.append(stack)
            self.means.append(mean)
            self.stds.append(std)
            z_max = t_len - z_patch - self.distance
            for z0 in range(0, z_max + 1, step(t_len, z_patch)):
                for y0 in range(0, y_len - h_patch + 1, step(y_len, w_patch)):
                    for x0 in range(0, x_len - w_patch + 1, step(x_len, h_patch)):
                        positions.append((img_idx, z0, y0, x0))
        self.positions = positions

    def __len__(self) -> int:
        return len(self.positions)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_idx, z0, y0, x0 = self.positions[index]
        stack = self.stacks[img_idx]
        inp = np.asarray(stack[z0 : z0 + self.z_patch, y0 : y0 + self.w_patch, x0 : x0 + self.h_patch], dtype=np.float32)
        tgt = np.asarray(
            stack[z0 + self.distance : z0 + self.distance + self.z_patch, y0 : y0 + self.w_patch, x0 : x0 + self.h_patch],
            dtype=np.float32,
        )
        mean, std = self.means[img_idx], self.stds[img_idx]
        inp = (inp - mean) / std
        tgt = (tgt - mean) / std
        return torch.from_numpy(inp), torch.from_numpy(tgt)


class SingleStackEvalDataset(Dataset):
    """评估数据集：与 evaluate_model_on_patches 兼容（5 元组 + indices + image_means/stds）。"""

    def __init__(
        self,
        stack_path: str | Path,
        mean: float,
        std: float,
        z_patch: int = 64,
        w_patch: int = 128,
        h_patch: int = 128,
        overlap: float = 0.1,
    ):
        self.stack = _open_stack(stack_path)
        self.z_patch, self.w_patch, self.h_patch = z_patch, w_patch, h_patch
        step = lambda size, patch: max(1, int(round(patch * (1.0 - overlap))))  # noqa: E731
        t_len, y_len, x_len = self.stack.shape
        self.indices: list[tuple[int, int, int, int]] = []
        for z0 in range(0, t_len - z_patch + 1, step(t_len, z_patch)):
            for y0 in range(0, y_len - w_patch + 1, step(y_len, w_patch)):
                for x0 in range(0, x_len - h_patch + 1, step(x_len, h_patch)):
                    self.indices.append((0, z0, y0, x0))
        self.image_means = [float(mean)]
        self.image_stds = [float(std)]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int, int, int]:
        _, z0, y0, x0 = self.indices[index]
        patch = np.asarray(
            self.stack[z0 : z0 + self.z_patch, y0 : y0 + self.w_patch, x0 : x0 + self.h_patch],
            dtype=np.float32,
        )
        patch = (patch - self.image_means[0]) / self.image_stds[0]
        return torch.from_numpy(patch), 0, z0, y0, x0


# --------------------------------------------------------------------------- #
# 训练
# --------------------------------------------------------------------------- #
def _phase_views(patch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """y/x 对角相位双视图：(0,0) 与 (1,1)，形状 (B,Z,Y//2,X//2)。"""
    return patch[:, :, 0::2, 0::2], patch[:, :, 1::2, 1::2]


def run_distance(cfg: dict[str, Any], distance: int) -> dict[str, Any]:
    seed = int(cfg.get("seed", 3407))
    smoke = bool(cfg.get("smoke", False))
    _set_seed(seed)
    train_cfg = cfg["training"]
    eval_cfg = cfg["evaluation"]
    d_dir = Path(cfg["run_dir"]).resolve() / f"d{distance}"
    for rel in ("logs", "metrics", "checkpoints"):
        (d_dir / rel).mkdir(parents=True, exist_ok=True)
    write_json(d_dir / "config.json", {**cfg, "distance": distance})

    _now = time.perf_counter()
    valid = _load_valid_components()
    stats = compute_train_stack_stats(cfg["data_root"], Path(cfg["run_dir"]) / STATS_FILENAME)
    dataset = PairDistanceDataset(
        data_root=cfg["data_root"],
        distance=distance,
        stats=stats,
        z_patch=int(train_cfg["z_patch"]),
        w_patch=int(train_cfg["w_patch"]),
        h_patch=int(train_cfg["h_patch"]),
        overlap=float(train_cfg.get("overlap", 0.1)),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(train_cfg.get("num_workers", 0)),
        pin_memory=True,
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

    # 评估环境：Validation 栈 + 真值 F0（仅评估/模型选择，不更新权重）。
    eval_sets: dict[str, SingleStackEvalDataset] = {}
    for level in ("F1", "F2", "F3"):
        mean_lvl, std_lvl = eval_stack_stats(cfg, level)
        eval_sets[level] = SingleStackEvalDataset(
            stack_path=Path(cfg["data_root"]) / "Validation" / EVAL_STACKS[level],
            mean=mean_lvl,
            std=std_lvl,
            z_patch=int(train_cfg["z_patch"]),
            w_patch=int(train_cfg["w_patch"]),
            h_patch=int(train_cfg["h_patch"]),
            overlap=float(train_cfg.get("overlap", 0.1)),
        )
    gt_path = Path(cfg["data_root"]) / "Validation" / GT_STACK

    def eval_level(model_: torch.nn.Module, level: str, max_patches: int, filename: str) -> dict[str, Any]:
        return evaluate_model_on_patches(
            model=model_,
            dataset=eval_sets[level],
            device=device,
            output_dir=d_dir,
            mean9_path=gt_path,
            max_patches=max_patches,
            foreground_percentile=float(eval_cfg.get("foreground_percentile", 75.0)),
            metrics_filename=filename,
        )

    total_epochs = 1 if smoke else int(train_cfg["epochs"])
    max_batches = 3 if smoke else int(train_cfg["max_batches_per_epoch"])
    eval_every = 1 if smoke else int(train_cfg.get("eval_every_epochs", 5))
    mid_level = str(eval_cfg.get("selection_level", "F2"))
    mid_patches = 2 if smoke else int(eval_cfg.get("mid_patch_count", 12))
    best_psnr = float("-inf")
    best_epoch = -1
    loss_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []

    model.train()
    for epoch in range(total_epochs):
        print(f"[d={distance}] epoch {epoch + 1}/{total_epochs} start", flush=True)
        for batch_idx, (inp, tgt) in enumerate(loader):
            inp = inp.to(device, non_blocking=True)
            tgt = tgt.to(device, non_blocking=True)
            v1, v2 = _phase_views(inp)
            v3, v4 = _phase_views(tgt)
            out1 = model(v1.unsqueeze(1))
            out2 = model(v2.unsqueeze(1))
            lll1, llh1, lhl1, _, hll1, _, _, _ = dwt3d(out1)
            lll2, llh2, lhl2, _, hll2, _, _, _ = dwt3d(out2)
            loss2neighbor = 0.5 * l2(out1, v3.unsqueeze(1)) + 0.5 * l2(out2, v4.unsqueeze(1))
            loss_idt = l2(out1, out2)
            loss_reg = hessian_loss(torch.cat([lll1, llh1, lhl1, hll1, lll2, llh2, lhl2, hll2], dim=1))
            loss_bg = 0.5 * (
                _background_variance(out1, v3.unsqueeze(1), float(train_cfg.get("bg_percentile", 25.0)))
                + _background_variance(out2, v4.unsqueeze(1), float(train_cfg.get("bg_percentile", 25.0)))
            )
            total_loss = (
                loss2neighbor
                + loss_idt
                + float(train_cfg["weight_reg"]) * loss_reg
                + float(train_cfg.get("bg_reg_weight", 0.0)) * loss_bg
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
            })
            if batch_idx + 1 >= max_batches:
                break

        if eval_every > 0 and ((epoch + 1) % eval_every == 0 or (epoch + 1) == total_epochs):
            summary = eval_level(model, mid_level, mid_patches, f"eval_mid_e{epoch + 1:03d}.json")
            fg = float(summary["regions"]["foreground"]["psnr_db"])
            eval_rows.append({"epoch": epoch + 1, "level": mid_level, "fg_psnr_db": fg})
            if fg > best_psnr:
                best_psnr = fg
                best_epoch = epoch + 1
                torch.save(
                    {"state_dict": model.state_dict(), "distance": distance, "epoch": epoch + 1, "fg_psnr_db": fg},
                    d_dir / "checkpoints" / f"tpair_d{distance}_best.pth",
                )
            print(f"[d={distance}] epoch {epoch + 1} mid-eval {mid_level} fg={fg:.3f} dB", flush=True)
            model.train()

    torch.save(
        {"state_dict": model.state_dict(), "distance": distance, "config": cfg, "loss_rows": loss_rows},
        d_dir / "checkpoints" / f"tpair_d{distance}_final.pth",
    )
    _write_loss_csv(d_dir / "logs" / "train_losses.csv", loss_rows)

    # 最终评估：final 与 best（F2 选择）两套权重各评 F1/F2/F3（18 patch）。
    final_summary: dict[str, Any] = {}
    best_summary: dict[str, Any] = {}
    full_patches = 2 if smoke else int(eval_cfg.get("patch_count", 18))
    for level in ("F1", "F2", "F3"):
        final_summary[level] = eval_level(model, level, full_patches, f"eval_final_{level}.json")
    if best_epoch > 0:
        ckpt = torch.load(d_dir / "checkpoints" / f"tpair_d{distance}_best.pth", map_location=device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        for level in ("F1", "F2", "F3"):
            best_summary[level] = eval_level(model, level, full_patches, f"eval_best_{level}.json")

    result = {
        "event": "t4b_rgrid_distance_complete",
        "distance": distance,
        "d_dir": str(d_dir),
        "seed": seed,
        "elapsed_seconds": time.perf_counter() - _now,
        "best_epoch": best_epoch,
        "best_mid_fg_psnr_db": best_psnr if best_epoch > 0 else None,
        "final_eval": {lvl: s["regions"] for lvl, s in final_summary.items()},
        "best_eval": {lvl: s["regions"] for lvl, s in best_summary.items()},
    }
    write_json(d_dir / "metrics" / "distance_summary.json", result)
    print(json.dumps({k: result[k] for k in ("distance", "elapsed_seconds", "best_epoch")}, indent=2), flush=True)
    return result


def eval_stack_stats(cfg: dict[str, Any], level: str) -> list[float]:
    """Validation 栈的 per-stack mean/std（缓存，仅用于输入归一化）。"""
    cache = Path(cfg["run_dir"]) / "eval_stack_stats.json"
    payload = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else {}
    if level not in payload:
        stack = _open_stack(Path(cfg["data_root"]) / "Validation" / EVAL_STACKS[level])
        payload[level] = list(_stack_stats(stack))
        del stack
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload[level]


# --------------------------------------------------------------------------- #
# H3 闭环分析
# --------------------------------------------------------------------------- #
def _spearman(x: list[float], y: list[float]) -> float | None:
    try:
        from scipy.stats import spearmanr

        rho, _ = spearmanr(x, y)
        return None if rho != rho else float(rho)
    except Exception:
        return None


def analyze(cfg: dict[str, Any]) -> dict[str, Any]:
    root = Path(cfg["run_dir"]).resolve()
    tau_s = float(cfg["analysis"].get("tau_s", 0.85))
    csv_path = Path(cfg["analysis"]["stage1_curves_csv"])
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    def curves(noise_source: str) -> dict[str, dict[int, dict[str, float]]]:
        out: dict[str, dict[int, dict[str, float]]] = {}
        for row in rows:
            if row["axis"] != "z" or row["region"] != "foreground" or row["noise_source"] != noise_source:
                continue
            level = row["noise_level"]
            d = int(float(row["distance_px"]))
            out.setdefault(level, {})[d] = {
                "Q": float(row["Q"]),
                "S_measurable": float(row["S_measurable"]),
                "S": float(row["S"]),
            }
        return out

    q_curves = {"proxy": curves("proxy"), "gt": curves("gt")}
    distances = sorted(cfg["analysis"].get("distances", [1, 2, 3, 4, 6, 8]))
    weight_set = str(cfg["analysis"].get("weight_set", "best"))

    table: list[dict[str, Any]] = []
    closure: dict[str, Any] = {"event": "t4b_h3_closure", "tau_s": tau_s, "distances": distances, "weight_set": weight_set, "levels": {}}
    for level in ("F1", "F2", "F3"):
        measured: dict[int, float] = {}
        for d in distances:
            path = root / f"d{d}" / "metrics" / f"eval_{weight_set}_{level}.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            measured[d] = float(payload["regions"]["foreground"]["psnr_db"])
        d_star = max(measured, key=lambda k: measured[k])
        level_out: dict[str, Any] = {"measured_psnr_db": measured, "measured_argmax_d": d_star}
        for source in ("proxy", "gt"):
            curve = q_curves[source].get(level, {})
            q_vals = [curve[d]["Q"] for d in distances if d in curve]
            d_list = [d for d in distances if d in curve]
            level_out[f"spearman_Q_{source}"] = _spearman(q_vals, [measured[d] for d in d_list]) if len(d_list) >= 3 else None
            level_out[f"predicted_d_max_{source}"] = max(
                [d for d, v in curve.items() if v["S_measurable"] >= tau_s], default=0
            )
        s_meas = q_curves["proxy"].get(level, {})
        level_out["spearman_S_measurable"] = _spearman(
            [s_meas[d]["S_measurable"] for d in distances if d in s_meas],
            [measured[d] for d in distances if d in s_meas],
        )
        closure["levels"][level] = level_out
        table.append((level, measured, d_star, level_out))

    print(f"\n=== H3 闭环（weight_set={weight_set}, τ_s={tau_s}）===")
    for level, measured, d_star, level_out in table:
        print(f"\n[{level}] fg PSNR(d): " + "  ".join(f"d{d}={v:.2f}" for d, v in sorted(measured.items())))
        print(f"  实测 argmax d*={d_star}   Q 预测 d_max: proxy={level_out['predicted_d_max_proxy']} gt={level_out['predicted_d_max_gt']}")
        print(f"  Spearman: Q_proxy={level_out['spearman_Q_proxy']} Q_gt={level_out['spearman_Q_gt']} S_measurable={level_out['spearman_S_measurable']}")
    out_path = root / "metrics" / "h3_closure.json"
    write_json(out_path, closure)
    print(f"\nwrote {out_path}")
    return closure


# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="T4b slice 2: temporal pairing-distance grid (r_a grid).")
    parser.add_argument("--config", required=True)
    parser.add_argument("--distance", type=int, default=None)
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    if args.smoke:
        cfg["smoke"] = True
    if args.analyze:
        analyze(cfg)
        return 0
    if args.distance is None:
        raise SystemExit("--distance is required unless --analyze")
    run_distance(cfg, args.distance)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
