"""A2（Plan v2 §7 P-A）：空间轴闭环——y/x 对角平移 s 网格训练变体（sgrid）。

设计（2026-09-14 定稿，见计划 §7 P-A 设计块；使用条款同 rgrid：
权重更新只用 Training 集 A1/B1/C2/D2，Validation F0–F3 仅评估/模型选择）：

- 配对定义：input = 帧 t..t+63 在 (y0,x0) 的 64×128×128 patch；
  target = 同一帧段在 (y0+s, x0+s) 的同尺寸窗口（对角平移）。
  s≥1 时 input/target 噪声独立（i.i.d. per-pixel），s=0 为恒等捷径对照。
- 相位视图配对：s 偶数 out1↔v3 / out2↔v4（与时间轴同构）；s 奇数交换
  （out1↔v4 / out2↔v3）——奇数平移把偶网格结构映射到奇网格，交换后才保持
  loss2neighbor 配对相位匹配视图。s 网格内唯一变量是 s。
- 协议与 rgrid 完全一致（20ep×100batch、同模型/优化器/损失框架、
  F1–F3 vs F0 评估 18 patch）。
- 分析（--analyze；GT 只做测量不更新权重）：实测 PSNR(s) 剂量-响应、
  F0 结构位移 nMSE0(s)、噪声数据可观测性 r_noisy(s)（i.i.d. 噪声只污染
  s=0 一点）、Spearman 与"平台起点 s ≈ nMSE0(s) 跨阈值处"一致性。

用法：
    python -m qmask.t4b_sgrid --config <cfg.yaml> --shift 2
    python -m qmask.t4b_sgrid --config <cfg.yaml> --analyze
    python -m qmask.t4b_sgrid --config <cfg.yaml> --shift 1 --smoke
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from aces.evaluate_stage2 import evaluate_model_on_patches
from aces.io_utils import load_config, write_json
from aces.train_stage2 import _background_variance, _load_valid_components, _set_seed, _write_loss_csv
from qmask.t4b_rgrid import (
    EVAL_STACKS,
    GT_STACK,
    TRAIN_STACKS,
    SingleStackEvalDataset,
    _open_stack,
    _phase_views,
    compute_train_stack_stats,
    eval_stack_stats,
)

STATS_FILENAME = "train_stack_stats.json"


# --------------------------------------------------------------------------- #
# 数据集
# --------------------------------------------------------------------------- #
class PairShiftDataset(Dataset):
    """空间对角平移 s 训练集：返回 (patch@(y0,x0), patch@(y0+s,x0+s))，同帧段、同栈 mean/std 归一化。"""

    def __init__(
        self,
        data_root: str | Path,
        shift: int,
        stats: dict[str, list[float]],
        z_patch: int = 64,
        w_patch: int = 128,
        h_patch: int = 128,
        overlap: float = 0.1,
    ):
        self.shift = int(shift)
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
            # 同帧配对：z 无时间余量；y/x 各留 s 边距（与 rgrid 的 y/x 窗口约定一致）
            z_max = t_len - z_patch
            for z0 in range(0, z_max + 1, step(t_len, z_patch)):
                for y0 in range(0, y_len - h_patch - self.shift + 1, step(y_len, w_patch)):
                    for x0 in range(0, x_len - w_patch - self.shift + 1, step(x_len, h_patch)):
                        positions.append((img_idx, z0, y0, x0))
        self.positions = positions

    def __len__(self) -> int:
        return len(self.positions)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_idx, z0, y0, x0 = self.positions[index]
        stack = self.stacks[img_idx]
        s = self.shift
        inp = np.asarray(stack[z0 : z0 + self.z_patch, y0 : y0 + self.w_patch, x0 : x0 + self.h_patch], dtype=np.float32)
        tgt = np.asarray(
            stack[z0 : z0 + self.z_patch, y0 + s : y0 + s + self.w_patch, x0 + s : x0 + s + self.h_patch],
            dtype=np.float32,
        )
        mean, std = self.means[img_idx], self.stds[img_idx]
        inp = (inp - mean) / std
        tgt = (tgt - mean) / std
        return torch.from_numpy(inp), torch.from_numpy(tgt)


# --------------------------------------------------------------------------- #
# 训练
# --------------------------------------------------------------------------- #
def run_shift(cfg: dict[str, Any], shift: int) -> dict[str, Any]:
    seed = int(cfg.get("seed", 3407))
    smoke = bool(cfg.get("smoke", False))
    _set_seed(seed)
    train_cfg = cfg["training"]
    eval_cfg = cfg["evaluation"]
    s_dir = Path(cfg["run_dir"]).resolve() / f"s{shift}"
    for rel in ("logs", "metrics", "checkpoints"):
        (s_dir / rel).mkdir(parents=True, exist_ok=True)
    write_json(s_dir / "config.json", {**cfg, "shift": shift, "axis": "spatial"})

    _now = time.perf_counter()
    valid = _load_valid_components()
    stats = compute_train_stack_stats(cfg["data_root"], Path(cfg["run_dir"]) / STATS_FILENAME)
    dataset = PairShiftDataset(
        data_root=cfg["data_root"],
        shift=shift,
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
    eval_sets: dict[str, Any] = {}
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
            output_dir=s_dir,
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
        print(f"[s={shift}] epoch {epoch + 1}/{total_epochs} start", flush=True)
        for batch_idx, (inp, tgt) in enumerate(loader):
            inp = inp.to(device, non_blocking=True)
            tgt = tgt.to(device, non_blocking=True)
            v1, v2 = _phase_views(inp)
            v3, v4 = _phase_views(tgt)
            # A2 唯一损失结构变更：奇数平移把偶网格结构映射到奇网格，交换配对保持相位匹配。
            if shift % 2 == 1:
                v3, v4 = v4, v3
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
                    {"state_dict": model.state_dict(), "shift": shift, "epoch": epoch + 1, "fg_psnr_db": fg},
                    s_dir / "checkpoints" / f"sshift_s{shift}_best.pth",
                )
            print(f"[s={shift}] epoch {epoch + 1} mid-eval {mid_level} fg={fg:.3f} dB", flush=True)
            model.train()

    torch.save(
        {"state_dict": model.state_dict(), "shift": shift, "config": cfg, "loss_rows": loss_rows},
        s_dir / "checkpoints" / f"sshift_s{shift}_final.pth",
    )
    _write_loss_csv(s_dir / "logs" / "train_losses.csv", loss_rows)

    # 最终评估：final 与 best（F2 选择）两套权重各评 F1/F2/F3（18 patch）。
    final_summary: dict[str, Any] = {}
    best_summary: dict[str, Any] = {}
    full_patches = 2 if smoke else int(eval_cfg.get("patch_count", 18))
    for level in ("F1", "F2", "F3"):
        final_summary[level] = eval_level(model, level, full_patches, f"eval_final_{level}.json")
    if best_epoch > 0:
        ckpt = torch.load(s_dir / "checkpoints" / f"sshift_s{shift}_best.pth", map_location=device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        for level in ("F1", "F2", "F3"):
            best_summary[level] = eval_level(model, level, full_patches, f"eval_best_{level}.json")

    result = {
        "event": "t4b_sgrid_shift_complete",
        "shift": shift,
        "s_dir": str(s_dir),
        "seed": seed,
        "elapsed_seconds": time.perf_counter() - _now,
        "best_epoch": best_epoch,
        "best_mid_fg_psnr_db": best_psnr if best_epoch > 0 else None,
        "final_eval": {lvl: s["regions"] for lvl, s in final_summary.items()},
        "best_eval": {lvl: s["regions"] for lvl, s in best_summary.items()},
    }
    write_json(s_dir / "metrics" / "shift_summary.json", result)
    print(json.dumps({k: result[k] for k in ("shift", "elapsed_seconds", "best_epoch")}, indent=2), flush=True)
    return result


# --------------------------------------------------------------------------- #
# 空间轴闭环分析
# --------------------------------------------------------------------------- #
SPATIAL_TILE = 128
SPATIAL_Z_FRAMES = 300  # 每栈最多采样的帧数（ stride 均匀取样，统计量足够）


def _spatial_curves(
    stack: np.ndarray,
    shifts: list[int],
    n_tiles: int,
    seed: int,
) -> dict[str, list[float]]:
    """对一个栈采样 tile×帧，计算 r(s)=corr(F(y),F(y+s)) 与 nMSE(s)=MSE/Var（(s,s) 对角平移，池化矩）。"""
    rng = np.random.default_rng(seed)
    t_len, y_len, x_len = stack.shape
    s_max = max(shifts)
    tiles = [
        (int(rng.integers(0, y_len - SPATIAL_TILE - s_max)), int(rng.integers(0, x_len - SPATIAL_TILE - s_max)))
        for _ in range(n_tiles)
    ]
    z_idx = np.unique(np.linspace(0, t_len - 1, min(SPATIAL_Z_FRAMES, t_len)).astype(int))
    # 池化矩：n, Σa, Σb, Σa², Σb², Σab, Σ(a-b)²
    mom = {s: np.zeros(7, dtype=np.float64) for s in shifts}
    for z in z_idx:
        frame = np.asarray(stack[z], dtype=np.float32)
        for y0, x0 in tiles:
            a_full = frame[y0 : y0 + SPATIAL_TILE, x0 : x0 + SPATIAL_TILE].astype(np.float64)
            for s in shifts:
                a = a_full[: SPATIAL_TILE - s, : SPATIAL_TILE - s].ravel()
                b = a_full[s:, s:].ravel()
                m = mom[s]
                m[0] += a.size
                m[1] += a.sum(); m[2] += b.sum()
                m[3] += (a * a).sum(); m[4] += (b * b).sum()
                m[5] += (a * b).sum()
                m[6] += ((a - b) ** 2).sum()
        del frame
    out_r, out_nmse = [], []
    for s in shifts:
        n, s1a, s1b, s2a, s2b, sab, ssd = mom[s]
        if n == 0:
            out_r.append(float("nan")); out_nmse.append(float("nan")); continue
        cov = n * sab - s1a * s1b
        var_a = max(n * s2a - s1a * s1a, 1e-9)
        var_b = max(n * s2b - s1b * s1b, 1e-9)
        r = float(cov / np.sqrt(var_a * var_b))
        mean_a = s1a / n
        var_total = max(s2a / n - mean_a * mean_a, 1e-9)
        nmse = float((ssd / n) / var_total)
        out_r.append(r)
        out_nmse.append(nmse)
    return {"r": out_r, "nmse": out_nmse}


def _spearman(x: list[float], y: list[float]) -> float | None:
    pairs = [(a, b) for a, b in zip(x, y) if np.isfinite(a) and np.isfinite(b)]
    if len(pairs) < 3:
        return None
    try:
        from scipy.stats import spearmanr

        rho, _ = spearmanr([p[0] for p in pairs], [p[1] for p in pairs])
        return float(rho)
    except Exception:
        return None


def analyze_spatial(cfg: dict[str, Any]) -> dict[str, Any]:
    """A2 闭环分析：实测 PSNR(s) vs F0 结构位移 nMSE0(s) vs 噪声可观测量 r_noisy(s)。"""
    root = Path(cfg["run_dir"]).resolve()
    analysis = cfg.get("analysis", {})
    shifts = sorted(int(s) for s in analysis.get("shifts", [0, 1, 2, 4, 8, 16]))
    weight_set = str(analysis.get("weight_set", "best"))
    nmse_threshold = float(analysis.get("nmse_threshold", 0.02))
    n_tiles = int(analysis.get("n_tiles", 8))
    seed = int(cfg.get("seed", 3407))
    data_root = Path(cfg["data_root"])

    def measured_psnr(level: str) -> list[float]:
        vals = []
        for s in shifts:
            path = root / f"s{s}" / "metrics" / f"eval_{weight_set}_{level}.json"
            if not path.exists():
                vals.append(float("nan"))
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            vals.append(float(data["regions"]["foreground"]["psnr_db"]))
        return vals

    # GT 结构位移（精确量）与噪声数据可观测性（A1 缺的对照面）。
    print("computing F0 structural shift curves ...", flush=True)
    gt_stack = _open_stack(data_root / "Validation" / GT_STACK)
    gt_curves = _spatial_curves(gt_stack, shifts, n_tiles, seed)
    del gt_stack
    noisy_curves: dict[str, dict[str, list[float]]] = {}
    for level in ("F1", "F2", "F3"):
        print(f"computing noisy r(s) curves for {level} ...", flush=True)
        stack = _open_stack(data_root / "Validation" / EVAL_STACKS[level])
        noisy_curves[level] = _spatial_curves(stack, shifts, n_tiles, seed + 1)
        del stack

    nmse0 = gt_curves["nmse"]
    threshold_s = next((s for s, v in zip(shifts, nmse0) if np.isfinite(v) and v >= nmse_threshold), None)

    closure: dict[str, Any] = {
        "event": "t4b_sgrid_closure",
        "shifts": shifts,
        "weight_set": weight_set,
        "nmse_threshold": nmse_threshold,
        "gt_structural": gt_curves,
        "noisy_observability": noisy_curves,
        "levels": {},
    }
    for level in ("F1", "F2", "F3"):
        psnr = measured_psnr(level)
        finite = [(s, v) for s, v in zip(shifts, psnr) if np.isfinite(v)]
        s_star = max(finite, key=lambda p: p[1])[0] if finite else None
        # 平台起点：最小 s 使 PSNR(s) ≥ 全程最大值 −1 dB
        plateau_s = next(
            (s for s, v in finite if v >= max(v for _, v in finite) - 1.0), None
        ) if finite else None
        rho = _spearman(noisy_curves[level]["r"], psnr)
        closure["levels"][level] = {
            "measured_psnr_db": dict(zip(map(str, shifts), psnr)),
            "measured_argmax_s": s_star,
            "plateau_start_s": plateau_s,
            "spearman_rnoisy_psnr": rho,
        }
        print(
            f"\n[{level}] fg PSNR(s): " + "  ".join(f"s{s}={v:.2f}" for s, v in zip(shifts, psnr))
            + f"\n  argmax s*={s_star}  plateau_start={plateau_s}  nmse0>={nmse_threshold} @ s={threshold_s}"
            + f"  Spearman(r_noisy, PSNR)={rho}",
            flush=True,
        )

    # 预判定（最终裁决以人工核对数值为准）：平台存在 且 平台起点与 nMSE0 阈值位移重合（差 ≤1 档）。
    verdicts = {}
    for level, out in closure["levels"].items():
        psnr_vals = [v for v in out["measured_psnr_db"].values() if np.isfinite(v)]
        has_gain = (max(psnr_vals) - min(psnr_vals)) >= 1.0 if psnr_vals else False
        out_plateau = out["plateau_start_s"]
        verdicts[level] = {
            "has_dose_response": has_gain,
            "plateau_matches_nmse_threshold": (
                out_plateau is not None and threshold_s is not None and abs(out_plateau - threshold_s) <= 1
            ),
        }
    closure["preliminary_verdict"] = verdicts
    out_path = root / "metrics" / "sgrid_closure.json"
    write_json(out_path, closure)
    print(f"\nwrote {out_path}")
    return closure


# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="A2: spatial diagonal-shift grid (sgrid).")
    parser.add_argument("--config", required=True)
    parser.add_argument("--shift", type=int, default=None)
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    if args.smoke:
        cfg["smoke"] = True
    if args.analyze:
        analyze_spatial(cfg)
        return 0
    if args.shift is None:
        raise SystemExit("--shift is required unless --analyze")
    run_shift(cfg, args.shift)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
