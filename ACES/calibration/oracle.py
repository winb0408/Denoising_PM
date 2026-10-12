"""Oracle 训练评估引擎（修正方案 M2 / §6 Exp C，Gate P2→P3）。

对每个 sampling family Γ 用与 micro-probe **完全相同**的训练协议
（同网络 / 同 lr / 同 patch 池 / 同 seed / 同 ``sample_views`` 视图构造）
训练一个 20ep pilot 去噪器，再在 3P 9-repeat 栈上以 AVG-9 伪真值评估
去噪增益 ``U_oracle(Γ)``，用于验证 U_probe 的预测力（方案 §3：U_oracle
只在拥有参考的开发数据上计算，**不得作为部署期 policy selector**）。

协议（方案 §8 oracle 成本控制：8–12 个精选 family × 20ep pilot）：

- 训练：镜像 ``calibration/micro_probe.py`` 的 ``_probe_family`` 逐行实现
  （seed 每 family 重置、Adam betas 0.9/0.999、grad-norm clip、
  loss2neighbor + weight_idt·loss_idt、视图来自 ``family_sampler.sample_views``
  —— 探针与 oracle 唯一的共同变量是 Γ，这是 probe 能预测 oracle 的前提）。
- 评估（U_oracle）：
    * 数据：3P repeat stack ``[K=9, Z=275, 512, 512]``；
    * 伪真值：K 帧 repeat 的逐区域均值（AVG-9，on-the-fly，float32）；
    * 评估网格：z 起点 ``z0 ∈ {0, stride, 2·stride, ...}``（``eval_stride_z``），
      y/x 为 patch 尺寸步长的非重叠网格（边缘余量丢弃）；
    * 每个 eval repeat 帧 k：raw patch 用 PatchPool 的全局 mean/std 归一化
      （与训练同源），过网络后反归一化，PSNR 在原始强度域计算
      （data_range = 参考patch 的 99.9%−0.1% 分位差，镜像
      ``aces.evaluate_stage2`` 约定）；
    * ``U_oracle(Γ) = mean_patches (PSNR_denoised − PSNR_raw)``（去噪增益，
      采样应影响的量），另记录绝对 PSNR、z 三分带（shallow/mid/deep）
      分解与 patch 数。
- 注意（协议说明，非严格 hold-out）：patch 深度 48 > eval_stride_z 16 时
  评估 patch 的 z 覆盖与训练 patch 重叠，无法做到 z 轴严格互斥；
  本引擎把 ``eval_stride_z`` 解释为**稀疏评估网格**（全部 family 用同一
  网格，横向比较仍公平），训练池与 micro_probe 完全一致（唯一变量 Γ）。

用法：
    PYTHONPATH=ACES python -m calibration.oracle \
        --data <repeat_stack.tif [K,Z,Y,X]> --run-dir <dir> \
        --families all --device cuda \
        [--probe-csv micro_probe.csv] \
        [--config "epochs=20,batches_per_epoch=200"]
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from aces.io_utils import write_json
from profiling.candidates import CandidateFamily

from .family_sampler import derive_block, sample_views
from .micro_probe import (
    NetFactory,
    PatchPool,
    _parse_override_value,
    _set_seed,
    load_volume,
    resolve_families,
)

BAND_NAMES = ("shallow", "mid", "deep")

# --------------------------------------------------------------------------- #
# 配置
# --------------------------------------------------------------------------- #
@dataclass
class OracleConfig:
    """Oracle 训练评估协议（20ep pilot，镜像 t4b_rgrid / micro_probe）。

    注意：VALID ``Network_CNR``（CRN）含 AvgPool3d(8)，要求每个视图的空间
    尺寸 ≥8/轴，即 ``patch_size ≥ 8 × block(Γ)``；默认库最大 block=4，
    默认 patch 48³ → 最小视图 12³，全部 family 安全。
    """

    epochs: int = 20
    batches_per_epoch: int = 200
    patch_size: tuple[int, int, int] = (48, 48, 48)
    batch_size: int = 1
    lr: float = 1e-4
    seed: int = 3407
    device: str = "cuda"
    # VALID Network_CNR 构造参数（与 micro_probe / t4b_rgrid 一致）
    base_features: int = 32
    n_groups: int = 8
    weight_idt: float = 1.0
    clip_gradients: float = 1.0
    # 训练 patch 池（全部 family 共享，与 micro_probe 相同的池）
    max_train_patches: int = 128
    overlap: float = 0.0
    # 评估
    eval_stride_z: int = 16
    eval_repeat_frames: list[int] | None = None  # None = 全部 repeat 帧
    max_eval_patches: int = 0  # 0 = 不限量（完整评估网格）
    # 训练后保存最终 checkpoint（families/{fid}_final.pth，定性渲染用）
    save_checkpoint: bool = False

    def __post_init__(self) -> None:
        size = tuple(int(v) for v in self.patch_size)
        if len(size) == 2:  # 2D：(Y,X) → (1,Y,X)
            size = (1, size[0], size[1])
        if len(size) != 3 or min(size) < 1:
            raise ValueError(f"patch_size 必须是 (Z,Y,X) 或 2D (Y,X), got {self.patch_size}")
        self.patch_size = size  # type: ignore[assignment]
        if int(self.eval_stride_z) < 1:
            raise ValueError(f"eval_stride_z 必须 >= 1, got {self.eval_stride_z}")


# --------------------------------------------------------------------------- #
# 训练（镜像 micro_probe._probe_family，仅去掉探针统计量收集）
# --------------------------------------------------------------------------- #
def _train_family(
    config: OracleConfig,
    family: CandidateFamily,
    pool: PatchPool,
    net_factory: NetFactory,
    device: torch.device,
) -> tuple[torch.nn.Module, list[dict[str, float]]]:
    """跑一个 family 的 pilot 训练，返回 (训练后的 net, 逐 batch loss 记录)。

    损失构造与 ``micro_probe._probe_family`` 逐行一致（此处复制而非 import：
    探针版不返回 net，且约定不改动既有模块）：
        out_i         = net(view_i)
        loss2neighbor = mean_{i≠j} MSE(out_i, view_j)
        loss_idt      = mean_{i<j} MSE(out_i, out_j)
        total         = loss2neighbor + weight_idt · loss_idt
    """
    _set_seed(config.seed)  # 每个 family 重置：初始化/数据顺序一致，只换 Γ
    net = net_factory().to(device)
    net.train()
    optimizer = torch.optim.Adam(net.parameters(), lr=float(config.lr), betas=(0.9, 0.999))
    mse = torch.nn.MSELoss()

    records: list[dict[str, float]] = []
    cursor = 0
    total_batches = int(config.epochs) * int(config.batches_per_epoch)
    for step in range(total_batches):
        epoch = step // max(1, int(config.batches_per_epoch)) + 1
        batch = pool.take(cursor, int(config.batch_size)).to(device)
        cursor += int(config.batch_size)
        views, _meta = sample_views(batch, family)
        k = len(views)
        ins = [v.unsqueeze(1) for v in views]  # (N,1,Z',Y',X')
        outs = [net(x) for x in ins]

        loss_neighbor = sum(
            mse(outs[i], ins[j]) for i in range(k) for j in range(k) if i != j
        ) / (k * (k - 1))
        loss_idt = (
            sum(mse(outs[i], outs[j]) for i in range(k) for j in range(i + 1, k))
            / (k * (k - 1) / 2)
        )
        total_loss = loss_neighbor + float(config.weight_idt) * loss_idt

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=float(config.clip_gradients))
        optimizer.step()

        records.append({
            "epoch": float(epoch),
            "loss_neighbor": float(loss_neighbor.detach()),
            "loss_idt": float(loss_idt.detach()),
            "total_loss": float(total_loss.detach()),
        })
    return net, records


# --------------------------------------------------------------------------- #
# 评估（U_oracle）
# --------------------------------------------------------------------------- #
def psnr_db(pred: np.ndarray, ref: np.ndarray) -> float:
    """原始强度域 PSNR（镜像 aces.evaluate_stage2 约定）。

    data_range = ref 的 99.9% − 0.1% 分位差（对同一参考 patch 的 raw 与
    denoised 完全同源，增益计算不受影响）；mse=0 → +inf。
    """
    pred64 = np.asarray(pred, dtype=np.float64)
    ref64 = np.asarray(ref, dtype=np.float64)
    if pred64.shape != ref64.shape:
        raise ValueError(f"PSNR 形状不匹配: {pred64.shape} vs {ref64.shape}")
    diff = pred64 - ref64
    mse = float(np.mean(diff * diff))
    if not math.isfinite(mse) or mse < 0.0:
        return float("nan")
    if mse == 0.0:
        return float("inf")
    data_range = float(np.percentile(ref64, 99.9) - np.percentile(ref64, 0.1))
    return float(10.0 * math.log10((max(data_range, 1e-6) ** 2) / mse))


def reference_patch(
    volume: np.ndarray,
    z0: int,
    y0: int,
    x0: int,
    patch_size: tuple[int, int, int],
) -> np.ndarray:
    """AVG-K 伪真值：K 个 repeat 帧在同一 patch 区域的均值（float32）。"""
    pz, py, px = (int(v) for v in patch_size)
    region = volume[:, z0 : z0 + pz, y0 : y0 + py, x0 : x0 + px]
    return np.asarray(region, dtype=np.float64).mean(axis=0).astype(np.float32)


def eval_positions(
    volume_shape: tuple[int, ...],
    patch_size: tuple[int, int, int],
    stride_z: int,
) -> tuple[list[int], list[int], list[int]]:
    """评估网格：z 起点按 stride_z 稀疏采样，y/x 为非重叠网格（余量丢弃）。"""
    if len(volume_shape) != 4:
        raise ValueError(f"oracle 需要 repeat stack [K,Z,Y,X], got shape {volume_shape}")
    _k, z_len, y_len, x_len = volume_shape
    pz, py, px = (int(v) for v in patch_size)
    zs = list(range(0, z_len - pz + 1, max(1, int(stride_z))))
    ys = list(range(0, y_len - py + 1, py))
    xs = list(range(0, x_len - px + 1, px))
    if not (zs and ys and xs):
        raise ValueError(
            f"评估网格为空: stack {volume_shape} vs patch {patch_size} (stride_z={stride_z})"
        )
    return zs, ys, xs


def _band_of(z_center: float, z_len: int) -> int:
    """z 三分带：shallow/mid/deep（z_center 落在 Z 的哪个三分之一）。"""
    if z_center < z_len / 3.0:
        return 0
    if z_center < 2.0 * z_len / 3.0:
        return 1
    return 2


def evaluate_family(
    net: torch.nn.Module,
    volume: np.ndarray,
    pool: PatchPool,
    config: OracleConfig,
    device: torch.device,
) -> dict[str, Any]:
    """单 family 的 U_oracle 评估：逐 eval patch 的 (PSNR_den − PSNR_raw) 增益。

    归一化与训练同源：raw patch 用 PatchPool 的全局 mean/std（探针训练在
    归一化域，评估必须同口径），网络输出反归一化后回原始强度域算 PSNR。
    """
    if volume.ndim != 4:
        raise ValueError(f"oracle 需要 repeat stack [K,Z,Y,X], got {volume.shape}")
    n_frames, z_len, _y_len, _x_len = volume.shape
    pz, py, px = (int(v) for v in config.patch_size)
    frames = config.eval_repeat_frames
    if frames is None:
        frames = list(range(n_frames))
    frames = [int(f) for f in frames]
    bad = [f for f in frames if not 0 <= f < n_frames]
    if bad:
        raise ValueError(f"eval_repeat_frames 越界: {bad} (K={n_frames})")

    zs, ys, xs = eval_positions(tuple(volume.shape), tuple(config.patch_size), int(config.eval_stride_z))
    net.eval()
    patches: list[dict[str, Any]] = []
    with torch.no_grad():
        # 位置外层（reference 每位置只算一次）、repeat 帧内层（AVG-K 伪真值
        # 与帧无关）；逐 patch 单独 forward，保持与训练 batch_size=1 同口径
        for z0 in zs:
            for y0 in ys:
                for x0 in xs:
                    ref = reference_patch(volume, z0, y0, x0, (pz, py, px))
                    for k in frames:
                        raw = np.asarray(
                            volume[k, z0 : z0 + pz, y0 : y0 + py, x0 : x0 + px],
                            dtype=np.float32,
                        )
                        norm = (raw - pool.mean) / pool.std
                        tensor = torch.from_numpy(norm).unsqueeze(0).unsqueeze(0).to(device)
                        out = net(tensor)
                        denorm = (
                            out.squeeze(0).squeeze(0).to("cpu").numpy().astype(np.float32)
                            * pool.std
                            + pool.mean
                        )
                        psnr_raw = psnr_db(raw, ref)
                        psnr_den = psnr_db(denorm, ref)
                        patches.append({
                            "frame": k,
                            "z0": z0,
                            "y0": y0,
                            "x0": x0,
                            "band": _band_of(z0 + pz / 2.0, z_len),
                            "psnr_raw": psnr_raw,
                            "psnr_denoised": psnr_den,
                            "gain": psnr_den - psnr_raw,
                        })
                        if int(config.max_eval_patches) > 0 and len(patches) >= int(config.max_eval_patches):
                            return _summarize_patches(patches, zs, frames, config)
    return _summarize_patches(patches, zs, frames, config)


def _summarize_patches(
    patches: list[dict[str, Any]],
    zs: Sequence[int],
    frames: Sequence[int],
    config: OracleConfig,
) -> dict[str, Any]:
    """逐 patch 记录 → 总体 / 分带 / 逐 z 聚合。"""
    if not patches:
        raise RuntimeError("评估未产生任何 patch（网格为空？）")

    def _agg(rows: list[dict[str, Any]]) -> dict[str, float]:
        gains = [float(r["gain"]) for r in rows]
        raws = [float(r["psnr_raw"]) for r in rows]
        dens = [float(r["psnr_denoised"]) for r in rows]
        return {
            "n_eval_patches": len(rows),
            "psnr_raw_mean": float(np.mean(raws)),
            "psnr_denoised_mean": float(np.mean(dens)),
            "u_oracle_gain": float(np.mean(gains)),
        }

    summary = _agg(patches)
    bands: dict[str, dict[str, float]] = {}
    for band_idx, name in enumerate(BAND_NAMES):
        rows = [r for r in patches if r["band"] == band_idx]
        bands[name] = _agg(rows) if rows else {
            "n_eval_patches": 0, "psnr_raw_mean": float("nan"),
            "psnr_denoised_mean": float("nan"), "u_oracle_gain": float("nan"),
        }
    # 逐 z 起点聚合（诊断深度剖面用，行数 = |zs|×|frames|）
    per_z: list[dict[str, Any]] = []
    for z0 in zs:
        rows = [r for r in patches if r["z0"] == z0]
        if rows:
            entry = _agg(rows)
            entry["z0"] = z0
            per_z.append(entry)
    summary["bands"] = bands
    summary["per_z"] = per_z
    summary["eval_frames"] = list(int(f) for f in frames)
    summary["eval_stride_z"] = int(config.eval_stride_z)
    return summary


# --------------------------------------------------------------------------- #
# probe-csv 子集选择（oracle 成本控制：8–12 个 family）
# --------------------------------------------------------------------------- #
def select_probe_subset(
    path: str | Path,
    n_top: int = 4,
    n_bottom: int = 4,
    n_middle: int = 2,
) -> list[str]:
    """从 micro_probe.csv 按 u_probe 选 oracle 子集：top + bottom + 中间。

    按 u_probe 降序排序（并列按 family_id 保证确定性）；family 数不足
    ``n_top+n_bottom+n_middle`` 时返回全部。返回保持 u_probe 降序。
    """
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        if "family_id" not in fieldnames or "u_probe" not in fieldnames:
            raise ValueError(f"probe csv 缺少 family_id/u_probe 列: {path} (columns={fieldnames})")
        rows = [(row["family_id"], float(row["u_probe"])) for row in reader]
    ranked = sorted(rows, key=lambda item: (-item[1], item[0]))
    ids = [fid for fid, _u in ranked]
    total_needed = n_top + n_bottom + n_middle
    if len(ids) <= total_needed:
        return ids
    top = ids[:n_top]
    bottom = ids[-n_bottom:]
    mid = len(ids) // 2
    middle = [ids[mid - 1], ids[mid]][:n_middle]
    selected = top + middle + bottom
    return sorted(selected, key=lambda fid: ids.index(fid))


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
class OracleRunner:
    """逐 family 训练 + AVG-9 评估，输出 U_oracle。"""

    def __init__(
        self,
        config: OracleConfig,
        net_factory: NetFactory | None = None,
    ):
        self.config = config
        self.net_factory = net_factory

    def _resolve_net_factory(self) -> NetFactory:
        """默认复用 VALID Network_CNR（与 micro_probe / t4b_rgrid 同一 import 路径）。"""
        if self.net_factory is not None:
            return self.net_factory

        def factory() -> torch.nn.Module:
            import sys

            from aces.paths import VALID_ROOT

            sys.path.insert(0, str(VALID_ROOT))
            from models.network import Network_CNR

            return Network_CNR(
                in_channels=1,
                out_channels=1,
                f_maps=int(self.config.base_features),
                n_groups=int(self.config.n_groups),
            )

        return factory

    def _resolve_device(self) -> torch.device:
        requested = str(self.config.device)
        if requested.startswith("cuda") and not torch.cuda.is_available():
            print("[warn] cuda 不可用，回退 cpu")
            return torch.device("cpu")
        return torch.device(requested)

    def _check_view_capacity(self, families: Sequence[CandidateFamily]) -> None:
        """默认 VALID CRN 的视图尺寸前置检查（AvgPool3d(8) → 每轴 ≥8）。

        注入自定义 net_factory 时不检查（约束是网络结构属性而非协议属性）。
        默认库中 axis_*_d2（block=7）要求 patch ≥56/轴，默认 48³ 不够，
        需 patch_size=(64,64,64) 一类配置。
        """
        if self.net_factory is not None:
            return
        patch = tuple(int(v) for v in self.config.patch_size)
        bad: list[str] = []
        for family in families:
            block, _norm = derive_block(family)
            view_dims = [p // b for p, b in zip(patch, block)]
            if min(view_dims) < 8:
                bad.append(f"{family.family_id}(block={block}→view={view_dims})")
        if bad:
            raise SystemExit(
                f"patch {patch} 对以下 family 的视图尺寸 <8/轴（CRN AvgPool3d(8) 限制）: "
                f"{bad}；请增大 patch_size（如 (64,64,64)）或排除这些 family"
            )

    def run(
        self,
        data: str | Path | np.ndarray,
        families: Sequence[CandidateFamily],
        run_dir: str | Path,
    ) -> list[dict[str, Any]]:
        """主入口：每个 family 一个 pilot 训练 + U_oracle 评估，写 JSON/CSV。"""
        run_dir = Path(run_dir).resolve()
        (run_dir / "families").mkdir(parents=True, exist_ok=True)
        volume = np.asarray(load_volume(data) if isinstance(data, (str, Path)) else data)
        if volume.ndim != 4:
            raise ValueError(
                f"oracle 评估需要 repeat stack [K,Z,Y,X]（AVG-K 伪真值），got {volume.shape}"
            )
        pool = PatchPool(
            volume,
            patch_size=tuple(int(v) for v in self.config.patch_size),  # type: ignore[arg-type]
            max_patches=int(self.config.max_train_patches),
            overlap=float(self.config.overlap),
            seed=int(self.config.seed),
        )
        print(
            f"[oracle] patch 池: {len(pool)} × {tuple(int(v) for v in self.config.patch_size)} "
            f"(mean={pool.mean:.4f}, std={pool.std:.4f})",
            flush=True,
        )
        device = self._resolve_device()
        net_factory = self._resolve_net_factory()
        self._check_view_capacity(families)

        rows: list[dict[str, Any]] = []
        started = time.perf_counter()
        for family in families:
            # 断点续跑：已完成的 family JSON 直接复用（同 run_dir 重启时跳过重训）。
            # JSON 结构见下方 write_json：u_oracle_gain 在 eval 子块中。
            done = run_dir / "families" / f"{family.family_id}.json"
            if done.exists():
                try:
                    import json as _json

                    cached = _json.loads(done.read_text(encoding="utf-8"))
                    ev = cached.get("eval", {})
                    if "u_oracle_gain" in ev:
                        row = {
                            "family_id": cached["family_id"],
                            "offsets_zyx": cached["offsets_zyx"],
                            "n_eval_patches": ev["n_eval_patches"],
                            "psnr_raw_mean": ev["psnr_raw_mean"],
                            "psnr_denoised_mean": ev["psnr_denoised_mean"],
                            "u_oracle_gain": ev["u_oracle_gain"],
                            "train_loss_final": cached["train_loss_final"],
                            "epochs": cached["epochs"],
                            "batches": cached["batches"],
                            "elapsed_seconds": cached["elapsed_seconds"],
                        }
                        for name in BAND_NAMES:
                            band = ev["bands"][name]
                            row[f"psnr_raw_{name}"] = band["psnr_raw_mean"]
                            row[f"psnr_denoised_{name}"] = band["psnr_denoised_mean"]
                            row[f"u_oracle_gain_{name}"] = band["u_oracle_gain"]
                        rows.append(row)
                        print(
                            f"[oracle] {family.family_id}: 已有结果，跳过 "
                            f"(U_oracle={row['u_oracle_gain']:+.4f} dB)",
                            flush=True,
                        )
                        continue
                except Exception:
                    pass  # 缓存损坏则重训
            t0 = time.perf_counter()
            net, records = _train_family(self.config, family, pool, net_factory, device)
            last_epoch = records[-1]["epoch"] if records else 0.0
            final_losses = [r["total_loss"] for r in records if r["epoch"] == last_epoch]
            train_loss_final = float(np.mean(final_losses)) if final_losses else float("nan")
            eval_summary = evaluate_family(net, volume, pool, self.config, device)
            elapsed = time.perf_counter() - t0
            if bool(self.config.save_checkpoint):
                ckpt_path = run_dir / "families" / f"{family.family_id}_final.pth"
                torch.save(
                    {
                        "state_dict": net.state_dict(),
                        "family_id": family.family_id,
                        "config": {
                            "epochs": int(self.config.epochs),
                            "batches_per_epoch": int(self.config.batches_per_epoch),
                            "patch_size": list(int(v) for v in self.config.patch_size),
                            "base_features": int(self.config.base_features),
                            "n_groups": int(self.config.n_groups),
                            "seed": int(self.config.seed),
                        },
                        "pool_stats": {"mean": float(pool.mean), "std": float(pool.std)},
                        "u_oracle_gain": eval_summary["u_oracle_gain"],
                    },
                    ckpt_path,
                )
                print(f"[oracle] {family.family_id}: checkpoint → {ckpt_path.name}", flush=True)

            row: dict[str, Any] = {
                "family_id": family.family_id,
                "offsets_zyx": [list(map(int, d)) for d in family.offsets_zyx],
                "n_eval_patches": eval_summary["n_eval_patches"],
                "psnr_raw_mean": eval_summary["psnr_raw_mean"],
                "psnr_denoised_mean": eval_summary["psnr_denoised_mean"],
                "u_oracle_gain": eval_summary["u_oracle_gain"],
                "train_loss_final": train_loss_final,
                "epochs": int(self.config.epochs),
                "batches": len(records),
                "elapsed_seconds": elapsed,
            }
            for name in BAND_NAMES:
                band = eval_summary["bands"][name]
                row[f"psnr_raw_{name}"] = band["psnr_raw_mean"]
                row[f"psnr_denoised_{name}"] = band["psnr_denoised_mean"]
                row[f"u_oracle_gain_{name}"] = band["u_oracle_gain"]
                row[f"n_eval_patches_{name}"] = band["n_eval_patches"]
            rows.append(row)

            # 逐 epoch 聚合 loss 曲线（20ep×200batch=4000 条逐 batch 记录太长）
            epoch_curve: list[dict[str, float]] = []
            for epoch in sorted({int(r["epoch"]) for r in records}):
                ep = [r for r in records if int(r["epoch"]) == epoch]
                epoch_curve.append({
                    "epoch": float(epoch),
                    "loss_neighbor_mean": float(np.mean([r["loss_neighbor"] for r in ep])),
                    "loss_idt_mean": float(np.mean([r["loss_idt"] for r in ep])),
                    "total_loss_mean": float(np.mean([r["total_loss"] for r in ep])),
                })
            write_json(run_dir / "families" / f"{family.family_id}.json", {
                "event": "oracle_family",
                "family_id": family.family_id,
                "offsets_zyx": row["offsets_zyx"],
                "train_loss_final": train_loss_final,
                "epochs": int(self.config.epochs),
                "batches": len(records),
                "train_loss_curve": epoch_curve,
                "eval": eval_summary,
                "elapsed_seconds": elapsed,
            })
            print(
                f"[oracle] {family.family_id}: U_oracle={row['u_oracle_gain']:+.4f} dB "
                f"(raw={row['psnr_raw_mean']:.3f} → den={row['psnr_denoised_mean']:.3f}, "
                f"n={row['n_eval_patches']}, {elapsed:.1f}s)",
                flush=True,
            )

        write_oracle_csv(run_dir / "oracle.csv", rows)
        write_json(run_dir / "families" / "summary.json", {
            "event": "oracle_complete",
            "config": asdict(self.config),
            "n_families": len(rows),
            "elapsed_seconds": time.perf_counter() - started,
            "rows": rows,
        })
        _print_table(rows)
        return rows


ORACLE_CSV_COLUMNS = [
    "family_id",
    "offsets_zyx",
    "psnr_raw_mean",
    "psnr_denoised_mean",
    "u_oracle_gain",
    "n_eval_patches",
    *(f"psnr_raw_{b}" for b in BAND_NAMES),
    *(f"psnr_denoised_{b}" for b in BAND_NAMES),
    *(f"u_oracle_gain_{b}" for b in BAND_NAMES),
    *(f"n_eval_patches_{b}" for b in BAND_NAMES),
    "train_loss_final",
    "epochs",
    "batches",
    "elapsed_seconds",
]


def write_oracle_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """合并 oracle.csv（family_id + U_oracle + 分带分解 + 训练终值）。"""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ORACLE_CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            out = dict(r)
            if "offsets_zyx" in out:
                out["offsets_zyx"] = ";".join(
                    ",".join(str(v) for v in d) for d in out["offsets_zyx"]
                )
            writer.writerow(out)


def _print_table(rows: list[dict[str, Any]]) -> None:
    print("\n=== oracle U_oracle 增益（降序）===")
    for r in sorted(rows, key=lambda x: -float(x.get("u_oracle_gain", 0.0))):
        bands = " ".join(
            f"{b}={float(r[f'u_oracle_gain_{b}']):+.2f}" for b in BAND_NAMES
        )
        print(
            f"  {r['family_id']:<16s} U_oracle={float(r['u_oracle_gain']):+.4f} dB "
            f"(raw={float(r['psnr_raw_mean']):.3f} → den={float(r['psnr_denoised_mean']):.3f}) "
            f"[{bands}] n={r['n_eval_patches']}"
        )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _split_config_spec(spec: str) -> list[str]:
    """按逗号切分 --config 项，但跳过括号内的逗号（支持 patch_size=(16,32,32)）。"""
    tokens: list[str] = []
    depth = 0
    current = []
    for ch in spec:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            tokens.append("".join(current))
            current = []
        else:
            current.append(ch)
    if current:
        tokens.append("".join(current))
    return [t for t in (tok.strip() for tok in tokens) if t]


def apply_config_overrides(config: OracleConfig, spec: str) -> OracleConfig:
    """应用 CLI 覆盖项：逗号分隔 key=value（括号内逗号不切分）。"""
    valid = {f.name for f in fields(OracleConfig)}
    for token in _split_config_spec(spec):
        token = token.strip()
        if not token:
            continue
        if "=" not in token:
            raise SystemExit(f"--config 项格式错误（需 key=value）: {token!r}")
        key, raw = token.split("=", 1)
        key = key.strip()
        if key not in valid:
            raise SystemExit(f"--config 未知字段 {key!r}（可用: {sorted(valid)}）")
        setattr(config, key, _parse_override_value(raw))
    return config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Oracle 训练评估：每 family 20ep pilot + AVG-9 伪真值 U_oracle（方案 M2 Exp C）。"
    )
    parser.add_argument("--data", required=True, help="repeat stack tif [K,Z,Y,X]")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--families", default="all", help='"all" 或逗号分隔 family_id（覆盖 --probe-csv 的自动选择）')
    parser.add_argument(
        "--config",
        default="",
        help='逗号分隔覆盖项，如 "epochs=20,batches_per_epoch=200,eval_stride_z=16"',
    )
    parser.add_argument("--device", default=None, help="覆盖 config.device")
    parser.add_argument(
        "--probe-csv",
        default=None,
        help="micro_probe.csv：按 u_probe 自动选 oracle 子集（top4+bottom4+2 中间）",
    )
    args = parser.parse_args(argv)

    config = OracleConfig()
    if args.config:
        config = apply_config_overrides(config, args.config)
    if args.device:
        config.device = args.device

    if args.probe_csv and args.families.strip().lower() in ("all", ""):
        selected = select_probe_subset(args.probe_csv)
        library = {f.family_id: f for f in resolve_families("all")}
        missing = [fid for fid in selected if fid not in library]
        if missing:
            raise SystemExit(f"probe csv 的 family 不在默认库中: {missing}")
        families = [library[fid] for fid in selected]
        print(f"[oracle] probe-csv 自动子集（top4+bottom4+2mid, u_probe 降序）: {selected}", flush=True)
    else:
        families = resolve_families(args.families)

    runner = OracleRunner(config)
    runner.run(args.data, families, args.run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
