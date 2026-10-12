"""Micro-probe 引擎（修正方案 M2 / §3 U_probe）。

极低预算训练探针：固定全部训练变量（网络、lr、patch 协议、seed），只换
sampling family Γ，用训练统计量 φ(Γ) 估计实际优化效用 U_probe（方案 §3）。

协议镜像 ``qmask/t4b_rgrid.py``（20ep×100batch pilot 的缩减版）：

- 网络：默认复用 VALID 的 ``Network_CNR``（与 t4b_rgrid / train_stage2 同一
  import 路径 ``models.network.Network_CNR``，经 ``sys.path`` 注入 VALID_ROOT）；
  允许 ``net_factory`` 注入以便单元测试 / 无 VALID 环境运行。
- 损失（N2N-style，t4b 的 4 视图对称推广，见 :func:`_probe_family`）：
    loss2neighbor = mean_{i≠j} MSE(out_i, view_j)   # t4b: 0.5·l2(out1,v3)+0.5·l2(out2,v4)
    loss_idt      = mean_{i<j} MSE(out_i, out_j)    # t4b: l2(out1, out2)
    total         = loss2neighbor + weight_idt · loss_idt
  t4b 的 Hessian/DWT/背景正则与 family 选择正交且引入 VALID 硬依赖，
  探针默认省略（保持自包含，全部 family 公平比较）。
- 无 validation、无 best checkpoint：探针只收训练统计量。

特征 φ(Γ)（方案 §13.2）：loss_neighbor_mean/slope、loss_identity_mean/slope、
residual_corr、output_input_corr、output_gradient_ratio、output_variance_ratio、
cross_view_consistency、loss_stability；外加可选 R_I（--risk-csv）。

U_probe（方案 §3 初始权重，robust z-score 跨 family 归一）：
    +0.25 z(−L_N mean) + 0.15 z(−L_I identity) + 0.15 z(−|residual_corr|)
    + 0.20 z(−|grad_ratio−1|) + 0.10 z(cross_view_consistency)
    + 0.05 z(−loss_stability) [+ 0.10 z(−R_I)（提供 risk csv 时）]
缺 R_I 时该项省略并按剩余权重重归一（÷0.90）。

用法：
    PYTHONPATH=ACES python -m calibration.micro_probe \
        --data <repeat_stack.tif|volume.tif> --run-dir <dir> \
        --families all --device cuda [--risk-csv risk_families.csv] \
        [--config "epochs=2,batches_per_epoch=100"]
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import random
import sys
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import tifffile
import torch

from aces.io_utils import write_json
from profiling.candidates import CandidateFamily, default_family_library

from .family_sampler import derive_block, sample_views

# --------------------------------------------------------------------------- #
# 配置
# --------------------------------------------------------------------------- #
NetFactory = Callable[[], torch.nn.Module]


@dataclass
class MicroProbeConfig:
    """Micro-probe 训练协议（镜像 t4b_rgrid pilot 协议的缩减版）。"""

    epochs: int = 2
    batches_per_epoch: int = 100
    patch_size: tuple[int, int, int] = (48, 48, 48)  # 2D 时传 (64,64) 或 (1,64,64)
    batch_size: int = 1
    lr: float = 1e-4
    seed: int = 3407
    device: str = "cuda"
    max_patches: int = 128
    # VALID Network_CNR 的构造参数（镜像 t4b_rgrid 的 base_features / n_groups）
    base_features: int = 32
    n_groups: int = 8
    # 损失权重：loss_idt 在 t4b 中与 loss2neighbor 同权（=1）
    weight_idt: float = 1.0
    clip_gradients: float = 1.0
    # patch 网格重叠率（0 = 不重叠网格）
    overlap: float = 0.0
    # view 最小维度护栏：默认 VALID Network_CNR（含 kernel-8 池化）要求 ≥8；
    # 测试注入的小网络可设 0 关闭
    min_view_dim: int = 8

    def __post_init__(self) -> None:
        size = tuple(int(v) for v in self.patch_size)
        if len(size) == 2:  # 2D：(Y,X) → (1,Y,X)
            size = (1, size[0], size[1])
        if len(size) != 3 or min(size) < 1:
            raise ValueError(f"patch_size 必须是 (Z,Y,X) 或 2D (Y,X), got {self.patch_size}")
        self.patch_size = size  # type: ignore[assignment]


def _set_seed(seed: int) -> None:
    """随机种子（与 aces.train_stage2._set_seed 同一实现）。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


# --------------------------------------------------------------------------- #
# 数据：volume / repeat-stack → 确定性 patch 池
# --------------------------------------------------------------------------- #
def load_volume(path: str | Path) -> np.ndarray:
    """读 tif（repeat-stack [K,Z,Y,X] 或单 volume [Z,Y,X] / 2D [Y,X]）。"""
    try:
        stack = np.asarray(tifffile.memmap(path))
    except Exception:
        stack = np.asarray(tifffile.imread(path))
    if stack.ndim not in (2, 3, 4):
        raise ValueError(f"不支持的数据维度 {stack.shape}（需 2/3/4 维 tif）")
    return stack


def volume_stats(volume: np.ndarray) -> tuple[float, float]:
    """全局 mean/std（分块累加，镜像 t4b_rgrid._stack_stats）。"""
    flat = np.asarray(volume).reshape(-1).astype(np.float64)
    total = 0.0
    total_sq = 0.0
    for i in range(0, flat.size, 10_000_000):
        block = flat[i : i + 10_000_000]
        total += block.sum()
        total_sq += (block**2).sum()
    mean = total / flat.size
    std = float(np.sqrt(max(total_sq / flat.size - mean**2, 0.0)))
    return float(mean), std


class PatchPool:
    """确定性 patch 池：volume（或 repeat-stack 逐帧）网格切块 + seeded 洗牌。

    归一化镜像 t4b_rgrid：patch = (raw − mean) / std（全局统计）。
    全部 family 共用同一池，保证探针的唯一变量是 Γ。
    """

    def __init__(
        self,
        volume: np.ndarray,
        patch_size: tuple[int, int, int],
        max_patches: int,
        overlap: float = 0.0,
        seed: int = 3407,
    ):
        z_patch, y_patch, x_patch = (int(v) for v in patch_size)
        self.patch_size = (z_patch, y_patch, x_patch)
        if volume.ndim == 4:  # repeat stack：逐帧收集
            frames = [np.asarray(volume[k]) for k in range(volume.shape[0])]
        elif volume.ndim == 3:
            frames = [np.asarray(volume)]
        elif volume.ndim == 2:
            frames = [np.asarray(volume)[np.newaxis]]
        else:
            raise ValueError(f"volume 维度错误: {volume.shape}")
        mean, std = volume_stats(volume)
        if std < 1e-12:
            raise ValueError("volume std≈0，无法归一化（全常数体）")

        step = lambda size, patch: max(1, int(round(patch * (1.0 - overlap))))  # noqa: E731
        positions: list[tuple[int, int, int, int]] = []
        for k, frame in enumerate(frames):
            if frame.ndim == 2:  # 2D 帧补 z 维
                frame = frame[np.newaxis]
            z_len, y_len, x_len = frame.shape
            if min(z_len - z_patch, y_len - y_patch, x_len - x_patch) < 0:
                raise ValueError(f"帧 {k} 尺寸 {frame.shape} 小于 patch {patch_size}（z,y,x）")
            for z0 in range(0, z_len - z_patch + 1, step(z_len, z_patch)):
                for y0 in range(0, y_len - y_patch + 1, step(y_len, y_patch)):
                    for x0 in range(0, x_len - x_patch + 1, step(x_len, x_patch)):
                        positions.append((k, z0, y0, x0))
        if not positions:
            raise ValueError("patch 网格为空：volume 尺寸不足")

        rng = np.random.default_rng(seed)
        order = rng.permutation(len(positions))
        positions = [positions[i] for i in order][: max(1, int(max_patches))]
        self.mean, self.std = mean, std

        patches = []
        for k, z0, y0, x0 in positions:
            frame = frames[k]
            if frame.ndim == 2:
                frame = frame[np.newaxis]
            raw = frame[z0 : z0 + z_patch, y0 : y0 + y_patch, x0 : x0 + x_patch]
            patches.append((raw.astype(np.float32) - mean) / std)
        self.patches = torch.from_numpy(np.stack(patches))  # (P,Z,Y,X)
        self.positions = positions

    def __len__(self) -> int:
        return int(self.patches.shape[0])

    def take(self, start: int, n: int) -> torch.Tensor:
        """循环取 n 个 patch，返回 (n,Z,Y,X)。"""
        total = len(self)
        if total == 0:
            raise RuntimeError("patch 池为空")
        idx = [(start + i) % total for i in range(n)]
        return self.patches[idx]


# --------------------------------------------------------------------------- #
# 数值工具
# --------------------------------------------------------------------------- #
def _pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    """展平后的 Pearson 相关（常量输入返回 0，避免 NaN）。"""
    a = a.reshape(-1).double()
    b = b.reshape(-1).double()
    a = a - a.mean()
    b = b - b.mean()
    denom = float(a.norm() * b.norm())
    if denom < 1e-12:
        return 0.0
    return float((a * b).sum().item() / denom)


def _grad_mag(t: torch.Tensor) -> torch.Tensor:
    """梯度幅值：各空间轴前向差分 |Δ| 均值，再对轴平均。t: (N,1,Z,Y,X)。"""
    total = t.new_zeros(())
    axes = 0
    for dim in (2, 3, 4):
        if t.shape[dim] > 1:
            total = total + t.diff(dim=dim).abs().mean()
            axes += 1
    if axes == 0:
        return t.new_zeros(())
    return total / axes


def _slope(values: list[float]) -> float:
    """对 batch 序号的线性拟合斜率（不足 2 点返回 0）。"""
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values), dtype=np.float64)
    return float(np.polyfit(x, np.asarray(values, dtype=np.float64), 1)[0])


def robust_zscore(values: Sequence[float]) -> np.ndarray:
    """robust z-score：median / (1.4826·MAD)；MAD 退化时回退 std，再退化置 0。"""
    arr = np.asarray(list(values), dtype=np.float64)
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med))) * 1.4826
    if mad < 1e-12:
        mad = float(arr.std())
    if mad < 1e-12:
        return np.zeros_like(arr)
    return (arr - med) / mad


# --------------------------------------------------------------------------- #
# 单 family 探针
# --------------------------------------------------------------------------- #
def _probe_family(
    config: MicroProbeConfig,
    family: CandidateFamily,
    pool: PatchPool,
    net_factory: NetFactory,
    device: torch.device,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    """跑一个 family 的微型训练，返回 (特征 φ(Γ), 逐 batch 记录)。

    损失构造（镜像 t4b_rgrid.py 的核心两项）：
        out_i          = net(view_i)                          # t4b: out1/out2
        loss2neighbor  = mean_{i≠j} MSE(out_i, view_j)        # t4b: 0.5·l2(out1,v3)+0.5·l2(out2,v4)
        loss_idt       = mean_{i<j} MSE(out_i, out_j)         # t4b: l2(out1, out2)
        total          = loss2neighbor + weight_idt · loss_idt
    t4b 的 v3/v4 来自时间配对距离 d 的另一帧；探针的 4 视图来自同一 patch 的
    共位相位分裂（family offset），邻域项取 K 视图对称形式（K=2 时等价双向）。
    """
    _set_seed(config.seed)  # 每个 family 重置：初始化/数据顺序一致，只换 Γ
    # fail-fast：view 任一维 < 8 时 VALID Network_CNR（含 kernel-8 池化）无法前向
    # （48³ patch 上 axis_*_d2 家族 block=7 → 6 体素，实测崩溃）。需加大 patch_size。
    block, _ = derive_block(family)
    view_dims = tuple(p // b for p, b in zip(pool.patch_size, block))
    if int(config.min_view_dim) > 0 and min(view_dims) < int(config.min_view_dim):
        raise ValueError(
            f"family {family.family_id}: patch {pool.patch_size} / block {block} → "
            f"view {view_dims}，最小维 <8，VALID 网络无法前向。"
            f"请增大 --config patch_size=(如 (96,96,96))。"
        )
    net = net_factory().to(device)
    net.train()
    optimizer = torch.optim.Adam(net.parameters(), lr=float(config.lr), betas=(0.9, 0.999))
    mse = torch.nn.MSELoss()

    records: list[dict[str, float]] = []
    cursor = 0
    total_batches = int(config.epochs) * int(config.batches_per_epoch)
    for _step in range(total_batches):
        batch = pool.take(cursor, int(config.batch_size)).to(device)
        cursor += int(config.batch_size)
        views, _meta = sample_views(batch, family)
        k = len(views)
        ins = [v.unsqueeze(1) for v in views]  # (N,1,Z',Y',X')，镜像 t4b 的 v.unsqueeze(1)
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

        with torch.no_grad():
            out_d = [o.detach() for o in outs]
            in_d = [x.detach() for x in ins]
            loss_identity = float(sum(mse(out_d[i], in_d[i]) for i in range(k)) / k)
            residual_corr = float(
                sum(_pearson(out_d[i] - in_d[i], in_d[i]) for i in range(k)) / k
            )
            output_input_corr = float(sum(_pearson(out_d[i], in_d[i]) for i in range(k)) / k)
            grad_out = float(_grad_mag(torch.cat(out_d, dim=0)))
            grad_in = float(_grad_mag(torch.cat(in_d, dim=0)))
            grad_ratio = grad_out / grad_in if grad_in > 1e-12 else 0.0
            var_ratios = []
            for i in range(k):
                v_in = float(in_d[i].var())
                var_ratios.append(float(out_d[i].var()) / v_in if v_in > 1e-12 else 0.0)
            cross_view = float(
                sum(_pearson(out_d[i], out_d[j]) for i in range(k) for j in range(i + 1, k))
                / (k * (k - 1) / 2)
            )
            records.append({
                "loss_neighbor": float(loss_neighbor.detach()),
                "loss_idt": float(loss_idt.detach()),
                "loss_identity": loss_identity,
                "residual_corr": residual_corr,
                "output_input_corr": output_input_corr,
                "output_gradient_ratio": grad_ratio,
                "output_variance_ratio": float(np.mean(var_ratios)),
                "cross_view_consistency": cross_view,
            })

    # φ(Γ)：逐 batch 统计量 → 聚合特征（方案 §13.2）
    nb = [r["loss_neighbor"] for r in records]
    ident = [r["loss_identity"] for r in records]
    n_half = max(1, len(nb) // 2)
    features: dict[str, float] = {
        "loss_neighbor_mean": float(np.mean(nb)),
        "loss_neighbor_slope": _slope(nb),
        "loss_identity_mean": float(np.mean(ident)),
        "loss_identity_slope": _slope(ident),
        "residual_corr": float(np.mean([r["residual_corr"] for r in records])),
        "output_input_corr": float(np.mean([r["output_input_corr"] for r in records])),
        "output_gradient_ratio": float(np.mean([r["output_gradient_ratio"] for r in records])),
        "output_variance_ratio": float(np.mean([r["output_variance_ratio"] for r in records])),
        "cross_view_consistency": float(np.mean([r["cross_view_consistency"] for r in records])),
        "loss_stability": float(np.std(nb[-n_half:])) if len(nb) >= 2 else 0.0,
    }
    return features, records


# --------------------------------------------------------------------------- #
# U_probe 组合（方案 §3 初始权重）
# --------------------------------------------------------------------------- #
# (weight, 特征列, 定向变换)：变换后的列越大越好，再跨 family robust z-score
U_PROBE_TERMS: list[tuple[float, str, Callable[[float], float]]] = [
    (0.25, "loss_neighbor_mean", lambda v: -v),  # 邻域损失改善
    (0.15, "loss_identity_mean", lambda v: -v),  # 恒等风险降低
    (0.15, "residual_corr", lambda v: -abs(v)),  # 残差独立
    (0.20, "output_gradient_ratio", lambda v: -abs(v - 1.0)),  # 梯度保持
    (0.10, "cross_view_consistency", lambda v: v),  # 跨视图一致
    (0.05, "loss_stability", lambda v: -v),  # 稳定性
    (0.10, "risk_ri", lambda v: -v),  # 捷径惩罚（--risk-csv 提供时）
]

FEATURE_COLUMNS = [
    "loss_neighbor_mean",
    "loss_neighbor_slope",
    "loss_identity_mean",
    "loss_identity_slope",
    "residual_corr",
    "output_input_corr",
    "output_gradient_ratio",
    "output_variance_ratio",
    "cross_view_consistency",
    "loss_stability",
    "risk_ri",
]


def compute_u_probe(
    rows: list[dict[str, Any]],
    risk_ri: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """跨 family robust z-score 并组合 U_probe（就地写入 rows 并返回）。

    定向变换（如 −|residual_corr|）先作用在原始列上再做 z-score；
    某项在某 family 缺值（如 risk csv 缺该 family 的 R_I）时该 family 的
    此项省略并按剩余权重重归一；整体缺 R_I 时全体重归一（÷0.90）。
    """
    if not rows:
        return rows
    if risk_ri is not None:
        missing = [r["family_id"] for r in rows if r["family_id"] not in risk_ri]
        if missing:
            print(f"[warn] risk csv 缺少 family R_I，对应 term 逐 family 省略: {missing}")
        for r in rows:
            r["risk_ri"] = risk_ri.get(r["family_id"])

    # 1) 定向列：orient_<col> = fn(raw)，缺值保持 None
    oriented_cols: list[str] = []
    for _w, col, fn in U_PROBE_TERMS:
        key = f"_oriented_{col}"
        oriented_cols.append(key)
        for r in rows:
            v = r.get(col)
            r[key] = fn(float(v)) if v is not None and np.isfinite(float(v)) else None

    # 2) 每个定向列在非缺值子集上 robust z-score → z_<col>
    for (_w, col, _fn), key in zip(U_PROBE_TERMS, oriented_cols):
        present = [r for r in rows if r[key] is not None]
        if not present:
            continue
        z = robust_zscore([r[key] for r in present])  # type: ignore[arg-type]
        for r, zval in zip(present, z.tolist()):
            r[f"z_{col}"] = float(zval)

    # 3) 非 U_probe 项的其余特征列（slope / corr / variance_ratio）也做
    #    identity robust z-score，随 CSV 一起输出供诊断
    term_cols = {col for _w, col, _fn in U_PROBE_TERMS}
    for col in FEATURE_COLUMNS:
        if col in term_cols or any(f"z_{col}" in r for r in rows):
            continue
        present = [
            r for r in rows
            if r.get(col) is not None and np.isfinite(float(r[col]))
        ]
        if not present:
            continue
        z = robust_zscore([r[col] for r in present])  # type: ignore[arg-type]
        for r, zval in zip(present, z.tolist()):
            r[f"z_{col}"] = float(zval)

    # 4) 加权组合（按 family 实际可用项重归一）
    for r in rows:
        num = 0.0
        den = 0.0
        for (w, col, _fn), key in zip(U_PROBE_TERMS, oriented_cols):
            if r.get(f"z_{col}") is not None:
                num += w * float(r[f"z_{col}"])
                den += w
        r["u_probe"] = num / den if den > 0 else 0.0
        for key in oriented_cols:
            r.pop(key, None)
    return rows


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
class MicroProbeRunner:
    """逐 family 跑 micro-probe，输出 φ(Γ) / z-score / U_probe。"""

    def __init__(
        self,
        config: MicroProbeConfig,
        net_factory: NetFactory | None = None,
    ):
        self.config = config
        self.net_factory = net_factory

    def _resolve_net_factory(self) -> NetFactory:
        """默认复用 VALID Network_CNR（镜像 t4b_rgrid / train_stage2 的 import 路径）。"""
        if self.net_factory is not None:
            return self.net_factory

        def factory() -> torch.nn.Module:
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

    def run(
        self,
        data: str | Path | np.ndarray,
        families: Sequence[CandidateFamily],
        run_dir: str | Path,
        risk_ri: dict[str, float] | None = None,
    ) -> list[dict[str, Any]]:
        """主入口：每个 family 一个微型训练，写 JSON/CSV，返回结果行。"""
        run_dir = Path(run_dir).resolve()
        (run_dir / "families").mkdir(parents=True, exist_ok=True)
        volume = load_volume(data) if isinstance(data, (str, Path)) else np.asarray(data)
        pool = PatchPool(
            volume,
            patch_size=tuple(int(v) for v in self.config.patch_size),  # type: ignore[arg-type]
            max_patches=int(self.config.max_patches),
            overlap=float(self.config.overlap),
            seed=int(self.config.seed),
        )
        print(
            f"[probe] patch 池: {len(pool)} × {tuple(int(v) for v in self.config.patch_size)} "
            f"(mean={pool.mean:.4f}, std={pool.std:.4f})",
            flush=True,
        )
        device = self._resolve_device()
        net_factory = self._resolve_net_factory()

        rows: list[dict[str, Any]] = []
        all_records: list[list[dict[str, float]]] = []
        started = time.perf_counter()
        for family in families:
            t0 = time.perf_counter()
            features, records = _probe_family(self.config, family, pool, net_factory, device)
            rows.append({
                "family_id": family.family_id,
                "offsets_zyx": [list(map(int, d)) for d in family.offsets_zyx],
                "n_batches": len(records),
                "elapsed_seconds": time.perf_counter() - t0,
                **features,
            })
            all_records.append(records)
            print(
                f"[probe] {family.family_id}: L_N={features['loss_neighbor_mean']:.4f} "
                f"L_I={features['loss_identity_mean']:.4f} ({rows[-1]['elapsed_seconds']:.1f}s)",
                flush=True,
            )

        rows = compute_u_probe(rows, risk_ri)

        # ---- 落盘：逐 family JSON（含逐 batch loss 曲线）----
        for row, records in zip(rows, all_records):
            write_json(run_dir / "families" / f"{row['family_id']}.json", {
                "event": "micro_probe_family",
                "family_id": row["family_id"],
                "offsets_zyx": row["offsets_zyx"],
                "features": {c: row[c] for c in FEATURE_COLUMNS if c in row},
                "z_scores": {k[2:]: row[k] for k in row if k.startswith("z_")},
                "u_probe": row.get("u_probe"),
                "loss_records": records,
            })
        write_json(run_dir / "families" / "summary.json", {
            "event": "micro_probe_complete",
            "config": asdict(self.config),
            "risk_csv_used": risk_ri is not None,
            "n_families": len(rows),
            "elapsed_seconds": time.perf_counter() - started,
            "rows": rows,
        })
        write_csv_rows(run_dir / "micro_probe.csv", rows)
        _print_table(rows)
        return rows


def write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    """合并 micro_probe.csv（family_id + 特征 + z + U_probe）。"""
    if not rows:
        return
    base_cols = ["family_id", "offsets_zyx", "n_batches", "elapsed_seconds"]
    feat_cols = [c for c in FEATURE_COLUMNS if any(c in r for r in rows)]
    z_cols = sorted({k for r in rows for k in r if k.startswith("z_")})
    fieldnames = base_cols + feat_cols + z_cols + ["u_probe"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            out = dict(r)
            if "offsets_zyx" in out:
                out["offsets_zyx"] = ";".join(
                    ",".join(str(v) for v in d) for d in out["offsets_zyx"]
                )
            writer.writerow(out)


def _print_table(rows: list[dict[str, Any]]) -> None:
    print("\n=== micro-probe U_probe（降序）===")
    for r in sorted(rows, key=lambda x: -float(x.get("u_probe", 0.0))):
        print(
            f"  {r['family_id']:<16s} U_probe={float(r['u_probe']):+.4f} "
            f"L_N={r['loss_neighbor_mean']:.4f} L_I={r['loss_identity_mean']:.4f} "
            f"resid_corr={r['residual_corr']:+.3f} grad_ratio={r['output_gradient_ratio']:.3f} "
            f"cross={r['cross_view_consistency']:+.3f}"
        )


# --------------------------------------------------------------------------- #
# risk csv：family → R_I
# --------------------------------------------------------------------------- #
def load_risk_ri(path: str | Path) -> dict[str, float]:
    """读 risk_families.csv（family 级 ri_mean）或 risk_field.csv（offset 级）。

    offset 级时对每个 family 的 pairwise Δ_ij 取 ri 均值（镜像
    profiling.risk_field.family_summary 的 mean 聚合）；pairwise offset
    不全的 family 打警告并跳过（其 R_I term 在 U_probe 中逐 family 省略）。
    """
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        rows = list(reader)
    if "family_id" in fieldnames and "ri_mean" in fieldnames:
        return {row["family_id"]: float(row["ri_mean"]) for row in rows}
    if "offset_voxel_z" in fieldnames:
        per_offset = {
            (
                int(float(row["offset_voxel_z"])),
                int(float(row["offset_voxel_y"])),
                int(float(row["offset_voxel_x"])),
            ): float(row["ri"])
            for row in rows
        }
        risk: dict[str, float] = {}
        for family in default_family_library():
            deltas = family.pairwise_offsets()
            missing = [d for d in deltas if d not in per_offset]
            if missing:
                print(f"[warn] risk csv 缺少 {family.family_id} 的 pairwise offset "
                      f"{missing}，该 family 的 R_I term 省略")
                continue
            risk[family.family_id] = float(np.mean([per_offset[d] for d in deltas]))
        return risk
    raise ValueError(f"无法识别的 risk csv 格式: {path} (columns={fieldnames})")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _parse_override_value(raw: str) -> Any:
    """解析 --config 的 value（literal_eval 失败时按原始字符串）。"""
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return raw


def apply_config_overrides(config: MicroProbeConfig, spec: str) -> MicroProbeConfig:
    """应用 CLI 覆盖项：逗号分隔的 key=value（patch_size 用 "(48,48,48)"）。

    括号内的逗号（元组值）不属于分隔符：先按括号深度切分。
    """
    valid = {f.name for f in fields(MicroProbeConfig)}
    tokens: list[str] = []
    depth, cur = 0, []
    for ch in spec:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            tokens.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    if cur:
        tokens.append("".join(cur))
    for token in tokens:
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


def resolve_families(spec: str) -> list[CandidateFamily]:
    """"all" 或逗号分隔的 family_id 列表。"""
    library = {f.family_id: f for f in default_family_library()}
    if spec.strip().lower() in ("all", ""):
        return list(library.values())
    wanted = [s.strip() for s in spec.split(",") if s.strip()]
    missing = [s for s in wanted if s not in library]
    if missing:
        raise SystemExit(f"未知 family: {missing}（可用: {sorted(library)}）")
    return [library[s] for s in wanted]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Micro-probe：极低预算训练探针估计 sampling family 的 U_probe（方案 M2）。"
    )
    parser.add_argument("--data", required=True, help="repeat-stack tif [K,Z,Y,X] 或单 volume tif")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--families", default="all", help='"all" 或逗号分隔 family_id')
    parser.add_argument(
        "--config",
        default="",
        help='逗号分隔覆盖项，如 "epochs=2,batches_per_epoch=100,patch_size=(48,48,48)"',
    )
    parser.add_argument("--device", default=None, help="覆盖 config.device")
    parser.add_argument("--risk-csv", default=None, help="risk_families.csv / risk_field.csv（R_I 特征）")
    args = parser.parse_args(argv)

    config = MicroProbeConfig()
    if args.config:
        config = apply_config_overrides(config, args.config)
    if args.device:
        config.device = args.device
    families = resolve_families(args.families)
    risk_ri = load_risk_ri(args.risk_csv) if args.risk_csv else None

    runner = MicroProbeRunner(config)
    runner.run(args.data, families, args.run_dir, risk_ri=risk_ri)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
