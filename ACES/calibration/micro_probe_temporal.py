"""时间轴 Micro-probe（CIDC25 反向验证，M2 收口项）。

在 CIDC25 时间序列上按 ``t4b_rgrid`` 的 Δt 配对协议跑极低预算探针，产出与
``micro_probe`` 同义的特征 φ(Δt)——用途：**反向验证** U_probe v2 的 regime
依赖（3P 空间上拟合的权重 w=[+1.45, +4.47] 迁移到时间轴应失效，H4 证据闭环）。

协议（镜像 micro_probe + t4b_rgrid）：

- 数据：CIDC25 Training 栈（A1/B1/C2/D2），patch (z_patch, w_patch, h_patch)，
  逐栈 mean/std 归一化（镜像 t4b_rgrid.PairDistanceDataset）。
- 配对：输入 patch(t0)，邻域 patch(t0 + dt)（唯一变量 = dt，对应空间探针的 Γ）。
- 视图：t4b 相位分裂（(0,0)/(1,1) 对角相位）——输入得 v1/v2（网络输入），
  邻域得 v3/v4（邻域视图）。
- 损失：loss2neighbor = mean_{i≠j} MSE(out_i, view_j)（micro_probe 的 K 视图
  对称形式，K=2）；loss_idt = MSE(out1, out2)；total = L_N + weight_idt·L_I。
  t4b 的 Hessian/DWT/背景正则与 dt 正交，探针省略（同 micro_probe）。
- 特征：与 micro_probe 完全同义（loss_neighbor_mean、cross_view_consistency 等）；
  每个 dt 前重置 seed（镜像逐 family 重置）。
- 网络：VALID Network_CNR(base_features=32, n_groups=8)——与 3P 空间探针
  同配置（权重迁移的可比性）；注意 CIDC25 rgrid oracle 用 16/4，网络差异
  记入结论保留项。

用法：
    PYTHONPATH=ACES python -m calibration.micro_probe_temporal \
        --data-root /data2/wjb/threePM/ThreePM_openDatasets/AI4Life-CIDC25 \
        --run-dir <dir> [--dt-list 1,2,3,4,6,8,16,32,64] \
        [--config "epochs=2,batches_per_epoch=100,patch_size=(64,128,128)"]
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
import torch

from aces.io_utils import write_json

from .micro_probe import (
    MicroProbeConfig,
    NetFactory,
    _grad_mag,
    _pearson,
    _set_seed,
    _slope,
)

# 与 qmask.t4b_rgrid 一致
TRAIN_STACKS = ("Training/A1.tif", "Training/B1.tif", "Training/C2.tif", "Training/D2.tif")
DEFAULT_DT_LIST = (1, 2, 3, 4, 6, 8, 16, 32, 64)


def _open_stack(path: str | Path) -> np.ndarray:
    try:
        return np.asarray(tifffile.memmap(path))
    except Exception:
        return np.asarray(tifffile.imread(path))


def _stack_stats(stack: np.ndarray) -> tuple[float, float]:
    """全局 mean/std（分块累加，镜像 t4b_rgrid._stack_stats）。"""
    flat = np.asarray(stack).reshape(-1).astype(np.float64)
    total = 0.0
    total_sq = 0.0
    for i in range(0, flat.size, 10_000_000):
        block = flat[i : i + 10_000_000]
        total += block.sum()
        total_sq += (block**2).sum()
    mean = total / flat.size
    std = float(np.sqrt(max(total_sq / flat.size - mean**2, 0.0)))
    return float(mean), std


def _phase_views(patch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """y/x 对角相位双视图：(0,0) 与 (1,1)，形状 (B,Z,Y//2,X//2)。同 t4b_rgrid。"""
    return patch[:, :, 0::2, 0::2], patch[:, :, 1::2, 1::2]


class TemporalPatchPool:
    """确定性 (patch_t, patch_t+dt) 配对池：逐栈网格 + seeded 洗牌 + 预取。

    镜像 micro_probe.PatchPool（全部 dt 共用协议，唯一变量是 dt）与
    t4b_rgrid.PairDistanceDataset（z 范围扣除 dt）。
    """

    def __init__(
        self,
        data_root: str | Path,
        dt: int,
        patch_size: tuple[int, int, int],
        max_patches: int,
        overlap: float = 0.0,
        seed: int = 3407,
    ):
        self.dt = int(dt)
        z_patch, y_patch, x_patch = (int(v) for v in patch_size)
        self.patch_size = (z_patch, y_patch, x_patch)
        step = lambda size, patch: max(1, int(round(patch * (1.0 - overlap))))  # noqa: E731
        positions: list[tuple[int, int, int, int]] = []
        self.stacks: list[np.ndarray] = []
        self.means: list[float] = []
        self.stds: list[float] = []
        for img_idx, name in enumerate(TRAIN_STACKS):
            stack = _open_stack(Path(data_root) / name)
            t_len, y_len, x_len = stack.shape
            if min(t_len - z_patch - self.dt, y_len - y_patch, x_len - x_patch) < 0:
                raise ValueError(
                    f"栈 {name} 尺寸 {stack.shape} 不足 patch {self.patch_size} + dt={self.dt}"
                )
            mean, std = _stack_stats(stack)
            self.stacks.append(stack)
            self.means.append(mean)
            self.stds.append(std)
            z_max = t_len - z_patch - self.dt
            for z0 in range(0, z_max + 1, step(t_len, z_patch)):
                for y0 in range(0, y_len - y_patch + 1, step(y_len, y_patch)):
                    for x0 in range(0, x_len - x_patch + 1, step(x_len, x_patch)):
                        positions.append((img_idx, z0, y0, x0))
        if not positions:
            raise ValueError(f"dt={self.dt} 的配对网格为空")

        rng = np.random.default_rng(seed)
        order = rng.permutation(len(positions))
        positions = [positions[i] for i in order][: max(1, int(max_patches))]

        inps, tgts = [], []
        for img_idx, z0, y0, x0 in positions:
            stack = self.stacks[img_idx]
            mean, std = self.means[img_idx], self.stds[img_idx]
            a = np.asarray(
                stack[z0 : z0 + z_patch, y0 : y0 + y_patch, x0 : x0 + x_patch],
                dtype=np.float32,
            )
            b = np.asarray(
                stack[
                    z0 + self.dt : z0 + self.dt + z_patch,
                    y0 : y0 + y_patch,
                    x0 : x0 + x_patch,
                ],
                dtype=np.float32,
            )
            inps.append((a - mean) / std)
            tgts.append((b - mean) / std)
        self.inp = torch.from_numpy(np.stack(inps))  # (P,Z,Y,X)
        self.tgt = torch.from_numpy(np.stack(tgts))
        self.positions = positions

    def __len__(self) -> int:
        return int(self.inp.shape[0])

    def take(self, start: int, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        """循环取 n 个配对，返回 (inp, tgt) 各 (n,Z,Y,X)。"""
        total = len(self)
        idx = [(start + i) % total for i in range(n)]
        return self.inp[idx], self.tgt[idx]


def probe_dt(
    config: MicroProbeConfig,
    dt: int,
    pool: TemporalPatchPool,
    net_factory: NetFactory,
    device: torch.device,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    """跑一个 dt 的微型训练，返回 (特征 φ(Δt), 逐 batch 记录)。

    损失构造与 micro_probe._probe_family 逐行对应：
        out_i          = net(view_i)            # view_i ∈ {v1, v2}（输入相位视图）
        loss2neighbor  = mean_{i≠j} MSE(out_i, view_j)   # view_j ∈ {v3, v4}（邻域相位视图）
        loss_idt       = MSE(out1, out2)
    """
    _set_seed(config.seed)  # 每个 dt 重置：初始化/数据顺序一致，只换 dt
    net = net_factory().to(device)
    net.train()
    optimizer = torch.optim.Adam(net.parameters(), lr=float(config.lr), betas=(0.9, 0.999))
    mse = torch.nn.MSELoss()

    records: list[dict[str, float]] = []
    cursor = 0
    total_batches = int(config.epochs) * int(config.batches_per_epoch)
    for _step in range(total_batches):
        batch, batch_tgt = pool.take(cursor, int(config.batch_size))
        cursor += int(config.batch_size)
        batch = batch.to(device)
        batch_tgt = batch_tgt.to(device)
        v1, v2 = _phase_views(batch)
        v3, v4 = _phase_views(batch_tgt)
        ins = [v1.unsqueeze(1), v2.unsqueeze(1)]  # (N,1,Z,Y',X')
        nbs = [v3.unsqueeze(1), v4.unsqueeze(1)]
        outs = [net(x) for x in ins]

        loss_neighbor = (
            sum(mse(outs[i], nbs[j]) for i in range(2) for j in range(2)) / 4.0
        )
        loss_idt = mse(outs[0], outs[1])
        total_loss = loss_neighbor + float(config.weight_idt) * loss_idt

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=float(config.clip_gradients))
        optimizer.step()

        with torch.no_grad():
            out_d = [o.detach() for o in outs]
            in_d = [x.detach() for x in ins]
            loss_identity = float(sum(mse(out_d[i], in_d[i]) for i in range(2)) / 2)
            residual_corr = float(
                sum(_pearson(out_d[i] - in_d[i], in_d[i]) for i in range(2)) / 2
            )
            output_input_corr = float(sum(_pearson(out_d[i], in_d[i]) for i in range(2)) / 2)
            grad_out = float(_grad_mag(torch.cat(out_d, dim=0)))
            grad_in = float(_grad_mag(torch.cat(in_d, dim=0)))
            grad_ratio = grad_out / grad_in if grad_in > 1e-12 else 0.0
            var_ratios = []
            for i in range(2):
                v_in = float(in_d[i].var())
                var_ratios.append(float(out_d[i].var()) / v_in if v_in > 1e-12 else 0.0)
            cross_view = float(_pearson(out_d[0], out_d[1]))
            records.append(
                {
                    "loss_neighbor": float(loss_neighbor.detach()),
                    "loss_idt": float(loss_idt.detach()),
                    "loss_identity": loss_identity,
                    "residual_corr": residual_corr,
                    "output_input_corr": output_input_corr,
                    "output_gradient_ratio": grad_ratio,
                    "output_variance_ratio": float(np.mean(var_ratios)),
                    "cross_view_consistency": cross_view,
                }
            )

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


def _resolve_net_factory(config: MicroProbeConfig) -> NetFactory:
    def factory() -> torch.nn.Module:
        import sys

        from aces.paths import VALID_ROOT

        sys.path.insert(0, str(VALID_ROOT))
        from models.network import Network_CNR

        return Network_CNR(
            in_channels=1,
            out_channels=1,
            f_maps=int(config.base_features),
            n_groups=int(config.n_groups),
        )

    return factory


def apply_config_overrides(config: MicroProbeConfig, spec: str) -> MicroProbeConfig:
    """逗号分隔 k=v 覆盖（元组括号感知，同 micro_probe 的解析规则）。"""
    if not spec.strip():
        return config
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
    for tok in tokens:
        tok = tok.strip()
        if not tok or "=" not in tok:
            continue
        key, _, val = tok.partition("=")
        key, val = key.strip(), val.strip()
        valid = {f.name for f in MicroProbeConfig.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        if key not in valid:
            raise SystemExit(f"未知 config 项: {key}（可用: {sorted(valid)}）")
        current = getattr(config, key)
        if isinstance(current, tuple):
            cleaned = val.strip("()").strip()
            setattr(config, key, tuple(int(v.strip()) for v in cleaned.split(",")))
        elif isinstance(current, bool):
            setattr(config, key, val.lower() in ("1", "true", "yes"))
        elif isinstance(current, int):
            setattr(config, key, int(val))
        elif isinstance(current, float):
            setattr(config, key, float(val))
        else:
            setattr(config, key, val)
    return config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="时间轴 micro-probe（CIDC25 反向验证）")
    parser.add_argument("--data-root", required=True, help="AI4Life-CIDC25 数据根目录")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--dt-list",
        default=",".join(str(d) for d in DEFAULT_DT_LIST),
        help="逗号分隔的时间配对距离列表",
    )
    parser.add_argument(
        "--config",
        default="",
        help='逗号分隔覆盖项，如 "epochs=2,batches_per_epoch=100,patch_size=(64,128,128)"',
    )
    parser.add_argument("--device", default=None, help="覆盖 config.device")
    parser.add_argument("--net-factory", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    config = MicroProbeConfig()
    config = apply_config_overrides(config, args.config)
    if args.device:
        config.device = args.device
    dt_list = [int(d) for d in args.dt_list.split(",") if d.strip()]

    device = torch.device(config.device if torch.cuda.is_available() or "cuda" not in config.device else "cpu")
    net_factory = _resolve_net_factory(config) if args.net_factory is None else None
    if args.net_factory is not None:
        raise SystemExit("--net-factory 仅测试内注入使用，CLI 不支持")

    write_json(run_dir / "config.json", {**{k: list(v) if isinstance(v, tuple) else v for k, v in config.__dict__.items()}, "data_root": str(args.data_root), "dt_list": dt_list})
    print(f"[probe-t] config: {config}", flush=True)
    print(f"[probe-t] dt 列表: {dt_list}", flush=True)

    rows: list[dict[str, Any]] = []
    for dt in dt_list:
        pool = TemporalPatchPool(
            data_root=args.data_root,
            dt=dt,
            patch_size=config.patch_size,
            max_patches=int(config.max_patches),
            overlap=float(config.overlap),
            seed=int(config.seed),
        )
        print(f"[probe-t] dt={dt}: 配对池 {len(pool)} × {pool.patch_size}", flush=True)
        t0 = time.perf_counter()
        features, _records = probe_dt(config, dt, pool, net_factory, device)
        elapsed = time.perf_counter() - t0
        row = {"dt": dt, "elapsed_seconds": elapsed, **features}
        rows.append(row)
        write_json(run_dir / f"dt_{dt}.json", row)
        print(
            f"[probe-t] dt={dt}: L_N={features['loss_neighbor_mean']:.4f} "
            f"L_I={features['loss_identity_mean']:.4f} "
            f"cross={features['cross_view_consistency']:.4f} ({elapsed:.1f}s)",
            flush=True,
        )

    csv_path = run_dir / "micro_probe_temporal.csv"
    fieldnames = ["dt", "elapsed_seconds"] + [
        k for k in rows[0] if k not in ("dt", "elapsed_seconds")
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[probe-t] 完成 → {csv_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
