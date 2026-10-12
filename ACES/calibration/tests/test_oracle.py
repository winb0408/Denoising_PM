"""oracle 引擎单元测试（M2 Exp C）：训练循环 / U_oracle 评估 / 子集选择。

覆盖：
- psnr_db 数值（已知 MSE 的解析值、mse=0 → inf）；
- AVG-K reference 数学（reference_patch == 逐帧均值）；
- 评估网格（z 稀疏 stride + y/x 非重叠网格、维度校验）；
- 训练循环：tiny conv 注入（loss 有限且下降）、同 seed 逐位确定性 /
  异 seed 权重不同；
- 评估：恒等网络 ⇒ 增益≈0；完美去噪器（输出归一化参考）⇒ 巨大增益；
  eval_repeat_frames 子集；z 三分带分解计数；
- Runner 落盘：oracle.csv 列齐全、逐 family JSON / summary.json；
- probe-csv 子集选择（top4+bottom4+2mid、不足 10 全选）；
- OracleConfig 校验（2D patch_size 归一、eval_stride_z>=1）。
"""

import csv
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from calibration.micro_probe import PatchPool
from calibration.oracle import (
    ORACLE_CSV_COLUMNS,
    OracleConfig,
    OracleRunner,
    _train_family,
    eval_positions,
    evaluate_family,
    psnr_db,
    reference_patch,
    select_probe_subset,
)
from profiling.candidates import default_family_library


def _tiny_net_factory():
    """测试用 2 层 conv 网络（不依赖 VALID 仓库）。"""

    def factory() -> torch.nn.Module:
        return torch.nn.Sequential(
            torch.nn.Conv3d(1, 8, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv3d(8, 1, 3, padding=1),
        )

    return factory


def _synthetic_stack(
    k: int = 5, z: int = 24, y: int = 32, x: int = 32,
    noise: float = 0.5, seed: int = 11,
) -> np.ndarray:
    """合成 repeat stack [K,Z,Y,X]：共享平滑结构 + 逐帧独立高斯噪声。"""
    rng = np.random.default_rng(seed)
    zz, yy, xx = np.meshgrid(
        np.arange(z), np.arange(y), np.arange(x), indexing="ij"
    )
    structure = np.sin(zz * 0.5) * 0.5 + np.cos(yy * 0.4) * 0.3 + np.sin(xx * 0.3) * 0.2
    frames = structure[np.newaxis] + rng.standard_normal((k, z, y, x)) * noise
    return frames.astype(np.float32)


def _oracle_config(**overrides) -> OracleConfig:
    """CPU + 极小预算的测试配置。"""
    base = dict(
        epochs=2,
        batches_per_epoch=4,
        patch_size=(8, 16, 16),
        batch_size=2,
        lr=5e-3,
        seed=3407,
        device="cpu",
        base_features=8,
        n_groups=8,
        max_train_patches=8,
        eval_stride_z=8,
    )
    base.update(overrides)
    return OracleConfig(**base)


def _library():
    return {f.family_id: f for f in default_family_library()}


class TestPsnrAndReference(unittest.TestCase):
    """PSNR 数值与 AVG-K 参考数学。"""

    def test_psnr_known_mse(self) -> None:
        """pred = ref + c ⇒ PSNR = 10·log10(range²/c²)（range = 99.9%−0.1% 分位差）。"""
        ref = np.linspace(0.0, 10.0, 4096, dtype=np.float64).reshape(16, 16, 16)
        rng = np.random.default_rng(0)
        rng.shuffle(ref.reshape(-1))  # 打乱后分位差仍由值域决定
        c = 0.25
        pred = ref + c
        data_range = float(np.percentile(ref, 99.9) - np.percentile(ref, 0.1))
        expected = 10.0 * math.log10((max(data_range, 1e-6) ** 2) / (c * c))
        self.assertAlmostEqual(psnr_db(pred, ref), expected, places=8)

    def test_psnr_zero_mse_inf(self) -> None:
        """完美预测（mse=0）→ +inf；形状不匹配报错。"""
        arr = np.random.default_rng(1).standard_normal((4, 4, 4))
        self.assertEqual(psnr_db(arr, arr), math.inf)
        with self.assertRaises(ValueError):
            psnr_db(arr, arr[:-1])

    def test_reference_patch_is_frame_mean(self) -> None:
        """AVG-K reference == K 帧同区域的逐 voxel 均值（float32）。"""
        stack = _synthetic_stack(k=6, z=12, y=20, x=20, seed=3)
        got = reference_patch(stack, 2, 4, 6, (6, 8, 10))
        expected = stack[:, 2:8, 4:12, 6:16].astype(np.float64).mean(axis=0)
        np.testing.assert_allclose(got, expected.astype(np.float32), rtol=0, atol=1e-5)


class TestEvalGrid(unittest.TestCase):
    """评估网格：z 稀疏 stride、y/x 非重叠。"""

    def test_positions_grid(self) -> None:
        zs, ys, xs = eval_positions((4, 20, 40, 50), (8, 16, 16), 4)
        self.assertEqual(zs, [0, 4, 8, 12])  # (20−8)//4+1 个稀疏 z 起点
        self.assertEqual(ys, [0, 16])  # 非重叠：余量 40−32=8 丢弃
        self.assertEqual(xs, [0, 16, 32])  # 余量 50−48=2 丢弃

    def test_positions_requires_repeat_stack(self) -> None:
        with self.assertRaises(ValueError):
            eval_positions((20, 40, 50), (8, 16, 16), 4)


class TestTraining(unittest.TestCase):
    """训练循环（tiny conv 注入，不依赖 VALID）。"""

    def test_train_loss_finite_and_decreases(self) -> None:
        """loss 全有限，且末 epoch 均值低于首 epoch（tiny conv 可学）。"""
        volume = _synthetic_stack()
        config = _oracle_config(epochs=5, batches_per_epoch=8)
        pool = PatchPool(volume, config.patch_size, config.max_train_patches, seed=config.seed)
        family = _library()["qmask_xy"]
        net, records = _train_family(config, family, pool, _tiny_net_factory(), torch.device("cpu"))
        self.assertEqual(len(records), config.epochs * config.batches_per_epoch)
        for r in records:
            for key in ("loss_neighbor", "loss_idt", "total_loss"):
                self.assertTrue(math.isfinite(float(r[key])), f"{key}={r[key]} 非有限")
        first = np.mean([r["total_loss"] for r in records if int(r["epoch"]) == 1])
        last = np.mean([r["total_loss"] for r in records if int(r["epoch"]) == config.epochs])
        self.assertLess(float(last), float(first), "训练 loss 应下降")

    def train_weights(self, seed: int) -> dict:
        volume = _synthetic_stack()
        config = _oracle_config(seed=seed, epochs=2, batches_per_epoch=4)
        pool = PatchPool(volume, config.patch_size, config.max_train_patches, seed=config.seed)
        family = _library()["qmask_xy"]
        net, _records = _train_family(config, family, pool, _tiny_net_factory(), torch.device("cpu"))
        return {k: v.clone() for k, v in net.state_dict().items()}

    def test_seed_determinism_bitwise(self) -> None:
        """同 seed 两次训练 → 权重逐位一致（每 family seed 重置的协议确定性）。"""
        w1 = self.train_weights(3407)
        w2 = self.train_weights(3407)
        self.assertEqual(set(w1), set(w2))
        for key in w1:
            self.assertTrue(torch.equal(w1[key], w2[key]), f"权重 {key} 不逐位一致")

    def test_different_seed_changes_weights(self) -> None:
        w1 = self.train_weights(3407)
        w2 = self.train_weights(1234)
        diff = any(not torch.equal(w1[key], w2[key]) for key in w1)
        self.assertTrue(diff, "异 seed 初始化应产生不同权重")


class _IdentityNet(torch.nn.Module):
    """恒等去噪器：原样返回输入（denorm(raw)≈raw ⇒ 增益≈0）。"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class _ConstantNet(torch.nn.Module):
    """完美去噪器：无视输入，输出固定（归一化域的参考 patch）。"""

    def __init__(self, out: torch.Tensor):
        super().__init__()
        self.register_buffer("out", out.clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out.expand_as(x)


class TestEvaluateFamily(unittest.TestCase):
    """evaluate_family：增益语义 / 帧子集 / 分带分解。"""

    def _eval(self, net, volume, config, **pool_kw):
        pool = PatchPool(
            volume, config.patch_size,
            pool_kw.get("max_patches", config.max_train_patches),
            seed=config.seed,
        )
        return evaluate_family(net, volume, pool, config, torch.device("cpu"))

    def test_identity_net_gain_zero(self) -> None:
        """denoised==raw ⇒ U_oracle 增益逐 patch 为 0（反归一化浮点误差级）。"""
        volume = _synthetic_stack()
        config = _oracle_config()
        summary = self._eval(_IdentityNet(), volume, config)
        self.assertTrue(math.isfinite(summary["psnr_raw_mean"]))
        self.assertLess(abs(summary["u_oracle_gain"]), 1e-3)
        self.assertEqual(summary["n_eval_patches"], 3 * 2 * 2 * 5)  # 12 位置（z stride 8 → zs=[0,8,16]）× K=5 帧

    def test_perfect_denoiser_large_gain(self) -> None:
        """单位置网格 + 输出参考本身 ⇒ psnr_denoised 巨大、增益远大于 0。"""
        # 单 patch 体积（网格恰 1 位置），强噪声使 raw PSNR ~20 dB
        volume = _synthetic_stack(k=6, z=8, y=8, x=8, noise=2.0, seed=5)
        config = _oracle_config(patch_size=(8, 8, 8), eval_stride_z=8, eval_repeat_frames=[0])
        pool = PatchPool(volume, config.patch_size, config.max_train_patches, seed=config.seed)
        ref = reference_patch(volume, 0, 0, 0, config.patch_size)
        net = _ConstantNet(torch.from_numpy((ref - pool.mean) / pool.std))
        summary = evaluate_family(net, volume, pool, config, torch.device("cpu"))
        self.assertEqual(summary["n_eval_patches"], 1)
        self.assertGreater(summary["psnr_denoised_mean"], 60.0)  # 反归一化误差级 MSE
        self.assertGreater(summary["u_oracle_gain"], 30.0)

    def test_eval_repeat_frames_subset(self) -> None:
        """只评指定 repeat 帧：eval_frames / patch 计数随帧数缩放。"""
        volume = _synthetic_stack()
        config = _oracle_config(eval_repeat_frames=[1, 2])
        summary = self._eval(_IdentityNet(), volume, config)
        self.assertEqual(summary["eval_frames"], [1, 2])
        self.assertEqual(summary["n_eval_patches"], 3 * 2 * 2 * 2)
        # 越界帧报错
        config_bad = _oracle_config(eval_repeat_frames=[0, 9])
        with self.assertRaises(ValueError):
            self._eval(_IdentityNet(), volume, config_bad)

    def test_band_breakdown(self) -> None:
        """z 三分带：每带 patch 数与总体计数 / 手工网格一致。"""
        volume = _synthetic_stack(k=5, z=24, y=32, x=32)
        config = _oracle_config()  # stride_z=8, patch (8,16,16) → zs=[0,8,16]
        summary = self._eval(_IdentityNet(), volume, config)
        zs, ys, xs = eval_positions(volume.shape, config.patch_size, config.eval_stride_z)
        # z_center = z0+4 → 4/12/20 恰落 shallow/mid/deep 三带
        per_z0 = len(ys) * len(xs) * 5
        for band_idx, name in enumerate(("shallow", "mid", "deep")):
            expected_n = per_z0 * sum(1 for z0 in zs if (z0 + 4) // 8 == band_idx)
            self.assertEqual(summary["bands"][name]["n_eval_patches"], expected_n, name)
        self.assertEqual(
            summary["n_eval_patches"],
            sum(summary["bands"][b]["n_eval_patches"] for b in ("shallow", "mid", "deep")),
        )
        self.assertEqual(len(summary["per_z"]), len(zs))


class TestRunnerOutputs(unittest.TestCase):
    """OracleRunner 端到端（tiny conv）：CSV/JSON 列与行数。"""

    def test_runner_writes_csv_and_json(self) -> None:
        volume = _synthetic_stack()
        config = _oracle_config(
            epochs=1, batches_per_epoch=2, max_train_patches=4, max_eval_patches=6
        )
        families = [_library()["qmask_xy"], _library()["axis_x_d1"]]
        with tempfile.TemporaryDirectory() as tmp:
            rows = OracleRunner(config, net_factory=_tiny_net_factory()).run(
                volume, families, Path(tmp) / "oracle_run"
            )
            self.assertEqual(len(rows), 2)
            for row in rows:
                for col in ORACLE_CSV_COLUMNS:
                    self.assertIn(col, row, f"行缺列 {col}")
                self.assertTrue(math.isfinite(float(row["u_oracle_gain"])))
                self.assertEqual(row["n_eval_patches"], 6)  # max_eval_patches 截断
                self.assertEqual(row["epochs"], 1)

            csv_path = Path(tmp) / "oracle_run" / "oracle.csv"
            self.assertTrue(csv_path.exists())
            with csv_path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(reader.fieldnames, ORACLE_CSV_COLUMNS)
                csv_rows = list(reader)
            self.assertEqual(len(csv_rows), 2)
            self.assertEqual(csv_rows[0]["family_id"], "qmask_xy")

            for fam in families:
                payload = json.loads(
                    (Path(tmp) / "oracle_run" / "families" / f"{fam.family_id}.json").read_text("utf-8")
                )
                self.assertEqual(payload["event"], "oracle_family")
                self.assertIn("train_loss_curve", payload)
                self.assertIn("u_oracle_gain", payload["eval"])
                self.assertIn("bands", payload["eval"])
            summary = json.loads(
                (Path(tmp) / "oracle_run" / "families" / "summary.json").read_text("utf-8")
            )
            self.assertEqual(summary["n_families"], 2)


class TestProbeSubset(unittest.TestCase):
    """probe-csv 子集选择（top4+bottom4+2mid）。"""

    def _write_csv(self, tmp: str, u_values: dict[str, float]) -> Path:
        path = Path(tmp) / "micro_probe.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["family_id", "u_probe"])
            writer.writeheader()
            for fid, u in u_values.items():
                writer.writerow({"family_id": fid, "u_probe": str(u)})
        return path

    def test_subset_top_bottom_middle(self) -> None:
        """12 个 family → top4 + bottom4 + 2 个正中，u_probe 降序返回。"""
        u_values = {f"fam{i:02d}": float(i) for i in range(12)}  # u_probe: 0..11
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_csv(tmp, u_values)
            selected = select_probe_subset(path)
        self.assertEqual(selected, ["fam11", "fam10", "fam09", "fam08",  # top4
                                    "fam06", "fam05",  # 正中 2 个（12 个的 mid）
                                    "fam03", "fam02", "fam01", "fam00"])  # bottom4
        self.assertEqual(len(selected), 10)

    def test_subset_small_library_all(self) -> None:
        """family 数不足 10 → 全选（保持 u_probe 降序）。"""
        u_values = {"a": 0.3, "b": -0.1, "c": 0.9, "d": 0.0, "e": 0.5}
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_csv(tmp, u_values)
            selected = select_probe_subset(path)
        self.assertEqual(selected, ["c", "e", "a", "d", "b"])

    def test_subset_bad_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.csv"
            path.write_text("family_id,other\nfam0,1\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                select_probe_subset(path)


class TestOracleConfig(unittest.TestCase):
    """配置校验。"""

    def test_patch_size_2d_normalized(self) -> None:
        config = OracleConfig(patch_size=(16, 16))
        self.assertEqual(tuple(config.patch_size), (1, 16, 16))

    def test_eval_stride_z_validation(self) -> None:
        with self.assertRaises(ValueError):
            OracleConfig(eval_stride_z=0)
        with self.assertRaises(ValueError):
            OracleConfig(patch_size=(0, 4, 4))


if __name__ == "__main__":
    unittest.main()
