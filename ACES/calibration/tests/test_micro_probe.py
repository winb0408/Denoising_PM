"""micro-probe 单元测试（M2）：family_sampler 逐值校验 + 特征/U_probe 数值性质。

覆盖：
- family_sampler：corner family / 自定义 offset family 的逐 voxel 手工索引
  对照、与 qmask.sampling.space_to_depth_multi 的等价性、形状一致、
  确定性（两次调用逐位相等）、2D patch 退化、patch 过小报错；
- micro_probe：注入 tiny 2 层 conv 的 net_factory（不依赖 VALID 仓库），
  合成体上跑 2ep×4batch，断言特征有限、z-score 均值≈0、U_probe 有限、
  CSV 落盘；compute_u_probe 的 R_I 重归一行为；load_risk_ri 两种格式。
"""

import csv
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from calibration.family_sampler import derive_block, sample_views
from calibration.micro_probe import (
    MicroProbeConfig,
    MicroProbeRunner,
    compute_u_probe,
    load_risk_ri,
    robust_zscore,
)
from profiling.candidates import CandidateFamily, default_family_library


def _make_patch(c: int, z: int, y: int, x: int, seed: int = 0) -> torch.Tensor:
    """确定性 patch：arange 平铺，保证手工索引可逐值对照。"""
    rng = np.random.default_rng(seed)
    return torch.from_numpy(rng.standard_normal((c, z, y, x)).astype(np.float32))


def _manual_view(patch: torch.Tensor, offset_zyx: tuple[int, int, int],
                 block_zyx: tuple[int, int, int]) -> torch.Tensor:
    """手工索引参照实现：view[c,iz,iy,ix] = P[c, δz+iz·bz, δy+iy·by, δx+ix·bx]。"""
    bz, by, bx = block_zyx
    dz, dy, dx = offset_zyx
    c_len, z_len, y_len, x_len = patch.shape
    n_z, n_y, n_x = z_len // bz, y_len // by, x_len // bx
    out = torch.empty((c_len, n_z, n_y, n_x), dtype=patch.dtype)
    for ic in range(c_len):
        for iz in range(n_z):
            for iy in range(n_y):
                for ix in range(n_x):
                    out[ic, iz, iy, ix] = patch[ic, dz + iz * bz, dy + iy * by, dx + ix * bx]
    return out


class TestFamilySampler(unittest.TestCase):
    """sample_views 的逐值 / 形状 / 确定性校验。"""

    def _check_against_manual(self, patch: torch.Tensor, family: CandidateFamily) -> None:
        block, norm_offsets = derive_block(family)
        views, meta = sample_views(patch, family)
        self.assertEqual(len(views), 4)
        self.assertEqual(meta["family_id"], family.family_id)
        self.assertEqual(tuple(meta["block_zyx"]), tuple(block))
        for view, delta in zip(views, norm_offsets):
            expected = _manual_view(patch, delta, block)
            self.assertTrue(torch.equal(view, expected),
                            f"view 不匹配: offset={delta} block={block}")
        for view in views[1:]:
            self.assertEqual(view.shape, views[0].shape)

    def test_corner_family_manual_indexing(self) -> None:
        """corner family（qmask_xy / valid_corners）逐 voxel 对照手工索引。"""
        patch = _make_patch(1, 6, 8, 8, seed=1)
        library = {f.family_id: f for f in default_family_library()}
        for fam_id in ("qmask_xy", "valid_corners", "qmask_xyz_tetra"):
            self._check_against_manual(patch, library[fam_id])

    def test_custom_offset_family_manual_indexing(self) -> None:
        """非对称自定义 offset family（span > 1 的混合距离变体）。"""
        family = CandidateFamily(
            "custom_mixed",
            [(0, 1, 2), (2, 0, 1), (1, 3, 0), (3, 2, 2)],
        )
        block, _ = derive_block(family)
        self.assertEqual(block, (4, 4, 3))
        patch = _make_patch(2, 16, 16, 16, seed=2)
        self._check_against_manual(patch, family)

    def test_equivalence_with_space_to_depth(self) -> None:
        """8-corner family 与 space_to_depth_multi 的 phase-split 逐值等价。"""
        from qmask.sampling import space_to_depth_multi

        patch = _make_patch(1, 8, 10, 12, seed=3)
        # valid_corners = 2×2×2 block 的 4 个真 corner；(dz,dy,dx) →
        # space_to_depth 通道序 (dz·by+dy)·bx+dx（block 全 2）。
        valid_family = CandidateFamily(
            "valid_corners", [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]
        )
        s2d, _ = space_to_depth_multi(patch, (2, 2, 2))  # (C, Z', Y', X', 8)
        views, _meta = sample_views(patch, valid_family)
        # 尺寸整除时本采样器（无 padding）与 s2d 的 coarse 站点数一致，逐值等价
        self.assertEqual(tuple(views[0].shape), tuple(s2d.shape[:4]))
        for view, delta in zip(views, valid_family.offsets_zyx):
            dz, dy, dx = delta
            channel = (dz * 2 + dy) * 2 + dx
            self.assertTrue(
                torch.equal(view, s2d[..., channel]),
                f"corner {delta} 与 space_to_depth 通道 {channel} 不等价",
            )

    def test_shapes_and_determinism(self) -> None:
        """两次调用逐位相等（无 RNG）；形状按 block 缩放。"""
        patch = _make_patch(1, 12, 12, 12, seed=4)
        family = CandidateFamily("diag_xyz", [(0, 0, 0), (1, 1, 1), (2, 2, 2), (3, 3, 3)])
        views_a, meta_a = sample_views(patch, family)
        views_b, meta_b = sample_views(patch, family)
        for va, vb in zip(views_a, views_b):
            self.assertTrue(torch.equal(va, vb))
        self.assertEqual(meta_a, meta_b)
        # block=(4,4,4) → view 形状 12//4 = 3 per axis
        self.assertEqual(tuple(views_a[0].shape), (1, 3, 3, 3))

    def test_2d_and_3d_conventions(self) -> None:
        """(Y,X) 2D patch 与 (Z,Y,X) 3D patch 的维度约定。"""
        family_xy = CandidateFamily("qmask_xy", [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1)])
        patch2d = _make_patch(1, 1, 8, 8, seed=5)[0, 0]  # (8,8)
        views2d, meta = sample_views(patch2d, family_xy)
        self.assertEqual(tuple(views2d[0].shape), (4, 4))
        self.assertEqual(meta["output_shape"], [4, 4])
        patch3d = _make_patch(1, 8, 8, 8, seed=6)[0]  # (8,8,8) 无通道
        views3d, meta3 = sample_views(patch3d, family_xy)
        self.assertEqual(tuple(views3d[0].shape), (8, 4, 4))

    def test_too_small_patch_raises(self) -> None:
        """patch 不足一个 coarse 站点时 fail-fast（z=3 < block=4）。"""
        family = CandidateFamily("diag_xyz", [(0, 0, 0), (1, 1, 1), (2, 2, 2), (3, 3, 3)])
        patch = _make_patch(1, 3, 10, 10, seed=7)  # z=3 → n_z = 3//4 = 0
        with self.assertRaises(ValueError):
            sample_views(patch, family)


def _tiny_net_factory():
    """测试用 2 层 conv 网络（不依赖 VALID 仓库）。"""

    def factory() -> torch.nn.Module:
        return torch.nn.Sequential(
            torch.nn.Conv3d(1, 8, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv3d(8, 1, 3, padding=1),
        )

    return factory


def _synthetic_volume(z: int = 8, y: int = 24, x: int = 24, seed: int = 11) -> np.ndarray:
    """合成噪声体：平滑结构 + 高斯噪声（保证非零梯度/方差）。"""
    rng = np.random.default_rng(seed)
    zz, yy, xx = np.meshgrid(
        np.arange(z), np.arange(y), np.arange(x), indexing="ij"
    )
    structure = np.sin(zz * 0.5) * 0.5 + np.cos(yy * 0.4) * 0.3 + np.sin(xx * 0.3) * 0.2
    noise = rng.standard_normal((z, y, x)) * 0.5
    return (structure + noise).astype(np.float32)


def _probe_config(**overrides) -> MicroProbeConfig:
    """CPU + 极小预算的测试配置。"""
    base = dict(
        epochs=2,
        batches_per_epoch=4,
        patch_size=(6, 12, 12),
        batch_size=2,
        lr=1e-3,
        seed=3407,
        device="cpu",
        max_patches=8,
        base_features=8,
        min_view_dim=0,  # 测试注入的 2 层小卷积无 kernel-8 约束，关闭护栏
    )
    base.update(overrides)
    return MicroProbeConfig(**base)


class TestMicroProbeFeatures(unittest.TestCase):
    """tiny conv + 合成体：特征有限性 / z-score / CSV 落盘。"""

    FAMILIES = ["qmask_xy", "valid_corners", "axis_x_d1", "sn2n_xy_diag", "diag_xyz"]

    def test_features_finite_and_zscore(self) -> None:
        volume = _synthetic_volume()
        library = {f.family_id: f for f in default_family_library()}
        families = [library[fid] for fid in self.FAMILIES]
        with tempfile.TemporaryDirectory() as tmp:
            runner = MicroProbeRunner(_probe_config(), net_factory=_tiny_net_factory())
            rows = runner.run(volume, families, Path(tmp) / "probe_run")
            self.assertEqual(len(rows), len(families))

            csv_path = Path(tmp) / "probe_run" / "micro_probe.csv"
            self.assertTrue(csv_path.exists())
            with csv_path.open(newline="", encoding="utf-8") as handle:
                csv_rows = list(csv.DictReader(handle))
            self.assertEqual(len(csv_rows), len(families))
            self.assertIn("u_probe", csv_rows[0])

            feature_keys = [
                "loss_neighbor_mean", "loss_neighbor_slope",
                "loss_identity_mean", "loss_identity_slope",
                "residual_corr", "output_input_corr",
                "output_gradient_ratio", "output_variance_ratio",
                "cross_view_consistency", "loss_stability",
            ]
            for row in rows:
                for key in feature_keys:
                    self.assertTrue(math.isfinite(float(row[key])),
                                    f"{row['family_id']}.{key}={row[key]} 非有限")
                self.assertTrue(math.isfinite(float(row["u_probe"])))

            # z-score 跨 family 归一：robust z 以 median 为中心（奇数个 family 时
            # median(z)=0 精确成立；均值受 outlier 影响可偏离，放宽为 |mean|<2.5）
            for row in rows:
                z_keys = [k for k in row if k.startswith("z_")]
                self.assertTrue(z_keys, "缺少 z_* 列")
                for key in feature_keys:
                    z_col = [r[f"z_{key}"] for r in rows if f"z_{key}" in r]
                    self.assertEqual(len(z_col), len(rows), f"z_{key} 应覆盖全部 family")
                    self.assertAlmostEqual(float(np.median(z_col)), 0.0, places=6,
                                           msg=f"z_{key} 中位数应为 0")
                    mean_z = float(np.mean(z_col))
                    self.assertLess(abs(mean_z), 2.5,
                                    f"z_{key} 均值 {mean_z:.3f} 偏离 0 过多")

            # 逐 family JSON 存在且含 loss 曲线
            for row in rows:
                fam_json = Path(tmp) / "probe_run" / "families" / f"{row['family_id']}.json"
                self.assertTrue(fam_json.exists())

    def test_determinism_same_seed(self) -> None:
        """同 seed 两次 run 的 L_N 逐位一致（协议确定性）。"""
        volume = _synthetic_volume()
        library = {f.family_id: f for f in default_family_library()}
        families = [library["qmask_xy"]]
        results = []
        with tempfile.TemporaryDirectory() as tmp:
            for _ in range(2):
                runner = MicroProbeRunner(_probe_config(), net_factory=_tiny_net_factory())
                rows = runner.run(volume, families, Path(tmp) / "run")
                results.append(rows[0]["loss_neighbor_mean"])
        self.assertEqual(results[0], results[1])


class TestUProbe(unittest.TestCase):
    """compute_u_probe / robust_zscore / load_risk_ri 的纯数值性质。"""

    def _rows(self, n: int = 5) -> list[dict]:
        rng = np.random.default_rng(0)
        rows = []
        for i in range(n):
            rows.append({
                "family_id": f"fam{i}",
                "loss_neighbor_mean": float(1.0 + 0.2 * rng.standard_normal()),
                "loss_identity_mean": float(0.8 + 0.1 * rng.standard_normal()),
                "residual_corr": float(0.3 * rng.standard_normal()),
                "output_gradient_ratio": float(1.0 + 0.1 * rng.standard_normal()),
                "output_variance_ratio": float(0.9 + 0.1 * rng.standard_normal()),
                "cross_view_consistency": float(0.5 + 0.2 * rng.standard_normal()),
                "loss_stability": float(abs(0.05 * rng.standard_normal()) + 1e-3),
            })
        return rows

    def test_robust_zscore_centered(self) -> None:
        values = [float(v) for v in np.linspace(0, 1, 9)]
        z = robust_zscore(values)
        self.assertAlmostEqual(float(np.median(z)), 0.0, places=6)
        self.assertLess(abs(float(np.mean(z))), 0.3)
        # 常数列 → 全 0（不产生 NaN/inf）
        z0 = robust_zscore([2.0, 2.0, 2.0])
        self.assertTrue(np.all(z0 == 0.0))

    def test_compute_u_probe_without_risk(self) -> None:
        rows = compute_u_probe(self._rows())
        for row in rows:
            self.assertTrue(math.isfinite(float(row["u_probe"])))
            self.assertNotIn("risk_ri", row)
        # 缺 R_I：权重 0.90 重归一后仍是加权平均（有限、跨 family 有差异）
        self.assertGreater(float(np.std([r["u_probe"] for r in rows])), 0.0)

    def test_compute_u_probe_with_risk_renormalized(self) -> None:
        rows_plain = compute_u_probe(self._rows())
        risk = {f"fam{i}": 0.1 * i for i in range(5)}
        rows_risk = compute_u_probe(self._rows(), risk_ri=risk)
        # R_I 项参与后排序/数值应改变（重归一 ≠ 原值）
        u_plain = [r["u_probe"] for r in rows_plain]
        u_risk = [r["u_probe"] for r in rows_risk]
        self.assertNotEqual(u_plain, u_risk)
        for row in rows_risk:
            self.assertIn("risk_ri", row)
            self.assertIn("z_risk_ri", row)
            self.assertTrue(math.isfinite(float(row["u_probe"])))
        # 部分 family 缺 R_I：该 family 该项省略且不崩
        risk_partial = {"fam0": 0.5}
        rows_partial = compute_u_probe(self._rows(), risk_ri=risk_partial)
        self.assertIn("risk_ri", rows_partial[0])
        self.assertNotIn("z_risk_ri", rows_partial[1])
        self.assertTrue(all(math.isfinite(float(r["u_probe"])) for r in rows_partial))

    def test_load_risk_ri_family_level(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "risk_families.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=["family_id", "n_pairs", "ri_mean", "ri_max"]
                )
                writer.writeheader()
                writer.writerow({"family_id": "qmask_xy", "n_pairs": "6", "ri_mean": "0.42", "ri_max": "0.5"})
                writer.writerow({"family_id": "axis_x_d1", "n_pairs": "6", "ri_mean": "0.11", "ri_max": "0.2"})
            risk = load_risk_ri(path)
            self.assertEqual(risk["qmask_xy"], 0.42)
            self.assertEqual(risk["axis_x_d1"], 0.11)

    def test_load_risk_ri_offset_level(self) -> None:
        """offset 级 risk_field.csv → 默认库 family 的 pairwise mean 聚合。"""
        # qmask_xy offsets [(0,0,0),(0,0,1),(0,1,0),(0,1,1)]，
        # pairwise Δ 去重后 = (0,0,1),(0,1,0),(0,1,1),(0,1,-1)
        family_id = "qmask_xy"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "risk_field.csv"
            fieldnames = ["offset_voxel_z", "offset_voxel_y", "offset_voxel_x", "ri"]
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                for (z, y, x), ri in [((0, 0, 1), 0.2), ((0, 1, 0), 0.4), ((0, 1, 1), 0.6), ((0, 1, -1), 0.8)]:
                    writer.writerow({
                        "offset_voxel_z": z, "offset_voxel_y": y,
                        "offset_voxel_x": x, "ri": ri,
                    })
            risk = load_risk_ri(path)
            self.assertIn(family_id, risk)
            expected = float(np.mean([0.2, 0.4, 0.6, 0.8]))
            self.assertAlmostEqual(risk[family_id], expected, places=6)


class TestConfigOverrides(unittest.TestCase):
    """apply_config_overrides 的括号内逗号（元组值）解析。"""

    def test_patch_size_tuple_parsed(self) -> None:
        from calibration.micro_probe import apply_config_overrides

        config = apply_config_overrides(
            MicroProbeConfig(), "epochs=1,patch_size=(96,96,96),max_patches=64"
        )
        self.assertEqual(config.epochs, 1)
        self.assertEqual(config.patch_size, (96, 96, 96))
        self.assertEqual(config.max_patches, 64)

    def test_invalid_token_fails(self) -> None:
        from calibration.micro_probe import apply_config_overrides

        with self.assertRaises(SystemExit):
            apply_config_overrides(MicroProbeConfig(), "epochs 1")


if __name__ == "__main__":
    unittest.main()
