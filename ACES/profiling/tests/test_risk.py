"""Pair-risk field 的 sanity 测试（修正方案 §42 Test 3/4/5/7 及候选库）。"""

import unittest

import numpy as np
from scipy import ndimage

from profiling.candidates import (
    CandidateFamily,
    CandidateOffset,
    default_family_library,
    default_offset_library,
)
from profiling.risk import (
    NoiseRiskEstimator,
    ShortcutRiskEstimator,
    StructureRiskEstimator,
)
from profiling.risk_field import family_summary

RNG = np.random.default_rng(3407)


def _smooth_structure(shape=(16, 32, 32)) -> np.ndarray:
    """平滑随机结构 S（模拟 neuron morphology 的低频成分）。"""
    raw = RNG.standard_normal(shape).astype(np.float32)
    return ndimage.gaussian_filter(raw, sigma=3.0).astype(np.float32)


class TestCandidateLibrary(unittest.TestCase):
    def test_offset_um(self):
        cand = CandidateOffset((1, 2, 3), spacing_um=(4.0, 0.5, 0.25))
        self.assertEqual(cand.offset_um, (4.0, 1.0, 0.75))
        self.assertIsNone(CandidateOffset((1, 2, 3)).offset_um)

    def test_offset_library_size_and_unique(self):
        library = default_offset_library()
        offsets = [cand.offset_zyx for cand in library]
        self.assertEqual(len(offsets), len(set(offsets)), "offset 库存在重复")
        self.assertTrue(20 <= len(offsets) <= 30, f"库大小 {len(offsets)} 不在 20-30")
        for offset in offsets:
            self.assertNotEqual(offset, (0, 0, 0))

    def test_family_pairwise_toy(self):
        # 手算 toy family 的 pairwise Δ_ij = δ_j − δ_i（i<j）
        family = CandidateFamily("toy", [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1)])
        deltas = set(family.pairwise_offsets())
        self.assertEqual(deltas, {(0, 0, 1), (0, 1, 0), (0, 1, 1), (0, 1, -1)})

    def test_family_library_shape(self):
        families = default_family_library()
        self.assertGreaterEqual(len(families), 10)
        for family in families:
            self.assertEqual(len(family.offsets_zyx), 4, f"{family.family_id} 不是 4 个 view")
            self.assertTrue(family.pairwise_offsets(), f"{family.family_id} 无 pairwise offset")


class TestNoiseRiskMonotonic(unittest.TestCase):
    """§42 Test 3：人为生成相关噪声时 R_N 应升高。"""

    OFFSET = (0, 0, 2)

    def _stack(self, correlated_noise: bool) -> np.ndarray:
        """构造 repeat stack：correlated=True 时每个 repeat 的噪声在空间上
        相关（高斯平滑噪声），ρ_N(Δ) 应显著高于白噪声情形。

        注意：跨 repeat 完全共享的噪声会被 Ȳ 消掉（等价于结构），
        因此“相关噪声”这里指 repeat 残差的空间自相关。
        """
        structure = _smooth_structure()
        k = 6
        repeats = []
        for _ in range(k):
            white = RNG.standard_normal(structure.shape).astype(np.float32)
            noise = ndimage.gaussian_filter(white, sigma=2.0) if correlated_noise else white
            repeats.append(structure + noise)
        return np.stack(repeats, axis=0)

    def test_correlated_noise_higher_rn(self):
        est_ind = NoiseRiskEstimator().fit(self._stack(correlated_noise=False))
        est_corr = NoiseRiskEstimator().fit(self._stack(correlated_noise=True))
        rn_ind = est_ind.risk(self.OFFSET)["rn"]
        rn_corr = est_corr.risk(self.OFFSET)["rn"]
        self.assertLess(rn_ind, 0.3, f"独立噪声的 R_N 应接近 0，得到 {rn_ind:.4f}")
        self.assertGreater(rn_corr, 0.5, f"空间相关噪声的 R_N 应显著更高，得到 {rn_corr:.4f}")

    def test_zero_offset_independent_noise(self):
        # Δ=0 时独立噪声残差与自身完全相关 → R_N≈1
        est = NoiseRiskEstimator().fit(self._stack(correlated_noise=False))
        self.assertGreater(est.risk((0, 0, 0))["rn"], 0.99)


class TestStructureRiskShift(unittest.TestCase):
    """§42 Test 4：人为平移（错位）结构后 R_S 应升高。"""

    def _striped_volume(self) -> np.ndarray:
        # 沿 x 方向周期 8 的条纹结构：对齐 offset=(0,0,8) 时 S(r)=S(r+Δ)
        x = np.arange(32, dtype=np.float32)
        profile = np.sin(2.0 * np.pi * x / 8.0)
        volume = np.broadcast_to(profile[None, None, :], (16, 32, 32)).copy()
        return volume.astype(np.float32)

    def test_shifted_structure_higher_rs(self):
        est = StructureRiskEstimator(sigma=0.5).fit(self._striped_volume())
        rs_aligned = est.risk((0, 0, 8))["rs"]    # 周期对齐
        rs_shifted = est.risk((0, 0, 4))["rs"]    # 半周期错位
        self.assertLess(rs_aligned, 0.4, f"对齐结构的 R_S 应低，得到 {rs_aligned:.4f}")
        self.assertGreater(rs_shifted, rs_aligned + 0.2,
                           f"错位结构 R_S 应显著更高: {rs_shifted:.4f} vs {rs_aligned:.4f}")

    def test_components_in_unit_range(self):
        est = StructureRiskEstimator().fit(_smooth_structure())
        parts = est.risk((1, 1, 0))
        for key in ("rs", "rs_s1", "rs_s3"):
            self.assertTrue(0.0 <= parts[key] <= 1.0, f"{key}={parts[key]} 超出 [0,1]")
        self.assertEqual(parts["rs_ssim"], 0.5)  # stub 常数


class TestShortcutRisk(unittest.TestCase):
    """§42 Test 5：复制相同图像（Δ=0）时 R_I ≈ 1。"""

    def test_identity_copies_ri_near_one(self):
        volume = _smooth_structure()
        # 完全相同的复制：nMSE(0)=0, ρ_Y(0)=1 → R_I = exp(0)*1 = 1
        est = ShortcutRiskEstimator().fit(volume)
        ri = est.risk((0, 0, 0))["ri"]
        self.assertGreater(ri, 0.999, f"Δ=0 的 R_I 应 ≈1，得到 {ri:.6f}")

    def test_far_offset_lower_ri(self):
        volume = _smooth_structure()
        est = ShortcutRiskEstimator().fit(volume)
        ri_near = est.risk((0, 0, 1))["ri"]
        ri_far = est.risk((4, 8, 8))["ri"]
        self.assertLess(ri_far, ri_near, "远 offset 的 shortcut 风险应更低")

    def test_ri_unit_range(self):
        volume = _smooth_structure()
        est = ShortcutRiskEstimator().fit(volume)
        for offset in [(0, 0, 1), (1, 0, 0), (0, 2, 2), (2, 1, 0)]:
            ri = est.risk(offset)["ri"]
            self.assertTrue(0.0 <= ri <= 1.0, f"R_I({offset})={ri} 超出 [0,1]")


class TestFamilyAggregation(unittest.TestCase):
    """§42 Test 7：pairwise risk → family risk 的 mean/max 聚合正确性。"""

    def test_mean_max_on_toy_family(self):
        family = CandidateFamily("toy", [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1)])
        deltas = family.pairwise_offsets()
        # 构造确定性 toy 风险：rn=|dx|/4, rs=|dy|/4, ri=(|dz|+1)/8, r_base=和
        per_offset = {}
        for delta in deltas:
            dz, dy, dx = (abs(v) for v in delta)
            row = {
                "rn": dx / 4.0,
                "rn_star": dx / 4.0,
                "rs": dy / 4.0,
                "ri": (dz + 1) / 8.0,
                "r_base": dx / 4.0 + dy / 4.0 + (dz + 1) / 8.0,
            }
            per_offset[delta] = row
        summary = family_summary(family, per_offset)
        self.assertEqual(summary["family_id"], "toy")
        self.assertEqual(summary["n_pairs"], len(deltas))
        self.assertAlmostEqual(summary["rn_mean"], float(np.mean([per_offset[d]["rn"] for d in deltas])))
        self.assertAlmostEqual(summary["rn_max"], 0.25)      # |dx| 最大为 1
        self.assertAlmostEqual(summary["rs_max"], 0.25)      # |dy| 最大为 1
        self.assertAlmostEqual(summary["ri_max"], 0.125)     # |dz| 最大为 0 → (0+1)/8
        expected_worst = max(per_offset[d]["r_base"] for d in deltas)
        self.assertAlmostEqual(summary["r_base_max"], expected_worst)

    def test_missing_pair_raises(self):
        family = CandidateFamily("toy", [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1)])
        with self.assertRaises(KeyError):
            family_summary(family, {(0, 0, 1): {"rn": 0.0, "rn_star": 0.0, "rs": 0.0, "ri": 0.0, "r_base": 0.0}})


if __name__ == "__main__":
    unittest.main()
