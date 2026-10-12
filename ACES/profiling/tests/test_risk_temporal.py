"""risk_field_temporal 的合成数据 sanity 测试（时间轴 R_N/R_S/R_I）。

设计（与 test_risk.py 的 §42 风格一致）：
  a) 静态结构 + iid 噪声 → R_N 低（gt 与 proxy 两档来源都验证）；
  b) 时间 AR(1) 噪声 → R_N(Δt=1) 显著升高；
  c) 静态结构 → R_S 对所有 Δt 都低；结构随时间漂移 → R_S 随 Δt 上升；
  d) 静态无噪序列 → R_I ≈ 1（identity shortcut 的极端情形）；
  e) ShortcutRiskEstimator 的 z 轴映射对照：offset (Δt,0,0) 等价于手动
     取帧对 (Y_t, Y_{t+Δt})；
  f) CLI 端到端：tif 输入 → csv/json 输出。
"""

import argparse
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage

from profiling.risk import ShortcutRiskEstimator
from profiling.risk_field_temporal import (
    CSV_FIELDNAMES,
    compute_temporal_table,
    run,
)

RNG = np.random.default_rng(3407)
T, Y, X = 40, 32, 32


def _smooth_frame() -> np.ndarray:
    """平滑随机结构（模拟 neuron morphology 低频成分），std 归一到 ~10。"""
    raw = RNG.standard_normal((Y, X)).astype(np.float32)
    frame = ndimage.gaussian_filter(raw, sigma=3.0).astype(np.float32)
    return frame / (frame.std() + 1e-9) * 10.0


def _ar1_frames(base: np.ndarray, phi: float, noise_scale: float) -> np.ndarray:
    """静态结构 base + 时间 AR(1) 噪声（phi=0 → iid 白噪声）。"""
    frames = []
    prev = np.zeros_like(base)
    for _ in range(T):
        white = RNG.standard_normal((Y, X)).astype(np.float32)
        if phi > 0.0:
            prev = phi * prev + np.sqrt(1.0 - phi * phi) * white
            noise = prev
        else:
            noise = white
        frames.append(base + noise_scale * noise)
    return np.stack(frames, axis=0).astype(np.float32)


def _static_gt(base: np.ndarray) -> np.ndarray:
    return np.repeat(base[None, ...], T, axis=0).astype(np.float32)


def _drifting_gt(base: np.ndarray, speed: float = 1.0) -> np.ndarray:
    """结构沿 x 以 speed px/帧 平移（形态随时间漂移）。"""
    frames = [
        ndimage.shift(base, (0.0, speed * t), order=1, mode="nearest")
        for t in range(T)
    ]
    return np.stack(frames, axis=0).astype(np.float32)


class TestNoiseRiskTemporal(unittest.TestCase):
    """a/b：iid 噪声 R_N 低；AR(1) 时间相关噪声 R_N(1) 高。"""

    def test_iid_noise_low_rn_gt(self):
        base = _smooth_frame()
        stack = _ar1_frames(base, phi=0.0, noise_scale=1.0)
        rows, source = compute_temporal_table(stack, _static_gt(base), [1, 4])
        self.assertEqual(source, "gt")
        for row in rows:
            self.assertLess(row["rn"], 0.2, f"iid 噪声 R_N(Δt={row['dt']}) 应 < 0.2，得到 {row['rn']:.4f}")

    def test_iid_noise_low_rn_proxy(self):
        # proxy：N = Y − 时间均值。静态结构下时间均值 ≈ 结构 → 残差 ≈ iid 噪声
        base = _smooth_frame()
        stack = _ar1_frames(base, phi=0.0, noise_scale=1.0)
        rows, source = compute_temporal_table(stack, None, [1])
        self.assertEqual(source, "proxy")
        self.assertLess(rows[0]["rn"], 0.25,
                        f"proxy iid 噪声 R_N(1) 应 < 0.25，得到 {rows[0]['rn']:.4f}")

    def test_ar1_noise_high_rn(self):
        base = _smooth_frame()
        stack = _ar1_frames(base, phi=0.9, noise_scale=1.0)
        rows, _ = compute_temporal_table(stack, _static_gt(base), [1, 4])
        # ρ_N(1) ≈ phi = 0.9 → R_N(1) 显著高；Δt=4 时 ρ ≈ 0.9^4 ≈ 0.66 仍 > 0.3
        self.assertGreater(rows[0]["rn"], 0.5, f"AR(1) R_N(1) 应 > 0.5，得到 {rows[0]['rn']:.4f}")
        self.assertGreater(rows[1]["rn"], 0.3, f"AR(1) R_N(4) 应 > 0.3，得到 {rows[1]['rn']:.4f}")
        self.assertLess(rows[1]["rn"], rows[0]["rn"], "AR(1) R_N 应随 Δt 衰减")


class TestStructureRiskTemporal(unittest.TestCase):
    """c：静态结构 R_S 低；漂移结构 R_S 随 Δt 上升。"""

    def test_static_structure_low_rs_all_dt(self):
        base = _smooth_frame()
        gt = _static_gt(base)
        rows, _ = compute_temporal_table(gt.copy(), gt, [1, 8, 16])
        for row in rows:
            self.assertLess(row["rs"], 0.05,
                            f"静态结构 R_S(Δt={row['dt']}) 应 ≈ 0，得到 {row['rs']:.4f}")

    def test_static_structure_proxy_rs_low(self):
        # 无 GT：结构代理 = 逐帧平滑 Y，噪声残留应被 sigma=0.8 平滑部分压掉
        base = _smooth_frame()
        stack = _ar1_frames(base, phi=0.0, noise_scale=1.0)
        rows, _ = compute_temporal_table(stack, None, [1, 8])
        for row in rows:
            self.assertLess(row["rs"], 0.3,
                            f"静态结构(proxy) R_S(Δt={row['dt']}) 应 < 0.3，得到 {row['rs']:.4f}")

    def test_drifting_structure_raises_rs(self):
        base = _smooth_frame()
        gt = _drifting_gt(base, speed=1.0)
        rows, _ = compute_temporal_table(gt.copy(), gt, [1, 8])
        rs_small, rs_large = rows[0]["rs"], rows[1]["rs"]
        self.assertLess(rs_small, 0.3, f"漂移结构 R_S(1) 应低，得到 {rs_small:.4f}")
        self.assertGreater(rs_large, rs_small + 0.2,
                           f"漂移结构 R_S(8) 应显著更高: {rs_large:.4f} vs {rs_small:.4f}")


class TestShortcutRiskTemporal(unittest.TestCase):
    """d/e：静态无噪 R_I ≈ 1；z 轴映射与手动帧对一致。"""

    def test_static_noiseless_ri_near_one(self):
        base = _smooth_frame()
        stack = _static_gt(base)
        rows, _ = compute_temporal_table(stack.copy(), stack.copy(), [1, 16])
        for row in rows:
            self.assertGreater(row["ri"], 0.95,
                               f"静态无噪 R_I(Δt={row['dt']}) 应 ≈ 1，得到 {row['ri']:.4f}")

    def test_drifting_ri_decreases(self):
        base = _smooth_frame()
        stack = _drifting_gt(base, speed=1.0)
        rows, _ = compute_temporal_table(stack.copy(), stack.copy(), [1, 8])
        self.assertLess(rows[1]["ri"], rows[0]["ri"], "结构漂移时 R_I 应随 Δt 下降")

    def test_z_axis_mapping_matches_manual_pairs(self):
        # [T,Y,X] 当 [Z,Y,X] 喂给 ShortcutRiskEstimator，offset (dt,0,0)
        # 必须等价于手动取帧对 (Y_t, Y_{t+dt}) 的池化统计
        base = _smooth_frame()
        stack = _drifting_gt(base, speed=0.5) + 0.5 * RNG.standard_normal((T, Y, X)).astype(np.float32)
        est = ShortcutRiskEstimator(tau_i=1.0, stride=1).fit(stack)
        parts = est.risk((4, 0, 0))
        left = stack[: T - 4].astype(np.float64)
        right = stack[4:].astype(np.float64)
        diff = np.mean((left - right) ** 2) / (stack.astype(np.float64).var() + 1e-6)
        self.assertAlmostEqual(parts["ri_nmse"], diff, places=6)
        self.assertAlmostEqual(parts["ri_corr_y"], np.corrcoef(left.ravel(), right.ravel())[0, 1], places=6)


class TestCliEndToEnd(unittest.TestCase):
    """f：tif 输入 → metrics/risk_field_temporal.csv + .json。"""

    def test_main_writes_outputs(self):
        base = _smooth_frame()
        stack = _ar1_frames(base, phi=0.0, noise_scale=1.0)
        gt = _static_gt(base)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            noisy_tif = tmp_path / "noisy.tif"
            gt_tif = tmp_path / "gt.tif"
            run_dir = tmp_path / "run"
            tifffile.imwrite(str(noisy_tif), stack.astype(np.int16))
            tifffile.imwrite(str(gt_tif), gt.astype(np.int16))
            # main() 只返回退出码，摘要从 run() 取（同 risk_field 约定）
            summary = run(argparse.Namespace(
                stack=str(noisy_tif),
                gt=str(gt_tif),
                run_dir=str(run_dir),
                dt_list="1,4",
                stride=2,
                frame_crop=None,
                tau_i=1.0,
                structure_sigma=0.8,
                noise_level_label="synthetic",
            ))
            csv_path = run_dir / "metrics" / "risk_field_temporal.csv"
            json_path = run_dir / "metrics" / "risk_field_temporal.json"
            self.assertTrue(csv_path.exists())
            self.assertTrue(json_path.exists())
            lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(lines[0].split(","), CSV_FIELDNAMES)
            self.assertEqual(len(lines), 3, "应为表头 + 2 个 Δt 行")
            self.assertEqual(summary["noise_source"], "gt")
            self.assertEqual(summary["config"]["noise_level_label"], "synthetic")
            self.assertEqual(len(summary["rows"]), 2)


if __name__ == "__main__":
    unittest.main()
