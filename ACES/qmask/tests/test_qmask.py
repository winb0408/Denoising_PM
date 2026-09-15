import unittest

import torch

from qmask.geometry import derive_qmask, QMaskGeometry, AxisGeometry
from qmask.sampling import (_allowed_corners, qmask_sampler, space_to_depth_multi,
                            TETRA_ROWS)


def _row(axis, d, n, s, q):
    return {
        "depth_bin": "global", "axis": axis, "distance_px": str(d),
        "distance_um": str(d), "region": "foreground", "noise_source": "repeat",
        "N": str(n), "S": str(s), "Q": str(q),
    }


class TestGeometry(unittest.TestCase):
    def test_monotonic_decay_chooses_min(self):
        rows = [
            _row("x", 1, 0.96, 0.97, 0.93),
            _row("x", 2, 0.99, 0.80, 0.79),
            _row("y", 1, 0.90, 0.90, 0.81),
            _row("y", 2, 0.96, 0.70, 0.67),
            _row("z", 1, 0.95, 0.90, 0.855),
            _row("z", 2, 0.97, 0.60, 0.58),
        ]
        g = derive_qmask(rows, region="foreground", noise_source="repeat",
                         tau_n=0.95, tau_s=0.85, tau_q=0.60)
        self.assertTrue(g.axes["x"].participate)
        self.assertEqual(g.axes["x"].d_min, 1)
        self.assertEqual(g.axes["x"].d_max, 1)
        self.assertFalse(g.axes["y"].participate)
        self.assertEqual(g.axes["y"].d_min, 2)
        self.assertTrue(g.axes["z"].participate)

    def test_promotion_ensures_two_axes(self):
        rows = [
            _row("x", 1, 0.96, 0.97, 0.93),
            _row("y", 1, 0.80, 0.90, 0.72),
            _row("z", 1, 0.80, 0.90, 0.72),
        ]
        g = derive_qmask(rows, region="foreground", noise_source="repeat",
                         tau_n=0.95, tau_s=0.85, tau_q=0.60)
        eff = g.effective_participate()
        self.assertGreaterEqual(sum(1 for v in eff.values() if v), 2)


class TestSampling(unittest.TestCase):
    def _geo(self, participate):
        g = QMaskGeometry(source="test", region="foreground", noise_source="repeat",
                          tau_n=0.95, tau_s=0.85, tau_q=0.60)
        for axis in ("z", "y", "x"):
            g.axes[axis] = AxisGeometry(
                axis=axis, participate=participate[axis], reason="test",
                d_min=1, d_max=1, n_d1=1.0, s_d1=1.0, q_d1=1.0,
            )
        g.fallback_order = ["x", "y", "z"]
        return g

    def test_z_off_preserves_z(self):
        g = self._geo({"z": False, "y": True, "x": True})
        img = torch.arange(1 * 8 * 8 * 8, dtype=torch.float32).reshape(1, 8, 8, 8)
        views, meta = qmask_sampler(img, g, seed=0)
        self.assertEqual(len(views), 4)
        for v in views:
            self.assertEqual(v.shape[0], 1)
            self.assertEqual(v.shape[1], 8)  # z preserved
            self.assertEqual(v.shape[2], 4)
            self.assertEqual(v.shape[3], 4)
        self.assertEqual(meta.mode, "qmask")

    def test_three_axis_on_halves_dims(self):
        g = self._geo({"z": True, "y": True, "x": True})
        img = torch.zeros((1, 8, 8, 8))
        views, _ = qmask_sampler(img, g, seed=0)
        self.assertEqual(len(views), 4)
        for v in views:
            self.assertEqual(v.shape, (1, 4, 4, 4))

    def test_deterministic_same_seed(self):
        g = self._geo({"z": True, "y": True, "x": True})
        img = torch.randn(1, 8, 8, 8)
        a, _ = qmask_sampler(img, g, seed=3)
        b, _ = qmask_sampler(img, g, seed=3)
        for va, vb in zip(a, b):
            self.assertTrue(torch.equal(va, vb))

    def test_allowed_corners_counts(self):
        corners, block = _allowed_corners({"z": False, "y": True, "x": True})
        self.assertEqual(block, (1, 2, 2))
        self.assertEqual(len(corners), 4)

    def test_space_to_depth_channel_order(self):
        img = torch.zeros((1, 2, 2, 2))
        for z in range(2):
            for y in range(2):
                for x in range(2):
                    img[0, z, y, x] = z * 4 + y * 2 + x
        s2d, _ = space_to_depth_multi(img, (2, 2, 2))
        self.assertEqual(s2d.shape, (1, 1, 1, 1, 8))
        for idx in range(8):
            z, y, x = idx // 4, (idx % 4) // 2, idx % 2
            self.assertEqual(float(s2d[0, 0, 0, 0, idx]), float(z * 4 + y * 2 + x))


if __name__ == "__main__":
    unittest.main()
