"""evaluate_stage2 单元测试（P0 评估协议清理）。

覆盖：
- P0-6：SSIM 只在 all 行报 full 值，fg/bg 行 SSIM 为 NaN；
- P0-5：cnr_db 有限且为正（前景/背景均值差明显时）；
- P0-7：checkpoint_selection 元数据块随指标一起写入 JSON。
"""

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import tifffile
import torch

from aces.evaluate_stage2 import _derive_selected_epoch, _region_metrics, evaluate_model_on_patches


def _make_pair() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """构造小体积合成数据：pred 在前景抬亮、背景压暗，fg/bg 统计差异明显。"""
    rng = np.random.default_rng(0)
    target = rng.normal(0.5, 0.05, size=(12, 16, 16)).astype(np.float32)
    mask = np.zeros(target.shape, dtype=bool)
    mask[4:8, 4:12, 4:12] = True
    bg_mask = ~mask
    pred = target.copy()
    pred[mask] += 0.4
    pred[bg_mask] -= 0.05
    return pred.astype(np.float32), target, mask, bg_mask


class TestRegionMetrics(unittest.TestCase):
    def test_ssim_nan_outside_all_region(self):
        pred, target, mask, bg_mask = _make_pair()
        all_row = _region_metrics(pred, target, np.ones_like(mask), bg_mask=bg_mask, region="all")
        fg_row = _region_metrics(pred, target, mask, bg_mask=bg_mask, region="foreground")
        bg_row = _region_metrics(pred, target, ~mask, bg_mask=None, region="background")
        self.assertTrue(math.isfinite(all_row["ssim"]))
        self.assertTrue(math.isnan(fg_row["ssim"]))
        self.assertTrue(math.isnan(bg_row["ssim"]))

    def test_cnr_db_finite_positive(self):
        pred, target, mask, bg_mask = _make_pair()
        all_row = _region_metrics(pred, target, np.ones_like(mask), bg_mask=bg_mask, region="all")
        fg_row = _region_metrics(pred, target, mask, bg_mask=bg_mask, region="foreground")
        # fg 行：前景/背景均值差明显 → cnr_db 有限且为正（线性 CNR > 1）
        self.assertTrue(math.isfinite(fg_row["cnr_db"]))
        self.assertGreater(fg_row["cnr_db"], 0.0)
        # all 行：μS 覆盖全 patch（含背景），与背景均值差被稀释，只要求有限
        self.assertTrue(math.isfinite(all_row["cnr_db"]))
        # background 行无 bg_mask 参照 → cnr_db 为 NaN（与 snr/sbr 同策略）
        bg_row = _region_metrics(pred, target, ~mask, bg_mask=None, region="background")
        self.assertTrue(math.isnan(bg_row["cnr_db"]))


class _IdentityModel(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class _TinyDataset:
    """与 evaluate_model_on_patches 兼容的最小合成 dataset（单 patch、image_idx=0）。"""

    def __init__(self, shape: tuple[int, int, int]):
        self.z_patch, self.w_patch, self.h_patch = shape
        self.image_means = [0.0]
        self.image_stds = [1.0]
        rng = np.random.default_rng(1)
        self.patch = rng.normal(0.5, 0.05, size=shape).astype(np.float32)
        self.indices = [(0, 0, 0, 0)]

    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int):
        return torch.from_numpy(self.patch).unsqueeze(0), 0, 0, 0, 0


class TestEvaluateModelOnPatches(unittest.TestCase):
    def test_checkpoint_selection_written_and_ssim_policy(self):
        with tempfile.TemporaryDirectory(prefix="eval_stage2_smoke_") as tmp:
            tmp_path = Path(tmp)
            rng = np.random.default_rng(2)
            mean9 = rng.normal(0.5, 0.05, size=(12, 16, 16)).astype(np.float32)
            mean9_path = tmp_path / "mean9.tif"
            tifffile.imwrite(mean9_path, mean9)
            summary = evaluate_model_on_patches(
                model=_IdentityModel(),
                dataset=_TinyDataset((12, 16, 16)),
                device=torch.device("cpu"),
                output_dir=tmp_path,
                mean9_path=mean9_path,
                max_patches=1,
                checkpoint_path=tmp_path / "checkpoints" / "valid_epoch12.pth",
            )
            block = summary["checkpoint_selection"]
            self.assertEqual(block["selected_epoch"], 12)
            self.assertIn(block["selection_metric"], ("fg_psnr", "none"))
            self.assertIn(block["selection_dataset"], ("val_patches", "test_patches"))
            self.assertIn(block["selection_protocol"],
                          ("official_eval_final_weights", "best_checkpoint_revalidation"))
            data = json.loads((tmp_path / "metrics" / "eval_metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(data["checkpoint_selection"]["selected_epoch"], 12)
            self.assertTrue(math.isfinite(data["regions"]["all"]["ssim"]))
            self.assertTrue(math.isnan(data["regions"]["foreground"]["ssim"]))
            self.assertTrue(math.isnan(data["regions"]["background"]["ssim"]))


class TestDeriveSelectedEpoch(unittest.TestCase):
    def test_epoch_from_filename(self):
        self.assertEqual(_derive_selected_epoch("runs/x/checkpoints/valid_epoch12.pth"), 12)
        self.assertEqual(_derive_selected_epoch("model_0030.pth"), 30)

    def test_fallback_to_stem_or_none(self):
        self.assertEqual(_derive_selected_epoch("checkpoints/final_weights.pth"), "final_weights")
        self.assertIsNone(_derive_selected_epoch(None))
        self.assertEqual(_derive_selected_epoch("a.pth", selected_epoch=7), 7)


if __name__ == "__main__":
    unittest.main()
