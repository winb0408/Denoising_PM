"""E2：VALID 原仓库 checkpoint 在 ACES 统一评估协议下的复算（Plan v2 §3.3）。

把 8/20 原仓库全协议训练（100 epoch，paper params）的 checkpoint 放进与
stage02 五档对比完全相同的评估管线（18 个前景丰富 patch、mean9 proxy、
分层 fg/bg PSNR），产出可直接并入内部归因表对照行的 eval_metrics.json。

只做评估端统一，不改动原仓库任何代码（Plan v2 E 通道纪律）。

用法（仓库根目录）：
    PYTHONPATH=ACES python -m aces.eval_valid_original
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

from .paths import VALID_ROOT

CKPT = Path(
    "/data2/wjb/Denoising_PM/Methods/VALID-v1.0/FDU-donglab-VALID-34f5637/"
    "checkpoint/train_repeat01_275z202608192223/train_repeat01_275z_Epoch_100.pth"
)
CKPT_CONFIG = Path(
    "/data2/wjb/Denoising_PM/Methods/VALID-v1.0/FDU-donglab-VALID-34f5637/"
    "checkpoint/train_repeat01_275z202608192223/config.json"
)
# 与 stage02_valid_40ep_seed3407（五档对比 valid 档）完全一致的评估协议
REFERENCE_CONFIG = Path("/data2/wjb/Denoising_PM/ACES/runs/stage02_valid_40ep_seed3407/valid/config.json")
OUTPUT_DIR = Path("/data2/wjb/Denoising_PM/ACES/runs/e2_valid_original_unified_3p")


def main() -> int:
    from .evaluate_stage2 import evaluate_model_on_patches
    from .io_utils import write_json

    sys.path.insert(0, str(VALID_ROOT))
    from datasets.dataset_fs import ReadDatasets, custom_collate_fn  # noqa: F401
    from models.network import Network_CNR

    ckpt_cfg = json.loads(CKPT_CONFIG.read_text(encoding="utf-8"))
    ref_cfg = json.loads(REFERENCE_CONFIG.read_text(encoding="utf-8"))
    train_cfg = ref_cfg["training"]
    eval_cfg = ref_cfg["evaluation"]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Network_CNR(
        in_channels=1,
        out_channels=1,
        f_maps=int(ckpt_cfg["base_features"]),
        n_groups=int(ckpt_cfg["n_groups"]),
    ).to(device)
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"])  # 键带 'net.' 前缀，与 Network_CNR 结构一致
    model.eval()
    print(f"loaded original VALID checkpoint: {CKPT.name} (epoch 100, paper params)")

    dataset = ReadDatasets(
        dataPath=train_cfg["train_folder"],
        mode="train",
        dataType="3D",
        dataExtension="tif",
        z_patch=int(train_cfg["z_patch"]),
        w_patch=int(train_cfg["w_patch"]),
        h_patch=int(train_cfg["h_patch"]),
        z_overlap=float(train_cfg["z_overlap"]),
        w_overlap=float(train_cfg["w_overlap"]),
        h_overlap=float(train_cfg["h_overlap"]),
        patch_num=int(train_cfg["patch_num"]),
        dataNum=int(train_cfg.get("train_frame_num", 10000)),
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = evaluate_model_on_patches(
        model=model,
        dataset=dataset,
        device=device,
        output_dir=OUTPUT_DIR,
        mean9_path=eval_cfg["mean9_proxy"],
        max_patches=int(eval_cfg.get("patch_count", 18)),
        foreground_percentile=float(eval_cfg.get("foreground_percentile", 75.0)),
        foreground_mask_path=eval_cfg.get("foreground_mask"),
        metrics_filename="eval_metrics.json",
    )
    write_json(
        OUTPUT_DIR / "e2_summary.json",
        {
            "event": "e2_valid_original_unified_eval",
            "checkpoint": str(CKPT),
            "checkpoint_config": str(CKPT_CONFIG),
            "reference_protocol": str(REFERENCE_CONFIG),
            "regions": summary["regions"],
        },
    )
    fg = summary["regions"]["foreground"]
    bg = summary["regions"]["background"]
    print(
        f"E2 original VALID unified eval: fg PSNR={fg['psnr_db']:.2f} dB, "
        f"bg PSNR={bg['psnr_db']:.2f} dB  -> {OUTPUT_DIR / 'eval_metrics.json'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
