"""E2-nf:VALID 原仓库 checkpoint 在 nf 统一评估协议下的复算。

与 run_e1_nf_infer_eval.sh step3c(E1-nf 同尺)完全一致:
  - 网格 PZ,PW,PH=64,128,128,stride 16,全栈枚举,fg-rich top-18
  - 区域 all/foreground(ROI mask 沿 z 广播)/background
  - _region_metrics per-patch p0.1–p99.9 data_range
差异:VALID 输出即目标尺度(栈级 mean/std 归一化训练),无 lstsq 对齐;
  推理端按 VALID 契约用训练栈 mean/std 归一化输入、输出去归一化。

用法: PYTHONPATH=ACES python -m aces.eval_valid_original_nf
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import tifffile
import torch

CKPT_DIR = Path("/data2/wjb/Denoising_PM/Methods/VALID-v1.0/FDU-donglab-VALID-34f5637/"
                "checkpoint/train_neurofinder_00_00202609041902")
STACK = Path("/data2/wjb/Denoising_PM/ACES/data/neurofinder.00.00/train/neurofinder_00_00_stack.tif")
PROXY = Path("/data2/wjb/Denoising_PM/ACES/data/neurofinder.00.00/proxy")
OUTPUT = Path("/data3/wjb/Denoising_PM/Results/valid_original_neurofinder/patch_eval_metrics.json")
PZ, PW, PH = 64, 128, 128
STRIDE = 16


def main() -> int:
    from .evaluate_stage2 import _mask_patch, _region_metrics

    sys.path.insert(0, "/data2/wjb/Denoising_PM/Methods/VALID-v1.0/FDU-donglab-VALID-34f5637")
    from models.network import Network_CNR

    ckpt_cfg = json.loads((CKPT_DIR / "config.json").read_text(encoding="utf-8"))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Network_CNR(in_channels=1, out_channels=1,
                        f_maps=int(ckpt_cfg["base_features"]),
                        n_groups=int(ckpt_cfg["n_groups"])).to(device)
    ckpt = torch.load(CKPT_DIR / "train_neurofinder_00_00_Epoch_100.pth",
                      map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print("loaded original VALID nf checkpoint (epoch 100)")

    stack = tifffile.imread(STACK).astype(np.float32)
    mean_v, std_v = float(stack.mean()), float(stack.std())
    stack_n = (stack - mean_v) / std_v
    print(f"stack {stack.shape}, mean={mean_v:.2f} std={std_v:.2f}")

    target = tifffile.imread(PROXY / "neurofinder_00_00_temporal_mean.tif").astype(np.float32)
    roi = tifffile.imread(PROXY / "neurofinder_00_00_roi_mask.tif").astype(bool)
    assert target.ndim == 2 and roi.ndim == 2
    z, y, x = stack.shape
    target = target[:y, :x]
    roi = roi[:y, :x]

    cands = []
    for z0 in range(0, z - PZ + 1, STRIDE):
        for y0 in range(0, y - PW + 1, STRIDE):
            for x0 in range(0, x - PH + 1, STRIDE):
                cands.append((int(_mask_patch(roi, z0, y0, x0, (PZ, PW, PH)).sum()), z0, y0, x0))
    cands.sort(reverse=True)
    selected = cands[:18]
    print(f"selected 18 fg-rich patches (fg frac {selected[0][0]/(PZ*PW*PH):.3f}..{selected[-1][0]/(PZ*PW*PH):.3f})")

    roi3 = np.broadcast_to(roi, (z, y, x)).astype(bool, copy=False)
    masks3 = {"all": np.ones_like(roi3), "foreground": roi3, "background": ~roi3}

    rows = []
    with torch.no_grad():
        for _, z0, y0, x0 in selected:
            inp = stack_n[z0:z0+PZ, y0:y0+PW, x0:x0+PH]
            t = torch.from_numpy(np.ascontiguousarray(inp)).unsqueeze(0).unsqueeze(0).to(device)
            out = model(t).squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32)
            out = out * std_v + mean_v
            tp = np.broadcast_to(target[y0:y0+PW, x0:x0+PH], out.shape).astype(np.float32, copy=False)
            for region, m3 in masks3.items():
                mp = m3[z0:z0+PZ, y0:y0+PW, x0:x0+PH]
                rows.append({"patch_index": [z0, y0, x0], "region": region,
                             **_region_metrics(out, tp, mp)})

    regions = {}
    for region in ("all", "foreground", "background"):
        sub = [r for r in rows if r["region"] == region]
        regions[region] = {"psnr_db_mean": float(np.mean([r["psnr_db"] for r in sub])),
                           "n_patches": len(sub)}
    summary = {
        "event": "e2_valid_original_nf_patch_eval",
        "checkpoint": str(CKPT_DIR),
        "grid": {"PZ": PZ, "PW": PW, "PH": PH, "stride": STRIDE},
        "selected_patch_zyx": [[s[1], s[2], s[3]] for s in selected],
        "regions": regions,
        "rows": rows,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(summary, indent=2))
    for region, st in regions.items():
        print(f"{region:12s} patch-mean psnr={st['psnr_db_mean']:.2f} dB ({st['n_patches']} patches)")
    print("saved", OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
