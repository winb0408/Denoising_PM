#!/bin/bash
# E1-NF 推理修复 + 统一 eval（GPU1）。与 3P 版（run_e1_3p_infer_eval.sh）同款 shim：
# 1) model_path/img_path 必须传目录（上游 os.walk 枚举，传文件静默空产出）
# 2) torch>=2.6 weights_only shim
# 3) uint8 契约：推理前把 uint16 原始栈按 0.1~99.9 分位归一化到 8-bit
# 在 run_e1_sn2n_neurofinder.sh（训练）完成后运行。
set -e
export CUDA_VISIBLE_DEVICES=1
SN2N_ROOT=/data2/wjb/SN2N-main
E1_DIR=/data3/wjb/Denoising_PM/Results/sn2n_original_neurofinder
PY=/data2/wjb/anaconda3/envs/threePM/bin/python

cd "$SN2N_ROOT"
export PYTHONPATH="$SN2N_ROOT"

if [ -f /data3/wjb/Denoising_PM/Results/sn2n_original_neurofinder/raw_data_8bit/neurofinder_00_00_stack_8bit.tif ]; then
  echo "=== [E1-nf] step3a: 8-bit stack exists, skip ==="
else
echo "=== [E1-nf] step3a: 8-bit normalization ==="
"$PY" - <<'EOF'
import glob, os
import numpy as np
import tifffile

src = glob.glob('/data3/wjb/Denoising_PM/Results/sn2n_original_neurofinder/raw_data/*.tif')[0]
dst_dir = '/data3/wjb/Denoising_PM/Results/sn2n_original_neurofinder/raw_data_8bit'
os.makedirs(dst_dir, exist_ok=True)
dst = dst_dir + '/neurofinder_00_00_stack_8bit.tif'
a = tifffile.imread(src).astype(np.float32)
lo, hi = np.percentile(a, [0.1, 99.9])
scale = max(float(hi - lo), 1e-6)
out = (np.clip((a - float(lo)) / scale, 0.0, 1.0) * 255.0).round().astype(np.uint8)
tifffile.imwrite(dst, out)
print('8-bit stack:', dst, out.shape, 'p1/p99.9 =', float(lo), float(hi))
EOF
fi

PRED_TIF=$(ls /data3/wjb/Denoising_PM/Results/sn2n_original_neurofinder/predictions/*.tif 2>/dev/null | head -1)
if [ -n "$PRED_TIF" ]; then
  echo "=== [E1-nf] step3b: prediction exists ($PRED_TIF), skip inference ==="
else
echo "=== [E1-nf] step3b: inference with final model ==="
"$PY" - <<'EOF'
import os, shutil, torch
_orig_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _orig_load(*args, **kwargs)
torch.load = _patched_load
print('[E1-nf] applied torch>=2.6 weights_only shim')

from SN2N.inference import Predictor3D

model_src = sorted(__import__('glob').glob('/data3/wjb/Denoising_PM/Results/sn2n_original_neurofinder/models/*_full.pth'))[-1]
model_dir = '/data3/wjb/Denoising_PM/Results/sn2n_original_neurofinder/models_final'
os.makedirs(model_dir, exist_ok=True)
dst = os.path.join(model_dir, os.path.basename(model_src))
if not os.path.exists(dst):
    shutil.copy(model_src, dst)

p = Predictor3D(img_path='/data3/wjb/Denoising_PM/Results/sn2n_original_neurofinder/raw_data_8bit',
                model_path=model_dir, infer_mode=1)
p.execute()
EOF
fi

echo "=== [E1-nf] step3c: unified patch eval vs temporal-mean proxy ==="
PYTHONPATH=/data2/wjb/Denoising_PM/ACES "$PY" - <<'EOF'
import glob, json
import numpy as np
import tifffile
from aces.evaluate_stage2 import _region_metrics, _mask_patch

pred_path = sorted(glob.glob('/data3/wjb/Denoising_PM/Results/sn2n_original_neurofinder/predictions/*.tif'))
assert pred_path, 'no prediction written'
pred = tifffile.imread(pred_path[0]).astype(np.float32)
target = tifffile.imread('/data2/wjb/Denoising_PM/ACES/data/neurofinder.00.00/proxy/neurofinder_00_00_temporal_mean.tif').astype(np.float32)
roi = tifffile.imread('/data2/wjb/Denoising_PM/ACES/data/neurofinder.00.00/proxy/neurofinder_00_00_roi_mask.tif').astype(bool)
print('pred', pred.shape, 'target', target.shape, 'roi', roi.shape, 'roi fg fraction', round(float(roi.mean()), 4))

# 与 qmask 内部 nf 评估同一把尺：2D temporal-mean 目标与 ROI 沿 z 广播（nf 的 z 是时间轴）
assert target.ndim == 2 and roi.ndim == 2, (target.shape, roi.shape)
z, y, x = pred.shape
target3 = np.broadcast_to(target, (z, y, x)).astype(np.float32, copy=False)
roi3 = np.broadcast_to(roi, (z, y, x)).astype(bool, copy=False)

A = np.stack([pred.ravel(), np.ones(pred.size)], axis=1)
coef, *_ = np.linalg.lstsq(A, target3.ravel(), rcond=None)
aligned = float(coef[0]) * pred + float(coef[1])
print(f'scale fit: {float(coef[0]):.4f}*pred + {float(coef[1]):.2f}')

# 18 个前景丰富 patch：与 evaluate_stage2 fg-rich 选点一致（64x128x128、步长 16 网格，
# 按广播 ROI 内前景计数排序；计数沿 z 不变，退化为同一 (y0,x0) 的不同 z 起点——
# 与内部协议 sorted() 的并列处理一致）
PZ, PW, PH = 64, 128, 128
cands = []
for z0 in range(0, z - PZ + 1, 16):
    for y0 in range(0, y - PW + 1, 16):
        for x0 in range(0, x - PH + 1, 16):
            cands.append((int(_mask_patch(roi, z0, y0, x0, (PZ, PW, PH)).sum()), z0, y0, x0))
cands.sort(reverse=True)
selected = cands[:18]
regions = {'all': np.ones_like(roi3, dtype=bool), 'foreground': roi3, 'background': ~roi3}
out = {'event': 'e1_sn2n_nf_patch_eval', 'prediction': pred_path[0],
       'scale_fit': {'a': float(coef[0]), 'b': float(coef[1])},
       'data_range': 'per-patch p0.1-p99.9 of masked region (same as _region_metrics)',
       'patches': []}
agg = {k: [] for k in regions}
for fg_count, z0, y0, x0 in selected:
    pp = aligned[z0:z0+PZ, y0:y0+PW, x0:x0+PH]
    tp = target3[z0:z0+PZ, y0:y0+PW, x0:x0+PH]
    entry = {'z0': z0, 'y0': y0, 'x0': x0, 'fg_count': fg_count}
    for name, m in regions.items():
        mp = m[z0:z0+PZ, y0:y0+PW, x0:x0+PH]
        entry[name] = _region_metrics(pp, tp, mp)['psnr_db']
        agg[name].append(entry[name])
    out['patches'].append(entry)
for name, vals in agg.items():
    out[name] = {'psnr_db_mean': float(np.mean(vals)), 'n_patches': len(vals)}
    print(f"{name:12s} patch-mean psnr={out[name]['psnr_db_mean']:.2f} dB ({len(vals)} patches)")
path = '/data3/wjb/Denoising_PM/Results/sn2n_original_neurofinder/patch_eval_metrics.json'
json.dump(out, open(path, 'w'), indent=2)
print('saved', path)
EOF
echo "E1-nf inference+eval done"
