#!/bin/bash
# E1-3P 推理修复 + 统一 eval（GPU3）。
# 背景两条上游事实（不改原仓库）：
# 1) Predictor3D.execute 用 os.walk(model_path) 枚举模型——传单个 .pth 文件会
#    静默产出空预测；故把最终模型放入独立目录再传目录路径。
# 2) Predictor3D 输入只做 astype('uint8')，期望 8-bit 数据；3P 原始栈为 uint16
#    （均值 4449），直接转换会模 256 回绕。训练侧 datagen 有 per-window min-max
#    归一化所以训练有效；推理侧需离线把原始栈按 0.1~99.9 分位归一化到 uint8。
set -e
export CUDA_VISIBLE_DEVICES=3
SN2N_ROOT=/data2/wjb/SN2N-main
E1_DIR=/data3/wjb/Denoising_PM/Results/sn2n_original_3p
PY=/data2/wjb/anaconda3/envs/threePM/bin/python

cd "$SN2N_ROOT"
export PYTHONPATH="$SN2N_ROOT"

echo "=== [E1-3p] step3a: 8-bit normalization of raw stack ==="
"$PY" - <<'EOF'
import glob
import numpy as np
import tifffile

src = glob.glob('/data3/wjb/Denoising_PM/Results/sn2n_original_3p/raw_data/*.tif')[0]
# Predictor3D 的 img_path 必须是目录（内部 os.walk 枚举 tif），故 8-bit 栈落盘为目录
dst_dir = '/data3/wjb/Denoising_PM/Results/sn2n_original_3p/raw_data_8bit'
import os as _os
_os.makedirs(dst_dir, exist_ok=True)
dst = dst_dir + '/3P_neuron_repeat01_275z_8bit.tif'
a = tifffile.imread(src).astype(np.float32)
lo, hi = np.percentile(a, [0.1, 99.9])
scale = max(float(hi - lo), 1e-6)
out = np.clip((a - float(lo)) / scale, 0.0, 1.0)
out = (out * 255.0).round().astype(np.uint8)
tifffile.imwrite(dst, out)
print('8-bit stack:', dst, out.shape, 'p1/p99.9 =', float(lo), float(hi))
EOF

echo "=== [E1-3p] step3b: inference with final model ==="
"$PY" - <<'EOF'
import os, shutil
# torch>=2.6 兼容 shim（不改原仓库源码）：上游 torch.load 未传 weights_only，
# 而其 checkpoint 序列化整个 Unet_3d 对象，2.6 默认拒绝反序列化。
# 本地自产 checkpoint 属可信来源，包一层默认 weights_only=False。
import torch
_orig_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _orig_load(*args, **kwargs)
torch.load = _patched_load
print('[E1] applied torch>=2.6 weights_only shim to torch.load')

from SN2N.inference import Predictor3D

model_src = '/data3/wjb/Denoising_PM/Results/sn2n_original_3p/models/model_9_3_full.pth'
model_dir = '/data3/wjb/Denoising_PM/Results/sn2n_original_3p/models_final'
os.makedirs(model_dir, exist_ok=True)
dst = os.path.join(model_dir, os.path.basename(model_src))
if not os.path.exists(dst):
    shutil.copy(model_src, dst)

p = Predictor3D(img_path='/data3/wjb/Denoising_PM/Results/sn2n_original_3p/raw_data_8bit',
                model_path=model_dir, infer_mode=1)
p.execute()
EOF

echo "=== [E1-3p] step3c: unified eval vs mean9 proxy ==="
PYTHONPATH=/data2/wjb/Denoising_PM/ACES "$PY" - <<'EOF'
import glob, json
import numpy as np
import tifffile
from scipy import ndimage

pred_path = sorted(glob.glob('/data3/wjb/Denoising_PM/Results/sn2n_original_3p/predictions/*.tif'))
assert pred_path, 'no prediction written'
pred = tifffile.imread(pred_path[0]).astype(np.float32)
target = tifffile.imread('/data2/wjb/Denoising_PM/Methods/FAST-main/reproduction/valid_comparison/experiments/valid_paper_3p_neuron/original_workflow_paper_params_repeat01_seed3407/data/gt_proxy/3P_neuron_mean9_gt_proxy_275z.tif').astype(np.float32)
print('pred', pred.shape, 'target', target.shape)
# 形状对齐（SN2N 拼块输出可能与 GT proxy 尺寸略异：裁到共同大小）
z = min(pred.shape[0], target.shape[0]); y = min(pred.shape[1], target.shape[1]); x = min(pred.shape[2], target.shape[2])
pred = pred[:z, :y, :x]; target = target[:z, :y, :x]

# 尺度对齐：全局线性拟合 pred -> target（自监督输出无绝对尺度，标准做法）
A = np.stack([pred.ravel(), np.ones(pred.size)], axis=1)
b = target.ravel()
coef, *_ = np.linalg.lstsq(A, b, rcond=None)
a1, a0 = float(coef[0]), float(coef[1])
aligned = a1 * pred + a0
print(f'scale fit: {a1:.4f}*pred + {a0:.2f}')

# 前景掩码：target 的 75 分位 + 形态学清理（与 qmask eval 一致）
norm = target / max(float(np.percentile(target, 99.8)) - float(np.percentile(target, 1)), 1e-6)
fg = norm > np.percentile(norm, 75.0)
fg = ndimage.binary_opening(fg, structure=np.ones((1, 3, 3), dtype=bool))
fg = ndimage.binary_closing(fg, structure=np.ones((1, 3, 3), dtype=bool))
bg = ~fg

def psnr(err_sq):
    mse = float(np.mean(err_sq))
    return 10.0 * np.log10(float(target.max() - target.min()) ** 2 / max(mse, 1e-12)) if mse > 0 else float('inf')

regions = {'all': np.ones_like(fg, dtype=bool), 'foreground': fg, 'background': bg}
out = {'event': 'e1_sn2n_3p_unified_eval', 'prediction': pred_path[0],
       'scale_fit': {'a': a1, 'b': a0},
       'psnr_reference_range': float(target.max() - target.min())}
for name, m in regions.items():
    err = (aligned[m] - target[m]) ** 2
    out[name] = {'psnr_db': psnr(err), 'mse': float(np.mean(err)),
                 'pred_mean': float(np.mean(aligned[m])), 'target_mean': float(np.mean(target[m])),
                 'pred_std': float(np.std(aligned[m])), 'target_std': float(np.std(target[m])),
                 'n_voxels': int(m.sum())}
    print(f"{name:12s} psnr={out[name]['psnr_db']:.2f} dB  std_ratio={out[name]['pred_std']/max(out[name]['target_std'],1e-9):.2f}")
path = '/data3/wjb/Denoising_PM/Results/sn2n_original_3p/unified_eval.json'
json.dump(out, open(path, 'w'), indent=2)
print('saved', path)
EOF
echo "E1-3p inference+eval done"
