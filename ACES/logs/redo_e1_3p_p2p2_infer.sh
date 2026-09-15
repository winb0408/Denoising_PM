#!/usr/bin/env bash
# 修复 E1-3P-p2p2 推理:首跑误用 uint16 raw_data(Predictor3D 契约 astype('uint8')
# 会 mod-256 回绕,预测与 target Pearson≈0,已证实作废)。
# 改用与原 3P leg 相同的 8-bit 归一化输入(p0.1–p99.9),重跑 Predictor3D。
set -euo pipefail
export CUDA_VISIBLE_DEVICES=3
PY=/data2/wjb/anaconda3/envs/threePM/bin/python
OUT=/data3/wjb/Denoising_PM/Results/sn2n_original_3p_p2p2

"$PY" - <<'EOF'
import glob, os
import numpy as np
import tifffile

OUT = '/data3/wjb/Denoising_PM/Results/sn2n_original_3p_p2p2'
raw_path = OUT + '/raw_data/3P_neuron_repeat01_275z.tif'
out_dir = OUT + '/raw_data_8bit'
os.makedirs(out_dir, exist_ok=True)
out_path = out_dir + '/3P_neuron_repeat01_275z_8bit.tif'
if not os.path.exists(out_path):
    a = tifffile.imread(raw_path).astype(np.float32)
    lo, hi = np.percentile(a, [0.1, 99.9])
    a8 = np.clip((a - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
    tifffile.imwrite(out_path, a8)
    print('wrote', out_path, a8.shape, a8.dtype)
else:
    print('8bit stack exists, skip')

import torch
_torch_load = torch.load
torch.load = lambda *a, **k: _torch_load(*a, **{**k, 'weights_only': False})  # torch>=2.6 shim
import shutil, sys
sys.path.insert(0, '/data2/wjb/SN2N-main')
from SN2N.inference import Predictor3D
if os.path.isdir(OUT + '/models_final'):
    shutil.rmtree(OUT + '/models_final')
os.makedirs(OUT + '/models_final')
full = sorted(glob.glob(OUT + '/models/*_full.pth'))[-1]
shutil.copy(full, OUT + '/models_final/')
print('inference with', full)
p = Predictor3D(img_path=out_dir, model_path=OUT + '/models_final', infer_mode=1)
p.execute()
EOF
echo "redo inference done"
ls -la "$OUT/predictions/"
