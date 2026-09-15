#!/bin/bash
# E1-3P 重跑（2026-09-04 修改小结 #1）：3D 管线内 P2Pmode 1->2 单变量重跑。
# 依据：P2Pmode=1 沿 t(=z) 轴互换与本项目归因结论冲突（3P 上 N_repeat(z,1)≈1,
# z 向过采样、配对无独立噪声实现 -> 训练信号弱 -> 过度平滑, 用户定性判 FAIL）。
# P2Pmode=2 = 单帧内 xy 配对, 是 3D 管线中匹配该数据几何的原始模式（零源码改动）。
# 除 P2Pmode 外与 run_e1_sn2n_3p.sh 完全同参（P2Pup=6, BAmode=0, SWsize=64,
# SWfilter=8, bs=4, lr=2e-4, epochs=100, seed 环境同前）, 隔离配对几何变量。
set -e
export CUDA_VISIBLE_DEVICES=3
SN2N_ROOT=/data2/wjb/SN2N-main
E1_DIR=/data3/wjb/Denoising_PM/Results/sn2n_original_3p_p2p2
RAW_TIF=/data2/wjb/Denoising_PM/Methods/FAST-main/reproduction/valid_comparison/experiments/valid_paper_3p_neuron/original_workflow_paper_params_repeat01_seed3407/data/train_repeat01_275z/3P_neuron_repeat01_275z.tif
PY=/data2/wjb/anaconda3/envs/threePM/bin/python

mkdir -p "$E1_DIR/raw_data"
[ -f "$E1_DIR/raw_data/3P_neuron_repeat01_275z.tif" ] || cp "$RAW_TIF" "$E1_DIR/raw_data/"

cd "$SN2N_ROOT"
export PYTHONPATH="$SN2N_ROOT"

echo "=== [E1-3P-p2p2] step1 datagen3D (P2Pmode=2) ==="
N_TRAIN_TIFS=$(ls "$E1_DIR"/datasets/*.tif 2>/dev/null | wc -l)
if [ "$N_TRAIN_TIFS" -gt 100 ]; then
  echo "datasets already generated ($N_TRAIN_TIFS tifs), skip datagen"
else
"$PY" - <<'EOF'
# numpy>=1.25 兼容 shim（不改原仓库源码, 与原 3P leg 同款, 已文档化）
import inspect
import textwrap
import SN2N.datagen as dg

_SRC = textwrap.dedent(inspect.getsource(dg.generator3D.random_interchange))
assert 'if imgb == []:' in _SRC, 'upstream random_interchange changed, aborting shim'
_NEW = _SRC.replace('if imgb == []:', 'if len(imgb) == 0:')
_ns = {}
exec(compile(_NEW, '<shim random_interchange>', 'exec'), _ns)
dg.generator3D.random_interchange = _ns['random_interchange']
print('[E1-3P-p2p2] applied numpy>=1.25 compat shim to generator3D.random_interchange')

from SN2N.get_options import datagen3D
from SN2N.datagen import generator3D
args = datagen3D([
    '--img_path', '/data3/wjb/Denoising_PM/Results/sn2n_original_3p_p2p2/raw_data',
    '--P2Pmode', '2', '--P2Pup', '6', '--BAmode', '0',
    '--SWsize', '64', '--SWfilter', '8',
])
print(args)
gen = generator3D(img_path=args.img_path, P2Pmode=args.P2Pmode, P2Pup=args.P2Pup,
                  BAmode=args.BAmode, SWsize=args.SWsize, SWmode=args.SWmode,
                  SWfilter=args.SWfilter, P2Ppatch=args.P2Ppatch,
                  vol_patch=args.vol_patch, ifx2=args.ifx2, inter_mode=args.inter_mode)
gen.execute()
EOF
fi

echo "=== [E1-3P-p2p2] step2 trainer3D (paper params: bs=4, lr=2e-4, epochs=100) ==="
"$PY" - <<'EOF'
import glob
from SN2N.trainer import net3D
OUT = '/data3/wjb/Denoising_PM/Results/sn2n_original_3p_p2p2'
if glob.glob(OUT + '/models/*_full.pth'):
    print('full model exists, skip training')
else:
    net = net3D(img_path=OUT + '/raw_data', sn2n_loss=1, bs=4, lr=2e-4, epochs=100)
    net.train()
EOF

echo "=== [E1-3P-p2p2] step3 inference3D (取 *_full.pth) ==="
"$PY" - <<'EOF'
import glob, shutil, os
import torch
torch_load = torch.load
torch.load = lambda *a, **k: torch_load(*a, **{**k, 'weights_only': False})  # torch>=2.6 shim, 同前
from SN2N.inference import Predictor3D
OUT = '/data3/wjb/Denoising_PM/Results/sn2n_original_3p_p2p2'
full = sorted(glob.glob(OUT + '/models/*_full.pth'))
assert full, 'no *_full.pth produced'
if os.path.isdir(OUT + '/models_final'):
    shutil.rmtree(OUT + '/models_final')
os.makedirs(OUT + '/models_final')  # Predictor3D 用 os.walk, 必须传目录
shutil.copy(full[-1], OUT + '/models_final/')
print('inference with', full[-1])
p = Predictor3D(img_path=OUT + '/raw_data_8bit',  # uint16 会 mod-256 回绕, 必须 8-bit 输入(小结 #4)
                model_path=OUT + '/models_final', infer_mode=1)
p.execute()
EOF

echo "E1-3P-p2p2 pipeline done"
