#!/bin/bash
# E1（Plan v2 §3.3）：SN2N 原仓库 3D 管线跑 3P repeat01。
# 原仓库零修改；三步：datagen → trainer → inference。
# 数据布局：img_path 指向 raw_data 目录，产物写到同级 datasets/models/images。
set -e
export CUDA_VISIBLE_DEVICES=3
SN2N_ROOT=/data2/wjb/SN2N-main
E1_DIR=/data3/wjb/Denoising_PM/Results/sn2n_original_3p
RAW_TIF=/data2/wjb/Denoising_PM/Methods/FAST-main/reproduction/valid_comparison/experiments/valid_paper_3p_neuron/original_workflow_paper_params_repeat01_seed3407/data/train_repeat01_275z/3P_neuron_repeat01_275z.tif
PY=/data2/wjb/anaconda3/envs/threePM/bin/python

mkdir -p "$E1_DIR/raw_data"
[ -f "$E1_DIR/raw_data/3P_neuron_repeat01_275z.tif" ] || cp "$RAW_TIF" "$E1_DIR/raw_data/"

cd "$SN2N_ROOT"
export PYTHONPATH="$SN2N_ROOT"

echo "=== [E1] step1 datagen3D ==="
N_TRAIN_TIFS=$(ls "$E1_DIR"/datasets/*.tif 2>/dev/null | wc -l)
if [ "$N_TRAIN_TIFS" -gt 100 ]; then
  echo "datasets already generated ($N_TRAIN_TIFS tifs), skip datagen"
else
"$PY" - <<'EOF'
# numpy>=1.25 兼容 shim（不改原仓库源码）：
# 上游 random_interchange 用 `if imgb == []` 判空，imgb 为 ndarray 时
# numpy>=1.25 直接 raise ValueError。此处以完全相同的分支逻辑替换该
# 15 行分发函数，仅把判空改为 len(imgb)==0，其余逐行一致。
import inspect
import textwrap
import SN2N.datagen as dg

_SRC = textwrap.dedent(inspect.getsource(dg.generator3D.random_interchange))
assert 'if imgb == []:' in _SRC, 'upstream random_interchange changed, aborting shim'
_NEW = _SRC.replace('if imgb == []:', 'if len(imgb) == 0:')
_ns = {}
exec(compile(_NEW, '<shim random_interchange>', 'exec'), _ns)
dg.generator3D.random_interchange = _ns['random_interchange']
print('[E1] applied numpy>=1.25 compat shim to generator3D.random_interchange')

from SN2N.get_options import datagen3D
from SN2N.datagen import generator3D
args = datagen3D([
    '--img_path', '/data3/wjb/Denoising_PM/Results/sn2n_original_3p/raw_data',
    '--P2Pmode', '1', '--P2Pup', '6', '--BAmode', '0',
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

echo "=== [E1] step2 trainer3D (paper params: bs=4, lr=2e-4, epochs=100) ==="
"$PY" - <<'EOF'
from SN2N.trainer import net3D
net = net3D(img_path='/data3/wjb/Denoising_PM/Results/sn2n_original_3p/raw_data',
            sn2n_loss=1, bs=4, lr=2e-4, epochs=100)
net.train()
EOF

echo "=== [E1] step3 inference3D ==="
"$PY" - <<'EOF'
import glob
from SN2N.inference import Predictor3D
model_path = sorted(glob.glob('/data3/wjb/Denoising_PM/Results/sn2n_original_3p/models/*.pth'))[-1]
print('inference with', model_path)
p = Predictor3D(img_path='/data3/wjb/Denoising_PM/Results/sn2n_original_3p/raw_data',
                model_path=model_path, infer_mode=1)
p.execute()
EOF

echo "E1 pipeline done"
