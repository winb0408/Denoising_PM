#!/bin/bash
# E1（Plan v2 §3.3）neurofinder.00.00 leg：SN2N 原仓库 3D 管线。
# 与 3P leg（run_e1_sn2n_3p.sh）完全同参（P2Pmode=1, P2Pup=6, SWsize=64,
# SWfilter=8; bs=4, lr=2e-4, epochs=100），仅换数据源。GPU1。
set -e
export CUDA_VISIBLE_DEVICES=1
SN2N_ROOT=/data2/wjb/SN2N-main
E1_DIR=/data3/wjb/Denoising_PM/Results/sn2n_original_neurofinder
RAW_TIF=/data2/wjb/Denoising_PM/ACES/data/neurofinder.00.00/train/neurofinder_00_00_stack.tif
PY=/data2/wjb/anaconda3/envs/threePM/bin/python

mkdir -p "$E1_DIR/raw_data"
[ -f "$E1_DIR/raw_data/neurofinder_00_00_stack.tif" ] || cp "$RAW_TIF" "$E1_DIR/raw_data/"

cd "$SN2N_ROOT"
export PYTHONPATH="$SN2N_ROOT"

echo "=== [E1-nf] step1 datagen3D ==="
"$PY" - <<'EOF'
# numpy>=1.25 兼容 shim（不改原仓库源码），同 3P leg。
import inspect
import textwrap
import SN2N.datagen as dg

_SRC = textwrap.dedent(inspect.getsource(dg.generator3D.random_interchange))
assert 'if imgb == []:' in _SRC, 'upstream random_interchange changed, aborting shim'
_NEW = _SRC.replace('if imgb == []:', 'if len(imgb) == 0:')
_ns = {}
exec(compile(_NEW, '<shim random_interchange>', 'exec'), _ns)
dg.generator3D.random_interchange = _ns['random_interchange']
print('[E1-nf] applied numpy>=1.25 compat shim to generator3D.random_interchange')

from SN2N.get_options import datagen3D
from SN2N.datagen import generator3D
args = datagen3D([
    '--img_path', '/data3/wjb/Denoising_PM/Results/sn2n_original_neurofinder/raw_data',
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

echo "=== [E1-nf] step2 trainer3D (bs=4, lr=2e-4, epochs=5=step-budget match) ==="
# 协议偏离说明：nf datagen 产出 ~74.5k 训练块（18,639 batch/epoch，~2.1h/epoch），
# 原仓库默认 100 epoch 需 ~200h 不可行。改 epochs=5，使总梯度步数
# 5×18,639≈93k 与 3P leg（100×891≈89k）匹配；其余参数（bs/lr/loss）不动。
"$PY" - <<'EOF'
from SN2N.trainer import net3D
net = net3D(img_path='/data3/wjb/Denoising_PM/Results/sn2n_original_neurofinder/raw_data',
            sn2n_loss=1, bs=4, lr=2e-4, epochs=5)
net.train()
EOF

echo "=== [E1-nf] step3 inference3D ==="
"$PY" - <<'EOF'
import glob
from SN2N.inference import Predictor3D
model_path = sorted(glob.glob('/data3/wjb/Denoising_PM/Results/sn2n_original_neurofinder/models/*.pth'))[-1]
print('inference with', model_path)
p = Predictor3D(img_path='/data3/wjb/Denoising_PM/Results/sn2n_original_neurofinder/raw_data',
                model_path=model_path, infer_mode=1)
p.execute()
EOF

echo "E1-nf pipeline done"
