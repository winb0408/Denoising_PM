#!/usr/bin/env bash
# E1-nf 重跑（2026-09-04 修改小结 #1）：neurofinder.00.00 是 2D+time 数据,
# 改用原仓库 2D 管线 datagen2D -> net2D -> Predictor2D（此前误用 3D 管线）。
# 2D 例程规范参数（Script_SN2Nexecute_2D.py / Script_SN2N_trainer_2D.py）:
#   P2Pmode=1(t 轴互换, 对 nf 的 t=时间轴正确), P2Pup=1, BAmode=1,
#   SWsize=64, img_patch=128, bs=32, lr=2e-4, sn2n_loss=1
# 协议偏离（同 nf-3D leg 理由）: 2D datagen 实测 312,908 patch -> 9,778 batch/epoch
#   （bs=32）, 100ep 不可行; epochs 按"总梯度步数 ≈ 3P-100ep ≈ 89k"约定取 9
#   （9,778×9≈8.8 万步, 首轮 epochs=4 仅 3.9 万步, 已按脚本注释约定上调）。
# 仓库源码零修改; launcher shim 仅两处（与 3D leg 相同, 已文档化）:
#   1) random_interchange 的 `if imgb == []` numpy>=1.25 判空崩溃 -> len(imgb)==0
#   2) （2D 管线推理 torch.load 已自带 weights_only=False, 无需 shim）
set -euo pipefail
export CUDA_VISIBLE_DEVICES=1

OUT=/data3/wjb/Denoising_PM/Results/sn2n_original_neurofinder_2d
NF_STACK=/data2/wjb/Denoising_PM/ACES/data/neurofinder.00.00/train/neurofinder_00_00_stack.tif
PY=/data2/wjb/anaconda3/envs/threePM/bin/python
mkdir -p "$OUT/raw_data"
ln -sfn "$NF_STACK" "$OUT/raw_data/neurofinder_00_00_stack.tif"

echo "=== [E1-nf-2D] step1 datagen2D ==="
if [ ! -d "$OUT/datasets" ] || [ -z "$(ls -A "$OUT/datasets" 2>/dev/null)" ]; then
  "$PY" - <<'EOF'
import sys
sys.path.insert(0, '/data2/wjb/SN2N-main')
import inspect, textwrap
import SN2N.datagen as dg
# shim 1: numpy>=1.25 判空兼容（2D generator 的 random_interchange, 源码零改动）
_SRC = textwrap.dedent(inspect.getsource(dg.generator2D.random_interchange))
_SRC = _SRC.replace('if imgb == []:', 'if len(imgb) == 0:')
_ns = {}
exec(compile(_SRC, '<shim>', 'exec'), _ns)
dg.generator2D.random_interchange = _ns['random_interchange']
print('[E1-nf-2D] applied numpy>=1.25 compat shim to generator2D.random_interchange')

from SN2N.get_options import execute2D
args = execute2D([
    '--img_path', '/data3/wjb/Denoising_PM/Results/sn2n_original_neurofinder_2d/raw_data',
    '--P2Pmode', '1', '--P2Pup', '1', '--BAmode', '1',
    '--SWsize', '64', '--bs', '32', '--lr', '2e-4', '--epochs', '9',
    '--model_path', '/data3/wjb/Denoising_PM/Results/sn2n_original_neurofinder_2d/models_final',
    '--infer_mode', '1',
])
print('[E1-nf-2D] datagen2D args:', vars(args))
gen = dg.generator2D(img_path=args.img_path, P2Pmode=args.P2Pmode, P2Pup=args.P2Pup,
                     BAmode=args.BAmode, SWsize=args.SWsize, SWmode=args.SWmode,
                     SWfilter=args.SWfilter, P2Ppatch=args.P2Ppatch,
                     img_patch=args.img_patch, ifx2=args.ifx2, inter_mode=args.inter_mode)
gen.execute()
EOF
else
  echo "datasets already generated, skip datagen"
fi
ls "$OUT/datasets" | wc -l

echo "=== [E1-nf-2D] step2 net2D 训练 (bs=32, lr=2e-4, epochs=9, sn2n_loss=1) ==="
"$PY" - <<'EOF'
import sys, glob, os
sys.path.insert(0, '/data2/wjb/SN2N-main')
import SN2N.datagen as dg
import inspect, textwrap
_SRC = textwrap.dedent(inspect.getsource(dg.generator2D.random_interchange))
_SRC = _SRC.replace('if imgb == []:', 'if len(imgb) == 0:')
_ns = {}
exec(compile(_SRC, '<shim>', 'exec'), _ns)
dg.generator2D.random_interchange = _ns['random_interchange']

from SN2N.trainer import net2D
OUT = '/data3/wjb/Denoising_PM/Results/sn2n_original_neurofinder_2d'
if glob.glob(OUT + '/models/*_full.pth'):
    print('full model exists, skip training')
else:
    n = len(glob.glob(OUT + '/datasets/*.tif'))
    print(f'[E1-nf-2D] dataset patches: {n}, batch/epoch ≈ {n // 32} (bs=32)')
    net = net2D(img_path=OUT + '/raw_data', sn2n_loss=1, bs=32, lr=2e-4, epochs=9,
                img_patch='128', if_alr=True)
    net.train()
EOF

echo "=== [E1-nf-2D] step3 Predictor2D 推理 (取 *_full.pth) ==="
"$PY" - <<'EOF'
import sys, glob, shutil, os
sys.path.insert(0, '/data2/wjb/SN2N-main')
OUT = '/data3/wjb/Denoising_PM/Results/sn2n_original_neurofinder_2d'
full = sorted(glob.glob(OUT + '/models/*_full.pth'))
assert full, 'no *_full.pth produced'
if os.path.isdir(OUT + '/models_final'):
    shutil.rmtree(OUT + '/models_final')
os.makedirs(OUT + '/models_final')  # Predictor2D 用 os.walk, 必须传目录
shutil.copy(full[-1], OUT + '/models_final/')
print('inference with', full[-1])
from SN2N.inference import Predictor2D
p = Predictor2D(img_path=OUT + '/raw_data',
                model_path=OUT + '/models_final', infer_mode=1)
p.execute()
EOF

echo "=== [E1-nf-2D] done ==="
ls -la "$OUT/images/" | head
