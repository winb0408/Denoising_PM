#!/usr/bin/env bash
# E2-nf: 原版 VALID (Methods/VALID-v1.0, 零源码改动) 在 neurofinder.00.00 上训练
# 对齐 3P leg 配置 (checkpoint/train_repeat01_275z202608192223/config.json):
#   epochs=100, z/w/h_patch=224, overlap=0.1, lr=1e-4, amsgrad, base_features=16,
#   n_groups=4, weight_reg=1e-4, bs=1, seed=3407, N2N, withGT=false
# 无头启动方式 (launcher 层 shim, 仓库源码不动):
#   - goTrainingVALID(params_path) 可直接调用, 无需 Qt 事件循环
#   - GUI_HANDLER 用 stub (epoch 100 验证块会调 update_volume_data)
#   - STOP_HANDLER 只需 _is_running=True
#   - json2args 内部 argparse.parse_args() 会解析 sys.argv -> 先置 sys.argv
# 存储: /data2 已满 -> 仓库 checkpoint/ 目录迁移到 /data3 镜像并留符号链接
#       (路径不变, E2 eval 脚本引用不受影响)
set -euo pipefail

export CUDA_VISIBLE_DEVICES=2

REPO=/data2/wjb/Denoising_PM/Methods/VALID-v1.0/FDU-donglab-VALID-34f5637
E2_DIR=/data3/wjb/Denoising_PM/Results/valid_original_neurofinder
CKPT3_DIR=/data3/wjb/Denoising_PM/Methods/VALID-v1.0/FDU-donglab-VALID-34f5637/checkpoint
NF_STACK=/data2/wjb/Denoising_PM/ACES/data/neurofinder.00.00/train/neurofinder_00_00_stack.tif

mkdir -p "$E2_DIR"

# ---- 1. checkpoint 目录迁移到 /data3 (cp -> 校验 -> 换成符号链接) ----
if [ ! -L "$REPO/checkpoint" ]; then
  mkdir -p "$(dirname "$CKPT3_DIR")"
  if [ ! -d "$CKPT3_DIR" ]; then
    cp -a "$REPO/checkpoint" "$CKPT3_DIR"
  fi
  # 校验: 逐文件字节 + 内容哈希 (du -sb 的目录记账跨文件系统不等, 不能用)
  SRC_SIZE=$(find "$REPO/checkpoint" -type f -printf '%s\n' | awk '{s+=$1} END{print s+0}')
  DST_SIZE=$(find "$CKPT3_DIR" -type f -printf '%s\n' | awk '{s+=$1} END{print s+0}')
  SRC_HASH=$(cd "$REPO/checkpoint" && find . -type f -print0 | sort -z | xargs -0 md5sum | md5sum | cut -d' ' -f1)
  DST_HASH=$(cd "$CKPT3_DIR" && find . -type f -print0 | sort -z | xargs -0 md5sum | md5sum | cut -d' ' -f1)
  if [ "$SRC_SIZE" != "$DST_SIZE" ] || [ "$SRC_HASH" != "$DST_HASH" ]; then
    echo "checkpoint copy mismatch (size $SRC_SIZE vs $DST_SIZE, hash $SRC_HASH vs $DST_HASH), aborting"; exit 1
  fi
  rm -rf "$REPO/checkpoint"
  ln -s "$CKPT3_DIR" "$REPO/checkpoint"
  echo "checkpoint -> $CKPT3_DIR (symlink)"
fi

# ---- 2. nf 训练数据目录 (符号链接指向只读 stack, 不复制 1.6G) ----
TRAIN_DIR=$E2_DIR/train_neurofinder_00_00
mkdir -p "$TRAIN_DIR"
ln -sfn "$NF_STACK" "$TRAIN_DIR/neurofinder_00_00_stack.tif"

# ---- 3. 训练参数 json (镜像 3P leg, 仅 train_folder/val_folder 换 nf) ----
CONFIG=$E2_DIR/train_config.json
cat > "$CONFIG" <<EOF
{
    "epochs": 100,
    "train_frame_num": 10000,
    "w_patch": 224,
    "h_patch": 224,
    "z_patch": 224,
    "w_overlap": 0.1,
    "h_overlap": 0.1,
    "z_overlap": 0.1,
    "patch_num": -1,
    "gpu_ids": [0],
    "save_freq": 100,
    "train_folder": "$TRAIN_DIR",
    "num_workers": 0,
    "data_extension": "tif",
    "withGT": false,
    "lr": 0.0001,
    "amsgrad": true,
    "base_features": 16,
    "n_groups": 4,
    "train": true,
    "test": false,
    "data_type": "3D",
    "denoising_strategy": "N2N",
    "seed": 3407,
    "clip_gradients": 100.0,
    "mode": "train",
    "batch_size": 1,
    "weight_reg": 0.0001,
    "val_folder": "$TRAIN_DIR"
}
EOF

# ---- 4. 无头训练 (stub GUI_HANDLER / STOP_HANDLER, sys.argv 置空) ----
cd "$REPO"
/data2/wjb/anaconda3/envs/threePM/bin/python - "$CONFIG" <<'PYEOF'
import sys
config_path = sys.argv[1]
sys.argv = ['e2_valid_nf_headless']  # json2args 内部 argparse.parse_args() 需要空 argv
sys.path.insert(0, '/data2/wjb/Denoising_PM/Methods/VALID-v1.0/FDU-donglab-VALID-34f5637')

import train_pipeline as tp


class _StubGUI:
    """epoch 100 验证块调用 update_volume_data(raw, valid) —— 无头模式下 no-op。"""

    def update_volume_data(self, *args, **kwargs):
        pass


class _StopHandler:
    _is_running = True  # goTrainingVALID 每个 batch 检查该属性, False 会中断训练


tp.goTrainingVALID(config_path, GUI_HANDLER=_StubGUI(), STOP_HANDLER=_StopHandler())
PYEOF
