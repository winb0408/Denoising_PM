#!/bin/bash
# Q-2/Q-3 neurofinder 验证 pilots（Plan v2 §4 item 1/3/7）。GPU2，顺序执行两个配置。
# 1) q2: n_decision=false（掩码 S/Q 决定，预期全开）+ fixed  → 隔离 Q-2
# 2) q3: n_decision=true（掩码 101=z.x 同原 pilot）+ permuted → 隔离槽位排列
# 对照锚 = 原有 qmask_neurofinder_pilot_seed3407（101 + fixed + valid/qmask）。
set -e
export CUDA_VISIBLE_DEVICES=2
cd /data2/wjb/Denoising_PM
PY=/data2/wjb/anaconda3/envs/threePM/bin/python

echo "=== [Q-nf] run1: q2 all-on + fixed ==="
PYTHONPATH=ACES "$PY" -m qmask.train --config ACES/qmask/configs/qmask_nf_q2_allon_fixed_seed3407.json

echo "=== [Q-nf] run2: q3 z.x + permuted ==="
PYTHONPATH=ACES "$PY" -m qmask.train --config ACES/qmask/configs/qmask_nf_q3_zx_permuted_seed3407.json

echo "Q-nf validation pilots done"
