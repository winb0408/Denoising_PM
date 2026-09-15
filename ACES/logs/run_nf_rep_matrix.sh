#!/bin/bash
# 多种子重复矩阵（Plan v2 Q-2/Q-3 方差控制）：4 臂 x 3 种子，GPU2 顺序执行。
# 臂：valid 锚 / qmask z.x+fixed / qmask z.x+permuted / qmask 全开(n_decision=false)+fixed
# 单次 3-epoch pilot 的 run-to-run 方差可达 1.5 dB（q2 vs 原 pilot 同配置实测），
# 任何配置间 delta 必须以跨种子均值±范围报告。
set -e
export CUDA_VISIBLE_DEVICES=2
cd /data2/wjb/Denoising_PM
PY=/data2/wjb/anaconda3/envs/threePM/bin/python

for cfg in ACES/qmask/configs/nf_rep/nf_rep_*.json; do
  echo "=== [nf-rep] $(basename "$cfg") ==="
  PYTHONPATH=ACES "$PY" -m qmask.train --config "$cfg"
done
echo "nf-rep matrix done"
