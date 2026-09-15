#!/usr/bin/env bash
# nf 重复矩阵 worker(Plan v2 §4 item 10a)。用法: run_nf_matrix.sh <gpu_id>
# 作业分配(每 GPU 4 个, 约 4-5h):
#   GPU1: valid×3 + iso3d(seed3407)
#   GPU2: sn2n_xy×3 + iso3d(seed3408)
#   GPU3: qmask×3 + iso3d(seed3409)
set -uo pipefail
GPU=${1:?usage: run_nf_matrix.sh <gpu_id>}
PY=/data2/wjb/anaconda3/envs/threePM/bin/python
CFG_DIR=/data3/wjb/Denoising_PM/ACES/configs
LOG_DIR=/data3/wjb/Denoising_PM/ACES/logs/nf_matrix
mkdir -p "$LOG_DIR"

run_job () {  # run_job <seed> <mode>
  local SEED=$1 MODE=$2
  local YAML=$CFG_DIR/stage02_nf_matrix4_seed${SEED}.yaml
  echo "[$(date +%H:%M:%S)] start gpu$GPU seed$SEED $MODE"
  CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=/data2/wjb/Denoising_PM/ACES \
    "$PY" -m qmask.train --config "$YAML" --modes "$MODE" \
    > "$LOG_DIR/gpu${GPU}_seed${SEED}_${MODE}.log" 2>&1
  echo "[$(date +%H:%M:%S)] done  gpu$GPU seed$SEED $MODE (exit $?)"
}

case $GPU in
  1) run_job 3407 valid;  run_job 3408 valid;  run_job 3409 valid;  run_job 3407 iso3d ;;
  2) run_job 3407 sn2n_xy; run_job 3408 sn2n_xy; run_job 3409 sn2n_xy; run_job 3408 iso3d ;;
  3) run_job 3407 qmask;  run_job 3408 qmask;  run_job 3409 qmask;  run_job 3409 iso3d ;;
  *) echo "gpu_id must be 1|2|3"; exit 1 ;;
esac
echo "[$(date +%H:%M:%S)] worker gpu$GPU all done"
