# ACES-QMask

ACES 方案复检实验：把原“采样半径/椭球”改为“共位相位分裂 + 轴参与掩码”，并隔离验证
`valid / sn2n_xy / iso3d / qmask` 四档。详见 `docs/TECHNICAL.md` 与 `docs/RESULTS.md`。

```bash
# 环境
PYTHONPATH=/data2/wjb/Denoising_PM/ACES
PY=/data2/wjb/anaconda3/envs/threePM/bin/python

# 几何预览（不训练）
PYTHONPATH=$PYTHONPATH $PY -m qmask.geometry \
  --stage1-run-dir /data2/wjb/Denoising_PM/ACES/runs/stage01_3p_repeat_pairability_seed3407 \
  --noise-source repeat --out runs/_preview_3p.json

# 单元测试
cd /data2/wjb/Denoising_PM && PYTHONPATH=$PYTHONPATH $PY -m unittest discover -s ACES/qmask/tests -v

# 冒烟 / pilot
cd /data2/wjb/Denoising_PM && CUDA_VISIBLE_DEVICES=1 PYTHONPATH=$PYTHONPATH $PY -m qmask.train \
  --config ACES/qmask/configs/qmask_3p_pilot.yaml

# 可视化（含反标准化自检）
cd /data2/wjb/Denoising_PM && PYTHONPATH=$PYTHONPATH $PY -m qmask.visualize \
  --run-dir ACES/qmask/runs/qmask_3p_pilot_seed3407

# 结果报告
cd /data2/wjb/Denoising_PM && PYTHONPATH=$PYTHONPATH $PY -m qmask.report \
  --run-dir ACES/qmask/runs/qmask_3p_pilot_seed3407

# 审图预览：先看 overview/tiles，再谈定量结论；不要直接把全分辨率大图传给模型。
cd /data2/wjb/Denoising_PM && PYTHONPATH=$PYTHONPATH $PY -m qmask.preview \
  --run-dir ACES/qmask/runs/qmask_3p_pilot_seed3407
```

目录：

- `geometry.py`：`[d_min, d_max]` 可用区间 + 轴参与掩码。
- `sampling.py`：共位相位分裂采样器。
- `train.py`：训练 runner（模型/损失复用 VALID）。
- `evaluate.py`：深度分箱前景 PSNR。
- `visualize.py`：几何曲线 + 定性图（标准化/反标准化自检）。
- `report.py`：Markdown 报告与门控判定。
