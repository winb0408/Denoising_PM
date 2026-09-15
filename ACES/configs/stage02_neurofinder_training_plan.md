# ACES Stage 2 Neurofinder 00.00 训练计划

## 背景

之前的 `stage02_*_40ep` 配置在 **3P_neuron repeat01 (275帧)** 上训练，在 **neurofinder (3024帧)** 上评估，导致：
- 训练数据不足：仅 125 patches/epoch × 40 = 5,000 updates
- 训练/评估数据不匹配
- DeepCAD-RT 使用 62,400 updates，差距 12 倍
- 结果：前景 PSNR 低 3dB，块伪影严重，分割失败

## 多方法训练配置对比

基于 `Results/neurofinder_00_00_multimethod_comparison_20260822/` 的实际运行配置：

| 方法 | Epochs | Patch 大小 | Overlap | 预估 Updates |
|------|--------|-----------|---------|--------------|
| **DeepCAD-RT** | 10 | (150,150,150) | 0.25 | **62,400** |
| **SRDTrans** | 20 | (160,160,160) | - | **~120,000** |
| **SUPPORT** | 100 | - | - | 未知 |
| **FAST original** | 100 | - | - | 未知 |
| **ACES (旧配置)** | 40 | (64,128,128) | 0.1 | **5,000** ❌ |
| **ACES (新配置)** | 80 | (64,128,128) | 0.1 | **66,560** ✅ |

## 新配置设计

### 训练预算

ACES 在 neurofinder (3024, 512, 512) 上理论可切：
- Z 方向：52 patches
- Y 方向：4 patches  
- X 方向：4 patches
- **总数：832 patches/epoch**

目标训练量：
- **80 epochs × 832 = 66,560 updates**
- 略高于 DeepCAD-RT (62,400)
- 低于 SRDTrans (~120,000)
- 与 VALID/FAST 的 100 epochs 精神一致

### 配置文件

已创建：`configs/stage02_neurofinder_5mode_80ep.yaml`

**关键参数**：
```yaml
training:
  train_folder: /data2/wjb/Denoising_PM/ACES/data/neurofinder.00.00/train
  epochs: 80
  z_patch: 64
  w_patch: 128
  h_patch: 128
  z_overlap: 0.1
  w_overlap: 0.1
  h_overlap: 0.1
  eval_every_epochs: 10
  
sampling:
  stage1_run_dir: /data2/wjb/Denoising_PM/ACES/runs/stage01_neurofinder_00_00_proxy_pairability_seed3407
  noise_source: proxy  # neurofinder 没有 repeat 数据
  modes: [valid, sn2n_xy, iso3d, empirical_block, ellipsoid]
```

## 运行命令

```bash
# 单GPU训练（推荐先跑一个模式验证）
PYTHONPATH=ACES CUDA_VISIBLE_DEVICES=0 \
  /data2/wjb/anaconda3/envs/threePM/bin/python -m aces.train_stage2 \
  --config ACES/configs/stage02_neurofinder_5mode_80ep.yaml

# 生成报告
PYTHONPATH=ACES /data2/wjb/anaconda3/envs/threePM/bin/python -m aces.make_stage2_report \
  --run-dir ACES/runs/stage02_neurofinder_5mode_80ep_seed3407
```

## 预期结果

如果训练不足是主因，80 epochs 后应该看到：

1. **损失收敛**：
   - total_loss 进入平台期
   - 不再像 40ep 时持续下降

2. **PSNR 恢复**：
   - 前景 PSNR 从 18.9-19.4 dB → **~22 dB**
   - 接近 valid/sn2n_xy baseline

3. **块伪影消失**：
   - 网络学会跨 patch 边界的平滑过渡
   - MIP 图像无明显矩形边界

4. **分割性能恢复**：
   - Cellpose 检测到 ~300 个神经元（接近 GT 330）
   - Instance F1 达到可比水平

## 时间估算

基于之前的训练速度：
- 40 epochs 完成时间：~数小时（取决于 GPU）
- 80 epochs 预估：~2倍时间
- 建议使用 `eval_every_epochs: 10` 监控收敛

## 风险与备选方案

**如果 80 epochs 仍不足**：
- 检查 eval_curve.csv，看 PSNR 是否仍在上升
- 继续训练到 100 epochs（与 SUPPORT/FAST 一致）
- 或者增加 patch overlap（0.1 → 0.2），增加每 epoch 样本数

**如果块伪影持续存在**：
- 问题可能在推理策略（硬平铺）而非训练不足
- 需要实现 overlap + blending 推理（如 DeepCAD-RT 的 0.6 overlap）