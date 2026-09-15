# VALID/FAST 3P Neuron 复现实验会话记录

日期：2026-08-20  
工作目录：`/data2/wjb`  
主要环境：`/data2/wjb/anaconda3/envs/threePM/bin/python`，GPU 使用 `CUDA_VISIBLE_DEVICES=2`

## 目标

本轮工作的核心目标是重新审视 `3P_neuron_raw.tif` 在 VALID 论文 Fig. 2 中的复现差异，并按照用户要求：

- FAST 和 VALID 尽量使用原仓库代码与原论文/原参数流程。
- VALID 使用原仓库的完整 `224 x 224 x 224` 3D patch 训练。
- VALID 测试使用原仓库 `10%` overlap 滑窗推理与重叠平均重建。
- Swin 只作为对应适配对照，若 224 下 window partition 不整除，则使用 256。
- 按论文 Fig. 2 风格生成层面图、XZ MIP 和深度伪彩显示。

## 关键数据理解

原始数据：

`/data2/wjb/Denoising_PM/VALID-v1.0/FDU-donglab-VALID-34f5637/data/3P_neuron_raw.tif`

之前复现差异的一个关键原因是 raw tif 的帧序理解问题。当前采用的解释为：

- 原始 tif 形状：`(2475, 512, 512)`
- 正确物理结构：`275 depth groups x 9 repeats x 512 x 512`
- 深度步长：`4.0 um`
- 深度范围：`0-1096 um`，与论文 Fig. 2 的 `0-1100 um` 显示范围一致。
- `mean9` 只作为评价和可视化 proxy，不用于自监督训练。

数据审计和 frame order 相关旧报告：

`/data2/wjb/Denoising_PM/FAST-main/reproduction/valid_comparison/experiments/valid_paper_3p_neuron/equal_updates_u64_p256_seed3407/report/3p_neuron_frame_order_audit.md`

## 为什么之前的“公平对比版”不够

之前的 `equal_updates` 或 `corrected_layout` 系列实验主要用于排查：

- frame order 是否正确；
- mean9 proxy 是否合理；
- FAST/Swin/VALID 在相同 update/patch 预算下的相对趋势；
- adapter 训练方式是否造成差异。

但这类对照不等于论文完整复现，因为它没有严格保留 VALID 原仓库的关键行为：

- 不是完整 `224 x 224 x 224` 3D patch；
- 不是原 `ReadDatasets` 3D patchify/reconstruct；
- 不是原 `goTrainingVALID/goTestingVALID` 训练测试入口；
- 训练单位和原论文设定存在偏差。

因此用户指出后，实验口径被修正为“原 workflow 复现”：FAST/VALID 调用原仓库代码，Swin 只作为适配基线。

## 新增脚本

新增 wrapper：

`/data2/wjb/Denoising_PM/FAST-main/reproduction/valid_comparison/src/valid_compare/original_workflow.py`

它做的事情：

1. 将 `3P_neuron_raw.tif` 拆成 corrected layout。
2. 生成 repeat01 完整 3D volume 用于训练/测试。
3. 保留全部 9 个 repeat volume 与 mean9 proxy 供评估。
4. 为 VALID/FAST 生成 config。
5. 调用原仓库训练/测试入口。
6. 收集输出到统一实验目录。
7. 生成 Fig. 2 风格可视化和指标报告。

注意：该 wrapper 不重写 VALID/FAST 核心训练逻辑，只负责参数、路径和自动化调用。

## 环境修复

`threePM` 环境原本缺少部分原仓库依赖，已安装：

```bash
/data2/wjb/anaconda3/envs/threePM/bin/python -m pip install scikit-image PyQt5 setproctitle
/data2/wjb/anaconda3/envs/threePM/bin/python -m pip install PyWavelets
/data2/wjb/anaconda3/envs/threePM/bin/python -m pip install einops
```

其中：

- `scikit-image`：FAST/VALID 原代码读写 tif 需要。
- `PyQt5`：VALID GUI pipeline 导入依赖，虽然实际使用 dummy GUI handler。
- `setproctitle`：VALID 训练入口依赖。
- `PyWavelets`：VALID `datasets/sampling.py` 依赖。
- `einops`：VALID sampling 依赖。

## 数据准备

新实验目录：

`/data2/wjb/Denoising_PM/FAST-main/reproduction/valid_comparison/experiments/valid_paper_3p_neuron/original_workflow_paper_params_repeat01_seed3407`

数据结构：

- 训练 volume：
  `data/train_repeat01_275z/3P_neuron_repeat01_275z.tif`
- 测试 volume：
  `data/test_repeat1_275z/3P_neuron_repeat01_275z.tif`
- 全部 repeats：
  `data/all_repeats_275z/3P_neuron_repeat01_275z.tif` 至 `repeat09`
- mean9 proxy：
  `data/gt_proxy/3P_neuron_mean9_gt_proxy_275z.tif`
- 数据 manifest：
  `logs/data_manifest.json`

采用 repeat01 训练/测试的原因：

- 更贴近原仓库“单个 raw 3D volume 零样本训练”的语义；
- 避免把 9 次重复采集误当成 9 个独立训练样本，导致训练量扩大 9 倍；
- 其他 repeats 仍用于 mean9 proxy 和潜在 SNR/稳定性评估。

曾短暂测试过 9 repeats 全部进训练集，VALID 可跑且显存约 36.3 GB，但训练量变为 `162 patches/epoch x 100 epochs`，不作为最终复现口径。

## VALID 原 workflow 复现

调用入口：

- 训练：原仓库 `goTrainingVALID`
- 测试：原仓库 `goTestingVALID`

关键参数：

- `epochs = 100`
- `w_patch = 224`
- `h_patch = 224`
- `z_patch = 224`
- `w_overlap = 0.1`
- `h_overlap = 0.1`
- `z_overlap = 0.1`
- `patch_num = -1`
- `batch_size = 1`
- `seed = 3407`
- `lr = 0.0001`
- `base_features = 16`
- `n_groups = 4`
- `clip_gradients = 100.0`
- `weight_reg = 0.0001`

说明：

原 `jsons/train_params.json` 中没有显式 `weight_reg`，但训练函数 `goTrainingVALID` 实际需要该属性：

```python
Total_loss = loss2neighbor + loss_idt + args.weight_reg * loss_reg
```

因此 wrapper 按论文 Hessian regularization 权重补入 `weight_reg=0.0001`，这是对原实现的配置补全，不改变训练逻辑。

实际训练规模：

- 输入：repeat01，`275 x 512 x 512`
- 原 224³ + 10% overlap patchify 后：
  `18 patches/epoch`
- 总训练：
  `18 x 100 = 1800` 个 3D patch batch

运行情况：

- 显存峰值约：`36.3 GB`
- 训练时间：约 `0.76 hours`
- 测试 patch 数：`18`
- 测试时间：约 `58 s`

VALID 输出：

`/data2/wjb/Denoising_PM/FAST-main/reproduction/valid_comparison/experiments/valid_paper_3p_neuron/original_workflow_paper_params_repeat01_seed3407/outputs/valid/202608192310_3P_neuron_repeat01_275z.tif`

原仓库 checkpoint：

`/data2/wjb/Denoising_PM/VALID-v1.0/FDU-donglab-VALID-34f5637/checkpoint/train_repeat01_275z202608192223`

## FAST 原 workflow 复现

调用入口：

- `FAST-main/main.py`
- 训练：`goTraining`
- 测试：`goTesting`

关键参数来自 FAST 原 `params.json` 并做路径/数据长度适配：

- `epochs = 100`
- `miniBatch_size = 4`
- `lr = 0.0001`
- `weight_decay = 0.9`
- `amsgrad = true`
- `train_frames = 275`
- `batch_size = 1`
- `seed = 123`
- `data_type = 3D`
- `denoising_strategy = FAST`

说明：

FAST 原代码是 temporal 2D 方法，当前将 corrected 3D stack 的 z 轴作为 temporal axis 使用。这符合对 temporal denoising 方法在 volumetric 数据上的适配方式。

实际运行：

- 训练时间：约 `12 min 15 s`
- 测试时间：约 `3.5 s`
- 每 epoch 内部沿 z 轴以 `miniBatch=4` 滑动训练。

FAST 输出：

`/data2/wjb/Denoising_PM/FAST-main/reproduction/valid_comparison/experiments/valid_paper_3p_neuron/original_workflow_paper_params_repeat01_seed3407/outputs/fast/202608192323_3P_neuron_repeat01_275z.tif`

原仓库 checkpoint：

`/data2/wjb/Denoising_PM/FAST-main/checkpoint/train_repeat01_275z202608192310`

## Swin 适配对照

Swin 不属于原 VALID/FAST 论文方法，因此本轮没有把它当作“原 workflow”方法。它作为架构适配对照，使用之前已完成的 corrected-layout Swin-256 结果：

`/data2/wjb/Denoising_PM/FAST-main/reproduction/valid_comparison/experiments/valid_paper_3p_neuron/corrected_layout_u2048_fast224_valid224_swin256_seed3407/predictions/fig2_corrected_full_stack/swin_full_stack.npy`

在新实验目录中转存为：

`/data2/wjb/Denoising_PM/FAST-main/reproduction/valid_comparison/experiments/valid_paper_3p_neuron/original_workflow_paper_params_repeat01_seed3407/outputs/swin/swin_adapted_u2048_p256_full_stack.tif`

说明：

- Swin 使用 256 patch 是为避免 224 下 window partition 尺寸不整除。
- 该 Swin 结果是适配对照，不应与 VALID 原 224³ workflow 混为同一种训练协议。

## Fig. 2 风格可视化

生成的主要图：

层面/ROI 对比：

`/data2/wjb/Denoising_PM/FAST-main/reproduction/valid_comparison/experiments/valid_paper_3p_neuron/original_workflow_paper_params_repeat01_seed3407/figures/fig2_corrected/fig2_corrected_layers.png`

XZ MIP + 900-1050 um 深度伪彩：

`/data2/wjb/Denoising_PM/FAST-main/reproduction/valid_comparison/experiments/valid_paper_3p_neuron/original_workflow_paper_params_repeat01_seed3407/figures/fig2_corrected/fig2_corrected_projection_pseudocolor.png`

报告：

`/data2/wjb/Denoising_PM/FAST-main/reproduction/valid_comparison/experiments/valid_paper_3p_neuron/original_workflow_paper_params_repeat01_seed3407/report/original_workflow_fig2_report.md`

可视化采用：

- 层面深度：`48, 572, 740, 900, 1050 um`
- 对应 z index 约：`12, 143, 185, 225, 262`
- XZ MIP：全深度投影
- 深度伪彩：`900-1048 um`，接近论文标注的深层范围

## 当前指标

指标均以 mean9 proxy 为参考，mean9 不参与训练。

| Method | MSE to mean9 | MAE to mean9 | PSNR to mean9 |
|---|---:|---:|---:|
| Raw repeat01 | 15576086.97 | 2615.22 | 15.72 dB |
| FAST original | 5045417.57 | 1636.18 | 20.62 dB |
| Swin adapted | 3832141.60 | 1324.89 | 21.81 dB |
| VALID original | 2964867.19 | 1238.58 | 22.93 dB |

完整指标 JSON：

`/data2/wjb/Denoising_PM/FAST-main/reproduction/valid_comparison/experiments/valid_paper_3p_neuron/original_workflow_paper_params_repeat01_seed3407/logs/visualization_summary.json`

## 主要观察

1. 按原 VALID `224³ + 10% overlap` workflow 后，VALID 结果明显优于之前 adapter 版，更符合论文中 VALID 应有的趋势。
2. VALID 对 mean9 proxy 的 PSNR 最高，约 `22.93 dB`。
3. FAST 原 workflow 相比 raw 有明显提升，但在该数据和当前配置下低于 VALID。
4. Swin 适配版数值介于 FAST 与 VALID 之间，但 XZ MIP 和伪彩图中仍可见明显块状/网格伪影。
5. VALID 原 workflow 显存需求较高，224³ batch 在 A6000 上约 36.3 GB，可稳定运行。

## 需要注意的实现细节

1. 原 VALID `jsons/train_params.json` 缺少 `weight_reg`，但训练函数需要它。本轮按论文/GUI 默认值补为 `0.0001`。
2. 原 VALID 使用 `setproctitle`、`PyQt5` 等依赖，即使不使用 GUI，也需要安装或提供 dummy handler。
3. 原 VALID 保存 checkpoint 时会在训练集上做一次重建预览，所以第 100 epoch 明显比普通 epoch 慢。
4. `save_freq` 设置为 `100`，避免每 10 epoch 都保存和重建，减少时间开销；最终 checkpoint 仍是 100 epoch。
5. FAST 的 `weight_decay=0.9` 看起来很大，但这是原 `params.json` 中的值，本轮按“原参数”保留。

## 关键复现实验命令

准备数据：

```bash
PYTHONPATH=/data2/wjb/Denoising_PM/FAST-main/reproduction/valid_comparison/src \
/data2/wjb/anaconda3/envs/threePM/bin/python \
-m valid_compare.original_workflow prepare --gpu 2
```

运行 VALID：

```bash
PYTHONPATH=/data2/wjb/Denoising_PM/FAST-main/reproduction/valid_comparison/src \
CUDA_VISIBLE_DEVICES=2 \
/data2/wjb/anaconda3/envs/threePM/bin/python \
-m valid_compare.original_workflow valid --gpu 2
```

运行 FAST：

```bash
PYTHONPATH=/data2/wjb/Denoising_PM/FAST-main/reproduction/valid_comparison/src \
CUDA_VISIBLE_DEVICES=2 \
/data2/wjb/anaconda3/envs/threePM/bin/python \
-m valid_compare.original_workflow fast --gpu 2
```

生成可视化：

```bash
PYTHONPATH=/data2/wjb/Denoising_PM/FAST-main/reproduction/valid_comparison/src \
CUDA_VISIBLE_DEVICES=2 \
/data2/wjb/anaconda3/envs/threePM/bin/python \
-m valid_compare.original_workflow visualize --gpu 2
```

一键完整流程：

```bash
PYTHONPATH=/data2/wjb/Denoising_PM/FAST-main/reproduction/valid_comparison/src \
CUDA_VISIBLE_DEVICES=2 \
/data2/wjb/anaconda3/envs/threePM/bin/python \
-m valid_compare.original_workflow all --gpu 2
```

## 后续建议

1. 如果要更接近论文 Fig. 2 的 SNR 统计，建议对 9 个 repeat 分别做推理，然后基于 denoised repeats 计算 pixel-wise mean/std 的 SNR，而不只是对 mean9 proxy 做 PSNR。
2. 可进一步核对论文 Fig. 2 的显示窗宽、伪彩 colormap、MIP 范围和 ROI 位置，当前可视化是“类似论文风格”，不是完全复刻排版。
3. FAST 若要完全公允，可进一步确认 FAST 论文或官方说明中 `weight_decay=0.9` 是否确为推荐参数，而不是仓库默认配置中的历史遗留值。
4. Swin 对照目前不是原 workflow，可单独建立 `swin_adapted_p256_repeat01` 实验目录，使用与 FAST 更接近的 temporal self-supervised signal 重新训练。
5. 可在报告中加入运行环境快照，例如 PyTorch/CUDA/scikit-image/tifffile 版本，增强可复现性。

