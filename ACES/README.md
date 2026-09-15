# ACES Stage 0-1

各向异性校准经验正交采样（Anisotropy-Calibrated Empirical orthogonal Sampling）的
Stage 0-1 独立工具。Stage 0-1 只做一件事：**在真实数据上检验"配对性前提"是否成立**
——也就是说，在训练任何采样模型之前，先用数据本身回答一个问题：噪声在各方向上
是否足够独立、结构在各方向上是否存在各向异性，从而判断自监督配对采样在这份数据上
是否有物理依据。如果这个前提不成立，就不应该进入 Stage 2 的采样训练。

## 快速运行

从仓库根目录运行，需设置 `PYTHONPATH=ACES`；也可直接在本 `ACES/` 目录内运行。

```bash
# 1) 环境/依赖门控：检查 torch、numpy、VALID、SN2N 等是否可导入
PYTHONPATH=ACES /data2/wjb/anaconda3/envs/threePM/bin/python -m aces.env_check \
  --run-dir ACES/runs/stage01_3p_repeat_pairability_seed3407

# 2) 各向异性剖析（GPU）：核心计算，产出所有 metrics/figures
PYTHONPATH=ACES CUDA_VISIBLE_DEVICES=3 /data2/wjb/anaconda3/envs/threePM/bin/python -m aces.profile_anisotropy \
  --config ACES/configs/stage01_3p_repeat.yaml

# 3) 汇总报告：把 metrics 汇总成人类可读的 markdown 报告
PYTHONPATH=ACES /data2/wjb/anaconda3/envs/threePM/bin/python -m aces.make_report \
  --run-dir ACES/runs/stage01_3p_repeat_pairability_seed3407
```

冒烟测试（小裁剪，秒级验证流程是否通）：

```bash
PYTHONPATH=ACES CUDA_VISIBLE_DEVICES=3 /data2/wjb/anaconda3/envs/threePM/bin/python -m aces.profile_anisotropy \
  --config ACES/configs/stage01_3p_repeat_smoke.yaml
```

## Stage 2 Sampling-Only

Stage 2 固定 VALID 的 CRN、DWT、Hessian 正则与 N2N 损失形式，只替换采样算子。
所有实现都在 `ACES/` 内完成，不修改 VALID/SN2N/FAST 原仓库源码。

先跑静态 sampler 检查：

```bash
PYTHONPATH=ACES CUDA_VISIBLE_DEVICES=3 /data2/wjb/anaconda3/envs/threePM/bin/python -m aces.check_stage2 \
  --stage1-run-dir ACES/runs/stage01_3p_repeat_pairability_seed3407
```

五档 smoke 训练闭环：

```bash
PYTHONPATH=ACES CUDA_VISIBLE_DEVICES=3 /data2/wjb/anaconda3/envs/threePM/bin/python -m aces.train_stage2 \
  --config ACES/configs/stage02_sampling_smoke.yaml

PYTHONPATH=ACES /data2/wjb/anaconda3/envs/threePM/bin/python -m aces.make_stage2_report \
  --run-dir ACES/runs/stage02_sampling_only_seed3407_smoke
```

低预算 pilot 与正式模板：

```bash
PYTHONPATH=ACES CUDA_VISIBLE_DEVICES=3 /data2/wjb/anaconda3/envs/threePM/bin/python -m aces.train_stage2 \
  --config ACES/configs/stage02_sampling_pilot.yaml

# Full 100-epoch template; do not start casually unless the GPU budget is planned.
PYTHONPATH=ACES CUDA_VISIBLE_DEVICES=3 /data2/wjb/anaconda3/envs/threePM/bin/python -m aces.train_stage2 \
  --config ACES/configs/stage02_sampling_full.yaml
```

Stage 2 五档采样：

- `valid`: 原 VALID Tetris sampler，作为 baseline。
- `sn2n_xy`: XY-only 四相位采样，保留 z 轴，不宣称等价 SN2N。
- `iso3d`: 确定性 2x2x2 tetrahedral corner sampler。
- `empirical_block`: 由 Stage 1 foreground repeat Q 曲线阈值导出整数半径，再采样 block corner。
- `ellipsoid`: 用同一半径构造椭球度量，在候选内选择高 Q offset。

## 数据与标定

- **数据来源**：`3P_neuron_raw.tif`（原始形状 `[2475, 512, 512]`）拆分为
  `[9 repeats, 275 z, 512 y, 512 x]` 的重复体积栈，位于
  `Methods/FAST-main/.../all_repeats_275z`。9 个 repeat 是同一场景的重复扫描，
  它们之间的差异被当作**真实噪声**（repeat-noise 分支的依据）。
- **XY 像素间距 = 0.8096639067 μm/px**：来自原始 TIFF Info 块的
  `XResolution=YResolution=12350.803731437707 px/cm`（单位 Centimeter），
  换算 `10000 μm / 12350.8037 = 0.8097 μm/px`。
  ⚠️ 之前配置里的 `0.0087890625` 是 ScanImage 的 `pixelToRefTransform`
  （参考坐标系归一化系数），**不是物理像素间距**，已更正。
- **Z 步长 = 4.0 μm/group**：原始 ImageJ wrapper 写的是 `spacing=0.4 μm`，但它把
  2475 帧当成平铺 z 栈；帧序审计确认真实布局是 275 个深度组 × 9 repeats。
  `dz=4.0 μm/group` 由更正后的布局 + 对齐论文 Fig.2 的 0–1100 μm 深度范围推得，
  **不是 raw TIFF 中干净给出的单一 z 标定**，属于推导值，使用时需谨慎。
- 标定值只用于把体素距离换算成 `distance_um` 输出列；**门控判据用的是无量纲的
  体素距离比值**，因此更正 XY 标定不改变门控结论，只让物理距离标注正确。

## 三个核心指标（每个方向 x / y / z 各算一遍）

对每条轴、每个体素间距 d，比较"相距 d 的两个体素"的配对关系：

- **N（Noise independence，噪声独立性）**：`N = 1 − |corr(噪声(p), 噪声(p+d))|`。
  N 越接近 1，说明相距 d 的两点噪声越独立，越适合做自监督配对（一个当输入、
  一个当目标时不会互相泄漏噪声）。有两个来源：
  - **repeat 分支**：用 9 个 repeat 之间的真实差异作为噪声，最可信。
  - **proxy 分支**：用单帧减高斯平滑得到的残差作为噪声代理，作为交叉验证。
- **S（Structure continuity，结构连续性）**：`S = corr(结构(p), 结构(p+d))`。
  S 越接近 1，说明相距 d 的两点结构越连续（还是"同一个东西"）。z 方向因
  ~4μm 大步长，结构衰减快，S 会明显低于 x/y——这正是各向异性的物理信号。
- **Q（综合配对性）**：把 N 与 S 结合的综合分数，代表"该方向、该距离上做配对采样
  的整体适配度"。门控用的就是 Q。

## 门控判据（是否放行到 Stage 2）

- 判据：**前景区** Q 的 **XY/Z 比值（d=1..3 平均）≥ 1.2** 即判 `observed=True`
  （存在各向异性，配对性前提成立）。当前结果 **1.3134，PASS**。
- **为什么只看前景**：背景约占 86% 体素、几乎无结构，会把全区域比值稀释到 ~1.03
  造成假阴性。因此门控基于前景区（按亮度分位数 `foreground_percentile=75` 划定），
  报告中同时保留全区域数据仅作参考。
- **为什么以 S 为主**：结构连续性 S 是物理上最稳健的各向异性信号；噪声独立性 N
  在 z 方向可能偏弱甚至反向（见下方相关性表 z 行为负），不能据此说"无各向异性"。
- 若门控为 `REVIEW`/不通过，Stage 2 采样训练不应启动，需先重查配对性前提。

## 产物解读

运行目录 `runs/stage01_3p_repeat_pairability_seed3407/` 下：

- **`report/stage01_pairability_report.md`**：主报告，先看这个。含门控结论、环境表、
  前景/全区域两套 Q/N/S 表、proxy-vs-repeat 相关性表。
- **`metrics/pairability_curves.csv`**：全局逐轴逐距离的 N/S/Q 原始值
  （列含 `distance_px`、`distance_um`、`region`、`noise_source` 等），可自行复算。
- **`metrics/depth_conditioned_pairability.csv`**：按深度分箱（`depth_bins=5`）的
  同类指标，用于看各向异性是否随成像深度变化。
- **`metrics/proxy_repeat_correlations.json`**：proxy 分支与 repeat 分支的
  Pearson/Spearman 相关性，验证"用单帧代理噪声"是否能代表真实重复噪声
  （前景 Q 相关达 0.997–0.999，说明代理可信）。
- **图 `figures/q_curves_xyz.png`**：横轴为体素距离 d，三条曲线分别是 x/y/z 的 Q
  （或 N/S）随距离衰减。z 曲线明显低于 x/y = 各向异性直观证据。
- **图 `figures/proxy_vs_repeat.png`**：proxy 指标 vs repeat 指标的散点，越贴近对角线
  说明代理越可信。
- **图 `figures/depth_q_heatmaps.png`**：深度 × 距离的 Q 热图，看各向异性随深度的分布。

## Stage 2 产物解读

每个采样档在 `runs/stage02_*/<mode>/` 下产出：

- **`logs/train_losses.csv`**：逐 batch 的 `total_loss / loss2neighbor / loss_idt / loss_reg`。
  `loss2neighbor` 是"输出对邻域视图"的 N2N 主损失，`loss_idt` 是两条输出分支的一致性，
  `loss_reg` 是 DWT 子带上的 Hessian 正则。看 total_loss 是否平滑下降、无发散。
- **`logs/eval_curve.csv`**（`eval_every_epochs>0` 时）：每评估轮的前景/全区域 PSNR。
  用来判断收敛点和是否过拟合（数据仅 125 patch，验证仅 18 patch，需盯过拟合）。
- **`logs/sampler_metadata.json`**：每次采样调用的 offset、半径、输入输出形状，验证采样算子行为。
- **`metrics/eval_metrics.json`**：全分辨率下 all/foreground/background 三区的 PSNR（mean9 代理为参考）。
- **`metrics/eval_metrics_aligned.json`**（仅下采样档 valid/sn2n_xy/iso3d 产出）：
  把模型输出、目标、mask 按训练时的实际 stride 下采样后再评估，消除"训练在低分辨率、
  评估在全分辨率"的尺度错配。偏移档（empirical_block/ellipsoid）stride=1、训练即全分辨率，
  对齐值等于全分辨率值，因此不产出该文件、报告中显示 `-`。
- **`checkpoints/<mode>_best.pth` / `<mode>_epochNNNN.pth`**：best 按前景 PSNR 选出。

报告 `report/stage02_sampling_report.md` 汇总所有档：`PSNR fg` 是全分辨率对比，
`PSNR fg (aligned)` 是对齐后对比，两列并存以便公平比较不同 stride 的采样档。

## 复现性说明

深度分箱的随机采样种子使用 `blake2b` 稳定哈希(而非 Python 内置 `hash()`,后者受
`PYTHONHASHSEED` 逐进程加盐、跨进程不可复现)。相同配置多次运行结果按位一致。

## Best vs Final Epoch 对比分析

**位置**: `runs/stage02_neurofinder_5mode_80ep_seed3407/best_vs_final_comparison/`

完整的best epoch vs final epoch (epoch 80)对比分析,结合定量指标和定性可视化:

### 核心发现
- **所有模式均过拟合**: 从best epoch到final epoch平均下降**4.54 dB**
- **iso3d最稳定**: 仅下降1.94 dB,综合性能与泛化能力最优
- **几何可用性场失败**: ellipsoid和empirical_block过拟合严重(-6.4至-6.8 dB)
- **Early stopping必要**: best epoch普遍在epoch 10-20,继续训练损害性能

### 生成的分析文件
```
best_vs_final_comparison/
├── COMPREHENSIVE_ANALYSIS.md          # 全面分析报告(11KB)
├── comparison_report.md               # 简化版报告
├── best_vs_final_metrics.csv          # 定量指标表
├── quantitative_comparison.png        # 定量对比图(172KB)
├── convergence_degradation.png        # 收敛与退化曲线(211KB)
├── qualitative_comparison.png         # 实际去噪图像对比(3.7MB)
└── difference_maps.png                # 误差热图(1.7MB)
```

### 重新生成分析
```bash
cd runs/stage02_neurofinder_5mode_80ep_seed3407

# 定量分析(秒级)
python compare_best_vs_final.py

# 定性可视化(需GPU,约10-30分钟)
python visualize_best_vs_final_qualitative.py
```

### 实践建议
1. **模型选择**: 优先使用**iso3d best.pth**(epoch 20, 22.92 dB)
2. **训练策略**: 实施early stopping,默认30 epochs + patience=10
3. **Checkpoint使用**: 始终使用best.pth而非final.pth进行推理
4. **避免几何可用性场**: ellipsoid和empirical_block不适合实际应用

**可用性场(Availability Field)**: 自监督降噪中的空间掩码,确定哪些体素可形成有效
噪声对。数据驱动可用性场(基于pairability分析,如iso3d/sn2n_xy/valid)显著优于
几何近似可用性场(如ellipsoid/empirical_block)。
