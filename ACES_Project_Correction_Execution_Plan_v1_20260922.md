# ACES 项目修正与下一阶段执行方案 v1.0

> **版本**：v1.0  
> **日期**：2026-09-22  
> **适用仓库**：`https://github.com/winb0408/Denoising_PM`  
> **当前核心目录**：`ACES/`，重点为 `ACES/qmask/`、`ACES/aces/`、`ACES/example/`  
> **文档定位**：项目总研究主线 + 方法重构方案 + 技术实施规格 + 实验协议 + 代码重构路线 + 阶段性 Gate  

---

## 🗺️ 文档颜色/层级约定

| 标记 | 含义 | 执行级别 |
|---|---|---|
| 🔴 | P0 / blocker | 必须先修，不通过不得进入新方法 |
| 🟠 | P1 / core | 主方法核心实现 |
| 🟡 | P2 / validation | 关键实验与机制验证 |
| 🟢 | P3 / application | 真实数据与生物学验证 |
| 🟣 | Future | 后续扩展，不进入 A3 v1 主线 |

---

## 🧭 0. 执行摘要

经过对当前仓库、QMask 代码、Stage 1/2 实验、已有实验记录以及 `ACES/example` 实际数据说明的综合审阅，本项目下一阶段不再继续沿着“QMask + 更多阈值/半径实验”线性扩展，而是进行一次**研究问题层面的升级**。

### 当前项目得到的核心认识

1. **2P/3P 体成像确实存在显著的空间各向异性**，尤其是 z 方向的结构连续性与 xy 方向不同。
2. 原始 ACES 的 `max{d: Q(d)>τ}` 型距离/椭球半径存在逻辑缺陷：它倾向于选择“最远仍满足阈值”的危险距离。
3. `empirical_block / ellipsoid` 的大位移 pair 会破坏 N2N 所需的“同一潜在内容 + 独立噪声”近似，因此其训练失败具有明确的机制解释。
4. QMask 将“半径”修正为“轴参与”是合理的，但它本质上只回答了“哪个轴参与”，没有回答“哪些具体 observation pair 最适合训练”。
5. 当前 r-grid / s-grid / shortcut-risk 实验进一步暴露了一个更重要的问题：
   
   $$
   \text{pair similarity / pairability}
   \neq
   \text{training utility}.
   $$
6. 因此，下一阶段真正值得研究的不是另一个固定采样模板，而是：
   
   > **在没有 clean ground truth 的情况下，从数据本身估计不同 observation pairs 的风险和训练效用，并据此自动形成 sampling policy。**

### 🎯 下一阶段总目标

> **面向 2P/3P 各向异性体成像，建立一种无需 clean ground truth 的数据校准自适应配对采样框架，从候选 observation pairs 中识别具有可靠结构一致性、低噪声依赖和低 shortcut 风险、同时对自监督优化真正有益的训练对，并使采样策略进一步随成像方向、深度和局部信号质量自适应，在抑制噪声的同时提高深层弱神经元信号的 SNR/CNR/SBR 与可检测性。**

### 🔑 最终主线

```text
Raw noisy volumetric data
        │
        ▼
Observation redundancy profiling
        │
        ▼
Candidate pair / sampling-family library
        │
        ▼
Pair-risk field
 ┌──────────────┬──────────────┬──────────────┐
 │ Noise risk   │ Structure    │ Shortcut     │
 │              │ mismatch     │ risk         │
 └──────────────┴──────────────┴──────────────┘
        │
        ▼
Optimization-aware micro-probe
        │
        ▼
Pair / family utility estimation
        │
        ▼
Adaptive sampling policy
        │
        ├── spatial anisotropy
        ├── depth condition
        └── local signal-quality condition
        │
        ▼
Self-supervised denoising
        │
        ▼
Deep-signal recovery
        ├── PSNR / SSIM
        ├── SNR / CNR / SBR
        ├── signal retention
        └── neuron detectability
```

---

# 📊 当前阶段证据定位

当前仓库已经形成了足以支撑“方法重构”的开发证据，但尚未形成最终论文级 blind benchmark。

### 已有证据可以支持的结论

| 发现 | 当前证据 | 定位 |
|---|---|---|
| 空间各向异性存在 | Stage 1 repeat/proxy pairability 曲线 | 机制证据 |
| naive radius/ellipsoid 不可靠 | empirical/ellipsoid 与 r-grid 结果 | 失败实验/动机 |
| co-located phase split 更合理 | QMask + VALID-like sampling | sampler 语义修正 |
| fixed view→corner mapping 有 shortcut 风险 | Q-3 permutation / mosaic 消融 | 机制证据 |
| pairability 不等于 training utility | r-grid / s-grid / shortcut 分析 | A3 核心动机 |
| QMask 可以正常训练 | Neurofinder 3-seed 80-epoch runs | 工程可行性 |

### 当前证据不能支持的结论

目前还不能据此声称：

- QMask 已经是最优采样策略；
- QMask 已经实现真正 distance-aware sampling；
- Neurofinder 的 `z` 结果代表 physical depth；
- 当前 PSNR 结果已经等价于 blind test；
- 当前 `ACES/example` 已经完成严格的深层信号恢复验证。

因此，下一阶段不是“把现有结果包装成论文”，而是利用已有失败/机制结果，把研究问题推进到 pair utility + adaptive policy。

# 🟦 1. 项目研究问题重新定义

## 1.1 原问题

原始 ACES 主要在解决：

> “针对各向异性的 2P/3P 体数据，应该给 x/y/z 三个方向多大的 sampling radius？”

数学上近似为：

$$
r_a = \max\{d:Q_a(d)>\tau\}.
$$

该定义已经被现有实验否证。

---

## 1.2 修正后的问题

下一阶段将研究问题提升为：

> **对于某个 noisy volumetric observation，哪些空间 observation pairs / sampling families 是适合用于 self-supervised denoising 的？这种适合性是否随方向、距离、深度和局部信号质量变化？**

设候选 offset 为：

$$
\Delta=(\Delta_x,\Delta_y,\Delta_z)
$$

以后若扩展动态成像：

$$
\Delta=(\Delta_x,\Delta_y,\Delta_z,\Delta_t).
$$

定义 pair utility：

$$
\Delta^*=
\arg\max_{\Delta\in\mathcal D}U(\Delta\mid q),
$$

其中：

- `\mathcal D`：candidate offset library；
- `q`：局部/深度/数据状态；
- `U`：pair utility。

最终方法不再是一个 binary mask，而是：

$$
\boxed{
\pi(\Delta\mid q)
}
$$

即**自适应 sampling policy**。

---

# 🟩 2. 项目最终定位：QMask 是中间方法，不是最终方法

当前研究路径应明确整理为：

```text
Original ACES
   │
   ├── Pairability Q
   │      ↓
   │   radius / ellipsoid
   │      ↓
   │   ❌ 机制失败
   │
   └── QMask
          ↓
      axis participation
          ↓
      ✅ 正确修正
          ↓
      但仍只回答 axis-level question
          ↓
   Pair-risk / Pair-utility
          ↓
   Adaptive sampling policy
          ↓
   最终方法
```

## 2.1 QMask 应保留的价值

QMask 不应删除，未来作为以下 baseline / ablation：

- `VALID`：固定 3D orthogonal sampling baseline；
- `SN2N-like XY`：固定 XY sampling baseline；
- `QMask`：data-calibrated hard axis policy；
- `Adaptive-hard`：根据 utility 选择单个最优 family；
- `Adaptive-soft`：根据 utility distribution 进行随机采样；
- `Depth-adaptive`：加入深度/局部信号条件。

因此 QMask 在论文中的地位应是：

> **证明从 fixed geometry → axis-level data calibration 是合理的，但 axis-level binary decision 仍不足以描述真正的 optimal self-supervised sampling。**

---

# 🟧 3. 主要创新点重新确定

下一阶段不要继续以“新的 Tetris pattern”作为核心贡献。最终创新建议分成三层。

## Innovation 1 — Pair-risk characterization

将 self-supervised pair 的可用性从单一 `Q` scalar 分解成多个独立风险维度：

$$
\mathbf R(\Delta)=
[R_N(\Delta),R_S(\Delta),R_I(\Delta)].
$$

其中：

- $R_N$：noise dependence risk；
- $R_S$：structural mismatch risk；
- $R_I$：identity / shortcut risk。

核心思想：

> **Noise independence、structural consistency 和 shortcut susceptibility 是三个不同问题，不能再简单压缩成一个 `N×S` scalar。**

---

## Innovation 2 — Pair utility calibration

进一步提出：

$$
\text{static pair validity}
\neq
\text{optimization utility}.
$$

针对每个候选 pair / sampling family，进行极低预算的 self-supervised micro-probe，用训练过程自身的统计量估计其实际 utility。

核心形式：

$$
\phi(\Delta)=
[L_N,L_{idt},R_{res},G_{pres},C_{stability},R_I,\ldots]
$$

然后：

$$
U_{probe}(\Delta)=g(\phi(\Delta)).
$$

第一版优先使用可解释的标准化组合；如果实验发现简单组合与实际 utility 相关性不足，再进入跨数据集的 utility predictor calibration。

---

## Innovation 3 — Signal-aware adaptive sampling

最终使：

$$
\pi(\Delta\mid q)
$$

随数据状态变化，其中：

$$
q=[q_{depth},q_{signal},q_{noise},q_{anisotropy}].
$$

最终目标不是简单提升平均 PSNR，而是：

> **在深层低信号、高背景、强噪声区域自动选择更可靠、更适于训练的 observation pairs，最大程度保持真实 neuron morphology，同时提升 SNR/CNR/SBR。**

---

# 🟥 4. 第一阶段必须完成的工程/理论纠偏（P0）

这些工作不属于“新方法”，而属于进入新方法之前必须清理的地基。

## P0-1：统一 spatial / temporal / depth 语义

Neurofinder 数据本质是：

$$
[T,Y,X]
$$

其中第一维为时间，不能继续在通用代码中命名为 `Z` 并解释成 physical depth。

统一约定：

```text
static volumetric imaging : [Z,Y,X]
dynamic imaging            : [T,Y,X]
volumetric time-lapse      : [T,Z,Y,X]
```

建议将所有 API 最终使用：

```python
axis_names = ("x", "y", "z", "t")
```

实际存储可继续使用 `ZYX`，但 metadata 中必须明确物理语义。

---

## P0-2：删除“计算了但没有真正使用”的 geometry

当前 QMask：

$$
d_{min},d_{max}
$$

被计算，但 v1 实际 phase split 仍固定使用 1 voxel。

必须做到二选一：

### 方案 A：进入新的 offset policy

让 `d_min/d_max` 成为 risk field 的辅助特征，而非最终采样距离。

### 方案 B：退出主方法

不再伪装成“distance-aware sampler”。

本项目选择 **A**。

---

## P0-3：彻底拆开 N / S / Q

禁止继续在文档中写：

> “N 不参与 decision，但 Q 仍决定 decision”。

后续主方法不再依赖：

$$
Q=N\times S
$$

作为唯一 decision score。

Q 只作为：

- legacy diagnostic；
- historical baseline；
- ablation。

---

## P0-4：修复 evaluation protocol

最终实验必须统一：

```text
Calibration
    ↓
Validation
    ↓
checkpoint / threshold / policy selection
    ↓
Blind Test
```

Blind Test 禁止参与：

- offset selection；
- threshold selection；
- geometry selection；
- checkpoint selection；
- policy temperature selection。

---

## P0-5：统一 SNR / CNR / SBR 定义

建议以后代码层面定义成 intensity-domain 指标。

### SNR

推荐主定义：

$$
SNR=
\frac{\mu_S-\mu_B}{\sigma_B+\epsilon}.
$$

如需 dB：

$$
SNR_{dB}=20\log_{10}
\left(\frac{\mu_S-\mu_B}{\sigma_B+\epsilon}\right).
$$

### CNR

推荐：

$$
CNR=
\frac{|\mu_S-\mu_B|}
{\sqrt{\sigma_S^2+\sigma_B^2}+\epsilon}.
$$

### SBR

定义：

$$
SBR=\frac{\mu_S}{\mu_B+\epsilon}.
$$

如果转 dB：

$$
SBR_{dB}=10\log_{10}(SBR).
$$

> 后续论文必须在 Methods 中明确 ROI/background 定义、统计区域以及是否使用 dB。

---

## P0-6：修复 SSIM region 统计

目前 `evaluate_stage2.py` 的 `_region_metrics()` 对 full patch 计算一次 SSIM，然后把相同 SSIM 值写进 all/foreground/background 行。

后续：

- `SSIM_full`：全 patch；
- `SSIM_fg`：若有足够连通区域，可使用 masked/local SSIM；
- 不再把重复数值伪装成 region-specific SSIM。

如果 masked SSIM 不稳定，则正式主表只报告 full SSIM，foreground 主要使用 PSNR/CNR/SBR/SNR。

---

## P0-7：统一 checkpoint 选择

最终统一使用：

> **Validation-selected best checkpoint**

不再使用：

- final checkpoint 作为主结果；
- test proxy 选择 best epoch；
- 多套 full/aligned 结果中人为挑一个。

正式结果必须保存：

```text
selected_epoch
selection_metric
selection_dataset
selection_protocol
```

---

# 🟨 5. A3 新方法总体架构

新方法暂命名：

> **Pair-Utility Calibrated Adaptive Sampling**

中文：

> **配对效用校准自适应采样**

名称暂不作为最终论文 acronym 锁定，避免过早限制方法发展。

---

## 5.1 方法输入输出

### Input

一个 noisy image / volume：

$$
Y=S+N.
$$

### Calibration data

优先：

- 9-repeat 3P 数据；
- 具有 25-frame 重复观测的数据；
- 其他存在 independent noisy repeats 的数据。

### Output

一个 data-conditioned policy：

$$
\pi(\Gamma\mid q),
$$

其中 `\Gamma` 是 sampling family，而不是简单单一 offset。

---

# 🟦 6. 为什么从“offset”进一步升级成“sampling family”

现有 VALID 是 4-view self-supervised training，因此下一阶段不能简单把一个 pair distance 换来换去，然后同时改变 view 数量与 loss。

因此定义：

$$
\Gamma=
\{\delta_1,\delta_2,\delta_3,\delta_4\}
$$

其中每个：

$$
\delta_i=(\delta_{ix},\delta_{iy},\delta_{iz}).
$$

每个 view：

$$
V_i(k)=Y(Bk+\delta_i),
$$

其中：

- `B`：sampling stride；
- `k`：coarse-grid coordinate。

因此四个 view 始终被重新映射到同一 coarse grid。

### Pairwise offset

任意两个 view 的相对 offset：

$$
\Delta_{ij}=\delta_j-\delta_i.
$$

所以一个 family 可以对应多个 pairwise offsets。

### Family-level risk

$$
R_N(\Gamma)=
\mathrm{mean}_{i<j}R_N(\Delta_{ij})
$$

$$
R_S(\Gamma)=
\mathrm{mean}_{i<j}R_S(\Delta_{ij})
$$

$$
R_I(\Gamma)=
\mathrm{mean}_{i<j}R_I(\Delta_{ij}).
$$

同时记录 worst-pair：

$$
R^\max_k(\Gamma)=
\max_{i<j}R_k(\Delta_{ij}).
$$

这样可以统一：

- VALID family；
- SN2N-like family；
- QMask family；
- 新 adaptive family。

---

# 🟩 7. Candidate Sampling Family Library

## 7.1 第一版原则

不要一开始搜索整个无限 3D offset space。

建立可控 candidate library：

$$
\mathcal G=
\{\Gamma_1,\Gamma_2,\ldots,\Gamma_M\}.
$$

第一版候选应覆盖三个层次：

### A. Axis families

```text
x-only
 y-only
 z-only
xy
xz
yz
xyz
```

用于覆盖 QMask/VALID/SN2N-like 端点。

### B. Diagonal families

包括：

```text
(1,1,0)
(1,0,1)
(0,1,1)
(1,1,1)
```

以及对应的互补 phase 组合。

### C. Multi-distance families

后续加入：

```text
(2,0,0)
(0,2,0)
(0,0,2)
(2,2,0)
(2,0,2)
(0,2,2)
(2,2,2)
```

再视结果加入 4/8 voxel scale。

---

## 7.2 Physical-space candidate library

真实 2P/3P 中不能只记录 pixel distance。

每个 candidate 必须同时保存：

```json
{
  "offset_voxel": [dx, dy, dz],
  "spacing_um": [sx, sy, sz],
  "offset_um": [dx*sx, dy*sy, dz*sz]
}
```

即：

$$
\Delta_{um}=
(\Delta_xs_x,
\Delta_ys_y,
\Delta_zs_z).
$$

这样才能真正讨论“各向异性”。

> 需要注意：当前 3P 仓库 README 中 z spacing 属于根据数据布局和图示推导的标定值，正式论文前必须进一步确认，不应把推导值包装成原始设备 metadata。

---

# 🟨 8. Pair-Risk Field：三个核心风险

---

## 8.1 Noise Risk $R_N$

### 8.1.1 有真实 repeat 时

对 repeat：

$$
Y_k=S+N_k.
$$

估计：

$$
\bar Y=\frac1K\sum_kY_k
$$

$$
\hat N_k=Y_k-\bar Y.
$$

对 offset $\Delta$：

$$
\rho_N(\Delta)=
\mathrm{corr}
(\hat N_k(r),
\hat N_k(r+\Delta)).
$$

定义：

$$
R_N(\Delta)=|
ho_N(\Delta)|.
$$

风险越低越好。

### 8.1.2 进一步加入 variance consistency

记录：

$$
R_{NV}(\Delta)=
\left|
\log
\frac{\sigma_N^2(r+\Delta)}
{\sigma_N^2(r)+\epsilon}
\right|.
$$

最终第一版可以：

$$
R_N^*(\Delta)=
\alpha_N\tilde R_N+
(1-\alpha_N)\tilde R_{NV}.
$$

### 8.1.3 没有 repeat 时

使用 proxy：

- high-pass residual；
- frame split residual；
- local residual；

但明确标记为：

```text
noise_source = repeat
noise_source = split
noise_source = proxy
```

不能把 proxy 当作 ground truth noise。

---

# 🟥 9. Structure Risk $R_S$

单一 Pearson 不足以描述 neuron morphology。

第一版至少采用三项：

### 9.1 Low-frequency structure correlation

对结构代理：

$$
S_{LF}=G_\sigma(Y)
$$

计算：

$$
\rho_{LF}(\Delta)=
\mathrm{corr}
(S_{LF}(r),
S_{LF}(r+\Delta)).
$$

定义：

$$
R_{S1}=1-\rho_{LF}.
$$

### 9.2 Local SSIM

$$
R_{S2}=1-SSIM_{local}(r,r+\Delta).
$$

### 9.3 Gradient structure consistency

$$
R_{S3}=1-
\mathrm{corr}
(\nabla S(r),\nabla S(r+\Delta)).
$$

最终：

$$
R_S=
\alpha_1\tilde R_{S1}
+\alpha_2\tilde R_{S2}
+\alpha_3\tilde R_{S3}.
$$

第一版建议权重：

```text
LF correlation : 0.5
local SSIM     : 0.3
gradient       : 0.2
```

这些权重不是最终论文固定参数，Phase 1 需要做 sensitivity test；如果结果明显依赖固定权重，应升级为 calibrated predictor。

---

# 🟧 10. Identity / Shortcut Risk $R_I$

如果：

$$
Y(r)\approx Y(r+\Delta),
$$

模型容易选择：

$$
\hat S\approx Y.
$$

因此计算：

$$
nMSE(\Delta)=
\frac{
E[(Y(r)-Y(r+\Delta))^2]
}
{Var(Y)+\epsilon}.
$$

定义高 similarity = 高 shortcut risk：

$$
R_I(\Delta)=
\exp\left(
-\frac{nMSE(\Delta)}{\tau_I}
\right).
$$

可进一步乘以：

$$
\max(\rho_Y(\Delta),0)
$$

以降低 flat background 误判。

### 核心物理关系

理想 pair 不是：

```text
距离越近越好
```

也不是：

```text
距离越远越好
```

而应存在：

```text
too close
   ↓
shortcut risk ↑

intermediate
   ↓
useful pair region

too far
   ↓
structure risk ↑
```

这正是后续实验应重点验证的 U-shaped / trade-off mechanism。

---

# 🟩 11. Composite Pair-Risk

风险向量：

$$
\mathbf R(\Delta)=
[R_N,R_S,R_I].
$$

第一版可以定义一个诊断 score：

$$
R_{base}(\Delta)=
\lambda_NR_N+
\lambda_SR_S+
\lambda_IR_I.
$$

但必须强调：

> **`R_base` 仅作为 feasibility / diagnostic score，不是最终 utility。**

推荐先设置：

```text
λN = 0.35
λS = 0.40
λI = 0.25
```

随后在 calibration set 上进行 sensitivity analysis。

若方法对这些权重高度敏感，则不能继续手工调权重，应进入 learned utility predictor。

---

# 🟦 12. Pair Utility：从“合理”走向“有用”

这是 A3 的真正核心。

定义：

$$
U(\Gamma)
$$

表示一个 sampling family 对自监督训练的实际价值。

基本思想：

```text
physical/statistical validity
             ≠
optimization utility
```

因此需要 micro-probe。

---

# 🟨 13. Micro-Probe 技术规格

## 13.1 基本协议

对 candidate family $\Gamma$：

```text
固定：
  network architecture
  optimizer
  learning rate
  loss
  patch size
  batch size
  random seed

唯一改变：
  sampling family Γ
```

训练：

```text
1–3 epochs
小规模 patch subset
每个 candidate 相同计算预算
```

建议第一版：

```text
micro epochs = 2
batches       = 100–200
patches       = 64–128
seeds         = 2
```

如果 GPU 预算允许，正式 calibration 使用 3 seeds；探索阶段 1 seed 即可。

---

## 13.2 Micro-Probe 输出

每个 family 保存：

```text
loss_neighbor_mean
loss_neighbor_slope
loss_identity_mean
loss_identity_slope
residual_corr
output_input_corr
output_gradient_corr
output_gradient_ratio
output_variance_ratio
cross_view_consistency
loss_stability
```

### 推荐构造 feature vector

$$
\phi(\Gamma)=
[
-z(L_N),
-z(L_I),
-z(R_{res}),
+z(G_{pres}),
+z(C_{view}),
-z(V_{loss}),
-z(R_I)
].
$$

所有量先在同一 calibration batch 内做 robust z-score。

---

# 🟥 14. Micro-Probe Utility v1

第一版不引入 RL / transformer policy。

使用：

$$
U_{probe}(\Gamma)=
\sum_jw_j\phi_j(\Gamma).
$$

初始建议：

```text
neighbor loss improvement     +0.25
identity-risk reduction       +0.15
residual independence         +0.15
gradient preservation         +0.20
cross-view consistency        +0.10
training stability            +0.05
shortcut penalty              +0.10
```

具体实现时统一成“高 utility = 好”。

### 注意

这些权重是**实验初始值，不应直接写进论文作为最终理论参数**。

Phase 1 必须检验：

$$
\mathrm{Spearman}
(U_{probe},
U_{oracle})
$$

是否显著优于：

$$
\mathrm{Spearman}
(Q,U_{oracle}).
$$

如果不能，禁止直接进入 adaptive policy。

---

# 🟩 15. Utility Oracle 只用于方法开发，不用于部署

为了验证：

> micro-probe 是否真的能预测最终训练价值？

在拥有真实参考的 development laboratory 中计算：

$$
U_{oracle}(\Gamma)=
Metric_{val}(\Gamma)-Metric_{reference}.
$$

优先使用：

1. PSNR gain；
2. CNR gain；
3. deep-region signal quality gain。

但最终必须保持：

```text
oracle utility
     ↓
仅用于验证 micro-probe 是否预测得准

micro-probe utility
     ↓
真实部署时做 policy selection
```

换言之：

> **GT/repeat reference 可以用于验证“我们的自监督信号是否真的有预测能力”，但不能作为 deployment-time policy selector。**

这是方法可信度非常重要的一条边界。

---

# 🟧 16. 如果 Micro-Probe Utility v1 失败怎么办

不要直接堆更复杂的网络。

进入：

## Utility Predictor v2

学习：

$$
\hat U=g_\theta(\phi(\Gamma),q).
$$

训练 target 来自 calibration laboratory 的 oracle utility，但在新数据推理时只需要：

$$
\phi(\Gamma),q.
$$

候选模型优先：

```text
Ridge
Elastic Net
Gradient Boosted Trees
Lightweight MLP
```

不建议第一版使用 Transformer / RL。

### v2 的科学意义

这将把方法升级成：

> **cross-dataset learned pair-utility predictor**

而不是某一个数据集上 manually tuned sampling heuristic。

---

# 🟦 17. Adaptive Sampling Policy

完成 utility 后构造：

$$
\pi(\Gamma\mid q)=
\frac{
\exp(U(\Gamma\mid q)/T)
}
{
\sum_{\Gamma'}
\exp(U(\Gamma'\mid q)/T)
}.
$$

其中：

- $T$：temperature；
- $q$：数据状态。

### 第一版 policy

先实现 hard：

$$
\Gamma^*=\arg\max_\Gamma U(\Gamma).
$$

再实现 soft：

$$
\Gamma\sim\pi(\Gamma).
$$

这样可以分离：

```text
utility estimation
        ↓
selection mechanism
```

---

# 🟨 18. 为什么推荐 soft policy

hard policy：

```text
每次永远选同一个 family
```

容易过拟合某一 geometry。

soft policy：

```text
high utility family    → high probability
medium utility family  → low probability
unsafe family           → near zero
```

同时可以定义安全约束：

$$
R_N<R_N^{max}
$$

$$
R_S<R_S^{max}.
$$

因此：

$$
\pi(\Gamma)=0
$$

对于明显非法的 family。

---

# 🟥 19. Depth-/Signal-Conditioned Policy

这是将方法与真实科学目标连接起来的关键步骤。

定义状态：

$$
q_p=
[q_d,q_s,q_n,q_a].
$$

其中：

### Depth state

$$
q_d=\frac{z}{z_{valid,max}}.
$$

### Signal state

不依赖 clean GT，而从 local signal/background statistics 获得：

$$
q_s=\frac{\mu_{local}-\mu_{bg}}
{\sigma_{bg}+\epsilon}.
$$

这是 deployment-time 的 proxy signal quality。

### Noise state

来自：

- repeat residual；
- frame split residual；
- local high-pass variance。

### Anisotropy state

来自：

$$
R_S(\Delta_x),R_S(\Delta_y),R_S(\Delta_z).
$$

---

## 19.1 最终 policy

$$
\boxed{
\pi(\Gamma\mid z,signal,noise,anisotropy)
}
$$

最终形成：

```text
shallow + high SNR
      ↓
普通 pair library 可用

intermediate depth
      ↓
更高信息量 pair

Deep + weak signal + high background
      ↓
保守结构匹配
+
低 shortcut risk
+
高 training utility
```

注意：最终哪种 offset 在深层最优必须由实验决定，不能预先假设“z 一定关闭”。

---

# 🟩 20. Patch-level policy 优先于 voxel-level policy

第一版不做每 voxel routing。

推荐：

$$
\pi(\Gamma|p)
$$

以 patch `p` 为决策单位。

原因：

- 计算可控；
- policy 更稳定；
- 容易与现有 `ReadDatasets` 对接；
- 更容易进行 depth bin / SNR bin 分析。

未来可扩展到：

$$
\pi(\Gamma|r)
$$

但不属于第一版主线。

---

## 20.1 Adaptive sampler v1 伪代码

```text
Input patch P
  │
  ├─ estimate state q(P)
  │      depth / local signal / noise / anisotropy
  │
  ├─ load candidate families Γ1...ΓM
  │
  ├─ retrieve calibrated risk features
  │      RN / RS / RI / overlap / utility features
  │
  ├─ hard safety filtering
  │      reject if RN > τN or RS > τS
  │
  ├─ estimate utility U(Γ | q)
  │
  ├─ hard mode:
  │      Γ* = argmax U
  │
  └─ soft mode:
         π(Γ|q) = softmax(U/T)
         sample Γ ~ π
              │
              ▼
        construct 4 views
              │
              ▼
        existing denoiser
```

### 第一版实现边界

- policy 在 patch level 决策；
- candidate family 在训练前生成；
- 不在每个 voxel 上做 policy inference；
- 不在线更新 policy；
- 不使用 reinforcement learning；
- 不修改主干 network；
- 不把 test 数据送入 policy calibration。

# 🟦 21. 训练阶段的 policy update：暂缓

暂时不实现：

```text
online RL
meta-learning
full bilevel optimization
policy gradient
```

后续可实验：

$$
\pi_0
\rightarrow
\pi_1
\rightarrow
\pi_2
$$

即 curriculum / online policy adaptation。

但只有在静态 policy 明确有效之后才值得做。

---

# 🟪 22. 数据使用策略重新设计

## 22.1 9-repeat 3P 数据

定位为：

> **Pair-risk / utility development laboratory**

因为它同时提供：

- true noise residual；
- noise covariance；
- structure reference；
- candidate family oracle utility。

建议在这个数据上完成：

```text
Δ / Γ
 ↓
R_N
R_S
R_I
 ↓
Micro-probe
 ↓
U_probe
 ↓
Oracle U
 ↓
验证相关性
```

---

## 22.2 Neurofinder

定位改为：

> **Temporal redundancy / mechanism validation benchmark**

统一数据语义：

```text
[T,Y,X]
```

主要研究：

- temporal pair utility；
- shortcut risk；
- training stability；
- self-supervised consistency。

不再使用它作为 physical-depth evidence。

---

# 🟨 23. 真实 2P/3P 数据：从 example 升级为核心应用验证

当前 `ACES/example/readme.txt` 已明确描述真实应用需求：

- 三维 XYZ 数据；
- 不同深度信号强度不同；
- 红色通道特定区域背景噪声较强；
- 绿色通道特定区域存在较强背景；
- 目标是改善深层信号并提高信背比。

这些信息使真实数据应该从“demo”升级成：

> **Real-data deep-signal validation benchmark**

---

# 🟥 24. 真实 25-frame 数据的正确用法

当前存在：

$$
AVG_{25}
$$

建议以后把它定位为：

> high-SNR proxy / reference

而不是训练 target。

---

## 24.1 Independent split

如果 25 帧代表同一体数据的重复观测，建议：

$$
A=\{1,3,5,\ldots\}
$$

$$
B=\{2,4,6,\ldots\}
$$

得到：

$$
Y_A,Y_B.
$$

分别进行：

$$
Y_A\rightarrow\hat S_A
$$

$$
Y_B\rightarrow\hat S_B.
$$

`AVG25` 只用于：

- validation / sanity check；
- signal preservation reference；
- final figure reference。

### 必须先检查

若 25 frame 之间存在运动、生理变化、漂移或 bleaching：

1. 先 registration；
2. 再 independent split；
3. 做 split-consistency analysis；
4. 若变化不能忽略，则不要把它们当作严格 i.i.d. repeats。

---

# 🟩 25. 真实数据预处理协议

建议统一：

```text
Raw frames
    ↓
Flat-field / invalid-plane check
    ↓
Registration (if needed)
    ↓
Independent frame split
    ↓
Intensity normalization metadata
    ↓
ROI / background definition
    ↓
Patch extraction
    ↓
Calibration / Validation / Blind Test
```

### Artifact exclusion

`obj.tif` 中用户已经指出约 1.8 mm 以下存在物镜卡住导致的伪影。

代码中应显式保存：

```json
{
  "valid_depth_um": [z0, z1],
  "excluded_artifact_reason": "objective obstruction"
}
```

不要把排除深度写死在模型代码里。

---

# 🟦 26. Deep-signal evaluation：从 PSNR 转向 depth-aware metrics

最终核心结果必须包括：

$$
SNR(z)
$$

$$
CNR(z)
$$

$$
SBR(z)
$$

以及结构保持。

建议深度统计单位为 physical `μm`，而不是单纯 voxel index。

---

## 26.1 Depth bins

实际 bin 应依据真实 z metadata 自动生成。

建议第一版：

```text
Bin 1: shallow
Bin 2: mid-1
Bin 3: mid-2
Bin 4: deep-1
Bin 5: deepest valid
```

避免在不同数据集上强制使用同一绝对深度。

同时报告：

$$
\Delta SNR(z)=
SNR_{method}(z)-SNR_{raw}(z).
$$

---

# 🟥 27. Signal retention：防止“网络只是把图变亮”

需要额外定义：

$$
SR(z)=
\frac{\mu_{denoised}(z)}
{\mu_{reference}(z)+\epsilon}.
$$

但不能只看均值。

加入：

### Gradient retention

$$
GR=
\frac{\|\nabla I_{denoised}\|}
{\|\nabla I_{reference}\|+\epsilon}.
$$

### Local morphology consistency

可用：

- SSIM；
- edge overlap；
- skeleton continuity；
- neuron segmentation F1。

最终要回答：

> “增加的是信号可见性，还是单纯 intensity amplification？”

---

# 🟨 28. Neuron-level validation

这是本项目应用价值的重要升级。

在统一 detector / segmentation pipeline 下，比较：

$$
Precision
$$

$$
Recall
$$

$$
F1
$$

进一步按深度报告：

$$
Recall_{neuron}(z).
$$

以及：

$$
Recall_{deep}.
$$

最终可以定义：

> **deepest reliable detection depth**

但必须预先定义可靠检测的统计标准，例如在盲测区域达到预设 precision/recall 门槛，不能事后挑选深度。

---

# 🟦 29. 真实数据中 SNR / CNR / SBR 的 ROI 设计

不能简单把“亮度高的地方”自动定义为 signal，再拿它证明 SNR 提升。

建议：

### Signal ROI

由独立 neuron/structure annotation 或稳定的 segmentation 产生。

### Background ROI

使用：

- 与 neuron 相邻但不包含 neuron 的 ring；
- 固定解剖背景区；
- 用户已发现的高背景区域。

红/绿色通道分别建立 ROI protocol。

需要在 blinded test 上预先固定 ROI 规则。

---

# 🟩 30. 实验总矩阵

## Experiment A — Geometry failure / motivation

```text
VALID
SN2N-like XY
Ellipsoid
Empirical-block
QMask
```

目的：

> 证明简单 fixed geometry / naive radius 不足。

---

## Experiment B — Risk field

对 candidate `Δ / Γ`：

```text
R_N
R_S
R_I
```

目的：

> 证明三个 risk 具有不同规律，不能由单一 Q 替代。

---

## Experiment C — Pair utility

比较：

```text
Q
R_base
U_probe
U_oracle
```

核心指标：

$$
\rho_{Spearman}(score,U_{oracle}).
$$

---

## Experiment D — Adaptive hard sampling

比较：

```text
VALID
QMask
Utility-hard
```

核心问题：

> 自动选出的 family 是否逼近 oracle。

---

## Experiment E — Adaptive soft sampling

比较：

```text
Utility-hard
Utility-soft
```

关注：

- mean performance；
- seed variance；
- robustness。

---

## Experiment F — Depth-conditioned policy

比较：

```text
Global policy
Depth-aware policy
Depth + signal-aware policy
```

核心指标：

$$
\Delta SNR(z),
\Delta CNR(z),
\Delta SBR(z).
$$

---

## Experiment G — Real 2P/3P validation

比较：

```text
Raw
AVG25
VALID
QMask
Proposed
```

以及可实现时：

```text
SN2N
FAST / other applicable baselines
```

注意：不同方法必须遵守各自合理的数据/训练假设，不能把“论文原方法”的数据需求错误地改写成等价方法。

---

# 🟥 31. Oracle Gap：最终方法必须增加的核心指标

令：

$$
\Gamma^*=
\arg\max_\Gamma U_{oracle}(\Gamma)
$$

模型预测：

$$
\hat\Gamma=
\arg\max_\Gamma U_{probe}(\Gamma).
$$

定义：

$$
G_{oracle}=
U_{oracle}(\Gamma^*)-
U_{oracle}(\hat\Gamma).
$$

理想情况：

$$
G_{oracle}\rightarrow0.
$$

这会成为比“我的 PSNR 比 baseline 高 0.X dB”更有方法学意义的证据。

---

# 🟧 32. Policy robustness：不仅看一次最优结果

需要报告：

### Seed robustness

至少：

```text
3 seeds
```

### Candidate perturbation

轻微改变：

- risk threshold；
- utility temperature；
- patch subset；
- calibration subset。

观察：

$$
KL(\pi_1\|\pi_2)
$$

以及最终性能变化。

### Deep-region robustness

在不同深度 bins 上分别观察 policy。

---

# 🟦 33. Training protocol：解决当前过拟合问题

现有 80 epoch Neurofinder 实验已经表明：best epoch 与 final epoch 可能显著不同。

因此后续统一：

```text
max_epochs: 30–50
validation: every 2–5 epochs
patience: 5–10 evaluations
```

实际数字由 pilot 决定。

原则：

> **early stopping 只能由 validation 决定。**

不要为了追求最低 training loss 继续训练。

---

# 🟨 34. 评估 protocol：full / aligned 必须彻底统一

目前存在：

```text
full-resolution evaluation
aligned evaluation
best checkpoint
final checkpoint
```

导致解释复杂。

正式论文主表建议只保留一个：

> **training-resolution-aligned, validation-selected test evaluation**

然后 supplementary 才报告 full-resolution auxiliary analysis。

---

# 🟩 35. Background / foreground evaluation

不能再只依赖“foreground-rich patches top-k”。

正式 blind test 应固定：

```text
predefined spatial tiles
or
predefined ROI mask
```

不能根据 test target 重新挑选“最适合展示的 patch”。

当前 foreground-rich top-k 方法可以继续用于 development diagnostic，但不能成为最终 blind benchmark。

---

# 🟪 36. 3D Wiener：保留，但定位为 classical frequency baseline

3D Wiener 不作为主创新。

## 36.1 理论

$$
H(f_x,f_y,f_z)=
\frac{S(f_x,f_y,f_z)}
{S(f_x,f_y,f_z)+N(f_x,f_y,f_z)}.
$$

其中：

- $S$：signal PSD estimate；
- $N$：noise PSD estimate。

---

## 36.2 noise PSD

有 repeat 时：

$$
N(f)=PSD(Y_k-\bar Y).
$$

真实数据可使用 independent frame split 的 residual。

---

## 36.3 signal PSD

可采用：

$$
S(f)=
\max(PSD(\bar Y)-N(f)/K,0)
$$

并加入小正则：

$$
H(f)=
\frac{S(f)}{S(f)+N(f)+\epsilon}.
$$

具体估计需要在实现阶段根据 repeat 数量校准。

---

## 36.4 3D Wiener 的主要作用

比较：

```text
Raw PSD
Wiener PSD
Network PSD
Reference PSD
```

重点看：

- high-frequency retention；
- z-frequency attenuation；
- deep-region spectral behavior；
- noise-floor suppression。

目的：

> 排除“网络只是 aggressive smoothing”的解释。

---

# 🟦 37. 代码重构总体方案

当前代码不建议继续在 `qmask/` 里无限追加脚本。

建议形成：

```text
ACES/
│
├── profiling/
│   ├── __init__.py
│   ├── redundancy.py
│   ├── noise_risk.py
│   ├── structure_risk.py
│   └── shortcut_risk.py
│
├── calibration/
│   ├── __init__.py
│   ├── candidate_offsets.py
│   ├── candidate_families.py
│   ├── micro_probe.py
│   ├── utility.py
│   └── policy_calibration.py
│
├── sampling/
│   ├── __init__.py
│   ├── fixed.py
│   ├── qmask.py
│   ├── adaptive.py
│   ├── policy_sampler.py
│   └── reconstruction.py
│
├── training/
│   ├── train_fixed.py
│   ├── train_adaptive.py
│   └── early_stop.py
│
├── evaluation/
│   ├── image_metrics.py
│   ├── signal_metrics.py
│   ├── depth_metrics.py
│   ├── neuron_metrics.py
│   └── benchmark.py
│
├── frequency/
│   └── wiener3d.py
│
├── real_data/
│   ├── preprocessing.py
│   ├── frame_split.py
│   ├── registration.py
│   └── roi_protocol.py
│
├── baseline/
│   ├── valid.py
│   ├── sn2n_like.py
│   └── qmask.py
│
├── legacy/
│   ├── pairability.py
│   ├── empirical_block.py
│   └── ellipsoid.py
│
└── tests/
```

不一定一次性创建全部目录；先按阶段逐步迁移。

---

# 🟨 38. 现有代码与新架构的对应关系

| 当前文件 | 当前定位 | 下一阶段处理 |
|---|---|---|
| `aces/profile_anisotropy.py` | Stage 1 N/S/Q | 重构为 redundancy/risk profiling |
| `aces/pairability.py` | legacy radius | 移入 `legacy/` |
| `aces/sampling_aniso.py` | sampler common logic | 提炼统一 sampler interface |
| `qmask/geometry.py` | axis decision | 保留为 `baseline/qmask.py`，不再作为主 policy |
| `qmask/sampling.py` | QMask phase split | 保留为 baseline sampler，并提取通用 phase sampler |
| `qmask/train.py` | QMask-only runner | 迁移成 baseline runner / test harness |
| `qmask/evaluate.py` | depth diagnostics | 迁移到 `evaluation/depth_metrics.py` |
| `qmask/report.py` | gate report | 重写为统一 benchmark report |
| `qmask/tau_sweep.py` | threshold diagnostic | 保留 development-only |
| `qmask/geometry_sweep.py` | threshold diagnostic | 保留 development-only |
| `qmask/t4b_rgrid.py` | distance utility exploration | 核心实验素材，提取为 `micro_probe.py` |
| `qmask/t4b_sgrid.py` | spatial utility exploration | 核心实验素材，提取为 `micro_probe.py` |
| `qmask/shortcut_risk.py` | shortcut diagnostics | 正式纳入 `shortcut_risk.py` |
| `aces/train_stage2.py` | fixed sampling runner | 重构为统一 training backend |
| `aces/evaluate_stage2.py` | mixed evaluation | 重写统一 evaluation pipeline |
| `ACES/example/*` | real-data demo | 升级为 real-data benchmark |

---

# 🟥 39. Unified sampler interface

所有 sampler 都必须实现相同接口：

```python
views, metadata = sampler.sample(
    image,
    policy_state,
    seed=seed,
)
```

其中 metadata 至少包括：

```json
{
  "family_id": "...",
  "offsets_voxel": [...],
  "offsets_um": [...],
  "stride_zyx": [...],
  "policy_probability": 0.42,
  "risk": {
    "noise": 0.12,
    "structure": 0.18,
    "shortcut": 0.07
  },
  "utility": 0.83
}
```

这样才能真正做到：

> **sampling policy 可追踪、可复现、可审计。**

---

# 🟩 40. Candidate family generator 技术要求

必须解决以下问题：

### 40.1 不允许非法 overlap

如果某 family 的两个 view 在同一 coarse grid 上大量重复同一 observation，必须记录：

```text
overlap_ratio
```

作为 risk feature。

### 40.2 边界处理

不能默默 zero-padding 后参与统计。

建议：

- valid crop；或
- reflection only when mathematically documented；
- 保存有效 sample count。

### 40.3 deterministic seed

所有 policy sampling 必须：

```text
same seed + same config
→ same family sequence
```

### 40.4 policy audit

每个 epoch 保存：

```text
family histogram
policy entropy
mean risk
mean utility
```

这样可以判断 policy 是否发生 collapse。

---

# 🟦 41. Policy collapse monitoring

需要定义：

$$
H(\pi)=
-\sum_\Gamma\pi_\Gamma\log\pi_\Gamma.
$$

如果：

$$
H(\pi)\rightarrow0
$$

说明 policy 几乎永远只选择一个 family。

这不一定错误，但必须报告。

另外记录：

$$
N_{active}
$$

即实际被使用的 family 数量。

---

# 🟨 42. Unit tests：必须新增的测试

除了当前 QMask tests，至少增加：

### Test 1 — offset invariance

同一个 family 在不同 seed 下如果 geometry 固定，应保持 offsets 不变。

### Test 2 — policy reproducibility

固定 seed 后 policy family sequence 相同。

### Test 3 — risk monotonic sanity

人为生成相关噪声时：

$$
R_N\uparrow.
$$

### Test 4 — structure mismatch

人为平移结构后：

$$
R_S\uparrow.
$$

### Test 5 — shortcut

复制相同图像：

$$
R_I\approx1.
$$

### Test 6 — risk decomposition

改变 N/S/identity 中任一因素时，其对应 risk 应变化，而不应由其他指标完全掩盖。

### Test 7 — family aggregation

pairwise risk → family risk 的平均与 worst-case 聚合正确。

### Test 8 — no test leakage

正式 test loader 不得调用 calibration/policy selection API。

---

# 🟪 43. Development Laboratory：优先使用已有实验闭环

目前已有：

```text
r-grid
s-grid
shortcut-risk
QMask
```

不需要抛弃。

应把这些实验统一成：

```text
Candidate family
      ↓
static descriptors
      ↓
micro-probe
      ↓
training utility
      ↓
oracle validation
```

### 关键新分析

对每个 candidate：

$$
Q
$$

$$
R_{base}
$$

$$
U_{probe}
$$

$$
U_{oracle}
$$

同时画：

```text
score vs oracle utility
```

并比较 Spearman / Kendall。

这将直接验证下一阶段理论是否成立。

---

# 🟥 44. 关键假设 H1–H6

## H1 — Anisotropic redundancy exists

不同方向具有不同结构/噪声统计。

当前已有数据支持，但需要正式跨数据验证。

---

## H2 — Radius is not a sufficient representation

最大 Q radius 无法稳定预测 optimal training geometry。

当前 ellipsoid/rgrid 已提供初步证据。

---

## H3 — Pair similarity is not equivalent to training utility

这是当前项目最重要的机制假设。

如果成立，才值得发展 utility calibration。

---

## H4 — Risk decomposition improves utility prediction

应满足：

$$
Corr(U_{probe},U_{oracle})
>
Corr(Q,U_{oracle}).
$$

---

## H5 — Utility-based adaptive sampling approaches oracle

$$
G_{oracle}
$$

应明显小于固定策略。

---

## H6 — Depth/signal-conditioned policy improves deep detectability

核心应用假设：

$$
\Delta CNR_{deep}>0
$$

以及：

$$
\Delta SBR_{deep}>0
$$

同时不能出现不可接受的 morphology degradation。

---

# 🟩 45. 阶段 Gate：什么时候允许进入下一阶段

## Gate P0 → P1

必须满足：

- evaluation protocol 统一；
- Z/T 语义统一；
- SNR/CNR/SBR 定义固定；
- QMask 与 legacy geometry 隔离；
- test 不参与选择。

---

## Gate P1 → P2：Risk Field

至少证明：

1. $R_N,R_S,R_I$ 具有不同规律；
2. Q 无法完整解释这些规律；
3. 至少一个真实数据集存在明显 trade-off。

---

## Gate P2 → P3：Micro-Probe

必须测试：

$$
Spearman(U_{probe},U_{oracle}).
$$

若没有明显正相关：

> 不进入 adaptive policy，先修改 utility definition。

---

## Gate P3 → P4：Adaptive Hard

必须满足：

$$
G_{oracle}
$$

显著小于 fixed/QMask。

---

## Gate P4 → P5：Soft Policy

需要观察：

- seed variance；
- policy entropy；
- candidate robustness。

若 soft sampling 没有稳定性收益，则保留 hard policy，不继续复杂化。

---

## Gate P5 → P6：Deep-aware Policy

必须看到：

$$
\Delta SNR_{deep}>0
$$

$$
\Delta CNR_{deep}>0
$$

或：

$$
\Delta SBR_{deep}>0
$$

同时保持结构质量。

否则不能声称“deep-signal enhancement”。

---

# 🟦 46. 推荐实验优先级

按照 GPU / 时间收益排序：

### Priority 1 ⭐⭐⭐⭐⭐

完成：

```text
candidate Δ / Γ
→ R_N
→ R_S
→ R_I
```

目标：验证 risk field。

### Priority 2 ⭐⭐⭐⭐⭐

完成：

```text
candidate Γ
→ 2-epoch micro-probe
→ U_probe
→ oracle comparison
```

这是整个项目的关键科研 Gate。

### Priority 3 ⭐⭐⭐⭐⭐

实现：

```text
hard adaptive sampler
```

### Priority 4 ⭐⭐⭐⭐

实现：

```text
soft policy
```

### Priority 5 ⭐⭐⭐⭐

实现：

```text
depth + signal conditioned policy
```

### Priority 6 ⭐⭐⭐⭐

真实 2P/3P：

```text
25-frame split
→ deep SNR/CNR/SBR
→ neuron-level analysis
```

### Priority 7 ⭐⭐

3D Wiener / spectral analysis。

### Priority 8 ⭐

online curriculum / RL / full 4D policy。

---

# 🟨 47. 不建议继续投入的方向

除非出现新的机制问题，否则以下方向不再作为主线：

```text
❌ 继续堆 τ_n / τ_s / τ_q sweep
❌ 继续寻找一个“更漂亮”的固定 ellipsoid
❌ 继续尝试更多固定 xyz binary mask
❌ 单纯增加 epoch 追求 PSNR
❌ 用 test proxy 反复挑 checkpoint
❌ 用单个 Pearson correlation 解释全部 pairability
❌ 直接上 RL / Transformer policy
❌ 仅用一张 qualitative figure 证明 deep enhancement
```

---

# 🟥 48. 建议论文最终 Figure 结构

## Figure 1 — Problem / concept

```text
Fixed sampling patterns
       ↓
Pair validity ≠ pair utility
       ↓
Pair-risk field
       ↓
Adaptive sampling policy
```

---

## Figure 2 — Pair-risk field

展示：

```text
R_N(Δ)
R_S(Δ)
R_I(Δ)
```

沿 x/y/z / distance 变化。

---

## Figure 3 — Utility calibration

展示：

```text
Q vs oracle
R_base vs oracle
U_probe vs oracle
```

加：

$$
Spearman/Kendall.
$$

---

## Figure 4 — Adaptive sampling

展示：

```text
VALID
QMask
Adaptive-hard
Adaptive-soft
```

以及 policy probability map。

---

## Figure 5 — Depth-aware policy

展示：

```text
π(Γ | depth, signal)
```

以及不同 depth/SNR 条件下最常选的 family。

---

## Figure 6 — Real deep-signal recovery

建议核心图：

```text
Raw
AVG25
VALID
QMask
Proposed
```

配：

$$
SNR(z),CNR(z),SBR(z).
$$

---

## Figure 7 — Biological utility

展示：

- neuron detection；
- neurite continuity；
- deepest reliable detection depth。

---

# 🟩 49. 真实应用结果的核心叙事

最终论文不要停留在：

> “Our method gives higher PSNR.”

而应形成：

```text
2P/3P imaging
    ↓
low photon / depth attenuation / anisotropic redundancy
    ↓
not every spatial pair is equally useful
    ↓
data-calibrated pair-risk
    ↓
optimization-aware pair utility
    ↓
adaptive sampling
    ↓
less noise + less shortcut + better structural preservation
    ↓
deep neuron signal remains detectable
    ↓
higher SNR/CNR/SBR
    ↓
better biological readout
```

这条主线才真正将“sampling-method innovation”和“深层神经元信号增强”统一起来。

---

# 🟦 50. 结果报告标准

以后每一个正式 run 目录至少包含：

```text
run/
├── config.json
├── metadata.json
├── calibration/
│   ├── candidate_families.json
│   ├── risk_field.csv
│   ├── utility.csv
│   └── policy.json
│
├── logs/
│   ├── train_losses.csv
│   ├── eval_curve.csv
│   └── policy_trace.csv
│
├── checkpoints/
│   ├── best.pth
│   └── final.pth
│
├── metrics/
│   ├── image_metrics.json
│   ├── signal_metrics.json
│   ├── depth_metrics.csv
│   └── neuron_metrics.csv
│
└── report/
    └── RESULTS.md
```

---

# 🟨 51. Policy trace 必须成为正式输出

每个训练 epoch 至少记录：

| 字段 | 含义 |
|---|---|
| `family_id` | sampling family |
| `probability` | policy probability |
| `sample_count` | 实际被使用次数 |
| `risk_noise` | noise risk |
| `risk_structure` | structure risk |
| `risk_shortcut` | shortcut risk |
| `utility_probe` | probe utility |
| `depth_bin` | depth condition |
| `signal_bin` | signal-quality condition |

这样最终可以画：

$$
P(\Gamma|depth).
$$

这是方法解释性的关键证据。

---

# 🟥 52. 实验统计要求

正式比较至少使用：

```text
3 random seeds
```

报告：

$$
mean\pm SD.
$$

最好同时报告 95% CI，尤其对：

- SNR gain；
- CNR gain；
- SBR gain；
- neuron detection。

如果以 patch 为统计单位，需要避免把同一个 volume 的大量相邻 voxels 当成独立 biological samples。

正式论文应明确统计单位：

```text
volume / animal / experiment / field / patch
```

避免 pseudo-replication。

---

# 🟪 53. 重要实验设计原则：同一个 backbone / loss / budget

为了证明是 sampling innovation：

第一阶段必须固定：

```text
backbone
loss
optimizer
learning rate
patch size
training epochs
training budget
augmentation
```

只改变：

```text
sampling policy
```

即：

$$
Performance=
F(\text{sampling})
$$

而不是同时改变：

$$
F(\text{sampling},
network,
loss,
training).
$$

等 sampling mechanism 被证明有效之后，才可以研究 signal-aware loss。

---

# 🟦 54. Deep-signal enhancement 的第二阶段：暂不和 sampler 同时改 loss

后续确实可以考虑：

$$
L=
L_{N2N}+
\lambda_{deep}L_{deep}.
$$

但第一阶段禁止。

原因：

如果 sampler 和 loss 同时改变：

> 无法知道深层收益究竟来自 sampling 还是 loss。

正确顺序：

```text
Sampling innovation
        ↓
prove deep SNR/CNR/SBR benefit
        ↓
then study deep-aware loss
```

---

# 🟩 55. 可能的第二阶段 Deep-aware loss

等 adaptive sampling 成立后，可以定义：

$$
L_{total}=
L_{ssl}
+
\lambda_dL_{deep}
+
\lambda_gL_{gradient}.
$$

其中：

$$
L_{deep}=
\sum_p w_{deep}(p)L_p
$$

但：

$$
 w_{deep}
$$

必须由 deployment-time observable signal quality 得到，而不是 GT intensity。

建议：

$$
 w_{deep}=f(depth,background\ variance,local\ signal).
$$

这部分属于 future extension，不进入 A3 v1 主方法。

---

# 🟥 56. 与现有文献的创新边界

现有高影响力 self-supervised microscopy denoising 工作已经反复证明：**训练 pair / redundancy 的构造方式本身可以成为方法核心**。

- DeepCAD 利用时空冗余进行 self-supervised spatiotemporal denoising。[DeepCAD, 2021]
- SN2N 通过 spatial diagonal resampling 以及 self-constrained learning 在单 noisy frame 情况下构造训练对。[SN2N, 2024]
- FAST 使用 frame-multiplexed spatiotemporal learning，同时面向高速 fluorescence neural imaging。[FAST, 2025]
- 近期 3D fluorescence denoising 工作继续利用新的高维 redundancy，例如 spatial-angular redundancy。[LF-denoising, 2025]

因此：

> 本项目不能仅仅变成“又一种固定 spatial mask”。

真正需要拉开的创新层次是：

$$
\boxed{
\text{fixed sampling pattern}
\rightarrow
\text{data-calibrated sampling policy}
}
$$

即：

> **研究“如何设计 sampling pattern”升级为研究“如何让数据自己决定应该使用哪些 sampling pairs”。**

---

# 🟨 57. 文献定位的谨慎原则

论文中不能直接声称：

> “这是第一个 adaptive sampling method。”

除非完成系统的相关工作检索。

更安全的写法是：

> “Existing self-supervised microscopy denoising approaches commonly rely on predefined forms of temporal, spatial, spatiotemporal, or volumetric redundancy. We investigate whether the training utility of such observation pairs can itself be estimated from the data and used to construct an adaptive sampling policy.”

最终投稿前应再系统检索：

```text
adaptive self-supervised sampling
pair selection denoising
learned sampling policy microscopy denoising
data-driven redundancy denoising
```

防止遗漏 2026 年之后的新工作。

---

# 🟦 58. 第一版 A3 的最小可实现版本（MVP）

为了避免方法改动过大，建议第一版只实现：

### MVP-1

```text
candidate families
     ↓
R_N
R_S
R_I
```

### MVP-2

```text
2-epoch micro-probe
     ↓
U_probe
```

### MVP-3

```text
hard argmax family
```

### MVP-4

```text
compare with:
VALID
SN2N-like
QMask
```

### MVP-5

真实数据只做：

```text
Raw
AVG25
QMask
Adaptive
```

验证：

```text
SNR(z)
CNR(z)
SBR(z)
```

只有 MVP 成功，才进入 soft policy 和 depth conditioning。

---

# 🟥 59. MVP 的最小数学闭环

真正开始 coding 时，必须首先让下面五个对象跑通：

$$
\boxed{
\Delta
\rightarrow
\mathbf R(\Delta)
\rightarrow
\Gamma
\rightarrow
U_{probe}(\Gamma)
\rightarrow
\hat\Gamma
}
$$

其中：

### Step 1

$$
\Delta=(dx,dy,dz)
$$

### Step 2

$$
\mathbf R=[R_N,R_S,R_I]
$$

### Step 3

$$
\Gamma=\{\delta_1,\delta_2,\delta_3,\delta_4\}
$$

### Step 4

$$
U_{probe}=g(L_N,L_I,R_{res},G_{pres},...)
$$

### Step 5

$$
\hat\Gamma=\arg\max U_{probe}.
$$

如果这一闭环无法稳定运行，就不应该进入更复杂的 policy learning。

---

# 🟩 60. 项目里程碑

## Milestone 0 — Repository cleanup

目标：

> 让代码、语义、evaluation protocol 可信。

输出：

```text
ACES_v1_cleanup
```

---

## Milestone 1 — Pair-risk field

输出：

```text
candidate_offsets.csv
risk_field.csv
risk_heatmaps
```

必须可以回答：

> 哪些 offset 的 noise / structure / shortcut risk 高？

---

## Milestone 2 — Pair utility laboratory

输出：

```text
micro_probe.csv
oracle_utility.csv
utility_correlation.json
```

核心判断：

$$
U_{probe}
\stackrel{?}{\sim}
U_{oracle}.
$$

---

## Milestone 3 — Adaptive-hard

输出：

```text
adaptive_hard_sampler
oracle_gap
```

---

## Milestone 4 — Adaptive-soft

输出：

```text
policy_distribution
policy_entropy
seed_robustness
```

---

## Milestone 5 — Deep-aware policy

输出：

```text
policy_vs_depth
policy_vs_signal_quality
SNR(z)
CNR(z)
SBR(z)
```

---

## Milestone 6 — Real-data biological validation

输出：

```text
25-frame independent split
3D reconstruction
neuron-level validation
deepest reliable detection depth
```

---

# 🟨 61. Git / 项目版本管理建议

以后每一个阶段不要直接覆盖旧实验。

推荐 branch / tag：

```text
baseline-qmask-final
risk-field-v1
utility-lab-v1
adaptive-hard-v1
adaptive-soft-v1
deep-aware-v1
real-data-v1
```

每个 milestone 完成后建立一个 README：

```text
What changed?
Why changed?
Which hypothesis tested?
What passed?
What failed?
Next gate?
```

这样可以避免后期论文写作时无法追溯“为什么从 ellipsoid 改到 QMask，再改到 adaptive policy”。

---

# 🟥 62. 最终论文主线建议

如果 A3 v1/v2 顺利成立，最终论文可以形成如下逻辑。

## Introduction

### Problem

Self-supervised microscopy denoising depends critically on observation redundancy.

### Gap

Existing methods largely design a useful redundancy pattern beforehand; however, pair similarity and actual optimization utility may not be equivalent in anisotropic volumetric imaging.

### Hypothesis

Training-pair utility can be estimated from the noisy data itself.

### Method

Risk decomposition + micro-probe utility + adaptive sampling policy。

### Application

Depth-aware adaptation improves deep neuronal signal detectability。

---

# 🟦 63. Methods 最终结构建议

```text
A. Problem formulation
B. Observation redundancy and candidate sampling families
C. Pair-risk field
   C1. Noise risk
   C2. Structure risk
   C3. Shortcut risk
D. Pair-utility calibration
   D1. Micro-probe
   D2. Utility estimation
E. Adaptive sampling policy
   E1. Hard policy
   E2. Soft policy
   E3. Depth/signal conditioning
F. Self-supervised denoising network
G. 3D Wiener frequency baseline
H. Real-data preprocessing
I. Evaluation metrics
   I1. image metrics
   I2. signal metrics
   I3. depth metrics
   I4. neuron-level metrics
```

---

# 🟩 64. Results 最终结构建议

```text
A. Failure of fixed geometric sampling
B. Data-dependent pair-risk landscape
C. Pair utility is not pair similarity
D. Utility-calibrated sampling closes the oracle gap
E. Adaptive policy improves robustness
F. Depth/signal-aware policy improves deep signal quality
G. Real 2P/3P biological validation
H. Frequency-domain / Wiener analysis
```

---

# 🟥 65. 最终成功标准

这个项目是否值得继续作为一篇独立的方法论文，不应由某一次 PSNR 是否高于 QMask 决定，而应满足下面的逻辑闭环：

### Level 1 — Mechanism

证明：

$$
\text{pairability}
\neq
\text{training utility}.
$$

### Level 2 — Prediction

证明：

$$
U_{probe}
\rightarrow
U_{oracle}
$$

具有可重复的预测关系。

### Level 3 — Method

证明：

$$
\hat\Gamma
$$

能够稳定逼近：

$$
\Gamma^*.
$$

### Level 4 — Application

证明：

$$
\Delta SNR_{deep}>0
$$

$$
\Delta CNR_{deep}>0
$$

$$
\Delta SBR_{deep}>0
$$

并保持结构 / morphology fidelity。

### Level 5 — Biological relevance

证明 neuron detection / reconstruction 在深层区域实际改善。

只有 Level 1–4 成立后，才建议把完整 biological claim 写进论文主结论。

---

# 📈 66. 当前已有实验结果的正确使用方式

仓库中的现有 run 应作为 development evidence，不作为最终论文主表直接引用。尤其是当前 Neurofinder QMask 80-epoch runs 使用 proxy/ROI 与 foreground-selected patches，并存在明显的 checkpoint 依赖，因此主要作用是验证 pipeline 与机制，而不是证明最终泛化性能。

当前已经可以保留的指标包括：

- best foreground PSNR；
- aligned/full-resolution PSNR；
- foreground standard-deviation ratio；
- SNR/SBR；
- depth/time-binned diagnostics。

正式论文中应重新在严格 Calibration/Validation/Blind Test 划分下生成一套统一结果，避免旧 run 的 full/aligned、best/final 混杂。

# 🟪 67. 当前项目的最终一句话定位

> **本项目研究的不是“哪一种 3D mask 最好”，而是“在没有 clean ground truth 的情况下，如何让各向异性生物医学成像数据自己决定哪些 noisy observations 最值得用于自监督学习”，并进一步使这种配对策略适应成像深度和局部信号质量，从而在去噪过程中最大限度恢复深层弱神经元信号。**

对应方法主线：

$$
\boxed{
Data
\rightarrow
Risk
\rightarrow
Utility
\rightarrow
Policy
\rightarrow
Denoising
\rightarrow
Deep-signal recovery
}
$$

---

# 📌 68. 下一次编码执行时的第一批具体任务

建议严格按照以下顺序开始，不提前复杂化：

### Task 1 — repository cleanup

完成：

```text
Z/T semantics
SNR/CNR/SBR
SSIM
checkpoint selection
legacy isolation
```

### Task 2 — unified offset library

实现：

```python
CandidateOffset
CandidateFamily
```

### Task 3 — risk engine

实现：

```python
NoiseRiskEstimator
StructureRiskEstimator
ShortcutRiskEstimator
```

### Task 4 — risk-field report

生成：

```text
risk_field.csv
risk_field.json
risk_heatmap.png
```

### Task 5 — micro-probe engine

实现：

```python
MicroProbeRunner
```

固定全部 training variables，只替换 sampling family。

### Task 6 — utility report

生成：

```text
probe_utility.csv
oracle_utility.csv
correlation_report.md
```

### Task 7 — Gate decision

只有当：

$$
Corr(U_{probe},U_{oracle})
$$

达到预期，才开始写：

```python
AdaptiveSampler
```

### Task 8 — hard adaptive

先实现：

$$
\Gamma^*=argmax U.
$$

### Task 9 — deep-aware policy

最后再实现：

$$
\pi(\Gamma|depth,signal,noise,anisotropy).
$$

---

# 📚 69. 关键文献与项目定位依据

1. **DeepCAD** — Reinforcing neuron extraction and spike inference in calcium imaging using deep self-supervised denoising. *Nature Methods*, 2021. DOI: `10.1038/s41592-021-01225-0`. 采用时空冗余进行 self-supervised denoising，并强调无需高 SNR observation。来源：Nature Methods。[DeepCAD, 2021]

2. **SN2N** — Self-inspired learning for denoising live-cell super-resolution microscopy. *Nature Methods*, 2024, 21:1895–1908. DOI: `10.1038/s41592-024-02400-9`. 采用 spatial diagonal resampling 与 self-constrained learning，可从单幅 noisy frame 构造自监督训练。[SN2N, 2024]

3. **FAST** — Real-time self-supervised denoising for high-speed fluorescence neural imaging. *Nature Communications*, 2025, 16:9396. 使用 frame-multiplexed spatiotemporal learning，并面向 fluorescence neural imaging、多种高速应用。[FAST, 2025]

4. **LF-denoising** — Leveraging spatial-angular redundancy for self-supervised denoising of 3D fluorescence imaging without temporal dependency. *Nature Communications*, 2025, 16:11608. DOI: `10.1038/s41467-025-66654-3`. 通过 spatial-angular redundancy 构造高维 self-supervised pairs。[LF-denoising, 2025]

> **说明**：以上文献用于说明领域中“利用新的 redundancy / sampling 构造 self-supervised pairs”的研究脉络。具体到最终论文投稿时，应再次完成截至投稿日期的 comprehensive literature search，尤其关注 2026 年新增的 adaptive / learned sampling 工作。

---

# 🔍 70. 当前仓库审阅依据

本方案基于 2026-09-22 对 GitHub 仓库 `winb0408/Denoising_PM` 当前 `main` 分支的代码、文档和已入库 metrics/JSON/CSV 产物审阅形成。

重点依据包括：

```text
ACES/README.md
ACES/qmask/README.md
ACES/qmask/docs/TECHNICAL.md
ACES/qmask/geometry.py
ACES/qmask/sampling.py
ACES/qmask/train.py
ACES/qmask/report.py
ACES/qmask/t4b_rgrid.py
ACES/qmask/t4b_sgrid.py
ACES/qmask/shortcut_risk.py
ACES/aces/profile_anisotropy.py
ACES/aces/train_stage2.py
ACES/aces/evaluate_stage2.py
ACES/example/readme.txt
ACES/runs/*/metrics/*.json
ACES/runs/*/logs/*.json
```

需要特别说明：`ACES/example` 中的 TIFF/JPG/PNG 属于二进制图像文件；本次项目审阅基于仓库中的说明、文件元数据以及已有数值产物，未把无法直接解析的二进制结果图宣称为已完成逐像素视觉审阅。

---

# ✅ 71. 最终执行原则

> **第一原则：先证明机制，再增加模型复杂度。**

> **第二原则：先证明 pair utility 能预测 training utility，再实现 adaptive policy。**

> **第三原则：先证明 sampling 本身能够提升 deep signal quality，再考虑 deep-aware loss。**

> **第四原则：真实数据最终评价以 depth-aware SNR/CNR/SBR 和 neuron detectability 为核心，而不是只看平均 PSNR。**

> **第五原则：任何 calibration、policy selection、checkpoint selection 都必须与 blind test 隔离。**

> **第六原则：QMask 的研究价值保留，但把它放回正确的位置——它是从 fixed geometry 向 data-driven sampling policy 过渡的关键中间方法，而不是最终终点。**

---

## 🚀 项目下一阶段唯一最重要的实验闭环

```text
Candidate Pair / Family
        ↓
   Noise Risk
   Structure Risk
   Shortcut Risk
        ↓
    Micro-Probe
        ↓
   Pair Utility
        ↓
 Compare with Oracle
        ↓
      Gate
        ↓
 Adaptive Sampling
        ↓
 Depth-aware Sampling
        ↓
 Deep Neuron SNR/CNR/SBR
        ↓
 Biological Detectability
```

**在这一闭环中，最关键的 Gate 是：`Pair-risk → Micro-probe utility → Oracle utility` 是否成立。**

如果成立，项目就从“QMask 的实验修正”真正升级成了一个独立的 sampling-method 研究；如果不成立，应在这一层修正 utility 定义，而不是继续堆更复杂的 adaptive network。

---

## 🔗 文献链接

- DeepCAD: https://www.nature.com/articles/s41592-021-01225-0
- SN2N: https://www.nature.com/articles/s41592-024-02400-9
- FAST: https://www.nature.com/articles/s41467-025-64681-8
- LF-denoising: https://www.nature.com/articles/s41467-025-66654-3
- Repository: https://github.com/winb0408/Denoising_PM
