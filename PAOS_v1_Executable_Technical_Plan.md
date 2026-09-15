# PAOS-v1 可落地技术方案：面向 2P/3P 体成像的各向异性感知正交自监督去噪

> **文档定位**：本文件是交给 Codex 执行的工程规格书。与 `PAOS_2P_3P_Anisotropy_Aware_Technical_Roadmap.md`（GPT 初版）相比，本文修正了三处会导致无法落地的问题：
> 1. 代码基座只使用本仓库 `Methods/` 下**真实存在**的开源库；
> 2. 按 VALID 源码的**真实采样机制**定义各向异性改造，而非虚构的三截面加权；
> 3. 将 DeepCAD-RT 的**时间/重复冗余**前置为 v1 的第二自监督信号。
>
> **一句话方法**：以 VALID 的 CRN + 2×2×2 邻域自监督为主干，把采样从"各向同性 block 下采样"改造为"**各向异性条件化配对**"，并叠加 DeepCAD-RT 式的**时间/repeat 冗余一致性**，形成 anisotropy-aware 的伪监督；backbone 与超参保持与 VALID 一致以保证消融可归因。

---

## 0. 现有资产盘点（务必先确认）

`Methods/` 下**真实存在**的仓库（Codex 只能依赖这些）：

| 仓库 | 角色 | 关键文件 |
|---|---|---|
| `Methods/VALID-v1.0/FDU-donglab-VALID-34f5637/` | **主干**（backbone + 采样 + 训练入口） | `datasets/sampling.py`, `train_pipeline.py`, `test_pipeline.py`, `models/network.py`, `models/model_CRN.py` |
| `Methods/DeepCAD-RT-main/DeepCAD_RT_pytorch/` | **时间冗余思想来源** | `deepcad/data_process.py`（interlaced 帧配对） |
| `Methods/FAST-main/` | 时间基线 / 复现脚手架 | `reproduction/valid_comparison/src/valid_compare/original_workflow.py` |
| `Methods/SRDTrans-main/` | 时序去噪对照 | `SRDTrans/` |
| `Methods/SUPPORT-main/` | 体/时序自监督对照 | `model/`, `src/` |

**不存在**（GPT 初版误列，禁止依赖）：SN2N、DeepInterpolation、SelfMirror 官方仓库。
- SN2N 的"XY 局部对称"思想**在 VALID mask 内部重新实现**，不导入其仓库。
- 若确需 DeepInterpolation/SelfMirror 作为额外基线，Codex 必须先向用户确认再联网获取，不得默认存在。

已验证可用数据（来自 8-20 复现记录，勿重新造数）：
- 3P：`3P_neuron_raw.tif`，`(2475,512,512)` = `275 depth × 9 repeats × 512 × 512`，Δz=4μm，深度 0–1096μm；`mean9` 仅作评价 proxy，**不参与训练**。
- 2P：neurofinder.00.00 钙成像栈（见 `Results/neurofinder_00_00_multimethod_comparison_20260822/`）。
- 环境：`/data2/wjb/anaconda3/envs/threePM/bin/python`，A6000，`CUDA_VISIBLE_DEVICES` 按空闲卡指定。

---

## 1. VALID 真实机制（改造前必须准确理解）

这是 GPT 初版描述错误、必须纠正的部分。VALID 训练循环（`train_pipeline.py::goTrainingVALID`）实际做的是：

对输入体块 `input`（形状 `(B,1,Z,H,W)`），先 `space_to_depth(block_size=2)` 把每个 `2×2×2` block 的 8 个体素展平为通道，再用 `generate_mask_pair` 从 8 个体素中随机抽取 4 个互斥子采样 `sub1..sub4`（各为 `Z/2×H/2×W/2`）：

$$
\text{block}(2{\times}2{\times}2)=\{v_0,\dots,v_7\}\ \xrightarrow{\text{mask}}\ (\text{sub}_1,\text{sub}_2,\text{sub}_3,\text{sub}_4)
$$

损失（源码实况）：

$$
L_{\text{VALID}} = \underbrace{0.5\,\|f(\text{sub}_1)-\text{sub}_3\|_2^2 + 0.5\,\|f(\text{sub}_2)-\text{sub}_4\|_2^2}_{L_{2neighbor}} + \underbrace{\|f(\text{sub}_1)-f(\text{sub}_2)\|_2^2}_{L_{idt}} + \lambda_{reg}\,L_{Hessian}
$$

其中 $f$=CRN，$L_{Hessian}$ 是对输出经 3D Haar DWT 后低频子带的 3D Hessian Frobenius 范数（`HessianConstraintLoss3D`），$\lambda_{reg}=$ `weight_reg` 默认 `1e-4`。

**关键结论**：
- VALID 的采样是**各向同性 2×2×2 block 邻域下采样**，"正交/orthogonal"体现在 **Hessian 正则**，不是采样权重。
- 因此"各向异性改造"必须作用在**两处真实可改点**：(a) `space_to_depth` 的 block 形状与 z 方向配对规则；(b) Hessian 各方向权重。
- 源码存在一处可疑写法：`LLL_n2..=DWT3D(noisy_output_1)`（第二次仍用 output_1）。**v1 保持原样以对齐基线**，仅在 ablation 附注中记录，不擅自"修复"，避免污染可比性。

---
## 2. PAOS-v1 核心创新（三个可落地模块）

以 VALID 为唯一主干，v1 只做三件事，全部可在 VALID 源码上小改实现。

### 2.1 模块 A：各向异性条件化 block 采样（Anisotropy-aware pairing）

**动机**：当 $\Delta z \gg \Delta x$ 或轴向 PSF 远宽于横向时，`2×2×2` block 内跨 z 的两个体素并非"同一结构的独立噪声观测"，把它们当 N2N 配对会引入结构偏差。

**做法**：不改 CRN，只改采样。定义各向异性指标

$$
\rho_z = \frac{\Delta z}{\Delta_{xy}},\qquad \rho_{\text{PSF}}=\frac{f_z}{f_{xy}}
$$

用一个可解释的门控标量 $g\in[0,1]$ 决定"是否允许 z 方向参与 block 配对"：

$$
g = \exp\!\Big(-\max(\rho_z,\rho_{\text{PSF}})/\kappa\Big),\quad \kappa\ \text{为超参（默认 1.0）}
$$

据此实现**两种 block 模式**（在 `sampling.py` 内新增，通过 config 选择）：

- **`iso` 模式**（$g$ 高，如密采样 2P）：完全等价 VALID 原始 `2×2×2` + `generate_mask_pair`（保证退化到基线）。
- **`xy_dominant` 模式**（$g$ 低，如稀疏 z 的 3P 深层）：block 改为 `1×2×2`（仅 XY 内 2×2 邻域配对，含 SN2N 式对角 $(a,d)/(b,c)$ 组合），z 方向不做 N2N 配对，仅由 Hessian 正则维持轴向平滑。

> 这就是 GPT 初版想表达的"XY 强局部对称、Z 条件化"，但落到 VALID 真实机制上：**改 block 形状，而非虚构 w_xy/w_xz/w_yz**。

**实现锚点**：改写 `datasets/sampling.py::space_to_depth` 支持非立方 block `(bz,by,bx)`，并新增 `generate_mask_pair_xy` 针对 `1×2×2` 的 4 体素生成对角配对；训练循环按 `args.block_mode` 分派。

### 2.2 模块 B：DeepCAD-RT 式时间/repeat 冗余一致性（你点名要结合的部分）

**动机**：3P 数据每个深度有 9 个 repeat，2P 是时间序列——这是比"推测性 z 配对"干净得多、噪声真正独立的观测对。DeepCAD-RT 的核心 (`data_process.py::trainset`) 正是**交错帧配对**：`input=stack[s:e:2]`, `target=stack[s+1:e:2]`，相邻帧互为噪声独立观测。

**做法**：在 VALID 的 block 自监督之外，增加一路 repeat/时间一致性损失。设同一空间位置的两次独立观测 $Y^{(a)},Y^{(b)}$（3P：两个 repeat；2P：交错帧）：

$$
L_{\text{temp}} = \big\|f(Y^{(a)})-Y^{(b)}\big\|_1 + \big\|f(Y^{(b)})-Y^{(a)}\big\|_1
$$

总损失：

$$
L = L_{\text{VALID}} + \lambda_{\text{temp}}\,L_{\text{temp}},\qquad \lambda_{\text{temp}}\in\{0,0.5,1\}\ \text{做 ablation}
$$

**实现锚点**：新建 `datasets/dataset_temporal.py`，从 275×9 结构里成对采样 repeat；训练循环叠加 `L_temp`。$\lambda_{\text{temp}}=0$ 时严格退化为 VALID，保证可比。

### 2.3 模块 C：采样诱导不确定度（低成本、只在推理阶段）

对同一体块用 $K$ 组合法采样/repeat 组合得到 $\hat X_1,\dots,\hat X_K$：

$$
\mu=\frac1K\sum_i\hat X_i,\qquad U_{\text{sampling}}(x)=\mathrm{Var}_i[\hat X_i(x)]
$$

v1 只把 $U_{\text{sampling}}$ 作为**输出图**并验证 $\mathrm{Corr}(U,\text{error})>0$（用 mean9 proxy 算 error）。**不训练、不做 penalty**，避免过平滑。physics forward operator、Poisson-Gaussian NLL、deep ensemble、learnable router、adaptive sampling **全部推迟到 v2**（见 §7）。

---

## 3. 目录与工程结构

在 `Methods/` 同级新建工作区，**不修改任何 baseline 源码**（用 monkey-patch/子类/wrapper）：

```text
/data2/wjb/Denoising_PM/PAOS/
├── paos/
│   ├── sampling.py        # 各向异性 block（模块 A）
│   ├── temporal.py        # repeat/时间配对（模块 B）
│   ├── uncertainty.py     # 采样方差（模块 C）
│   ├── profiler.py        # 从 metadata 算 rho_z, rho_psf, g
│   └── train_paos.py      # 复用 VALID CRN+loss，注入 A/B
├── configs/               # 每个实验一个 json，含 seed/block_mode/lambda_temp
├── scripts/               # 数据准备、run、评估
└── experiments/<exp_id>/  # config/checkpoint/metrics.json/输出 volume/可视化
```

复用 VALID 通过 `PYTHONPATH` 指向其目录并 import `models.network`、`datasets.sampling` 等，不复制粘贴其代码。

---

## 4. Codex 执行顺序（严格按序，每步有验收）

| Task | 内容 | 验收标准 |
|---|---|---|
| T0 | 环境自检：激活 `threePM`，确认 VALID `python run.py` 依赖齐全（`setproctitle/PyQt5/PyWavelets/einops`），记录 torch/CUDA 版本 | 能 import VALID 全部模块无报错 |
| T1 | 复现 VALID 基线（3P repeat01，224³，参数同 8-20 记录），产出 denoised volume + PSNR/SSIM/CNR to mean9 | 复现 PSNR≈22.9dB（±0.5），与历史记录一致 |
| T2 | 实现 `profiler.py`，对 3P/2P 各算 $\rho_z,\rho_{\text{PSF}},g$，落盘 yaml | 数值合理（3P 的 g 明显低于 2P） |
| T3 | 实现模块 A（`iso`/`xy_dominant`），单测：`iso` 模式输出与 VALID 原 mask **逐元素一致** | `iso` 严格等价基线 |
| T4 | 实现模块 B（repeat/交错帧配对 + $L_{temp}$），$\lambda_{temp}=0$ 时等价 T1 | $\lambda_{temp}=0$ 复现 T1 指标 |
| T5 | 跑 PAOS-v1（`xy_dominant`+$\lambda_{temp}=0.5$）对比 VALID | 见 §6 成功标准 |
| T6 | 实现模块 C，产出 uncertainty 图并算 $\mathrm{Corr}(U,\text{error})$ | Corr>0 且可视化对应失败区 |
| T7 | 汇总：指标表 + 深度分层曲线 + photon-budget 曲线 + 神经元分割下游 | 生成 §6 全部产物 |

---

## 5. 消融矩阵（固定 CRN，只换采样/损失）

| 配置 | block | XY-diagonal | 各向异性门控 g | $L_{temp}$ | uncertainty |
|---|---|---|---|---|---|
| VALID（基线） | 2×2×2 iso | ✗ | ✗ | ✗ | ✗ |
| +XY diagonal | 1×2×2 | ✓ | ✗ | ✗ | ✗ |
| +各向异性门控 | 自适应 | ✓ | ✓ | ✗ | ✗ |
| +时间冗余 | 自适应 | ✓ | ✓ | ✓ | ✗ |
| PAOS-v1 full | 自适应 | ✓ | ✓ | ✓ | ✓（仅推理） |

所有行共用同一 CRN、同 seed、同 patch 预算、同 epoch，确保"提升来自采样/损失而非网络容量"。

---

## 6. 成功标准与产物

**算法有效（至少满足其一即继续 v2）**：
- PAOS-v1 在 CNR / EPI / 深度分层 PSNR 中至少一项稳定优于 VALID（同算力）；
- 神经元分割（下游）F1 或 Jaccard 优于 VALID；
- 优势在低 photon / 深层（>600μm）更明显。

**必产物**：
1. 定量对比表（Raw / VALID / SRDTrans / SUPPORT / PAOS-v1）；
2. 深度分层曲线（0–200 / 200–400 / … μm 的 PSNR/SSIM/CNR）；
3. photon-budget 曲线（用 repeat 子采样模拟 100/50/20/10/5%）；
4. uncertainty–error 校准图；
5. 代表性 3D 渲染 + 单神经元 ROI zoom；
6. 神经元分割对比。

**评价口径**：3P 用 mean9 proxy（不进训练）；2P 有 GT 用 GT，否则用时间平均 proxy。禁止只报 PSNR/SSIM。

---

## 7. 推迟到 v2/v3 的内容（v1 明确不做）

- Physics forward operator / Poisson-Gaussian NLL（需 detector gain、read-noise 标定，你目前没有）；
- Deep ensemble 的 model uncertainty、physics residual uncertainty；
- Learnable sampling router、uncertainty-guided adaptive sampling；
- XYZT 四维扩展、硬件自适应采集；
- Anisotropic separable conv（backbone 改造）。

理由：v1 必须先证明"各向异性采样 + 时间冗余"本身有效且可归因，再逐项增量。

---

## 8. 研究假设（须实验证实，不得写成已证实）

- H1：2P/3P 的 XY–Z 各向异性使各向同性 block 采样非最优。
- H2：SN2N 式 XY 局部对称是 VALID 的补充而非替代。
- H3：repeat/时间冗余（DeepCAD-RT 思想）比推测性 z 配对更干净，能提升稳定性。
- H4：采样方差可作为局部不确定度 proxy（$\mathrm{Corr}(U,\text{err})>0$）。
- H5：低 photon / 深层条件下 PAOS 优势随难度增强而更明显。

---

## 9. 工作纪律

1. 先复现（T1）后创新；baseline 源码零修改，改动全部在 `PAOS/`。
2. 每步保存 config/seed/checkpoint/metrics.json/输出 volume。
3. `iso` 模式与 $\lambda_{temp}=0$ 必须逐项验证等价基线，否则不得进入下一步。
4. 任何新采样必须输出可视化 mask/配对关系图。
5. 保留 VALID 源码中 `DWT3D(noisy_output_1)` 的原写法，仅记录不修改。
6. 3P→2P 泛化用完全独立数据验证，不共享超参调优。
7. 记录每个开源库的 commit/version/license/环境快照。

---

## 10. 参考文献

1. **FAST** — Wang et al., *Nat. Commun.* (2025). DOI: 10.1038/s41467-025-64681-8
2. **VALID** — Gu et al., *Sci. Adv.* (2026). DOI: 10.1126/sciadv.ady9194
3. **SN2N** — Qu et al., *Nat. Methods* (2024). DOI: 10.1038/s41592-024-02400-9
4. **DeepCAD-RT** — Li et al., *Nat. Biotechnol.* (2023). 本仓库 `Methods/DeepCAD-RT-main/`（时间冗余思想来源）
