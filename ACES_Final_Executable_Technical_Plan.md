# ACES 最终执行方案：噪声各向异性标定的自监督体去噪

> **文档定位**：交给 Codex/agent 执行的最终研究规格书。本文**取代**以下全部历史文档：
> `PAOS_2P_3P_Anisotropy_Aware_Technical_Roadmap.md`（GPT 初版）、
> `PAOS_v1_Executable_Technical_Plan.md`（第一次修订）、
> `ACES_Anisotropy_Calibrated_Sampling_Technical_Plan.md`（上一版 ACES）。
> 取代原因见 §1、§11（GPT 审查采纳记录）。
>
> **方法名**：ACES = **A**nisotropy-**C**alibrated **E**mpirical orthogonal **S**ampling。
>
> **第一性创新（一句话）**：**不假设**光学前向模型，而是**从数据直接测量"每个空间轴、每个采样间距 d 上，一对自监督观测是否合法"**——即噪声独立性 × 结构一致性构成的**配对可用度场 $Q_a(d)$**——再由这个实测场决定各向异性采样几何。椭球只是对该标定结果的一种高效参数化，**不是第一性定义**。

---

## 1. 为什么取代前几版（基于本地真实证据的硬约束）

Codex 必须先接受这四条来自**真实代码/历史实验**的证据，它们决定方法支点：

1. **SN2N 的 3D 采样本身回避 Z 轴。** 源码 `/data2/wjb/SN2N-main/SN2N/datagen.py::block3d`：即使 3D 模式，配对也只在 XY 做对角平均
   $$\text{left}=\tfrac12(I_{0::2,0::2}+I_{1::2,1::2}),\quad \text{right}=\tfrac12(I_{0::2,1::2}+I_{1::2,0::2})$$
   帧/z 轴**原样保留**。→ "不要把对角复制到 Z" 是 SN2N 作者的既定做法，各向异性问题真实存在。

2. **VALID 与 SN2N 的采样在算法上并不同构（已读源码确认）。**
   - VALID `datasets/sampling.py`：`space_to_depth(block_size=2)` 把 2×2×2 展成 8 通道，再用固定 `idx_pair`（8×4 组合表）+ mask **挑单个体素**做子图。
   - SN2N `block3d`：XY **对角两体素求平均**，另有 Fourier 上采样管线。
   → **结论（采纳 GPT）**：`r_z→0` 只能表示"采样支撑对 Z 退化"，`r=1` 只能表示各向同性配置；**不能宣称椭球在数学上严格统一/等价这两个已发表方法**。本文改用"**连续各向异性采样族，包含 VALID-like 与 SN2N-like 两个可比较基准配置**"这一稳妥表述。

3. **失败的是"PSF 作训练损失约束"，不是"物理建模"本身。** `/data2/wjb/threePM/archive/failed_experiments_20260413/stage4_psf_unsuccessful/`：PSF 通道/PSF loss/PSF full 三变体在同等预算下均未超 baseline（约 20.8 PSNR 持平）。教训是**不要把 PSF 塞进损失端**。ACES 若用物理标定，只影响**采样几何/先验**，不进损失，且可切换、可消融。论文措辞须为"**在当前数据/模型/预算下，PSF 直接作训练约束未见优势**"，而非"PSF 学习失败"。

4. **N2N 家族唯一硬前提是"配对观测的噪声独立"**，这是**可测量、可证伪**的量，无需 PSF/detector 标定。ACES 把创新支点放在此。

---

## 2. 现有资产（Codex 只能依赖这些真实路径）

**代码基座**（均已本地存在，禁止假设联网仓库）：

| 仓库 | 真实路径 | 角色 |
|---|---|---|
| VALID | `Methods/VALID-v1.0/FDU-donglab-VALID-34f5637/` | **主干**：CRN + N2N 子采样 + Hessian 正则；采样已抽象在 `datasets/sampling.py` |
| SN2N | `/data2/wjb/SN2N-main/` | XY 对角采样 + 自约束损失（`SN2N/datagen.py`, `SN2N/trainer.py`） |
| DeepCAD-RT | `Methods/DeepCAD-RT-main/` | 交错帧时间冗余（**可选**，见 §7） |
| SRDTrans / SUPPORT | `Methods/SRDTrans-main/`, `Methods/SUPPORT-main/` | 对照方法 |

**数据资产**：

| 数据 | 路径 | 关键属性 | 用途 |
|---|---|---|---|
| VALID 3P neuron（9-repeat） | 见 `VALID_FAST_3P_neuron_reproduction_session_notes_2026-08-20.md` | 每深度 **9 repeats**，Δz=4μm，0–1096μm | **各向异性主战场**：噪声金标准 + 深层压力测试 |
| CIDC25 钙成像 | `threePM/ThreePM_openDatasets/AI4Life-CIDC25/` | `[1500,490,490]`，**有干净 GT**；第三轴是 **t 非 z**，无 PSF 标定 | **辅助锚点**：验证可用度预测 + 时间轴自动收缩 |
| pollen 真实 3P | `threePM/ThreePM_openDatasets/pollen_test/` | 有横向标定 `2.051μm/px`；**无轴向 PSF 标定** | 泛化验证 |
| 各向异性稀疏集 | `threePM/ThreePM_openDatasets/200um-f10-z5_sparse/` | 2x/4x × X/Y/Z/XY/XYZ + dense + mask | 各向异性天然实验台 |

**可复用的方法学纪律**（来自 threePM 历史，务必继承）：
- 全图 PSNR/SSIM 会被大背景稀释产生**假提升**；必须分层报告 global / foreground / background / local-profile（见 `threePM/report/SUCCESS_PATH_2D_2P5D_3D.md`）。
- 严防 **3D→2D 静默坍缩**：fail-fast 断言体块 z 维度。
- 环境：`/data2/wjb/anaconda3/envs/threePM/bin/python`，A6000，按空闲卡设 `CUDA_VISIBLE_DEVICES`。

---

## 3. 核心：配对可用度场 $Q_a(d)$（第一性定义）

> **关键修正（采纳 GPT 意见①⑤⑦）**：上一版把 $r_a=\gamma L_a\eta_a$ 当核心理论公式。此形式是启发式、非最优推导（$L_a\eta_a$ 相同可对应完全不同的真实可配对性）。本文改为**直接测量按距离分辨的可用度曲线**，几何量从曲线导出。

### 3.1 逐轴、逐间距的两个可测量

对一段体/时序数据，沿轴 $a\in\{x,y,z,(t)\}$、对每个间距 $d$：

**(1) 噪声独立性 $N_a(d)$**（N2N 是否成立的直接依据）：
$$
N_a(d) = 1 - \big|C^{\text{noise}}_a(d)\big|,\qquad
C^{\text{noise}}_a(d)=\frac{\mathbb{E}[\,n(\mathbf p)\,n(\mathbf p+d\,\mathbf e_a)\,]}{\mathbb{E}[\,n(\mathbf p)^2\,]}
$$
$N_a(d)\to1$：间距 $d$ 上噪声近似独立，可安全配对；$\to0$：噪声高度相关，配对会把相关噪声当信号学。

**噪声估计的两条路径（采纳 GPT⑥：残差自相关 ≠ 噪声自相关）**：
- **proxy 分支**：残差 $r=Y-\tilde Y$（$\tilde Y$ 为 3×3 中值/小高斯 proxy），记 $N_a^{\text{proxy}}(d)$。**必须分区估计**：在结构边缘（树突/血管/胞膜）残差强烈携带真实信号，须按前景/背景掩膜分别统计，避免把信号相关误判为噪声相关。
- **金标准分支**：9-repeat 数据用重复观测差分直接估计**纯噪声**相关，记 $N_a^{\text{repeat}}(d)$。

**(2) 结构一致性 $S_a(d)$**（配对是否破坏结构）：
$$
S_a(d)=C^{\text{structure}}_a(d)=\mathrm{Corr}\big(\tilde Y(\mathbf p),\tilde Y(\mathbf p+d\,\mathbf e_a)\big)
$$
或用局部 SSIM / 梯度一致性 $1-\|\nabla\tilde Y_{\mathbf p}-\nabla\tilde Y_{\mathbf p+d\mathbf e_a}\|$。$S_a(d)$ 高 → 该间距两 patch 内容仍匹配，配对合法。

### 3.2 配对可用度场（核心指标）

$$
\boxed{\;Q_a(d) \;=\; N_a(d)\cdot S_a(d)\;}
$$

含义：既要求噪声解相关（$N$ 大）、又要求结构仍匹配（$S$ 大）时，该轴该间距的自监督配对才合法。这是 ACES 唯一的第一性量，**可用 CIDC25 GT / 9-repeat 直接证伪**。

### 3.3 从可用度场导出采样几何（椭球降为参数化，非定义）

先做**物理坐标归一化**（采纳 GPT Layer-1）：不要把体素索引当物理距离，显式保留
$$x_{\text{phys}}=x\,\Delta x,\quad y_{\text{phys}}=y\,\Delta y,\quad z_{\text{phys}}=z\,\Delta z,\qquad(\text{3P: }\Delta z\gg\Delta x)$$
无标定时用相对比例占位并在日志显式标注"未标定"。

再由可用度场取阈值导出各轴半轴：
$$
\boxed{\;d_a^\ast(z)=\max\{\,d:\;Q_a(d,z)>\tau\,\},\qquad r_a(z)=d_a^\ast(z)\;}
$$
最后（**仅作高效参数化**）构造度量张量与椭球（马氏）距离：
$$
M(z)=\mathrm{diag}\big(r_z(z)^{-2},r_y(z)^{-2},r_x(z)^{-2}\big),\qquad
d_M(p,q)=\sqrt{(p-q)^\top M(z)\,(p-q)}
$$
逻辑链是 **data → pairability $Q_a(d)$ → geometry**，而非"先发明椭球"。

### 3.4 采样算子：v1 用确定性候选 + 打分 + 拒绝（采纳 GPT③）

> **不在 v1 引入随机高斯加权采样**——它会同时改变配对距离、频率、采样概率与训练分布，结果变好也无法归因。第一版走确定性：

1. **候选枚举**：以椭球 $\{q:d_M(p,q)\le1\}$ 为邻域，枚举满足 noise 不重叠约束的候选配对 $\mathcal{C}_a$。
2. **打分**：每个候选按 $Q_a(d_{pq})$ 打分。
3. **Top-K 选择 + 拒绝**：优先高 $Q$ 配对；保留各轴正交覆盖与方向平衡；
   $$Q>\tau\ \text{用配对};\quad \tau_{\text{low}}<Q\le\tau\ \text{弱化配对};\quad Q\le\tau_{\text{low}}\ \text{拒绝，回退 VALID 标准采样}$$

**落地方式**（已核对 VALID `sampling.py` 可行）：改动**只在新采样模块**（新建 `ACES/sampling_aniso.py`），CRN 主干与损失沿用 VALID 原实现。三档可一键切换：`立方 baseline`（复现 VALID 组合式 idx_pair）/ `整数各向异性 block`（每轴 $B_a=\lceil2r_a\rceil$）/ `连续椭球度量`。随机高斯加权作为**后期增量**，在确定性版成立后再研究。

**端点定位（采纳 GPT②⑨，删除"等价"表述）**：ACES 是一族连续各向异性采样；**各向同性配置 ≈ VALID-like 基准**，**Z 退化配置 ≈ SN2N-like 基准**。二者作为**可比较的基准配置**纳入消融，**不宣称算法等价**。

### 3.5 损失（v1 复用已验证项，不动 Hessian 权重规则）

> **修正（采纳 GPT⑮⑯）**：上一版 $\lambda_{\text{reg},a}\propto(1-w_a)$ 有风险——低可用度轴（常是细长树突所在的 z）会被过度平滑，抹掉真实精细结构。**v1 保持 VALID 原 Hessian 正则不变**。

$$
L = \underbrace{L_{2neighbor}+L_{idt}+\lambda_{reg}L_{Hessian}}_{\text{VALID 原损失（不改权重规则）}} + \lambda_{sc}\underbrace{\|f(\text{sub}_a)-f(\text{sub}_b)\|_1}_{\text{SN2N 自约束（后期加入，见 §6 Stage4）}}
$$

各向异性/不确定性感知正则 $\lambda_H(p)=f(U_s(p),G(p))$（低不确定+强结构→弱正则；高不确定+平背景→强正则；高不确定+强生物结构→**不盲目加平滑**）作为**后期消融**，非 v1。physics loss / Poisson-Gaussian NLL / PSF 通道**一律不进 v1**。

### 3.6 采样不确定性 $U_s$（增量，采纳 GPT⑰⑱，但非 v1 核心）

继承 SN2N 自约束思想，用 $K\ge4$ 个可行采样配置的预测方差定义**采样不确定性**（无需额外 uncertainty head）：
$$
\mu=\tfrac1K\textstyle\sum_k\hat X_k,\qquad U_s=\tfrac1K\textstyle\sum_k\|\hat X_k-\mu\|^2
$$
必须验证 $U_s$ 与真实误差相关（$\mathrm{Corr}(U_s,|X-\hat X|)$ 显著）才算有价值，否则只是方差图。可驱动 §3.4 的 pair rejection。放在 Stage5。

### 3.7 深度自适应（3P 关键，写成待验证假设）

> **修正（采纳 GPT⑭）**：不承诺"越深 $r_z$ 越小"为算法必然行为。深度增加使 SNR↓、像差改变轴向 PSF，但 $S_z(d)$ 未必单调下降。

按深度分 bin 分别估计 $Q_a(d,z)$，得 $r_a(z)$，使采样几何随深度自动调整。**这是待实验估计的假设**（"depth-conditioned sampling is hypothesized to be beneficial and will be estimated"），Stage6 验证。

### 3.8 两个来源分支：实测 vs 标定先验

| 分支 | 触发条件 | $r_a$ 来源 | 定位 |
|---|---|---|---|
| **B-empirical**（默认） | 无 PSF/体素标定（CIDC25、pollen 轴向） | $r_a=d_a^\ast$（由 $Q_a(d)$ 阈值） | 不依赖任何光学参数，普适 |
| **B-calibrated**（增量） | 有 PSF/体素物理尺寸标定 | $r_a$ 由物理 FWHM/体素比作为**采样先验/初值** | **sampling/calibration prior，非 physical model**；只影响采样几何，不进损失 |

两分支产生可比较的 $r_a$，直接消融"实测 vs 标定 vs 一致性"。CIDC25 无标定 → 只走 B-empirical，验证普适性。

---

## 4. 与历史版本的关系（明确改了什么）

| 维度 | 历史版本 | 本最终版 |
|---|---|---|
| 第一性量 | 启发式 $r_a=\gamma L_a\eta_a$（当核心公式） | **可用度场 $Q_a(d)=N_a(d)S_a(d)$**，几何由 $r_a=d_a^\ast$ 导出 |
| 椭球定位 | 核心定义，宣称"统一并超越 VALID/SN2N" | **参数化表达**；端点为"可比较基准配置"，不宣称等价 |
| 采样实现 | 一上来高斯核加权随机采样 | **v1 确定性候选+打分+拒绝**，随机采样后置 |
| 噪声估计 | 残差自相关 ≈ 噪声 | 区分 proxy/金标准，分区估计，9-repeat 正式验证 |
| Hessian 权重 | $\lambda\propto1-w_a$（有过度平滑风险） | v1 不改；不确定性/结构感知正则后置为消融 |
| 深度→$r_z$ | 写成必然收缩 | 待验证假设 |
| 不确定性 | 仅 $L_{sc}$ | 增量的采样方差 $U_s$ + 与真实误差相关性验证 |
| 物理建模 | 塞损失（已失败） | 仅作采样先验，且标注"当前预算下 PSF-loss 未见优势" |

---

## 5. 研究假设（须证实，勿写成已证实）

- **H1**：2P/3P 噪声独立性存在显著各向异性，$Q_x(d)\approx Q_y(d)\gg Q_z(d)$，可从数据测得。
- **H2**：由 $Q_a(d)$ 驱动的自适应采样，优于 VALID-like 各向同性与 SN2N-like XY-only 两个基准端点。
- **H3**：$Q_a(d)$ 能在有 GT 的 CIDC25 上预测最优采样几何（核心可证伪主张）；且时间轴在快钙瞬变处自动收缩（$r_t$↓）。
- **H4**：并非所有自监督配对都值得学；pair rejection（$Q>\tau$）在质量/数据利用率间存在有价值权衡。
- **H5**：深度条件采样 $r_a(z)$ 在 3P 深层带来比固定策略更大的增益（待估计，非必然）。
- **H6**：采样方差 $U_s$ 与真实重建误差显著相关，可作可靠性指标。

---

## 6. Codex 执行顺序（分阶段，每步有验收门；采纳 GPT 二十节重排）

> **重排理由**：先证"各向异性真实存在"再训练；采样-only 实验单独隔离归因；不确定性/深度/跨模态逐层加，避免一次引入过多变量。

### Stage 0 · 复现与配对分析
| Task | 内容 | 验收 |
|---|---|---|
| T0 | 环境自检；VALID/SN2N 均可 import；记录 torch/CUDA/commit | 无 import 错误 |
| T1 | 复现 VALID 基线（3P repeat01，参数同 8-20 记录） | PSNR≈22.9dB(±0.5) |
| T2 | 复现 SN2N-3D 基线；分析两者实际生成配对的结构/统计性质 | 两 baseline 均跑通并记录配对分布 |

### Stage 1 · 先回答"各向异性是否真实"（不训练）
| Task | 内容 | 验收 |
|---|---|---|
| T3 | 实现 `anisotropy_profiler`：对 x,y,z 输出 $N_a(d),S_a(d),Q_a(d)$ 及 $Q_a(d,z)$ 深度图；9-repeat 算 $N_a^{\text{repeat}}$ | 产出 $Q_x,Q_y,Q_z$ 曲线与深度图 |
| T3b | proxy vs 金标准正式验证：$\mathrm{Corr}(N_a^{\text{proxy}},N_a^{\text{repeat}})$，前景/背景分区 | 报告 Spearman/Pearson，非仅"趋势一致" |
| **门** | 若未出现 $Q_x\approx Q_y\gg Q_z$ 或深度依赖，暂停并复议方法支点 | 各向异性经验基础成立 |

### Stage 2 · Sampling-only 实验（固定 CRN+VALID loss，只换采样）
| Task | 内容 | 验收 |
|---|---|---|
| T4 | 五档采样一键切换并各自跑通：①VALID ②SN2N-like XY ③各向同性 3D 候选 ④各向异性经验候选 ⑤连续椭球 | 五档跑通，指标可比 |
| T4b | CIDC25（有 GT）标定：验证 $Q_a(d)$ 预测最优采样几何 = 实测最优；观察 $r_t$ 是否自动收缩 | 相关性显著（记录 Spearman） |
| **门** | 回答"采样本身是否有效"，禁止在此阶段引入 uncertainty/PSF/新架构 | 能归因到采样 |

### Stage 3 · Pair rejection
| T5 | 扫 $\tau\in\{0,0.2,0.4,0.6,0.8\}$，研究质量↔数据利用率权衡 | 产出 τ-质量曲线 |

### Stage 4 · 自约束一致性
| T6 | 加入 SN2N $L_{sc}$，验证 $\Delta$PSNR/$\Delta$SSIM/$\Delta U$ | 记录增益与代价 |

### Stage 5 · 采样不确定性
| T7 | 计算 $U_s$，验证 $\mathrm{Corr}(U_s,|X-\hat X|)$ 显著；可选驱动 rejection | 相关性显著否则标注为无效增量 |

### Stage 6 · 深度自适应
| T8 | 由 $Q_a(d,z)$ 得 $r_a(z)$，比较 fixed-ACES vs depth-adaptive-ACES | 深层增益 > 浅层 |

### Stage 7 · 2P→3P 跨模态泛化
| T9 | 2P 上开发→3P 深组织压力测试（不混训）；分层指标 + 深度曲线 + photon-budget(9-repeat 子采样) + 各向异性/采样 mask 可视化 | 全部产出 |

工程结构：baseline 源码零修改，改动全在新 `ACES/` 目录；每步存 config/seed/checkpoint/metrics。

---

## 7. 可选增量：DeepCAD-RT 时间冗余（不绑定主创新）

$Q_a(d)$ 框架天然可扩展到 t 轴：把时间当作又一轴测 $N_t(d),S_t(d),Q_t(d)$。快事件使某些时刻 $S_t$ 骤降 → $Q_t$ 自动下调，回避 DeepCAD/FAST 都警告的"时间窗过大抹掉快动态"。若主创新在 Stage2 后成立，作为自然 ablation 分支加入，非 v1 必做。

---

## 8. 成功标准

**有效（至少其一）**：ACES 在 foreground-PSNR / SSIM_fg / local-profile / 深度分层 或 神经元分割下游中，稳定优于 VALID-like 与 SN2N-like 两端点（同 CRN、同预算）。
**可信**：$Q_a(d)$ 在 CIDC25 上确实预测最优采样几何（核心可证伪主张）；低 photon / 深层优势更明显。
**必产物**：分层指标表、深度曲线、photon-budget 曲线、$Q_a(d)$ 与采样 mask 可视化、代表性 3D 渲染、分割对比。**禁止只报全图 PSNR**（历史假提升教训）。

---

## 9. 风险与纪律

- **R1 假提升**：全图指标涨但前景/profile 不涨 → 判背景主导伪提升（沿用分层评价）。
- **R2 静默坍缩**：3D pipeline 退化成 2D → fail-fast 断言体块 z 维。
- **R3 归因不清**：固定 CRN，Stage2 只换采样；不确定性/正则/深度分阶段单独加。
- **R4 残差污染**：$N_a^{\text{proxy}}$ 在结构区被信号污染 → 强制分区估计并与 9-repeat 金标准对齐后方可采信。
- **R5 过度平滑**：禁止 v1 用 $\lambda\propto1-w_a$；正则改动须有 $U_s$/结构证据支撑。
- **R6 过拟合单数据**：3P(9-repeat) 开发 → CIDC25 验证 → pollen 泛化，独立数据不共享调参。
- baseline 源码零修改；VALID 中 `DWT3D(noisy_output_1)` 原写法仅记录不改。

---

## 10. 参考

1. FAST — Wang et al., *Nat. Commun.* (2025). DOI: 10.1038/s41467-025-64681-8
2. VALID — Gu et al., *Sci. Adv.* (2026). DOI: 10.1126/sciadv.ady9194；本地 `Methods/VALID-v1.0/`
3. SN2N — Qu et al., *Nat. Methods* (2024). DOI: 10.1038/s41592-024-02400-9；本地 `/data2/wjb/SN2N-main/`
4. DeepCAD-RT — Li et al., *Nat. Biotechnol.* (2023)；本地 `Methods/DeepCAD-RT-main/`
5. 本课题历史：`threePM/report/SUCCESS_PATH_2D_2P5D_3D.md`、`threePM/archive/failed_experiments_20260413/`（PSF-loss 失败证据）

---

## 11. GPT 审查采纳记录（`ChatGPT-董必勤团队论文分析-20260824-1828.md`）

| GPT 建议 | 决定 | 落点 |
|---|---|---|
| ① $r_a=\gamma L_a\eta_a$ 降为启发式，核心改 $Q_a(d)=N\cdot S$ | **接受** | §3.1–3.3 |
| ② 删"$r_z\to0$ 等价 SN2N"，改"端点包含/基准" | **接受**（源码已证不等价） | §1.2、§3.4 |
| ③ v1 不用随机高斯 gather，先确定性候选+打分+拒绝 | **接受** | §3.4 |
| ④ 不用 $\lambda_{H,a}\propto1-w_a$，改结构/不确定性感知 | **接受**（v1 不改，后置消融） | §3.5 |
| ⑤ 残差自相关 ≠ 噪声，9-repeat 正式验证 | **接受** | §3.1、T3b |
| ⑥ $\sigma_a$ 改为按距离分辨的结构相关 $S_a(d)$ | **接受** | §3.1(2) |
| ⑦ 用 pairability 曲线取代单一 $L_a$ | **接受** | §3.1–3.2 |
| ⑧ pair rejection 机制 | **接受为增量**（Stage3，非 v1 核心） | §3.4、T5 |
| ⑨ 采样不确定性 $U_s$ + 与真实误差相关性 | **接受为增量**（Stage5） | §3.6、T7 |
| ⑩ 物理坐标归一化 Layer-1 | **接受** | §3.3 |
| ⑪ CIDC25 定位为辅助锚点，9-repeat 为主战场 | **接受** | §2、§6 |
| ⑫ 深度→$r_z$ 收缩写成待验证假设 | **接受** | §3.7 |
| ⑬ 实验顺序重排（先证各向异性再训练，采样单独隔离） | **接受** | §6 |
| ⑭ PSF 措辞"当前预算下未见优势" | **接受** | §1.3、§3.8 |

**未全盘照搬之处**：GPT 建议把 $U_s$、rejection、不确定性正则都纳入核心方法；本文将它们定位为**Stage2 采样-only 结论成立后的增量**，以保证归因清晰——这与 GPT 自己"Stage2 不要引入 uncertainty"的建议一致，故为一致性收敛而非分歧。
