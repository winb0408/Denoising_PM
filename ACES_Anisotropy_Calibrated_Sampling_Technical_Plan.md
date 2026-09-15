# ACES 技术路线：噪声各向异性标定的自监督体去噪

> **文档定位**：交给 Codex 执行的研究规格书。本文**取代** `PAOS_2P_3P_Anisotropy_Aware_Technical_Roadmap.md`（GPT 初版）与 `PAOS_v1_Executable_Technical_Plan.md`（第一次修订）。取代原因见 §1。
>
> **方法名**：ACES = **A**nisotropy-**C**alibrated **E**mpirical self-supervised **S**ampling。
>
> **一句话核心创新**：不用假设的 PSF/前向物理模型（该路线在本课题历史实验中已验证失败），而是**从数据直接测量每个空间轴的噪声独立性与结构连续性**，用实测统计量自动决定自监督配对在各轴上的强度——把 VALID 的各向同性 2×2×2 采样与 SN2N 的 XY-only 对角采样，统一为一个"按数据/按深度自适应"的连续采样策略。

---

## 1. 为什么取代前两版（基于本地证据的硬约束）

Codex 必须先理解这三条来自真实代码与历史实验的证据，它们决定了方法支点：

1. **SN2N 的 3D 采样本身就回避 Z 轴。** 源码 `/data2/wjb/SN2N-main/SN2N/datagen.py::block3d`：即使 3D 模式，配对也只在 XY 做对角
   $$\text{left}=\tfrac12(I_{0::2,0::2}+I_{1::2,1::2}),\quad \text{right}=\tfrac12(I_{0::2,1::2}+I_{1::2,0::2})$$
   帧/z 轴**原样保留**。→ "不要把对角复制到 Z" 不是猜想，是 SN2N 作者的既定做法。各向异性问题真实存在。

2. **失败的是"PSF 作训练损失约束"，不是"物理建模"本身。** `/data2/wjb/threePM/archive/failed_experiments_20260413/stage4_psf_unsuccessful/`：PSF 通道、PSF loss、PSF full 三变体在同等预算下均未超过 baseline。教训是**不要把 PSF 塞进损失端**；ACES 改为把物理建模移到**采样几何端**（§3.6 B-calibrated 分支），且默认走无标定实测分支，二者可切换、可消融。

   **CIDC25 无 PSF 标定（已实测确认）**：`Training/A1.tif` 为 `[1500,490,490]` int16，元数据仅 `{"shape":[1500,490,490]}`，无 PSF/NA/波长/体素尺寸；且它是 **2D+时间**（1500 帧），第三轴是 t 不是 z，GT 为多帧平均参考。→ CIDC25 只能走 B-empirical 分支，恰好验证普适性。

3. **N2N 家族的唯一硬前提是"配对观测的噪声独立"**，这是**可测量**的量，无需 PSF/detector 标定。ACES 把创新支点放在这个可测量、可证伪的量上。

---

## 2. 现有资产（Codex 只能依赖这些真实路径）

**代码基座**（均已本地存在，禁止假设联网仓库）：

| 仓库 | 真实路径 | 角色 |
|---|---|---|
| VALID | `Methods/VALID-v1.0/FDU-donglab-VALID-34f5637/` | **主干**：CRN + 2×2×2 N2N + Hessian 正则 |
| SN2N | `/data2/wjb/SN2N-main/` | XY 对角采样 + 自约束损失（`SN2N/datagen.py`, `SN2N/trainer.py`） |
| DeepCAD-RT | `Methods/DeepCAD-RT-main/` | 交错帧时间冗余（**可选**，见 §6） |
| SRDTrans / SUPPORT | `Methods/SRDTrans-main/`, `Methods/SUPPORT-main/` | 对照方法 |

**数据资产**：

| 数据 | 路径 | 关键属性 | 用途 |
|---|---|---|---|
| CIDC25 钙成像 | `threePM/ThreePM_openDatasets/AI4Life-CIDC25/` | `[1500,490,490]`，**有干净 GT F0** | **标定/监督验证锚点** |
| VALID 3P neuron | 见 8-20 复现记录 | 每深度 **9 repeats**，Δz=4μm，0–1096μm | 真实噪声测量 + 深层压力测试 |
| pollen 真实 3P | `threePM/ThreePM_openDatasets/pollen_test/` | 有横向标定 `2.051μm/px`、帧率、功率、PMT gain；**无轴向 PSF 标定** | 真实数据泛化 |
| 各向异性稀疏集 | `threePM/ThreePM_openDatasets/200um-f10-z5_sparse/` | 2x/4x × X/Y/Z/XY/XYZ + dense + mask | **各向异性天然实验台** |

**可复用的方法学纪律**（来自 threePM 历史，务必继承）：
- 全图 PSNR/SSIM 会被大背景稀释产生**假提升**；必须分层报告 global / foreground / background / local-profile（见 `threePM/report/SUCCESS_PATH_2D_2P5D_3D.md`）。
- 严防 **3D→2D 静默坍缩**（`threePM/dataset.py` 的 fail-fast 思路）。
- 环境：`/data2/wjb/anaconda3/envs/threePM/bin/python`，A6000，按空闲卡设 `CUDA_VISIBLE_DEVICES`。

---

## 3. 核心创新：噪声各向异性标定 → 采样几何

### 3.1 可测量的两个逐轴统计量（无需任何物理标定）

对一段体/时序数据 $Y$，沿每个轴 $a\in\{x,y,z,(t)\}$ 估计两个量：

**(1) 噪声独立性 $\eta_a$**（N2N 是否成立的直接依据）。先用一个快速去噪 proxy（如 3×3 中值或小高斯）得到 $\tilde Y$，残差 $r=Y-\tilde Y$，计算残差沿轴 $a$ 的 lag-1 归一化自相关：

$$
\phi_a = \frac{\mathbb{E}[\,r(\mathbf{p})\,r(\mathbf{p}+\mathbf{e}_a)\,]}{\mathbb{E}[\,r(\mathbf{p})^2\,]},\qquad \eta_a = 1-|\phi_a|\in[0,1]
$$

$\eta_a\to1$ 表示该轴相邻体素噪声近似独立（可安全做 N2N 配对）；$\eta_a\to0$ 表示噪声高度相关（配对会把相关噪声当信号学，有害）。

> **对 9-repeat 数据的加强版**：直接用重复观测估计真实噪声的逐轴相关，比 proxy 残差更准，作为 $\eta_a$ 的金标准校验。

**(2) 结构连续性 $\sigma_a$**（配对是否会破坏结构）。沿轴 $a$ 的一阶梯度能量占比：

$$
\sigma_a = \frac{\mathbb{E}[\,(\partial_a \tilde Y)^2\,]}{\sum_{b}\mathbb{E}[\,(\partial_b \tilde Y)^2\,]}
$$

$\sigma_a$ 大表示该轴结构变化剧烈，跨该轴配对更易引入结构偏差。

### 3.2 逐轴相关长度 → 椭球半轴（把统计量变成采样几何）

先定义逐轴**配对可行度**（noise 独立 × 结构平坦）：

$$
w_a = \eta_a\cdot\exp(-\beta\,\sigma_a),\qquad \beta\ \text{默认 1.0}
$$

再把它转成一个**物理有意义的相关长度** $L_a$（残差/信号自相关衰减到 $1/e$ 的 lag），并定义每个轴的**椭球半轴**：

$$
\boxed{\;r_a \;=\; \gamma\cdot L_a\cdot \eta_a\;}\qquad\text{（无标定分支）},\qquad
r_a \;=\; \gamma\cdot \mathrm{FWHM}_a\ \text{（有标定分支，见 §3.6）}
$$

含义：半轴 ∝ "信号还近似平坦、但噪声已解相关"的尺度。3P 因 Δz=4μm 常 $>$ 轴向相关长度 → $r_z$ 自动坍缩；CIDC25 的快钙瞬变使时间轴相关长度骤降 → $r_t$ 自动收缩（回避把瞬变当噪声抹掉）。

### 3.3 各向异性椭球采样算子（核心创新，非"两端插值"）

**动机**：VALID 的 `datasets/sampling.py` 用 `space_to_depth(block_size=2)` 把体切成 `2×2×2` 立方块 + 固定 `idx_pair` 选配对；SN2N 用 XY 整数对角。二者都默认"**体素索引空间里的立方/方形邻域 = 合理配对邻域**"。但 3P 物理上 xy 是亚微米、z 是 4μm，索引空间的立方体在物理空间里是被强烈拉长的长方体——**这个各向同性假设本身就是可攻击的创新点**。

**定义**：用逐轴半轴 $r=(r_z,r_y,r_x)$ 构造度量张量 $M=\mathrm{diag}(r_z^{-2},r_y^{-2},r_x^{-2})$，两体素 $p,q$ 的**椭球（马氏）距离**

$$
d_M(p,q)=\sqrt{(p-q)^\top M\,(p-q)}
$$

配对邻域为椭球 $\{q: d_M(p,q)\le 1\}$；块内候选配对按高斯核 $\exp(-\tfrac12 d_M^2)$ **加权采样**。这样"邻域"是**连续的各向异性椭球**，而非离散立方。

**为什么这比"插值"强**：立方 `2×2×2` 只是 $r_x{=}r_y{=}r_z$ 的一个特例，SN2N XY-only 是 $r_z{\to}0$ 的特例——两个已发表方法都被这一个椭球度量**统一并超越**。端点退化仅作 sanity check（可单测），不再是卖点。

**落地方式**（已核对 VALID 源码可行）：外层用每轴整数 block $B_a=\lceil 2r_a\rceil$ 控制 `space_to_depth` 粒度，块内把固定 `idx_pair` 替换为"按 $\exp(-\tfrac12 d_M^2)$ 加权的邻域 gather + 配对采样"。$r$ 各向同性=1 → 逐元素等价 VALID；$r_z\to0$ → 等价 SN2N。改动**只在新采样模块**，CRN 主干与损失不改。

**必做消融（证明椭球这一步的边际收益）**：`立方 baseline` vs `整数各向异性 block` vs `连续椭球度量` 三档，缺一不可，否则无法归因椭球相对粗粒度开关的增益。

### 3.4 损失（复用 VALID + SN2N 已验证项，不引入未验证物理项）

$$
L = \underbrace{L_{2neighbor}+L_{idt}+\lambda_{reg}L_{Hessian}}_{\text{VALID 原损失}} + \lambda_{sc}\underbrace{\|f(\text{sub}_a)-f(\text{sub}_b)\|_1}_{\text{SN2N 自约束（多视图一致）}}
$$

其中 Hessian 正则在低配对轴（如 z）上**加权提高**，用正则而非配对来维持该轴平滑：$\lambda_{reg,a}\propto(1-w_a)$。physics loss、Poisson-Gaussian NLL、PSF 通道**一律不进 v1**（历史已证失败）。

### 3.5 深度自适应（3P 的关键）

3P 深层 SNR 下降、噪声相关性随深度变化。按深度分 bin 分别估计 $r_a(z)$，使椭球采样几何**随深度自动调整**：浅层可保留较大 $r_z$，深层自动坍缩为 XY 主导椭圆。这是"各向异性标定"相对固定策略的直接价值点。

### 3.6 两个来源分支：无标定实测 vs 有标定物理建模（关键修正）

历史 PSF 实验失败的是**把 PSF 当训练损失/前向约束**这一用法（`stage4_psf_unsuccessful/`），**不等于**"采样阶段做物理建模无意义"。ACES 把物理建模**从损失端移到采样端**，且设为两个可切换分支，二者输出同一套椭球半轴 $r_a$：

| 分支 | 触发条件 | $r_a$ 来源 | 说明 |
|---|---|---|---|
| **B-empirical**（默认） | 无 PSF/体素标定（如 CIDC25，已确认 tiff 无标定；pollen 仅横向标定） | $r_a=\gamma L_a\eta_a$（数据实测） | 不依赖任何光学参数，普适 |
| **B-calibrated**（增量） | 有 PSF/体素物理尺寸标定时 | $r_a=\gamma\,\mathrm{FWHM}_a$ 或物理体素比 | 物理建模只影响**采样几何**，不进损失，规避历史失败模式 |

两分支产生可比较的 $r_a$，因此能直接做消融："实测椭球 vs 物理标定椭球 vs 两者一致性"。这既回应了你"采样阶段物理建模可能仍有意义"的判断，又不重蹈 PSF-loss 覆辙。CIDC25 无标定 → 只能走 B-empirical，正好验证普适性。

---

## 4. 与前两版方案的关系（明确改了什么）

| 维度 | GPT 初版 / 第一次修订 | ACES（本文） |
|---|---|---|
| 各向异性来源 | 人工假设 PSF 公式 $w\propto e^{-d/f}$ | **数据实测** $L_a,\eta_a$，有标定时可选物理 FWHM |
| 物理建模位置 | 塞进训练损失（已失败） | 移到**采样几何端**（B-calibrated 分支），不进损失 |
| 采样几何 | 描述与源码不符；沿用立方/方形邻域 | **连续各向异性椭球度量**，统一并超越 VALID 立方 / SN2N 方形 |
| 可证伪性 | 假设难验证 | $w_a$ 可用 CIDC25 GT 直接标定验证 |
| 时间冗余 | v1 必做 / Phase IV | **可选增量**（§6），不绑定主创新 |

---

## 5. Codex 执行顺序（每步有验收门）

| Task | 内容 | 验收 |
|---|---|---|
| T0 | 环境自检；VALID 与 SN2N 均可 import；记录 torch/CUDA/commit | 无 import 错误 |
| T1 | 复现 VALID 基线（3P repeat01，参数同 8-20 记录） | PSNR≈22.9dB（±0.5） |
| T2 | 实现 `anisotropy_profiler`：输入体块，输出 $L_a,\eta_a,\sigma_a$ 及椭球半轴 $r_a$；9-repeat 校验 $\eta_z$ | proxy 与 repeat 金标准趋势一致 |
| T3 | 实现椭球采样算子（度量 $M$ + 高斯核加权 gather）；单测两端退化：$r$ 各向同性=1 逐元素等于 VALID，$r_z{=}0$ 等于 SN2N XY 对角 | 两端严格等价 |
| T3b | 三档消融骨架就绪：`立方 baseline` / `整数各向异性 block` / `连续椭球` 可一键切换 | 三档均可跑通 |
| T4 | 在 CIDC25（有 GT，无 PSF）上标定：验证 $r_a$ 预测最优采样几何 = 实测最优 | 相关性显著（记录 Spearman） |
| T5 | 跑 ACES(椭球) vs 整数各向异性 block vs VALID(立方) vs SN2N-3D vs SRDTrans/SUPPORT | 见 §7，须能归因椭球边际收益 |
| T6 | 深度自适应 $w_a(z)$；3P 深度分层评估 | 深层增益 > 浅层 |
| T7 | 汇总：分层指标表 + 深度曲线 + photon-budget（用 9-repeat 子采样）+ 各向异性可视化 | 全部产出 |

工程结构、目录隔离（baseline 零修改，改动全在新 `ACES/` 目录）、每步存 config/seed/checkpoint/metrics 同前一版规范。

---

## 6. 可选增量：DeepCAD-RT 时间冗余（不绑定主创新）

按你的说明，时间冗余仅作参考。ACES 的 $\eta_a$ 框架**天然可扩展到 t 轴**：把时间当作又一个轴测 $\eta_t,\sigma_t$。对钙成像，快事件会使某些时刻 $\sigma_t$ 骤升，$w_t$ 自动下调，正好回避 DeepCAD/FAST 都警告的"时间窗过大抹掉快动态"问题。→ 若 T5 后主创新成立，可作为一个自然的 ablation 分支加入，而非 v1 必做。

---

## 7. 成功标准

**有效（至少其一）**：ACES 在 foreground-PSNR / SSIM_fg / local-profile / 深度分层指标 或 神经元分割下游中稳定优于 VALID 与 SN2N-3D（同 CRN、同预算）。

**可信**：$w_a$ 在 CIDC25 上确实预测最优采样几何（可证伪的核心主张）；低 photon / 深层优势更明显。

**必产物**：分层指标表、深度曲线、photon-budget 曲线、各向异性 $w_a$ 可视化、采样 mask 可视化、代表性 3D 渲染、分割对比。禁止只报全图 PSNR（历史假提升教训）。

---

## 8. 研究假设（须证实，勿写成已证实）

- H1：2P/3P 噪声独立性存在显著各向异性（$\eta_z\ll\eta_{xy}$），可从数据测得。
- H2：由 $\eta_a,\sigma_a$ 驱动的自适应采样优于 VALID 各向同性与 SN2N XY-only 两个端点。
- H3：$w_a$ 能在有 GT 的 CIDC25 上预测最优采样几何（核心可证伪主张）。
- H4：深度自适应 $w_a(z)$ 在 3P 深层带来比固定策略更大的增益。
- H5：删除假设式物理约束不损害、甚至优于含 PSF 的变体（与历史失败实验呼应）。

---

## 9. 风险与纪律

- **R1 假提升**：全图指标涨但前景/profile 不涨 → 判为背景主导伪提升（沿用 threePM 分层评价）。
- **R2 静默坍缩**：3D pipeline 退化成 2D → fail-fast 断言体块维度。
- **R3 端点不等价**：若两端退化未逐元素对齐基线，禁止进入 T4。
- **R4 过拟合单数据**：3P→CIDC25→pollen 用独立数据验证，不共享调参。
- **R5 归因不清**：固定 CRN，只换采样/权重做消融。
- baseline 源码零修改；保留 VALID 中 `DWT3D(noisy_output_1)` 原写法仅记录不改。

---

## 10. 参考

1. FAST — Wang et al., *Nat. Commun.* (2025). DOI: 10.1038/s41467-025-64681-8
2. VALID — Gu et al., *Sci. Adv.* (2026). DOI: 10.1126/sciadv.ady9194
3. SN2N — Qu et al., *Nat. Methods* (2024). DOI: 10.1038/s41592-024-02400-9；本地 `/data2/wjb/SN2N-main/`
4. DeepCAD-RT — Li et al., *Nat. Biotechnol.* (2023)；本地 `Methods/DeepCAD-RT-main/`
5. 本课题历史：`threePM/report/SUCCESS_PATH_2D_2P5D_3D.md`、`threePM/archive/failed_experiments_20260413/`（PSF 失败证据）
