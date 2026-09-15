# PAOS-2P/3P：面向双光子/三光子体成像的各向异性物理感知正交自监督去噪技术路线

> **Working title:** Physics-Aware Anisotropic Orthogonal Sampling for Self-Supervised Denoising of Low-Photon Two-/Three-Photon Volumetric Imaging
>
> **Working abbreviation:** PAOS-2P/3P
>
> **Purpose:** 本文作为 Codex 的研究与实现指导规格书。目标不是直接实现最终全部功能，而是按“复现基线 → 采样创新 → 物理约束 → uncertainty → 自适应采样 → 生物学验证”的顺序构建可验证的研究原型。

---

## 0. 核心结论

本项目聚焦 **two-photon / three-photon volumetric fluorescence microscopy**。不应把 SN2N 的 2D diagonal sampling 直接沿 Z 轴复制，因为 2P/3P 体成像通常具有明显的 **XY–Z anisotropy**：横向采样、轴向步长、PSF/OTF 和 SNR 随深度变化并不等价。

因此推荐的新方法不是“SN2N + VALID 网络拼接”，而是：

> **以 VALID 的 3D orthogonal/Tetris self-supervision 为基础，将 SN2N 的“局部物理对称采样 + prediction self-consistency + uncertainty 思想”重新设计为一个 dimension-aware、anisotropy-aware 的 sampling framework。**

核心创新假设：

\[
\boxed{\text{Local physical redundancy} + \text{3D orthogonal redundancy} + \text{photon/PSF physics} \rightarrow \text{reliable pseudo-supervision}}
\]

随后将 sampling-induced uncertainty、model uncertainty 和 physics consistency 结合起来：

\[
\boxed{U = \alpha U_{sampling}+\beta U_{model}+\gamma U_{physics}}
\]

最终进一步探索：

\[
\boxed{\text{uncertainty} \rightarrow \text{adaptive sampling}}
\]

这条路线比直接替换 Transformer/backbone 更符合已有 FAST、VALID、SN2N 的技术演进逻辑，也更容易依托现有开源代码快速落地。

---

# 1. 研究对象与应用边界

## 1.1 首要应用：双光子体成像

第一阶段建议锁定：

> **low-photon two-photon volumetric calcium / structural imaging of mouse cortex**

原因：

- VALID 已直接在 two-photon volumetric imaging 上验证；
- FAST 已在 two-photon calcium imaging 上验证；
- DeepInterpolation 有成熟的 two-photon calcium imaging 公开实现和数据生态；
- SelfMirror 等 volumetric self-supervised 方法可作为 baseline；
- 下游可以使用 neuron segmentation、calcium event detection、trace correlation 等成熟评价任务。

## 1.2 第二应用：三光子体成像

在方法稳定后扩展到：

> **deep-tissue three-photon multimodal volumetric imaging**

推荐场景：

- hippocampus / deep cortex
- neuron + vessel + THG 等多模态成像
- 800–1100 μm 等深层低 photon 条件

VALID 已在 3P 数据上展示很强的低 SNR 恢复能力，因此 3P 是 PAOS 的自然第二验证场景，而不是另起一套算法。

## 1.3 不建议第一阶段做的事情

暂不优先：

- 单纯“Transformer + VALID”；
- 单纯增大 3D U-Net；
- 同时引入 2P、3P、OCT、LFM、wide-field 等大量模态；
- 一开始就同时处理 XYZT、角度 U/V、spectral λ；
- 一开始就做 fully learnable sampling。

这些会使 novelty attribution 和 ablation 失控。

---

# 2. 现有方法的技术基础

## 2.1 SN2N

SN2N 的核心不是 U-Net，而是：

1. 基于 SR 成像物理属性构造 spatial diagonal resampling；
2. 用 Fourier interpolation 恢复 structural scale；
3. 产生内容高度相似而 noise realization 不同的 pseudo-pairs；
4. 用 self-constrained learning 强制不同 noisy views 的预测一致；
5. 用 Patch2Patch 提高 data efficiency。

关键思想：

\[
\text{physical symmetry}
\rightarrow
\text{paired noisy observations}
\rightarrow
\text{N2N-style self-supervision}
\]

SN2N 官方仓库已提供 2D/3D data generation、training、inference 和 Patch2Patch 等组件，可作为采样逻辑参考和代码参考。

## 2.2 FAST

FAST 重点在：

- spatiotemporal frame multiplexing；
- temporal window / shift 的动态权衡；
- 极轻量 2D CNN；
- real-time inference。

FAST 的启发是：

> sampling design 可以替代 network complexity。

第一阶段 PAOS 不直接引入 FAST temporal branch，但保留其“信息采样优先于网络堆叠”的设计哲学。

## 2.3 VALID

VALID 的核心是：

- Tetris 3D orthogonal sampling；
- X-Y / X-Z / Y-Z cross-plane redundancy；
- CRN；
- denoising + identity + regularization；
- 3D low-frequency Hessian constraint；
- zero-shot volumetric denoising。

VALID 是 PAOS 的主体基线。

## 2.4 SN2N 对 VALID 的重要补充

SN2N 当前最值得引入 VALID 的不是“2D diagonal mask”本身，而是三个理念：

- **局部 physical symmetry**；
- **prediction-level consistency**；
- **不同 pseudo-views 的 prediction variance 可以作为 uncertainty proxy**。

---

# 3. 最核心的技术矛盾：为什么不能直接把 SN2N 复制到 Z 轴

假设体数据：

\[
Y \in \mathbb{R}^{X\times Y\times Z}
\]

但通常：

\[
\Delta x \neq \Delta z
\]

而且：

\[
PSF_{xy} \neq PSF_z
\]

同时 depth 增大后：

\[
SNR(z) \downarrow
\]

scattering / aberration / attenuation 往往进一步增强。

因此不能简单认为：

\[
(1,4) \leftrightarrow (2,3)
\]

这种二维 diagonal symmetry 可以等价推广为 Z 方向的 voxel symmetry。

真正要建立的是：

\[
\boxed{\text{dimension-aware redundancy}}
\]

即不同轴使用不同的 redundancy prior。

---

# 4. PAOS 的核心概念：Dimension-aware Physical Redundancy

## 4.1 三种冗余分别承担不同角色

### A. XY local physical redundancy

来源：

- 局部 PSF 的横向连续性；
- 横向像素采样密度；
- 局部结构平滑/纹理连续性；
- detector noise 的近似像素独立性。

这一部分借鉴 SN2N。

### B. XYZ volumetric orthogonal redundancy

来源：

- neuron soma/dendrite/vessel 的连续三维结构；
- X-Y、X-Z、Y-Z 三个正交截面的一致性。

这一部分借鉴 VALID。

### C. 2P/3P imaging physics

来源：

- Poisson-dominated photon statistics；
- Gaussian read noise；
- depth-dependent SNR；
- anisotropic PSF/OTF；
- depth-dependent attenuation / scattering / aberration。

这一部分是 PAOS 的主要“物理升级”。

因此训练 pair 必须满足：

\[
\boxed{
\text{content consistency}
+
\text{noise independence}
+
\text{physics consistency}
}
\]

---

# 5. 核心采样创新：Anisotropy-aware Hybrid Orthogonal Sampling

这是本项目最重要的 algorithmic contribution。

## 5.1 第一层：建立 physical coordinate system

不要只根据 voxel index 处理。

从 metadata 或 calibration 读取：

- lateral pixel size: \(d_x,d_y\)
- axial step: \(d_z\)
- estimated PSF FWHM: \(f_{xy},f_z\)
- optional wavelength / refractive index / objective NA
- per-depth intensity / SNR proxy

计算轴向尺度比：

\[
\rho_z = \frac{d_z}{d_{xy}}
\]

并定义 PSF anisotropy：

\[
\rho_{PSF}=\frac{f_z}{f_{xy}}
\]

这两个参数决定 sampling 是否允许沿 Z 方向建立强配对关系。

---

## 5.2 第二层：VALID-style Tetris base sampling

对局部：

\[
2\times2\times2
\]

block 构造 Tetris sampling pool。

保留 VALID 的核心原则：

- pseudo-pair 的中心 voxel 不重叠；
- 相邻 subvolume 保持 3D structural continuity；
- 利用 X-Y、X-Z、Y-Z 的 orthogonal correlation。

但不要默认三轴贡献相同。

---

# 6. PAOS 的关键变化：Axis-weighted Orthogonal Sampling

对三个正交方向分别赋权：

\[
w_{xy}, w_{xz}, w_{yz}
\]

权重不仅由 network 学习，而首先由物理可解释的 anisotropy 指标初始化。

例如：

\[
w_{xz},w_{yz}
\propto
\exp\left(-\frac{d_z}{f_z}\right)
\]

而：

\[
w_{xy}
\propto
\exp\left(-\frac{d_{xy}}{f_{xy}}\right)
\]

然后归一化：

\[
w_i=\frac{\tilde w_i}{\sum_j\tilde w_j}
\]

**注意：这里不是最终必须使用的公式，而是 Codex 第一版应该实现和 ablation 的合理起点。**

实际实验应比较：

1. equal weighting；
2. voxel-spacing weighting；
3. PSF weighting；
4. spacing + PSF weighting。

目的：验证“各向异性物理信息是否真的比单纯 voxel spacing 更有效”。

---

# 7. SN2N diagonal sampling 如何正确融入 2P/3P

## 7.1 不沿 Z 复制 diagonal

这是一个明确的设计原则。

SN2N-style diagonal sampling 主要应用于 XY plane。

对每一个 XY slice / local XY patch：

```text
[a b]
[c d]
```

形成：

\[
D_1=(a,d),
\quad
D_2=(b,c)
\]

但其有效性必须受到以下条件约束：

\[
S_{xy} \ge \tau_{structure}
\]

且：

\[
|\rho_{noise}|\le\tau_{noise}
\]

如果局部结构变化过快，则不使用 diagonal pairing。

---

# 8. 更重要的创新：从“固定 diagonal”变成“candidate sampling pool”

对于一个 local volume，不只生成一种 sampling。

定义候选池：

\[
\mathcal P
=
\mathcal P_{Tetris}
\cup
\mathcal P_{XY-diagonal}
\cup
\mathcal P_{Z-aware}
\]

其中：

### \(\mathcal P_{Tetris}\)

VALID 风格的 3D orthogonal patterns。

### \(\mathcal P_{XY-diagonal}\)

SN2N 风格的局部横向 physical-symmetry patterns。

### \(\mathcal P_{Z-aware}\)

不是直接使用相邻 Z voxel，而是：

- 按真实物理距离选择 Z neighbors；
- 根据 axial PSF overlap 判断是否可以认为具有结构一致性；
- 可以允许 z-offset = 1、2、3 voxel，但必须由 physical distance / PSF criterion 决定，而非固定 index。

这一步是 PAOS 与现有 VALID 的关键差异之一。

---

# 9. Z-aware sampling：最值得深入的具体设计

## 9.1 用“PSF support”而不是 voxel adjacency 定义 Z 邻域

传统方法：

\[
z_i \leftrightarrow z_{i+1}
\]

PAOS：

\[
|z_i-z_j| \le \kappa f_z
\]

只有当两个位置在 PSF support / physical receptive range 内时，才允许建立强 Z-structural pairing。

定义：

\[
d_{ij}=|z_i-z_j|
\]

然后：

\[
w_{ij}^{z}
=
\exp(-d_{ij}^2/2\sigma_z^2)
\]

其中 \(\sigma_z\) 与实验 PSF / axial resolution 对齐。

---

## 9.2 为什么这样比“3D diagonal”合理

如果 \(\Delta z\) 很大：

直接把：

\[
(x,y,z)
\leftrightarrow
(x+1,y+1,z+1)
\]

当作结构一致 pair 可能过于激进。

而：

\[
(x,y,z)
\leftrightarrow
(x+1,y+1,z)
\]

主要使用 XY local symmetry。

与此同时：

\[
(x,y,z)
\leftrightarrow
(x,y,z+k)
\]

只有在 physical support 足够时才使用。

因此形成：

\[
\boxed{
XY\ strong\ local\ symmetry,
\quad
Z\ conditional\ structural\ redundancy
}
\]

这正是 2P/3P anisotropy-aware 的核心。

---

# 10. Candidate Pair Validity Score

对于每个候选 pair \((P_i,P_j)\)，定义：

\[
S_{pair}
=
\lambda_sS_{structure}
+
\lambda_nS_{noise}
+
\lambda_pS_{physics}
\]

其中：

## 10.1 Structural similarity

可用：

- local gradient correlation
- local normalized cross-correlation
- SSIM（主要用于 validation / candidate analysis）

训练时推荐使用 gradient/feature-level similarity，而不是把 SSIM 直接作为主 loss。

---

## 10.2 Noise independence

估计局部 residual correlation：

\[
\rho_n=Corr(r_i,r_j)
\]

目标：

\[
|\rho_n|\rightarrow0
\]

如果 pair 的 residual correlation 太高，则降低其 sampling weight 或直接拒绝该 pair。

---

## 10.3 Physical consistency

用 imaging forward operator \(H\)：

\[
S_{physics}
=
-\|H(P_i)-H(P_j)\|
\]

或者归一化：

\[
S_{physics}
=
1-rac{\|H(P_i)-H(P_j)\|_1}
{\|H(P_i)\|_1+\epsilon}
\]

第一版可以先采用近似 PSF convolution，而不需要完整 Monte Carlo light propagation。

---

# 11. 建议的三种采样模式

## Mode A：XY-dominant

适用于：

- \(\rho_z\) 很大；
- axial sampling 较稀；
- deep-tissue low SNR；
- 结构主要在 XY 平面可辨识。

形式：

\[
P \approx P_{XY-diagonal}+P_{Tetris}^{weak-Z}
\]

---

## Mode B：Balanced anisotropic 3D

适用于：

- Z sampling 相对密；
- dendrites / vessels / soma 跨层连续；
- \(f_z\) 支持足够的 cross-plane correlation。

形式：

\[
P=P_{Tetris}+P_{XY-diagonal}+P_{Z-aware}
\]

---

## Mode C：Z-aware dominant

适用于：

- 结构在轴向连续性明显；
- z-stack 密采样；
- 3P 深层体数据中局部结构在轴向有明显延续。

此时可以增强 X-Z/Y-Z pair，但仍不得使用未经物理验证的 isotropic diagonal assumption。

---

# 12. 第一阶段不要做 Fully Learnable Sampling

第一阶段 sampling 必须具有显式可解释规则。

推荐顺序：

\[
P_{fixed}
\rightarrow
P_{physics-weighted}
\rightarrow
P_{uncertainty-guided}
\rightarrow
P_{learnable}
\]

原因：

如果一开始 sampling、network、loss 都可学习，那么无法判断最终提升究竟来自哪里，也很难证明 sampling novelty。

---

# 13. Self-constrained Learning：继承 SN2N

对于同一 underlying structure 的多个 pseudo-views：

\[
Y_1,Y_2,\ldots,Y_K
\]

共享一个 network：

\[
\hat X_i=f_\theta(Y_i)
\]

加入：

\[
L_{SC}
=
\frac{1}{K(K-1)}
\sum_{i\neq j}
\|\hat X_i-\hat X_j\|_1
\]

它继承 SN2N 的核心思想，但从 two-view 扩展为 multi-view 3D self-consistency。

---

# 14. Sampling-induced uncertainty：把 SN2N uncertainty 推向 PAOS

对同一 local volume 使用 K 个不同 valid sampling patterns，得到：

\[
\hat X_1,\ldots,\hat X_K
\]

估计：

\[
\mu=\frac1K\sum_i\hat X_i
\]

\[
U_{sampling}(x)
=Var_i[\hat X_i(x)]
\]

解释：

> 如果不同 physically valid views 最终产生高度一致的 prediction，则该区域 self-supervision stability 较高；如果 prediction 差异大，则该区域可能存在 insufficient redundancy、极端低 SNR、结构快速变化或 sampling violation。

这是非常重要的输出之一。

---

# 15. Model uncertainty

第一版推荐最简单的 deep ensemble：

\[
f_{\theta_1},...,f_{\theta_M}
\]

例如 M=3。

估计：

\[
U_{model}(x)=Var_m[f_{\theta_m}(x)]
\]

不要一开始使用复杂 Bayesian architecture。

---

# 16. Physics uncertainty

输出 \(\hat X\) 后重新通过 forward model：

\[
\hat Y=H(\hat X)
\]

定义：

\[
U_{physics}
=
\|H(\hat X)-Y\|
\]

可进一步分解为：

- PSF mismatch
- photon likelihood residual
- depth-dependent residual

---

# 17. 最终 uncertainty

\[
U
=
\alpha U_{sampling}
+
\beta U_{model}
+
\gamma U_{physics}
\]

第一版先固定：

\[
\alpha=\beta=\gamma=1/3
\]

后续通过 validation calibration 调整。

注意：不要直接把 uncertainty 当成 penalty 全部压低，否则可能导致过度平滑。

正确目标是：

> **uncertainty should be calibrated, not minimized blindly.**

---

# 18. Physics-aware Loss

第一版建议总 loss：

\[
L=
L_{pair}
+\lambda_{sc}L_{SC}
+\lambda_{phys}L_{phys}
+\lambda_{str}L_{struct}
+\lambda_{id}L_{identity}
\]

其中：

## 18.1 Pair loss

\[
L_{pair}
=
\frac{1}{K}
\sum_i
\|\hat X_i-Y_{target(i)}\|_1
\]

---

## 18.2 Self-consistency

\[
L_{SC}
=
Var_i(\hat X_i)
\]

或 L1 pairwise consistency。

---

## 18.3 Physics loss

低 photon 情况建议优先尝试 Poisson-Gaussian likelihood，而非简单 MSE：

\[
Y\sim Poisson(G(X))+N(0,\sigma_r^2)
\]

第一版可做近似负对数似然：

\[
L_{PG}
\approx
\frac{(Y-G(\hat X))^2}
{G(\hat X)+\sigma_r^2+\epsilon}
+
\log(G(\hat X)+\sigma_r^2+\epsilon)
\]

其中 G 可以是 intensity gain / calibration mapping。

如果真实 detector calibration 不明确，则先退化成经过验证的 robust photon-weighted residual。

---

## 18.4 Structural loss

推荐：

\[
L_{grad}
=
\|\nabla\hat X_a-\nabla\hat X_b\|_1
\]

或者使用：

- 3D Hessian
- low-frequency wavelet

保持与 VALID 的 structural regularization 思路连续。

---

# 19. CRN backbone：第一版不要创新网络

第一版直接复用 VALID CRN。

理由：

- CRN 已在 volumetric biomedical denoising 上验证；
- VALID 本身强调 sampling 是 model-agnostic；
- 可以让论文 novelty 集中在 sampling/physics/uncertainty；
- 有利于做严格 ablation。

第二阶段才考虑：

- lightweight anisotropic convolution
- axial attention
- separable 3D operator
- hybrid CNN/Transformer

---

# 20. 进一步的网络改进：Anisotropic CRN（Phase II）

如果 PAOS-v1 已经有效，再研究：

\[
Conv_{3D}
\rightarrow
Conv_{xy}+Conv_z
\]

或者：

\[
K_{xy}\times1
\quad + \quad
1\times1\times K_z
\]

形成 anisotropic separable convolution。

网络每层可以显式输入：

\[
(d_x,d_y,d_z,f_{xy},f_z)
\]

这样 architecture 才真正感知物理 anisotropy。

---

# 21. 2P 与 3P 的差异化设计

## 21.1 2P

重点：

- moderate depth
- calcium dynamics
- soma/dendrite structure
- 30 Hz 左右 time-lapse 等公开范式

优先验证：

\[
XYZ
\rightarrow
Denoising
\rightarrow
Neuron segmentation
\rightarrow
Calcium event extraction
\]

## 21.2 3P

重点：

- deeper penetration
- lower photon availability
- stronger depth-dependent degradation
- more severe anisotropy / scattering / aberration concerns

因此 PAOS 的 physics-aware sampling 应进一步让权重依赖 depth：

\[
w(P,z)=f(SNR(z),PSF(z),depth)
\]

建议把 3P 当作 **generalization / stress test**：

> 如果 PAOS 真的利用的是 physical redundancy，而不是只对某一种 2P 数据过拟合，那么它应该在 3P 深层数据上展示更明显的优势。

---

# 22. 重要：sampling 应该与 depth 联动

定义 depth bins：

\[
z_1,z_2,\ldots,z_M
\]

每个深度估计：

- mean signal
- background variance
- local SNR
- estimated PSF / blur proxy

得到：

\[
SNR(z),\quad PSF_z(z)
\]

然后：

\[
P_z=\pi(P\mid SNR(z),PSF(z),d_z)
\]

即不同深度使用不同 sampling mixture。

这会比一个 global sampling pattern 更符合 2P/3P 的实际成像物理。

---

# 23. Uncertainty-guided adaptive sampling：Phase III

这是最终最有潜力的升级。

第一次：

\[
Y \rightarrow \{P_1,P_2,P_3\}
\rightarrow
U_1
\]

如果：

\[
U_1<\tau
\]

停止生成更多 pseudo-views。

否则：

\[
Y \rightarrow P_4,P_5,...
\]

直到：

\[
U<\tau
\]

形成：

\[
\boxed{Adaptive Redundancy Acquisition / Processing}
\]

这里的“adaptive”第一阶段可以先是 **computational sampling**；不要一开始就让 microscope hardware 改变 acquisition。

如果后期要升级，再研究 acquisition-level adaptive illumination。

---

# 24. 最重要的 ablation matrix

必须严格执行以下实验矩阵：

| Model | Tetris | XY-Diagonal | Anisotropy weighting | Physics | Self-consistency | Uncertainty |
|---|---:|---:|---:|---:|---:|---:|
| N2V / standard baseline | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| VALID | ✓ | ✗ | ✗ | ✓/原始 | ✓ | ✗ |
| SN2N-style 3D | ✗ | ✓ | ✗ | ✗ | ✓ | ✗ |
| Hybrid sampling | ✓ | ✓ | ✗ | ✗ | ✓ | ✗ |
| + anisotropy | ✓ | ✓ | ✓ | ✗ | ✓ | ✗ |
| + physics | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ |
| + uncertainty | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

注意：这里“VALID Physics”要按论文真实实现和代码实现核对后再确定，不得在最终论文中夸大 VALID 原始方法的物理 forward consistency 范围。

---

# 25. Sampling ablation 必须比 network ablation 更重要

至少比较：

1. random 3D masking；
2. standard blind-spot；
3. SN2N-like XY diagonal；
4. VALID Tetris；
5. Tetris + diagonal；
6. Tetris + diagonal + anisotropy weighting；
7. Tetris + diagonal + anisotropy + physics screening。

保持同一个 CRN。

这样才能证明：

> improvement comes from sampling, not network capacity.

VALID 本身已经通过 sampling ablation 证明其主要收益来自 Tetris sampling 而非 backbone；PAOS 必须延续这种实验逻辑。

---

# 26. 关键评价指标

## Image-level

- PSNR
- SSIM
- CNR
- EPI
- FRC / spatial resolution proxy
- depth-wise metrics

## Noise / uncertainty

- sampling variance
- model variance
- physics residual
- calibration error
- uncertainty–error correlation
- selective risk / coverage-risk curve

## Biological task

### Neuron
- segmentation Accuracy
- Recall
- F1
- Jaccard
- neuron count recovery

### Calcium
- event detection precision/recall
- event timing error
- ΔF/F correlation
- trace SNR

### Structure
- dendrite continuity
- vessel continuity
- soma morphology preservation

---

# 27. 最关键的实验设计：Photon-budget curve

建议把核心结果之一设计成：

\[
Performance=f(Photon\ Budget)
\]

测试：

- 100%
- 50%
- 20%
- 10%
- 5%
- 1% photon / effective exposure level

关注：

\[
\boxed{Biological\ utility\ per\ photon}
\]

而不仅仅是最高 PSNR。

这一点与 SN2N 的 photon-efficiency 逻辑及 2P/3P photon-limited imaging 问题高度一致。

---

# 28. 第二个核心结果：Depth robustness

把整个 volume 按 depth 划分：

\[
0-200\mu m,
200-400\mu m,
400-600\mu m,
...
\]

分别报告：

- PSNR/SSIM/CNR/EPI
- neuron recovery
- uncertainty

理想结果应该是：

> PAOS 随 depth 增大性能下降速度比 VALID 更慢，并且 uncertainty 在真正困难区域升高，而不是产生虚假高置信度。

---

# 29. 第三个核心结果：Failure / hallucination analysis

必须专门寻找：

- extremely low photon
- dark regions
- fine dendrite
- crossing structures
- deep region
- abrupt motion / calcium event

比较：

Raw / VALID / SN2N-like / PAOS

同时展示：

\[
Output + uncertainty
\]

验证 uncertainty 是否真正对应 failure。

---

# 30. 开源基础与代码实施策略

## 30.1 VALID

优先作为主代码基座：

- environment：Python 3.9
- PyTorch 2.x
- CRN
- Tetris sampler
- 3D patch inference
- GUI 可先忽略

官方 GitHub：
https://github.com/FDU-donglab/VALID

## 30.2 SN2N

提取：

- diagonal resampling
- self-constrained loss
- Patch2Patch
- 3D data generation参考

官方 GitHub：
https://github.com/WeisongZhao/SN2N

## 30.3 FAST

暂时只作为：

- temporal baseline
- future Phase IV
- 2P dynamic signal preservation reference

官方 GitHub：
https://github.com/FDU-donglab/FAST

## 30.4 DeepInterpolation

用于 2P calcium imaging baseline/data pipeline。

GitHub：
https://github.com/AllenInstitute/deepinterpolation

## 30.5 SelfMirror

用于 volumetric 2P self-supervised baseline。

Codex 应在开始实验前确认当前仓库、license、environment 和数据可用性。

---

# 31. Codex 第一阶段执行任务

## Task 1：环境检查

建立：

```text
paos/
├── baselines/
│   ├── VALID/
│   ├── SN2N/
│   ├── FAST/
│   ├── DeepInterpolation/
│   └── SelfMirror/
├── paos/
│   ├── sampling/
│   ├── physics/
│   ├── uncertainty/
│   ├── models/
│   ├── losses/
│   └── evaluation/
├── configs/
├── scripts/
└── experiments/
```

先不要修改 baseline 代码。

## Task 2：复现 VALID 2P baseline

要求：

- 在公开/允许使用的 2P volume 上跑通；
- 输出 denoised volume；
- 保存 PSNR/SSIM/CNR/EPI；
- 保存 segmentation downstream 结果；
- 记录实际 GPU memory / inference time。

## Task 3：复现 SN2N spatial sampling

不要直接把整个 SN2N network 移植到 PAOS。

只提取：

- diagonal pair generation
- self-consistency loss

验证它在 anisotropic 2P volume 上的行为。

## Task 4：实现 anisotropy profiler

输入 volume metadata：

```yaml
voxel_size_xyz:
psf_fwhm_xyz:
depth_um:
photon_level:
```

输出：

```yaml
rho_spacing:
rho_psf:
axis_weights:
depth_snr_profile:
```

## Task 5：实现 Hybrid Sampler

第一版仅：

```text
Tetris sampler
+
XY diagonal sampler
```

暂不加入 learnable router。

## Task 6：实现 Pair Validity Score

至少实现：

```python
structure_score
noise_independence_score
physics_consistency_score
```

先离线统计，不直接进入 training。

## Task 7：加入 self-consistency

验证：

\[
L_{SC}
\]

对不同 K 个 valid sampling views 的稳定性。

## Task 8：加入 photon physics

先实现 Poisson-Gaussian synthetic noise benchmark，再处理真实数据。

## Task 9：uncertainty

实现：

- sampling variance
- 3-model ensemble variance
- forward-model residual

暂不做 Bayesian network。

## Task 10：最终评估

必须生成：

1. quantitative comparison table；
2. depth-wise curves；
3. photon-budget curves；
4. uncertainty/error calibration plot；
5. representative 3D renderings；
6. neuron segmentation comparison。

---

# 32. Phase II：如果 PAOS-v1 成功

加入：

## A. Z-aware candidate generation

根据：

\[
PSF_z,
\Delta z,
SNR(z)
\]

动态决定 Z-neighbor candidates。

## B. Adaptive sampling mixture

学习：

\[
p(P|x,z)
\]

但保留 hard physical constraints，不能让网络产生任意 sampling。

## C. uncertainty-guided additional views

根据：

\[
U(x)
\]

决定是否需要更多 self-supervised views。

---

# 33. Phase III：3P generalization

使用 3P deep volumetric imaging。

重点验证：

\[
Performance(depth)
\]

以及：

\[
Performance(photon\ budget)
\]

假设：

> PAOS 的 physical anisotropic sampling 在 3P 深层低 photon 条件下应该比仅利用固定 3D redundancy 的方法更加稳健。

这一点必须通过实验验证，不得预设结果。

---

# 34. Phase IV：4D extension（非首发版本）

在 PAOS-XYZ 稳定后，再加入 FAST-like temporal redundancy：

\[
XYZ+T
\]

形成：

\[
P=P_{space}+P_{orthogonal}+P_{temporal}
\]

并让：

\[
w_{space},w_{time}
\]

依赖 local dynamics。

特别注意 FAST 对 temporal causality / fast dynamics 的警告；不能简单扩大 temporal window。

---

# 35. 预期论文故事

如果 PAOS-v1/v2 成功，推荐论文主线：

### Problem
Low-photon 2P/3P volumetric imaging suffers from anisotropic spatial resolution, depth-dependent SNR and lack of clean GT.

### Gap
Existing self-supervised methods generally use spatial, temporal or volumetric redundancy, but do not explicitly model **dimension-dependent physical validity of pseudo-pairs** and their associated uncertainty.

### Method
PAOS introduces:

1. anisotropy-aware candidate sampling;
2. XY physical-symmetry sampling inspired by SN2N;
3. XYZ orthogonal sampling inspired by VALID;
4. physics-aware pair screening / forward consistency;
5. multi-view self-consistency;
6. sampling/model/physics uncertainty estimation.

### Outcome
在低 photon、深层、各向异性体成像条件下：

- better structure preservation;
- lower hallucination;
- better quantitative downstream analysis;
- calibrated uncertainty;
- improved photon efficiency。

### Second paper direction
uncertainty-guided adaptive sampling / 4D dynamic extension。

---

# 36. 当前创新点的优先级

| 创新点 | 必须实现 | 优先级 | 风险 |
|---|---:|---:|---:|
| Hybrid Tetris + XY diagonal | ✓ | S | 低 |
| anisotropy-aware axis weighting | ✓ | S | 低 |
| PSF-aware pair screening | ✓ | S | 中 |
| self-consistency | ✓ | S | 低 |
| sampling uncertainty | ✓ | S | 中 |
| Poisson-Gaussian physics | ✓ | A | 中 |
| physics forward consistency | ✓ | A | 中 |
| uncertainty-guided adaptive sampling | ✗ | A | 中高 |
| learnable sampling router | ✗ | B | 高 |
| XYZT extension | ✗ | B | 高 |
| hardware adaptive acquisition | ✗ | C | 很高 |

---

# 37. 必须避免的技术风险

## Risk 1：把 SN2N diagonal symmetry 错误解释为 XYZ isotropy

必须明确：

> SN2N diagonal sampling 是提供 **XY local physical redundancy** 的来源，不意味着 Z 轴可以等价复制。

## Risk 2：physics loss 只是形式包装

必须进行 ablation：

- no physics
- PSF only
- photon statistics only
- PSF + photon statistics

证明物理约束实际减少 error / hallucination。

## Risk 3：uncertainty 只是 visualization

必须验证：

\[
Corr(U,error)>0
\]

并做 coverage-risk。

## Risk 4：sampling novelty 无法证明

固定 CRN，全面替换 sampling。

## Risk 5：第一版方法太复杂

不要同时实现：

- Transformer
- learnable router
- temporal attention
- Bayesian network
- multimodal fusion

第一版只做：

\[
\boxed{
Sampling + Physics + Self-consistency + Uncertainty
}
\]

---

# 38. 当前最重要的科研假设

### H1
2P/3P 的 XYZ anisotropy 使得 naive isotropic self-supervised sampling 不是最优。

### H2
SN2N-style XY local physical symmetry 可以作为 VALID Tetris 的补充，而不是替代。

### H3
利用 PSF / voxel spacing / depth SNR 可以筛除不可靠 pseudo-pairs，提高 self-supervised training stability。

### H4
不同 valid sampling views 的 prediction variance 可以成为可靠的 local uncertainty proxy。

### H5
physics residual 能够帮助识别 hallucinated / physically inconsistent structures。

### H6
在低 photon / deep tissue 条件下，PAOS 的优势应随着 challenge 增强而更明显；如果只在高 SNR 条件有效，则研究价值有限。

---

# 39. Codex 工作纪律

1. **先复现，后创新。**
2. baseline 代码与 PAOS 代码分目录保存，避免污染 baseline。
3. 每一步实验必须保存 config、seed、checkpoint、metric JSON 和输出 volume。
4. 任何新 sampling 都必须提供可视化 mask / pair relation。
5. 每个创新模块必须有 ablation。
6. 不得把“研究假设”写成“已经证实”。
7. 不得只报告 PSNR/SSIM，必须包含 biological downstream metrics。
8. 对 2P → 3P 的 generalization 必须使用完全独立的数据/实验条件进行验证。
9. 对公开代码应记录 commit/version、license 和环境。
10. 如果现有代码无法直接运行，先建立 compatibility layer，不要大规模重写原仓库。

---

# 40. 推荐的最小可行版本（MVP）

如果需要最快判断 idea 是否成立，只实现：

```text
VALID CRN
   +
VALID Tetris sampler
   +
XY diagonal physical sampler
   +
axis-weighted candidate selection
   +
SN2N-style self-consistency
```

在同一套 2P 体数据上比较：

```text
Raw
N2V
SN2N-like
VALID
Hybrid Tetris+Diagonal
PAOS-v1
```

如果 PAOS-v1 已经在：

- CNR
- EPI
- depth robustness
- neuron segmentation

中稳定超过 VALID，那么再加入 physics + uncertainty。

这是最稳妥的研发顺序。

---

# 41. 研究成功标准

## Stage 1：算法可运行

- [ ] VALID baseline reproducible
- [ ] SN2N-inspired diagonal sampler reproducible
- [ ] Hybrid sampler generates valid pseudo-pairs

## Stage 2：算法有效

- [ ] PAOS > VALID in at least one image-quality metric under matched compute
- [ ] PAOS > VALID in at least one biological downstream metric
- [ ] advantage persists under low-photon condition

## Stage 3：算法可信

- [ ] uncertainty correlates with reconstruction error
- [ ] failure cases are identifiable
- [ ] physics residual reduces hallucination / false structures

## Stage 4：跨模态

- [ ] 2P validation complete
- [ ] 3P validation complete
- [ ] same algorithmic framework works with only calibrated physical parameters changed

## Stage 5：高级创新

- [ ] uncertainty-guided adaptive sampling improves efficiency
- [ ] optional XYZ+T extension preserves fast dynamics

---

# 42. 参考与代码入口

## User-provided papers

1. **FAST** — Wang et al., *Nature Communications* (2025), “Real-time self-supervised denoising for high-speed fluorescence neural imaging.” DOI: https://doi.org/10.1038/s41467-025-64681-8
2. **VALID** — Gu et al., *Science Advances* (2026), “Enhancing biomedical optical volumetric imaging via self-supervised orthogonal learning.” DOI: https://doi.org/10.1126/sciadv.ady9194
3. **SN2N** — Qu et al., *Nature Methods* (2024), “Self-inspired learning for denoising live-cell super-resolution microscopy.” DOI: https://doi.org/10.1038/s41592-024-02400-9

## Open-source starting points

- VALID: https://github.com/FDU-donglab/VALID
- FAST: https://github.com/FDU-donglab/FAST
- SN2N: https://github.com/WeisongZhao/SN2N
- DeepInterpolation: https://github.com/AllenInstitute/deepinterpolation
- SelfMirror: verify current public repository and license before integration

---

# 43. 最终建议

当前最推荐的研究顺序不是“先做一个大模型”，而是：

\[
\boxed{
2P/3P\ anisotropy
\rightarrow
physics-aware\ sampling
\rightarrow
hybrid\ Tetris+XY\ diagonal
\rightarrow
self-consistency
\rightarrow
physics\ screening
\rightarrow
uncertainty
\rightarrow
adaptive\ sampling
}
\]

其中真正应该争取形成论文核心 novelty 的部分是：

> **Anisotropy-aware, physics-constrained pseudo-pair generation for volumetric self-supervision.**

其次是：

> **Sampling-induced uncertainty as a measure of intrinsic self-supervision reliability.**

最终升级为：

> **Uncertainty-guided adaptive redundancy utilization.**

如果前三步成立，这条技术路线就可以自然从 2P 扩展到 3P，再扩展到 XYZT 动态成像，而不需要重新设计整个方法体系。
