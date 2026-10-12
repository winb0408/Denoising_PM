# Denoising_PM

**User:** Anonymous  
**Created:** 9/17/2026 16:34:31  
**Updated:** 9/22/2026 22:11:01  
**Exported:** 9/22/2026 22:12:09  
**Link:** [https://chatgpt.com/c/6aaba616-d1bc-83ec-bfc8-eb6e1c16d99a](https://chatgpt.com/c/6aaba616-d1bc-83ec-bfc8-eb6e1c16d99a)  

## Prompt:
2026/9/22 17:18:06

https://github.com/winb0408/Denoising_PM.git该仓库下为按照你给出的项目方案推进执行的阶段性产物，尤其是ACES目录下的qmask。仓库中有具体的实验记录报告、代码和部分结果。你先帮我审阅一下这个项目当前的产物，复述研究目标、思路，然后审阅距离初始方案是否偏移，若有修改是否恰当，并指出当前不足和下一步创新建议。另外我需要补充一点：ACES/example中有我使用我自己的实际数据进行测试的对比结果，我当前的目标更倾向于在去噪的同时，能够有效加强深层神经元信号，进一步提高深层信号的信背比。

## Response:
2026/9/22 17:18:24

我先按“研究目标 → 实际代码/实验数据流 → 与初始方案对照 → 偏移是否合理 → 当前证据强弱 → 面向深层神经元 SNR 的下一阶段创新”这条线审阅，并重点核对 `ACES/example` 的真实数据结果。🔍

从仓库文本和现有 run 看，项目已经出现一个很重要的转折：**QMask 本身更像“机制验证版”，而不是最终论文方法**；同时你的真实数据目标明确把问题从“去噪 PSNR”推进到了“深层信号增强/SNR”。

> **审阅示例数据**
>
> 我正在确认示例中的二进制图像与仓库内容，但当前文本读取并不适合直接检查这类文件。我会先依据可访问的说明和结果文件，评估流程、对比指标与示例是否一致。
>
> **审阅了示例数据**
>
> 
>
> **获取精确示例与计划名称**
>
> 
>
> **下载示例图像文件**
>
>
> **下载了示例图像文件**
>
> 
>
> **获取并核对QMask配置、报告、训练评估及实验汇总**
>
> 
>
> **正在搜索网页**
>
> 
>
> **已搜索 1 个网站**
>
> 
>
> **评估信号指标**
>
> 我正在判断示例结果是否足以支持深层信号增强。现有指标更像均值比值和背景波动比较，可能不足以反映真实对比度；我会建议结合原始帧、独立平均结果与深度剖面验证。
>
> **评估信号指标**
>
> 
>
> **评估QMask信号增强指标并核验种子几何配置及图像文件**
>
> 
>
> 思考了 2m 28s

我审阅了当前 GitHub `main` 分支中的 ACES/qmask、Stage 1/2 实验产物，以及你刚刚补入的 `ACES/example` 实际数据说明。仓库最近一次提交是 **2026-09-22**，已经把 `obj_2mm.tif`、`AVG_red-1100.tif`、`AVG_green-1300.tif` 等实际数据样例纳入仓库。

先给结论：

> **目前项目没有偏离原研究主线，反而通过失败实验把原方案中“距离/椭球半径”这一错误假设证伪了。QMask 是一次合理的中间修正，但它还不能作为最终论文方法。真正值得继续做的，应从“数据驱动选择参与哪些轴”升级为“数据驱动决定什么样的观测对值得被拿来训练”，并且最终评价目标应从单纯 PSNR 明确转向“深度条件下的信号保持 + 背景抑制 + SNR/CNR/SBR 提升”。**

---

# 1. 先重新明确：现在这个项目到底在研究什么

结合最初 ACES 方案、现在仓库中的实验，以及你新增的实际数据，我认为项目真正的问题已经非常清楚了。

## 1.1 原始目标

最初的 ACES 想解决的是：

> **在没有 clean ground truth 的情况下，针对 2P/3P 各向异性三维成像数据，能否从数据本身判断哪些空间方向、哪些空间距离适合构造 Noise2Noise 式自监督训练对，并据此自动选择采样几何。**

它的基本思想是：

$$
\text{数据本身}
\rightarrow
\text{噪声独立性}
+
\text{结构连续性}
\rightarrow
\text{pairability}
\rightarrow
\text{采样几何}
\rightarrow
\text{自监督去噪}
$$

这和 DeepCAD、SN2N、FAST、VALID 的共同核心是吻合的：**不是首先改变网络，而是改变“如何从原始数据构造训练对”。**

DeepCAD 主要利用时间冗余；SN2N利用空间 diagonal redundancy；FAST把时空冗余组织成 frame-multiplexed sampling；而 2026 年发表的 VALID 则直接把 3D spatial redundancy 组织成 2×2×2 Tetris orthogonal sampling，并明确把 sampling strategy 作为其性能提升的核心来源。([Nature](https://www.nature.com/articles/s41592-021-01225-0?utm_source=chatgpt.com))

---

# 2. 现在其实多了一个非常重要的“应用目标”

你这次补充的 `ACES/example` 非常关键。

仓库 `ACES/example/readme.txt` 明确写了：

- `obj.tif`：约 1.8 mm 深度的 XYZ 三维数据，每层 5 μm；
- 1.8 mm 以下受到物镜卡住等伪影影响，需要剔除；
- `AVG` 是 **25 帧叠加**的三维数据；
- 当前数据的核心问题不是单纯“看起来噪声大”，而是：
  - **越深信号越弱**
  - 红色通道在胼胝体附近背景噪声明显
  - 绿色通道在海马附近背景较强
  - 最终需要提高深层信号的可见性和信背比。

因此，我现在更建议把整个项目的目标重新表达为：

> **面向各向异性三维/四维生物医学光学成像，研究无需 clean ground truth 的数据驱动自监督采样与配对机制，在抑制噪声的同时最大程度保持并增强深层弱神经元信号，从而提升深度相关的 SNR、CNR 和 SBR。**

这里“增强”一定要理解成 **improve detectability / recover latent signal**，而不是简单把亮度放大。

这点非常重要。否则容易变成“看起来更亮”，而不是科学意义上的信号恢复。

---

# 3. 当前实际代码已经发生了什么变化

把目前仓库真实代码流和原计划摆在一起，其实非常清楚。

## 3.1 原始计划

```text
3P/2P raw volume
      │
      ▼
各向异性分析
  N / S / Q
      │
      ▼
估计 rx, ry, rz
      │
      ▼
Ellipsoid / empirical sampling
      │
      ▼
构造自监督 pair
      │
      ▼
VALID-like denoiser
      │
      ▼
去噪结果
```

---

## 3.2 当前真实实现

现在实际上变成：

```text
raw / repeated volume
      │
      ▼
Stage 1
N(d), S(d), Q(d)
      │
      ▼
QMask geometry
 ┌──────────────────────┐
 │ 哪个轴允许参与 phase │
 │ z / y / x = on / off│
 └──────────────────────┘
      │
      ▼
2×2×2 / 2×2×1 / ...
co-located phase split
      │
      ▼
4-view / VALID-like CRN
      │
      ▼
L_neighbor
+ L_identity
+ Hessian
      │
      ▼
checkpoint
      │
      ▼
mean9 / temporal mean proxy
      │
      ├── PSNR
      ├── SSIM
      ├── SNR/SBR
      └── depth/time bin
```

这个改变我认为总体上是**正确的**，因为你已经通过实验发现原先的“半径”概念本身有问题。

---

# 4. 原来的 ellipsoid 为什么失败，这个修正是合理的

这是当前项目最有价值的一部分。

仓库 `qmask/docs/TECHNICAL.md` 对失败原因的总结基本是成立的。

原先：

$$
r_a=\max\{d:Q_a(d)>\tau\}
$$

实际上选择的是：

> “最远还勉强满足阈值的位置”。

对于单调衰减的 Q 曲线，这实际上会系统性地把 pair 推向“最危险边缘”。

更严重的是，3P 数据里你发现：

$$
N(d=1)\approx 1
$$

但是：

$$
S_z(d=1)<S_x(d=1),S_y(d=1)
$$

也就是说，**z 方向真正的问题不是噪声不独立，而是结构在 z 方向衰减得更快。**

因此：

```text
原方案：
z 方向半径更大
        ↓
更远的 z pair
        ↓
结构错配更严重
```

显然是反的。

而 QMask 改成：

```text
每一轴：
d_min = noise independence 的下界
d_max = structure continuity 的上界

然后问：
d = 1 是否安全？
```

在逻辑上明显更合理。

所以：

### 这次从“ellipsoid”转成“axis participation mask”

我认为是**恰当的科研修正，而不是偏题**。

甚至这应该保留为论文中的一个方法学发现：

> **A distance-threshold-derived geometric radius is insufficient for self-supervised pair construction in anisotropic volumetric microscopy because noise independence and structural correspondence impose different-and sometimes opposing-constraints.**

这个发现本身是有价值的。

---

# 5. 但是 QMask 目前有一个根本问题：它把问题“简化过头了”

这可能是现在最重要的判断。

QMask 现在实际解决的是：

> **“这一整个轴要不要参与？”**

而不是：

> **“哪一种 pair 最适合训练？”**

这是完全不同的两个问题。

---

## 5.1 d_min / d_max 目前实际上没有进入真正的采样距离

这是代码层面非常明确的问题。

`geometry.py` 计算：

$$
d_{\min}, d_{\max}
$$

但是 `sampling.py` 最终依然做的是：

```text
block = 2 or 1
phase = 0 / 1
```

也就是说最终仍然是：

$$
d=1
$$

的 phase split。

所以现在实际上是：

```text
测量很多距离
      ↓
得到 d_min / d_max
      ↓
最后只问：
d=1 能不能？
      ↓
YES → axis ON
NO  → axis OFF
```

因此：

> **当前 QMask 是“distance-aware calibration → binary axis decision”，不是“distance-aware sampling”。**

这就是它和原始 ACES 目标之间最大的剩余差距。

---

# 6. 第二个理论/实现矛盾：`use_n_decision=False`

这一点代码中非常明显。

`geometry.py` 的注释说：

> `use_n_decision=False`：N threshold 不参与决策。

但实际代码是：

$$
Q=N\times S
$$

而最终依然要求：

$$
Q(d=1)\ge\tau_q
$$

所以：

> **N 虽然不再作为单独的门槛，但是仍然通过 Q 间接参与了决策。**

这不算严重 bug，但它确实是理论表述不一致。

现在实际上应该二选一：

### 方案 A
真正定义：

$$
\text{participate}=f(S,Q)
$$

并明确承认 Q 中包含 N。

### 方案 B
真正改成：

$$
\text{participate}=f(S,\text{other structure metrics})
$$

彻底移除 Q。

我更建议后者，甚至进一步：

> **不要再把 N/S/Q 压成一个 scalar Q。**

因为后续真正需要的是一个 **pair-risk vector**，而不是单一“配对性分数”。

---

# 7. 第三个更大的问题：QMask 在 Neurofinder 上其实没有真正证明“各向异性自适应”

你当前 80 epoch 的 geometry：

```text
z: participate = true
y: participate = true
x: participate = true
```

也就是：

$$
111
$$

换句话说，QMask 最终退化成了：

> **全轴参与。**

那么它在这个数据集上的行为，本质上就非常接近 iso3d / VALID-like sampling。

所以目前：

> **Neurofinder 80 epoch 实验不能证明 QMask 的“自适应几何”起作用。**

它只能证明：

> 这套修改后的采样器在某个数据上可以正常训练。

这是两个层级完全不同的结论。

---

# 8. 而且 80 epoch 的结果已经暴露出“训练目标 ≠ 最终性能”的问题

这个发现实际上非常重要。

三个 seed 的当前 best foreground PSNR 大约：

| Seed | Best FG PSNR |
|---|---:|
| 3407 | 23.23 dB |
| 3408 | 23.13 dB |
| 3409 | 22.89 dB |

平均约：

$$
23.08\pm0.18\text{ dB}
$$

但它们的 best epoch 并不稳定。

seed 3407 在 epoch 70 才达到当前 best，而 seed 3408 的 best 在 epoch 20 左右。

这说明：

> **当前模型性能对训练轨迹高度敏感。**

而且 final checkpoint 经常退化。

因此，后续论文不能使用：

```text
train 80 epoch
→ final checkpoint
→ test
```

作为标准协议。

必须改成：

```text
Calibration
   ↓
Training
   ↓
Validation
   ↓
early stopping / policy selection
   ↓
Blind test
```

否则很容易把验证集调成训练集。

---

# 9. 你现在最应该警惕的是：把 Neurofinder “z” 当成真实深度

这个问题我认为必须尽早纠正。

你的 Neurofinder 数据本质是：

$$
[T,Y,X]
$$

这里第三维是 temporal axis。

但目前一些代码把它统一写成：

```text
Z
depth
```

例如当前 `depth_binned` 输出：

```text
z_range
```

对于 Neurofinder，这其实不是 physical depth。

因此：

> **Neurofinder 的分箱结果不能作为你“深层神经元增强”的证据。**

它只能用来验证：

> 不同 temporal positions 下，sampling/denoising 的行为是否稳定。

真正的“deep-signal claim”必须回到你的实际 2P/3P XYZ 数据。

---

# 10. 另外一个问题：目前 SNR/SBR 的定义还不够适合你的最终目标

这点尤其重要，因为你现在已经明确提出“深层信号增强”。

当前代码类似：

$$
SNR=20\log_{10}\frac{\mu_{fg}}{\sigma_{bg}}
$$

以及：

$$
SBR=20\log_{10}\frac{\mu_{fg}}{\mu_{bg}}
$$

这里存在两个问题。

## 10.1 SBR 直接用 20log 不太合适

荧光强度属于 intensity/count-like quantity，通常更自然的是：

$$
SBR=\frac{\mu_{signal}}{\mu_{background}}
$$

如需 dB：

$$
SBR_{dB}=10\log_{10}
\left(
\frac{\mu_{signal}}{\mu_{background}}
\right)
$$

而不是 20 log。

---

## 10.2 对“深层弱神经元”而言，单纯 SBR 还不够

例如深层：

```text
signal = 10
background = 2
```

浅层：

```text
signal = 100
background = 20
```

两者 SBR 都是 5。

但是深层明显更难检测。

所以最终至少要同时使用：

$$
SBR=\frac{\mu_{ROI}}{\mu_{BG}}
$$

$$
CNR=
\frac{\mu_{ROI}-\mu_{BG}}
{\sigma_{BG}}
$$

以及：

$$
SNR=
\frac{\mu_{ROI}-\mu_{BG}}
{\sigma_{noise}}
$$

再配合 depth-dependent curves。

对于你的论文，我反而建议：

> **把 CNR 作为核心指标，把 SBR 作为辅助指标。**

因为你的科学问题实际上是：

> 深层 neuron 从背景中能不能被更可靠地区分出来？

这比单纯“平均亮度提高多少”更科学。

---

# 11. 你这次加入 `ACES/example`，其实使项目的故事发生了一个非常好的变化

当前实际数据让我觉得，论文最终不应该写成：

> “We propose a new anisotropic self-supervised sampler and obtain better PSNR.”

这个故事和 VALID 很接近，而且很难形成足够明显的 methodological separation。VALID 2026 已经明确使用 2×2×2 Tetris sampling，并把 3D sampling 作为核心贡献。([PubMed Central (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC13082324/?utm_source=chatgpt.com))

更好的故事应该变成：

> **如何在不同深度、不同局部信号强度、不同各向异性条件下，自动找到“对模型真正有用”的自监督训练 pair。**

这一步非常关键。

---

# 12. 我建议把 QMask 定义为“中间实验”，不要继续把它当最终方法

我会重新给当前阶段定位：

```text
A0 物理可行性验证
       ↓
A1 Pairability measurement
       ↓
A2 Radius / Ellipsoid 失败
       ↓
A3 QMask
       ↓
证明：
“binary axis participation”
仍然不足
       ↓
A4 Pair-Utility Adaptive Sampling
       ↓
最终方法
```

也就是说：

> **QMask 不是失败项目，而是一个非常有用的 ablation / intermediate method。**

它已经帮助你证明：

1. “大半径”是不正确的；
2. `global radius` 不适合各向异性 3D SSL；
3. axis-level participation 比 ellipsoid 更合理；
4. fixed geometry 仍不能解释不同数据、不同深度下的 training utility。

第四点正好成为下一篇工作的出发点。

---

# 13. 下一阶段真正值得做的创新：不要再研究“mask”，研究“pair utility”

我建议把 A3 正式升级为：

## **Pair-Risk / Pair-Utility Calibrated Adaptive Sampling**

暂时不急着固定英文缩写。

核心问题变成：

$$
\boxed{
\text{Given a noisy volume, which observation pairs should be used for self-supervised training?}
}
$$

而不是：

$$
\text{Which axes should be turned on?}
$$

---

# 14. 新 A3 的基本数学形式

对每一个候选 offset：

$$
\Delta=(\Delta_x,\Delta_y,\Delta_z)
$$

甚至以后扩展成：

$$
\Delta=(\Delta_x,\Delta_y,\Delta_z,\Delta_t)
$$

不再只是：

```text
x on/off
y on/off
z on/off
```

而是建立：

$$
\mathcal D=
\{
(1,0,0),
(0,1,0),
(0,0,1),
(1,1,0),
(1,0,1),
...
\}
$$

甚至：

$$
|\Delta|=1,2,4,8,\ldots
$$

形成完整的 candidate pair library。

---

# 15. 对每一个 pair，不再计算一个 Q，而是一个 risk vector

定义：

$$
\mathbf R(\Delta)=
[
R_N(\Delta),
R_S(\Delta),
R_I(\Delta),
R_O(\Delta)
]
$$

其中：

### ① Noise risk

$$
R_N
$$

衡量两个 observation 的噪声依赖程度。

不能再只用 Pearson。

应逐步加入：

- covariance
- normalized covariance
- conditional residual correlation
- variance consistency

---

### ② Structure risk

$$
R_S
$$

不要只用：

$$
corr(x,y)
$$

可以使用：

$$
1-\rho_{LF}
$$

加上：

$$
1-\text{SSIM}
$$

加上 gradient consistency。

---

### ③ Identity / shortcut risk

这是当前 QMask 实验已经暴露出来的一个真正有价值的问题。

例如：

$$
R_I(\Delta)
=
\mathrm{sim}(x,x_\Delta)
$$

或者 normalized MSE：

$$
nMSE(\Delta)=
\frac{
\|x-x_\Delta\|^2
}{
Var(x)
}
$$

pair 太相似：

```text
input ≈ target
       ↓
identity shortcut
```

pair 太远：

```text
input ≠ target
       ↓
structural mismatch
```

所以真正存在一个中间最优区间。

这是你当前项目非常值得深入挖掘的一条线。

---

# 16. 更关键：引入“training utility”

这是我现在认为最值得把项目往前推进的一步。

你现在已经通过 r-grid / s-grid 看到一个很关键的事实：

> **静态 pair similarity 并不能完全预测最终 denoising utility。**

也就是说：

$$
Q(\Delta)\not\equiv PSNR(\Delta)
$$

甚至：

$$
corr(pairability,PSNR)
$$

可以是错误方向。

这意味着：

> **pair selection 不是纯物理判定问题，而是一个 optimization-aware problem。**

所以可以定义：

$$
U(\Delta)
=
f(
R_N,R_S,R_I,
L_{\text{neighbor}},
L_{\text{identity}},
R_{\text{residual}},
R_{\text{gradient}}
)
$$

---

# 17. 最实用的做法：micro-probe

不需要每个候选距离都训练 80 epoch。

可以：

```text
candidate offset
      ↓
1-3 epoch micro training
      ↓
training-only statistics
      ↓
utility estimate
```

例如：

$$
U(\Delta)=
-\alpha L_{neighbor}
-\beta L_{identity}
-\gamma R_{res}
+\eta G_{structure}
$$

这样就形成：

```text
物理统计
   +
优化行为
   ↓
Pair Utility
   ↓
Sampling Policy
```

这是和现在 QMask 最大的区别。

---

# 18. 然后从 hard mask 升级成 soft sampling policy

QMask：

$$
m_a\in\{0,1\}
$$

下一阶段：

$$
\pi(\Delta)
=
\frac{
e^{-R(\Delta)/T}
}{
\sum_{\Delta'}e^{-R(\Delta')/T}
}
$$

这样同一份数据可以得到：

```text
Δ=(1,0,0)   0.31
Δ=(0,1,0)   0.28
Δ=(0,0,1)   0.05
Δ=(1,1,0)   0.20
Δ=(1,0,1)   0.04
Δ=(0,1,1)   0.06
...
```

这时你的方法才真正从：

> **QMask**

变成：

> **Adaptive Sampling Policy**

---

# 19. 而你的“深层神经元增强”恰好可以成为这个 sampling policy 的第二层

我非常建议你不要直接对输出做“deep enhancement”。

应该让：

$$
\pi(\Delta|v)
$$

依赖于局部信号状态：

$$
\pi(\Delta|v)
=
\pi(
\Delta
\mid
depth,
SNR,
background,
structure
)
$$

也就是说：

```text
浅层、高 SNR
    ↓
可以采用普通 pair

深层、低 SNR
    ↓
更重视：
结构稳定方向
+
更低 shortcut risk
+
更高 pair utility
```

这就变成了真正的：

> **depth-/signal-conditioned self-supervised sampling**

而不是简单的“深层亮度增强”。

---

# 20. 对你自己的数据，我建议马上改变实验思路

这一点非常重要。

你现在的 `AVG` 是 **25 帧平均**。

因此：

> **不要把 25-frame AVG 单纯作为“输入图像”来训练，然后再拿它自己证明自己变好了。**

更好的方法是：

```text
25 raw frames
      │
      ├──── subset A
      │
      ├──── subset B
      │
      └──── full 25-frame average
```

例如：

$$
A=\{1,3,5,\ldots\}
$$

$$
B=\{2,4,6,\ldots\}
$$

得到两个相对独立的 noisy observations。

然后：

```text
A → self-supervised network → denoised A
B → self-supervised network → denoised B
```

同时：

$$
GT_{proxy}=AVG_{25}
$$

只用于：

> **validation / analysis**

而不是用于训练 target。

这样你就拥有了真正更可靠的 N2N 实验闭环。

当然，如果这 25 帧存在明显运动或生理变化，应先做 registration，再决定如何拆分。

---

# 21. 你的最终真实数据评价，我建议直接围绕“深度信号”设计

建议最终形成一张核心图：

## Fig. X - Depth-dependent signal recovery

横轴：

$$
Depth(\mu m)
$$

纵轴分别画：

$$
SNR
$$

$$
CNR
$$

$$
SBR
$$

同时加入：

```text
Raw
25-frame AVG
VALID
FAST
SN2N
Your method
```

这样会比一张普通“去噪前后图”有说服力得多。

---

# 22. 更进一步：用 neuron-level analysis，而不是只算 voxel metrics

你的数据是神经元成像，这其实是天然优势。

最终可以做：

```text
raw
 ↓
denoising
 ↓
neuron detection
 ↓
deep-neuron detection rate
```

例如：

$$
Recall_{deep}
$$

$$
F1_{deep}
$$

以及：

$$
\text{detectable depth}_{95\%}
$$

比如回答：

> 原始数据能够可靠检测神经元到 900 μm，而新方法能够延伸到 1200 μm。

这类结果比：

> PSNR +1.4 dB

更直接对应你的真实应用价值。

当然，这些数字现在还不能提前假设，必须由实际数据测出来。

---

# 23. 当前 `ACES/example` 的定位，我建议立即从“demo”升级为“验证集”

目前它还是：

```text
example/
    obj
    AVG
    result image
```

从论文角度看还比较弱。

建议以后变成：

```text
example/
├── raw/
├── splitA/
├── splitB/
├── avg25/
├── denoising/
├── roi/
├── metrics/
│   ├── depth_snr.csv
│   ├── depth_cnr.csv
│   ├── depth_sbr.csv
│   └── neuron_detection.csv
└── figures/
```

这样你的真实数据就进入正式 scientific pipeline，而不是“最后放一张效果图”。

---

# 24. 当前项目还存在几个代码/实验层面的必须修正项

我建议优先级如下。

| 优先级 | 问题 | 影响 |
|---|---|---|
| P0 | Neurofinder 时间轴仍大量被当作 Z/depth | 理论表述错误 |
| P0 | `d_min/d_max` 没真正控制 offset | QMask 与其理论不一致 |
| P0 | 主实验仍可能使用 fixed view→slot mapping | shortcut 风险 |
| P0 | geometry/checkpoint 选择与 test 混用 | 泛化证据不足 |
| P1 | N→Q→decision 的定义不一致 | 方法可解释性问题 |
| P1 | SBR 定义需要重构 | 与最终应用目标不匹配 |
| P1 | region-specific SSIM 实现实际上不是 region-specific | 指标报告问题 |
| P1 | 80 epoch 过拟合明显 | 训练协议不稳 |
| P2 | legacy ellipsoid 代码仍在主代码树 | 方法边界混乱 |
| P2 | README / docs 部分路径和实际产物不完全一致 | 复现性问题 |

---

# 25. 当前“科学证据链”我会这样评价

现在已有的东西实际上已经可以组成一个很不错的 research progression：

### 已经证明的

**证据 1**

各向异性确实存在。

$$
X/Y \neq Z
$$

---

**证据 2**

简单 `Q=N×S` + 最大半径不可靠。

---

**证据 3**

几何偏移过大，会破坏 N2N 的 same-content assumption。

---

**证据 4**

co-located phase split 是更安全的采样语义。

---

**证据 5**

fixed view→corner mapping 会出现 shortcut。

---

**证据 6**

pair similarity 与实际 training utility 并不等价。

---

### 还没有证明的

**缺失 1**

能否自动找到最佳 pair？

**缺失 2**

能否做到 depth-conditioned？

**缺失 3**

能否做到 local / patch-conditioned？

**缺失 4**

能否在完全无 GT 条件下预测这个 sampling policy？

**缺失 5**

最终能否提高深层神经元 SNR/CNR，而不牺牲浅层结构？

这五个就是下一阶段真正应该解决的问题。

---

# 26. 与当前文献相比，你下一步创新应该“跳一层”

现在的研究路线大致已经演化成：

```text
DeepCAD
  temporal redundancy
       ↓
SN2N
  spatial diagonal redundancy
       ↓
FAST
  spatiotemporal multiplexed redundancy
       ↓
VALID
  volumetric orthogonal redundancy
       ↓
LF-denoising
  spatial-angular redundancy
       ↓
你的下一步
  ─────────────────────
  pair utility / adaptive redundancy
```

FAST明确通过时空 frame-multiplexed sampling 设计训练 pair，SN2N通过 spatial diagonal resampling 建立 pair，VALID 则进一步将体数据的 3D orthogonal redundancy 组织成 Tetris sampling；近期工作也继续在探索新的高维 redundancy。([Nature](https://www.nature.com/articles/s41467-025-64681-8?utm_source=chatgpt.com))

因此单纯：

> “QMask = 根据数据决定 z 开不开”

从方法创新层级上讲还是不够远。

真正应该跨过去的是：

> **从 sampling pattern 转向 sampling policy。**

---

# 27. 我目前最推荐的最终论文主线

可以概括成一句话：

> **Self-supervised denoising should not assume a fixed sampling pattern; it should learn which noisy observations are useful training partners from the data itself.**

然后你的方法主链：

```text
Raw anisotropic volume
        │
        ▼
Local / global redundancy profiling
        │
        ▼
Candidate pair field
Δx, Δy, Δz, [Δt]
        │
        ▼
Noise risk
Structure risk
Shortcut risk
Optimization utility
        │
        ▼
Pair Utility Field
        │
        ▼
Adaptive Sampling Policy
π(Δ | depth, signal, structure)
        │
        ▼
Self-supervised denoiser
        │
        ▼
Depth-aware signal preservation
        │
        ├── PSNR / SSIM
        ├── SNR
        ├── CNR
        ├── SBR
        └── neuron detectability
```

这个故事比“QMask 改进 VALID”完整得多。

---

# 28. 还有一个我认为非常关键的判断

你现在不应该为了追求“像 DeepCAD/SN2N/FAST/VALID 一样有一个漂亮的新 sampler”，再硬设计一个新的固定几何。

因为现有实验已经告诉你：

> **固定几何本身可能就是问题。**

所以最有潜力的创新不是：

```text
Tetris
↓
QMask
↓
再设计一个六角形/椭球/八面体
```

而是：

```text
固定 sampling pattern
        ↓
数据驱动 sampling
        ↓
pair utility
        ↓
adaptive policy
```

这会让你的方法论层级明显提高。

---

# 29. 对“深层信号增强”再做一个概念上的限定

我建议你从现在开始在项目内部统一使用：

> **deep-signal preservation / recovery / detectability enhancement**

而尽量不要直接说：

> **signal amplification**

因为真正严谨的目标是：

$$
\text{remove noise}
\rightarrow
\text{recover latent signal}
\rightarrow
\text{increase detectability}
$$

而不是：

$$
\text{multiply intensity}
$$

否则 reviewer 很容易问：

> “Did the network really recover signal, or did it simply increase intensity?”

所以后面的 SNR/CNR/SBR + raw/independent-half/full-average 对照就非常重要。

---

# 30. 最终判断

### 关于“有没有偏离初始方案”

**没有。**

你实际上是在通过实验淘汰初始方案中错误的几何假设，然后向更一般的 sampling-policy 问题演进。

### 关于“QMask 的修改是否合理”

**合理，而且是必要修正。**

尤其是：

- 放弃最大 Q 半径；
- 放弃大 offset；
- 恢复 co-located pair；
- 用 axis participation 替代 ellipsoid。

这些都是正确方向。

### 关于“QMask 是否已经足够做最终论文”

**目前还不够。**

原因不是它做错了，而是它解决的问题太窄：

$$
\text{axis selection}
$$

还没有到：

$$
\text{pair selection}
$$

更没有到：

$$
\text{adaptive pair utility}
$$

### 关于你新增的实际数据

这部分我认为**非常重要，而且应该成为下一阶段主验证对象**。

目前仓库已经确认存在 `obj_2mm.tif`、`AVG_red-1100.tif`、`AVG_green-1300.tif` 以及对应结果图；README 也明确记录了深层弱信号和区域性高背景噪声这一真实应用需求。但当前 GitHub connector 无法直接把这些二进制 TIFF/PNG/JPG 的像素内容送出来，因此我这里只能基于仓库中的元数据、说明和已有数值产物判断，**不会假装已经完成了对 `obj_result.png`/`AVG_result.jpg` 的视觉逐像素审图**。

---

## 我建议下一步直接进入 A3 正式设计

下一阶段不要再继续扩 qmask 的 threshold sweep 了，而是正式开始设计：

> **A3：Pair Utility Field + Signal-Aware Adaptive Sampling**

具体应一次性定下来：

1. candidate offset space；
2. Noise / Structure / Shortcut / Optimization 四类指标；
3. micro-probe 的具体训练方式；
4. pair utility 的数学定义；
5. soft sampling policy；
6. depth/SNR-conditioned routing；
7. 与 QMask/VALID/SN2N/FAST 的严格统一实验协议；
8. 针对你 `ACES/example` 的 **25-frame independent split + deep SNR/CNR/SBR + neuron detectability** 验证体系；
9. 以及直接映射到现有 `ACES/qmask` 代码的文件级重构方案。

**这一步做完以后，QMask 就可以正式降级为 A3 的 baseline/ablation，而不是继续把它当主创新硬往前推。**

---

**Sources:**

- [Reinforcing neuron extraction and spike inference in calcium imaging using deep self-supervised denoising | Nature Methods](https://www.nature.com/articles/s41592-021-01225-0?utm_source=chatgpt.com)
- [Enhancing biomedical optical volumetric imaging via self-supervised orthogonal learning - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC13082324/?utm_source=chatgpt.com)

## Prompt:
2026/9/22 18:04:07

我认为你的分析非常中肯。帮我详细梳理一下下一阶段需要修改完善的地方，形成一个完整的项目修正方案（不光要包含方法修正细节如公式等，还要说明当前修改后的项目主线等信息）。先以回复的方式展示给我，我确定之后再让你生成md文档

## Response:
2026/9/22 18:04:34

可以。基于这次对仓库现状、QMask 实现、已有实验记录，以及你补充的真实 2P/3P 数据应用目标的审阅，我建议不要继续在现有 QMask 上做零散修补，而是把下一阶段定义成一次**方法主线级别的重构**。

我先把它整理成一个完整的“项目修正方案 v1”。这里先不给你生成 MD，而是把研究主线、方法、代码、实验和评价体系全部讲清楚。等你确认后，再把它整理成正式项目文档。

---

# 一、首先重新确定项目的主线

这是整个修正最重要的一步。

## 1.1 原来的主线

最初 ACES 的核心问题是：

> 针对 2P/3P 各向异性体数据，从数据中估计不同方向的 pairability，再据此设计自监督采样几何。

形式上：

$$
\text{anisotropy measurement}
\rightarrow
N,S,Q
\rightarrow
\text{sampling radius / ellipsoid}
\rightarrow
\text{N2N training}
$$

后来发现：

$$
Q(d)
$$

并不能直接告诉我们：

$$
\text{training utility}(d)
$$

于是原来的 radius/ellipsoid 被证伪。

QMask 是对此的第一次修正：

$$
\text{axis-wise pairability}
\rightarrow
\text{axis participation}
\rightarrow
\text{phase split}
$$

这个修正是正确的，但它只解决了：

> **哪些轴参与。**

还没有解决真正更一般的问题：

> **哪些 observation pair 值得被拿来训练。**

---

# 二、下一阶段正式把研究问题升级

我建议把整个项目从：

> **Anisotropy-Calibrated Empirical Sampling**

升级为更一般的：

> **Data-Calibrated Adaptive Pair Sampling for Self-Supervised Volumetric Denoising**

中文可以暂定：

> **面向自监督体成像去噪的数据校准自适应配对采样方法**

最终并不一定要马上定死英文缩写。

---

# 三、重新定义科学问题

下一阶段真正要回答的不是：

> 哪个方向可以采样？

而是：

> **对于给定的无 GT 体成像数据，什么样的 observation pair 最适合用于自监督去噪？这种 pair 是否会随空间方向、距离、深度和局部信号状态而变化？**

于是整个问题可以写成：

$$
\boxed{
\Delta^*
=
\arg\max_{\Delta}
U(\Delta\mid x)
}
$$

其中：

- $\Delta$：两个 observation 之间的 offset；
- $U$：这个 pair 对自监督训练的 utility。

这样，QMask 就变成这个框架中的一个特殊情况：

$$
\text{QMask}
\approx
\text{binary axis-level policy}
$$

而不是最终方法。

---

# 四、最终方法主线，我建议确定为五层

以后所有代码、实验、论文内容都围绕下面这条线组织：

```text
Raw noisy volumetric data
        │
        ▼
1. Observation redundancy profiling
        │
        ▼
2. Pair-risk estimation
        │
        ├── Noise dependence
        ├── Structure mismatch
        ├── Identity shortcut
        └── local uncertainty
        │
        ▼
3. Pair-utility calibration
        │
        └── short self-supervised micro-probe
        │
        ▼
4. Adaptive sampling policy
        │
        ├── spatial anisotropy
        ├── depth condition
        └── signal/noise condition
        │
        ▼
5. Self-supervised denoising
        │
        ▼
Deep signal recovery
        │
        ├── PSNR / SSIM
        ├── SNR
        ├── CNR
        ├── SBR
        ├── signal retention
        └── neuron detectability
```

这是以后项目的总架构。

---

# 五、第一部分修正：彻底废弃“采样半径 = 方法核心”的思路

## 5.1 legacy ellipsoid / empirical_block

这两个东西现在可以保留，但定位必须改变。

不再作为：

> 当前主方法

而作为：

> **failed baseline / negative result**

即：

```text
legacy radius
legacy ellipsoid
QMask
adaptive pair policy
```

以后论文里甚至可以形成：

> Fixed geometric assumptions can be systematically suboptimal for anisotropic volumetric self-supervised denoising.

这会成为方法动机。

---

# 六、第二部分修正：QMask 不删除，而是降级为 baseline / ablation

QMask 很值得保留。

因为它已经告诉我们：

$$
\text{axis-level adaptation}
>
\text{naive large-offset ellipsoid}
$$

但下一步应该把它重新定义成：

### QMask = Hard Axis Policy

$$
m_a\in\{0,1\}
$$

例如：

$$
m=(1,1,0)
$$

或者：

$$
m=(1,1,1)
$$

下一阶段变成：

### Adaptive Pair Policy

$$
\pi(\Delta|x)
$$

所以方法逻辑就变成：

$$
\boxed{
\text{Hard axis mask}
\rightarrow
\text{pair utility field}
\rightarrow
\text{adaptive sampling policy}
}
$$

---

# 七、第三部分修正：从 XYZ 三个轴升级为完整 offset space

这是下一阶段最核心的代码和理论变化。

当前：

$$
x/y/z
$$

每个轴只看：

$$
d=1,2,\ldots
$$

然后最后压缩成一个 Boolean。

下一阶段不再把轴作为最终 decision unit。

定义：

$$
\Delta=(\Delta_x,\Delta_y,\Delta_z)
$$

如果以后考虑 dynamic imaging，则自然扩展：

$$
\Delta=(\Delta_x,\Delta_y,\Delta_z,\Delta_t)
$$

对于 2P/3P 静态 volumetric imaging，第一版先只实现：

$$
(\Delta_x,\Delta_y,\Delta_z)
$$

避免一次改得太大。

---

# 八、候选 offset 集怎么设计

第一版不建议把整个三维空间全部扫完。

建立一个 controlled candidate set：

$$
\mathcal D=
\{
\Delta_1,\Delta_2,\ldots,\Delta_K
\}
$$

建议包括：

### 单轴

$$
(1,0,0),(2,0,0),(4,0,0),(8,0,0)
$$

$$
(0,1,0),(0,2,0),(0,4,0),(0,8,0)
$$

$$
(0,0,1),(0,0,2),(0,0,4),(0,0,8)
$$

### 对角

$$
(1,1,0),(2,2,0),(4,4,0)
$$

$$
(1,0,1),(2,0,2)
$$

$$
(0,1,1),(0,2,2)
$$

以及少量 3D offsets：

$$
(1,1,1),(2,2,1),(2,2,2)
$$

这一阶段的目的不是无限搜索，而是建立一个：

> **candidate pair library**

---

# 九、第四部分修正：不再使用单一 Q=N×S 作为最终判据

这是方法理论上的重大修改。

当前：

$$
Q=N\times S
$$

这个设计的问题在于：

$$
Q
$$

把性质不同的风险压缩成了一个 scalar。

下一阶段改成：

$$
\mathbf R(\Delta)
=
[
R_N(\Delta),
R_S(\Delta),
R_I(\Delta)
]
$$

至少三个维度。

---

# 十、Noise Risk：噪声独立性

当前：

$$
N(\Delta)
=
1-|\mathrm{corr}(n_i,n_{i+\Delta})|
$$

可以保留作为第一版指标。

但要注意：

> 它只是一个 independence proxy，而不是 N2N correctness 的完整证明。

因此建议定义：

$$
R_N(\Delta)
=
|\rho_n(\Delta)|
$$

这样：

$$
R_N\downarrow
\Rightarrow
\text{noise dependence decreases}
$$

同时保留：

- covariance；
- residual correlation；
- variance ratio。

在 repeat 数据中进一步直接估计：

$$
\mathrm{Cov}
\left[
n(\mathbf r),
n(\mathbf r+\Delta)
\right]
$$

这会比现在的单一 Pearson 更有说服力。

---

# 十一、Structure Risk：结构错配风险

当前：

$$
S(\Delta)
=
\mathrm{corr}(s_i,s_{i+\Delta})
$$

这个也不应该完全删除，但需要升级。

建议：

$$
R_S(\Delta)
=
\alpha(1-\rho_{LF})
+
\beta(1-\mathrm{SSIM}_{local})
+
\gamma(1-\rho_{\nabla})
$$

其中：

- $\rho_{LF}$：低频结构相关性；
- SSIM：局部结构一致性；
- $\rho_\nabla$：边缘/梯度一致性。

原因很简单：

两个位置强度相关，并不一定意味着 neuron morphology 一致。

---

# 十二、Shortcut Risk：这是当前实验暴露出来、值得正式纳入的方法组成部分

这是这次项目推进中一个非常好的发现。

pair 太相似：

$$
X_t\approx X_{t+\Delta}
$$

模型很容易学习：

$$
f(X)\approx X
$$

而不是：

$$
f(X)\approx E[S|X]
$$

因此定义：

$$
R_I(\Delta)
=
\mathrm{sim}(X,X_\Delta)
$$

或者使用：

$$
R_I(\Delta)
=
1-
\frac{
\|X-X_\Delta\|^2
}{
\mathrm{Var}(X)
}
$$

这里需要注意方向定义，最终实现时统一成：

> risk 越大 = 越危险。

比如：

$$
R_I(\Delta)
=
\exp(-k\cdot nMSE(\Delta))
$$

这样：

- offset 太小 → shortcut risk 高；
- offset 太大 → structure risk 高；
- 中间存在适合区间。

这正是下一阶段最重要的机制假设之一。

---

# 十三、第五部分：加入“Pair Utility”

这是整个 A3 真正区别于 QMask 的关键。

仅仅知道：

$$
R_N,R_S,R_I
$$

还不够，因为你现在已经有实验说明：

$$
\text{static pair similarity}
\neq
\text{training utility}
$$

因此定义：

$$
U(\Delta)
$$

---

# 十四、Pair Utility 不直接学习一个大型网络，先采用 micro-probe

这是为了让方法可控、可解释。

对于每个 candidate $\Delta$：

```text
candidate offset
      ↓
构造 self-supervised pairs
      ↓
跑 1-3 epoch
      ↓
记录 training-only signals
      ↓
estimate utility
```

不使用 blind test。

---

# 十五、Micro-probe 记录什么

至少记录：

### 1. Neighbor loss

$$
L_N(\Delta)
$$

---

### 2. Identity loss

$$
L_I(\Delta)
$$

---

### 3. Residual statistics

例如：

$$
R_{out}
=
\mathrm{corr}
(
X-\hat X,
X_\Delta-\hat X_\Delta
)
$$

---

### 4. Gradient preservation

$$
G(\Delta)
=
\frac{
\|\nabla \hat X\|
}{
\|\nabla X\|
}
$$

---

### 5. Output-input similarity

用于监测：

> 模型是不是只学 identity。

---

# 十六、最终定义 Pair Utility

第一版可以采用线性形式：

$$
U(\Delta)
=
-\lambda_N R_N(\Delta)
-\lambda_S R_S(\Delta)
-\lambda_I R_I(\Delta)
-\lambda_L L_N(\Delta)
+\lambda_G G(\Delta)
$$

更建议最终改成归一化形式：

$$
\tilde U(\Delta)
=
w^\top
\phi(\Delta)
$$

其中：

$$
\phi(\Delta)
=
[
R_N,
R_S,
R_I,
L_N,
L_I,
G
]
$$

这样以后可以从手工权重升级成 regression / ranking model。

---

# 十七、第一版先不要直接做 RL 或复杂 policy network

这点我建议强行控制住。

当前项目最容易犯的错误就是：

> 一发现 binary mask 太简单，就直接上 reinforcement learning / transformer policy / meta-learning。

我不建议。

因为这样会一下引入：

- policy optimization；
- second-order gradient；
- instability；
- computational cost；
- reviewer 难解释。

第一版应该采用：

$$
U(\Delta)
\rightarrow
\text{softmax}
$$

即：

$$
\pi(\Delta|x)
=
\frac{
\exp(U(\Delta)/T)
}{
\sum_{\Delta'}
\exp(U(\Delta')/T)
}
$$

已经足够形成真正的 adaptive sampling。

---

# 十八、进一步引入 depth-conditioned policy

这一步是为了真正贴合你的实际目标。

定义：

$$
\pi(\Delta|z,\rho)
$$

其中：

- $z$：depth；
- $\rho$：local signal quality / noise level。

于是：

$$
\pi(\Delta|z,\rho)
=
\mathrm{Softmax}
\left(
U(\Delta|z,\rho)/T
\right)
$$

---

# 十九、为什么这一点特别适合你的真实数据

你的实际数据明确表现出：

> signal decreases with depth.

因此：

浅层：

$$
SNR_{shallow}\gg SNR_{deep}
$$

深层：

$$
SNR_{deep}\downarrow
$$

那么最优 pair 很可能不是全体数据恒定的。

例如可能出现：

```text
浅层
    Δz 可以适当增加
    → 更丰富的独立信息

深层
    Δz 必须更加保守
    → 优先结构稳定 pair
```

这才真正把：

> **2P/3P anisotropy**

和：

> **deep signal recovery**

结合起来。

---

# 二十、最终 sampling 不一定是“每次只选一个 offset”

这是另一个重要设计。

可以允许：

$$
\pi(\Delta)
$$

每次训练从多个高 utility pair 中随机采样：

$$
\Delta\sim\pi
$$

这会产生两个好处：

1. 不会过度依赖单一 offset；
2. policy 本身变成一个 distribution，而不是 heuristic。

最终方法的形式就很漂亮：

$$
\boxed{
\Delta\sim
\pi(\Delta|z,\rho)
}
$$

---

# 二十一、一个更进一步的版本：training-stage adaptive policy

这是后期可以做的，不作为第一版主干。

训练过程中：

$$
\pi_0(\Delta)
\rightarrow
\pi_1(\Delta)
\rightarrow
\pi_2(\Delta)
$$

根据模型自身的 training statistics 更新。

例如：

```text
early training
→ emphasize easier / reliable pairs

later training
→ introduce informative harder pairs
```

这可以叫：

> curriculum pair sampling

但我建议放到第二阶段，暂时不要同时实现。

---

# 二十二、你之前提出的 3D Wiener，现在如何处理

我仍然认为：

> **3D Wiener 很值得做，但绝对不要作为主创新。**

它的角色应该是：

> classical frequency-domain reference。

基本形式：

$$
H(f_x,f_y,f_z)
=
\frac{S(f_x,f_y,f_z)}
{S(f_x,f_y,f_z)+N(f_x,f_y,f_z)}
$$

对你来说最重要的不是“Wiener 比 deep learning 好不好”。

而是分析：

```text
raw PSD
      ↓
Wiener PSD
      ↓
network PSD
      ↓
reference PSD
```

特别比较：

- 高频能量保留；
- z 方向频率响应；
- deep region signal attenuation；
- noise suppression。

这会帮助回答：

> 新方法究竟是在“正确恢复结构”，还是仅仅在强 smoothing。

因此 3D Wiener 应该作为：

### Frequency-domain sanity baseline

而不是主方法。

---

# 二十三、真实数据部分需要重新设计为核心验证，而不是 example

这是我认为此次项目需要最大的应用侧升级。

现在：

```text
ACES/example
├── obj_2mm
├── AVG_red
├── AVG_green
└── result image
```

以后应变成：

> **Real-world application benchmark**

---

# 二十四、25-frame AVG 要重新定义角色

你现在有：

$$
AVG_{25}
$$

这个非常宝贵。

但它不应该只作为“好看的参考图”。

应该构造：

$$
A=\text{odd frames}
$$

$$
B=\text{even frames}
$$

得到：

$$
X_A,\quad X_B
$$

然后：

$$
X_A\rightarrow f\rightarrow\hat X_A
$$

$$
X_B\rightarrow f\rightarrow\hat X_B
$$

而：

$$
AVG_{25}
$$

作为 high-quality proxy。

如果存在帧间运动，则先进行 registration。

这样就形成：

> **真实数据的 independent noisy pair 实验。**

这非常重要。

---

# 二十五、真实数据最终应该同时回答三个问题

## 问题 A：有没有真正去噪？

看：

$$
PSNR,\ SSIM
$$

---

## 问题 B：有没有提升 deep signal quality？

看：

$$
SNR(z)
$$

$$
CNR(z)
$$

$$
SBR(z)
$$

---

## 问题 C：有没有提升 biological detectability？

看：

$$
\text{neuron detection rate}(z)
$$

$$
\text{deepest detectable depth}
$$

以及：

$$
\text{neurite continuity}
$$

这才是最终应用价值。

---

# 二十六、SNR / CNR / SBR 需要重新定义

这一部分建议在代码中统一。

## SNR

不要简单定义成：

$$
20\log_{10}
\frac{\mu_{fg}}{\sigma_{bg}}
$$

建议至少报告：

$$
SNR
=
\frac{\mu_{signal}-\mu_{bg}}
{\sigma_{bg}}
$$

如果需要 dB：

$$
SNR_{dB}
=
20\log_{10}
\left(
\frac{
\mu_{signal}-\mu_{bg}
}{
\sigma_{bg}
}
\right)
$$

---

## CNR

建议作为核心：

$$
CNR
=
\frac{
|\mu_{signal}-\mu_{bg}|
}{
\sqrt{
\sigma_{signal}^2+\sigma_{bg}^2
}
}
$$

或者采用你最终文献体系中统一的一种标准定义。

---

## SBR

采用：

$$
SBR
=
\frac{\mu_{signal}}{\mu_{bg}}
$$

如果转成 dB：

$$
SBR_{dB}
=
10\log_{10}(SBR)
$$

不要再使用当前代码中的：

$$
20\log
$$

形式。

---

# 二十七、真正重要的是做 depth-dependent curves

最终最关键的结果不是一个平均值。

应该：

$$
SNR=f(z)
$$

$$
CNR=f(z)
$$

$$
SBR=f(z)
$$

例如：

```text
Depth
│
├── Raw
├── AVG25
├── VALID
├── SN2N
├── FAST
└── Proposed
```

然后观察：

$$
\Delta SNR(z)
=
SNR_{proposed}(z)-SNR_{raw}(z)
$$

尤其关注：

$$
z>z_{critical}
$$

区域。

---

# 二十八、“深层增强”不能只看 intensity

必须额外加入：

## Signal retention

$$
SR
=
\frac{\mu_{denoised}}
{\mu_{reference}}
$$

以及 morphology retention，例如：

$$
MR
=
\frac{
\|\nabla\hat X\|
}{
\|\nabla X_{ref}\|
}
$$

这样可以防止 reviewer 质疑：

> 你是不是把整个图变亮了？

---

# 二十九、建议增加 neuron-level evaluation

这会显著提升项目的完整性。

例如：

$$
Recall_{neuron}
$$

$$
Precision_{neuron}
$$

$$
F1_{neuron}
$$

和：

$$
Recall_{deep}
$$

重点不是构建一个极复杂的 neuron detector，而是建立一个一致的 detection pipeline。

真正想证明的是：

$$
\text{denoising}
\rightarrow
\text{better detectability}
$$

而不是：

$$
\text{denoising}
\rightarrow
\text{better PSNR only}
$$

---

# 三十、数据集重新划分

这是后续必须严格执行的。

建议明确分成：

```text
Calibration
Validation
Blind Test
```

---

## Calibration

用于：

- pairability；
- candidate offsets；
- risk model；
- micro-probe；
- policy construction。

---

## Validation

用于：

- threshold；
- temperature；
- stopping；
- model selection；
- ablation。

---

## Blind Test

只在最后一次使用。

用于：

- PSNR;
- SSIM;
- SNR;
- CNR;
- SBR;
- deep signal;
- neuron detection。

---

# 三十一、对于 9-repeat 数据，我建议把它升级成“method development laboratory”

你现有：

$$
9\ repeat
$$

非常珍贵。

不要仅仅拿来做一个 Stage 1 N。

应该允许：

$$
\text{repeat covariance}
$$

直接估计：

$$
N_{true}(\Delta)
$$

甚至：

$$
\Sigma_n(\Delta_x,\Delta_y,\Delta_z)
$$

这样可以用于验证：

> proxy risk → true risk

这会成为你方法可靠性的一个重要实验。

---

# 三十二、Neurofinder 的角色重新定位

以后不要再把：

$$
[T,Y,X]
$$

中的 T 称为 Z。

统一定义：

$$
(t,y,x)
$$

即：

> temporal stack

这样可以保留 Neurofinder，但它的作用变成：

### temporal-domain mechanism validation

而不是：

### physical-depth validation

这会让整个论文概念清楚很多。

---

# 三十三、代码架构也应该重构

我建议不要在现在的 `qmask/` 里继续无限加脚本。

最终组织成：

```text
ACES/
│
├── profiling/
│   ├── redundancy.py
│   ├── noise_risk.py
│   ├── structure_risk.py
│   └── shortcut_risk.py
│
├── calibration/
│   ├── candidate_offsets.py
│   ├── micro_probe.py
│   ├── utility.py
│   └── policy_calibration.py
│
├── sampling/
│   ├── fixed.py
│   ├── qmask.py
│   ├── adaptive.py
│   └── policy_sampler.py
│
├── models/
│   └── ...
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
│   └── neuron_metrics.py
│
├── frequency/
│   └── wiener3d.py
│
├── real_data/
│   ├── preprocessing.py
│   ├── frame_split.py
│   └── registration.py
│
└── legacy/
    ├── ellipsoid/
    └── empirical_block/
```

---

# 三十四、现有 QMask 代码如何处理

建议不是删除，而是冻结。

### 保留

```text
qmask/geometry.py
qmask/sampling.py
qmask/train.py
```

作为：

> `baseline_qmask`

---

### 不再继续扩展

例如：

```text
tau_sweep
geometry_sweep
各种 threshold trick
```

这些可以保留作实验历史，但不再作为研究主线。

---

# 三十五、当前几个明确的代码修复必须先做

这些属于 P0，不等新的 adaptive method。

### P0-1

`d_min/d_max` 必须真正进入采样逻辑。

如果最终方法不再需要 d_min/d_max，也要：

> 删除它们在主方法中的伪存在。

不能再出现：

```text
computed
but unused
```

---

### P0-2

修复：

```text
use_n_decision=False
```

与实际 Q gate 的逻辑矛盾。

最终应明确：

$$
R_N
$$

和：

$$
R_S
$$

分别存在。

---

### P0-3

Neurofinder 全部变量统一：

```text
time
```

不要再写：

```text
z
depth
```

---

### P0-4

所有 checkpoint 选择必须来自 Validation。

---

### P0-5

正式 test 禁止参与：

- mask construction；
- offset selection；
- threshold selection；
- early stopping。

---

# 三十六、评价代码必须重新整理

当前 `evaluate_stage2.py` 有两个值得尽快处理的问题。

### 问题 1

SSIM 当前是 full patch SSIM，然后在不同 region 行重复。

因此：

```text
all SSIM
foreground SSIM
background SSIM
```

不能当作独立 region metrics。

---

### 问题 2

当前：

```text
best checkpoint
+
final checkpoint
+
aligned
+
full
```

协议比较复杂，容易造成结果解释混乱。

以后正式 benchmark 只保留：

> **Validation-selected checkpoint + predefined test protocol**

---

# 三十七、最终主实验应该分成四组

这会是论文最清晰的实验框架。

## Experiment 1：Why fixed sampling fails

比较：

```text
VALID
SN2N-like
Ellipsoid
QMask
```

目的：

证明固定/简单几何不能稳定适配各向异性数据。

---

## Experiment 2：Pair utility calibration

在 9-repeat / validation 数据上：

$$
\Delta
\rightarrow
R_N,R_S,R_I,U
$$

比较：

$$
U(\Delta)
$$

和真实训练效果。

这里要回答：

> pair utility 是否比简单 pairability 更准确。

这是方法理论核心。

---

## Experiment 3：Adaptive sampling

比较：

```text
fixed
QMask
adaptive-hard
adaptive-soft
depth-conditioned adaptive
```

真正证明：

$$
\pi(\Delta|z,\rho)
$$

有效。

---

## Experiment 4：Real 2P/3P deep-signal recovery

最后回到：

```text
你的真实数据
```

比较：

- Raw；
- AVG25；
- existing methods；
- fixed baseline；
- QMask；
- proposed adaptive。

指标：

$$
PSNR
$$

$$
SSIM
$$

$$
SNR(z)
$$

$$
CNR(z)
$$

$$
SBR(z)
$$

$$
neuron\ detectability(z)
$$

---

# 三十八、最关键的 ablation matrix

我建议最终形成这样一张矩阵：

| Sampling | Risk | Utility | Depth-conditioned | 作用 |
|---|---:|---:|---:|---|
| VALID | × | × | × | 基础 baseline |
| QMask | ✓ | × | × | axis-level adaptation |
| Risk-hard | ✓ | × | × | 验证 risk |
| Utility-hard | ✓ | ✓ | × | 验证 utility |
| Utility-soft | ✓ | ✓ | × | 验证 probabilistic policy |
| Depth-adaptive | ✓ | ✓ | ✓ | 最终方法 |
| Depth+signal adaptive | ✓ | ✓ | ✓ | 完整版本 |

这个实验矩阵会非常漂亮。

---

# 三十九、建议增加一个非常重要的指标：Oracle Gap

这是我特别推荐你加入的。

定义：

$$
\Delta^*
=
\arg\max_{\Delta}
PSNR_{val}(\Delta)
$$

但模型不知道它。

预测得到：

$$
\hat\Delta
$$

那么：

$$
G_{oracle}
=
PSNR(\Delta^*)-
PSNR(\hat\Delta)
$$

理想情况：

$$
G_{oracle}\rightarrow0
$$

这会直接回答：

> 你的 data-driven policy 到底能不能接近最优 sampling。

这比单独报告“我的模型比 baseline 高 0.5 dB”有更强的方法学意义。

---

# 四十、再进一步，可以定义“pair-policy consistency”

不同数据：

$$
D_1,D_2,\ldots,D_n
$$

分别得到：

$$
\pi_1(\Delta),\pi_2(\Delta),\ldots
$$

研究：

- 2P；
- 3P；
- 不同深度；
- 不同 SNR；
- 不同组织区域。

是否出现有规律的 policy variation。

这样最终论文不仅是：

> “我的模型效果更好”

而是：

> **We characterize how the optimal self-supervised pair distribution changes with imaging anisotropy and signal quality.**

这会让文章的 scientific depth 明显提升。

---

# 四十一、真实数据最终最有价值的一张图，我已经比较明确了

建议最终核心 figure 之一：

```text
                         Depth
                         ───────────────►

Raw             ────────╲
VALID                    ╲
QMask                     ╲
Ours                       ╲────
                              ↑
                       deep signal preserved
```

实际不是画成这样，而是：

### Panel A

三维重建：

```text
Raw
VALID
QMask
Ours
```

### Panel B

$$
SNR(z)
$$

### Panel C

$$
CNR(z)
$$

### Panel D

$$
SBR(z)
$$

### Panel E

neuron detection rate vs depth

### Panel F

deepest reliable detection depth

这比“PSNR 表格”更接近你真正想解决的问题。

---

# 四十二、另外建议新增“deep-region loss”

这里需要谨慎。

我不建议简单：

$$
L_{deep}
$$

直接把深层像素权重大幅提高。

这样可能导致模型牺牲浅层。

更合理的是先定义一个局部 signal-quality weighting：

$$
w(z,\rho)
$$

然后：

$$
L
=
\sum_i
w_i
L_i
$$

但这个应该放在后续阶段。

第一阶段必须先回答：

> sampling policy 本身是否能提高深层 signal quality。

否则 sampler 和 loss 同时改，就无法知道收益来自哪里。

---

# 四十三、项目开发顺序必须严格分阶段

我建议不要现在立即进入“adaptive sampler 全部实现”。

按照下面顺序推进。

## Phase 0：代码纠偏

修复：

- Z/T 语义；
- evaluation；
- checkpoint；
- QMask N/Q 逻辑；
- legacy code 隔离。

---

## Phase 1：建立 Pair Utility Laboratory

实现：

$$
\Delta
\rightarrow
R_N,R_S,R_I
$$

然后：

$$
\Delta
\rightarrow
micro-probe
\rightarrow
U(\Delta)
$$

这一阶段先不做 adaptive sampling。

最终目标：

> 证明 utility field 存在，且具有预测能力。

---

## Phase 2：Hard adaptive sampling

直接：

$$
\Delta^*
=
\arg\max U(\Delta)
$$

得到一个 deterministic sampler。

这是最容易 debug 的版本。

---

## Phase 3：Soft adaptive sampling

变成：

$$
\Delta\sim\pi(\Delta)
$$

验证是否比 single best offset 更稳定。

---

## Phase 4：Depth/signal-conditioned sampling

变成：

$$
\pi(\Delta|z,\rho)
$$

这一步开始直接服务你的深层神经元目标。

---

## Phase 5：真实数据 benchmark

开始：

- 25-frame split；
- real 2P/3P；
- depth SNR/CNR/SBR；
- neuron detection。

---

# 四十四、每个阶段应该设“继续/停止”标准

这个很重要，可以避免项目再一次陷入大量无效训练。

## Phase 1 Gate

只有满足：

$$
corr(U,PSNR_{val})
$$

明显高于：

$$
corr(Q,PSNR_{val})
$$

才继续。

如果没有：

> 重新定义 utility，而不是直接做 adaptive sampler。

---

## Phase 2 Gate

预测 offset 的：

$$
G_{oracle}
$$

如果很大：

> 不进入 soft policy。

---

## Phase 3 Gate

soft policy 如果不能降低 seed variance：

> 暂时不继续复杂化。

---

## Phase 4 Gate

必须看到：

$$
\Delta CNR_{deep}>0
$$

或：

$$
\Delta SNR_{deep}>0
$$

否则不能宣称 depth-aware policy 成功。

---

# 四十五、关于论文创新性，我建议从现在开始统一使用三个层次

最终方法不是一个“新 sampler”而已。

而是：

### Contribution 1
**Pair-risk characterization**

提出：

$$
R_N,R_S,R_I
$$

描述自监督 pair 的不同风险。

---

### Contribution 2
**Pair-utility calibration**

发现：

$$
\text{pair similarity}
\neq
\text{training utility}
$$

并通过无 GT micro-probe 估计实际 utility。

---

### Contribution 3
**Adaptive sampling policy**

利用：

$$
\pi(\Delta|z,\rho)
$$

自动适配空间各向异性及深层低 SNR 条件。

---

# 四十六、最终的方法学创新点，我建议这样定调

不是：

> “We propose a better Tetris sampling pattern.”

而是：

> **We propose a data-calibrated pair sampling framework that estimates the reliability and training utility of candidate observation pairs and adaptively selects self-supervised training partners according to spatial anisotropy and local signal quality.**

这才是我认为目前最有机会拉开与 VALID / SN2N / FAST 差距的方向。

---

# 四十七、当前 QMask 的最终定位，我建议正式确定为

```text
Original ACES
     │
     ├── empirical radius
     │        ↓
     │      failed
     │
     ├── ellipsoid
     │        ↓
     │      failed
     │
     └── QMask
              ↓
       successful correction
              ↓
      axis-level adaptation
              ↓
      still insufficient
              ↓
       Pair-Utility Policy
              ↓
       final method
```

这样过去这些失败实验都不会浪费。

它们全部成为最终论文的 method evolution / motivation evidence。

---

# 四十八、我特别建议增加一个“Method Map”

以后项目文档顶部直接放：

```text
Research Question
      ↓
Which noisy observations are valid training partners?
      ↓
Are validity and utility the same?
      ↓
No
      ↓
Separate
Noise Risk
Structure Risk
Shortcut Risk
      ↓
Estimate Pair Utility
      ↓
Construct Adaptive Policy
      ↓
Condition on Depth / Signal Quality
      ↓
Improve Deep Neural Signal Detectability
```

这个图会把整个项目一下变得非常清楚。

---

# 四十九、最终完整研究主线可以浓缩成这一句话

> **本项目不是单纯研究“如何设计一个新的三维采样模板”，而是研究在没有 clean ground truth 的条件下，如何从各向异性体成像数据中自动判断“哪些观测应该彼此配对用于自监督去噪”，并进一步使这种配对策略随空间方向、成像深度和局部信号质量自适应，从而在抑制噪声的同时尽可能保留和恢复深层弱神经元信号。**

我认为这句话现在已经可以作为整个项目的**总研究目标**。

---

# 五十、我对下一阶段具体修改优先级的最终排序

### P0：必须先改

**代码与实验协议**

- 修复 Z/T 语义；
- 修复 evaluation；
- 修复 checkpoint / validation/test 隔离；
- 修复 QMask `N/Q` 逻辑；
- 清理 legacy geometry；
- 统一 SNR/CNR/SBR 定义。

### P1：真正开始新方法

**Pair-risk field**

$$
\Delta
\rightarrow
(R_N,R_S,R_I)
$$

### P2：方法核心

**Micro-probe**

$$
\Delta
\rightarrow
U(\Delta)
$$

### P3：建立 adaptive sampler

$$
\Delta^*=\arg\max U
$$

再：

$$
\Delta\sim\pi
$$

### P4：与你真正的应用目标结合

$$
\pi(\Delta|z,\rho)
$$

### P5：真实 2P/3P 验证

$$
SNR(z),CNR(z),SBR(z),\text{neuron detectability}(z)
$$

### P6：高级扩展

- online policy update；
- curriculum pair sampling；
- 4D $(x,y,z,t)$；
- frequency-domain analysis；
- 3D Wiener reference。

---

## 最后一个非常重要的判断

我建议你现在**不要继续大量跑 QMask 的 threshold / seed / epoch 实验**了。

现有结果已经足够支持它作为：

> “从 radius/ellipsoid 到 axis-level adaptation 的过渡方法”。

继续堆 QMask 数字，科研收益会越来越低。

下一步最值得投入 GPU 时间的其实是：

$$
\boxed{
\text{Candidate Offset}
\rightarrow
\text{Risk}
\rightarrow
\text{Micro-probe Utility}
}
$$

只要这个闭环能够成立，整个项目的研究层级就真正往前迈了一步。

而你最新加入的 `ACES/example` 实际 25-frame 数据，应当从这一阶段开始就进入设计中心：**最终的成功标准不应只是“去噪后 PSNR 更高”，而应该是“深度越深，原本不可见的神经元信号仍能被更可靠地恢复和区分出来”。**

下一步我建议直接把上面这套方案进一步落到**“A3 v1 技术实施规格”**：包括每个公式的最终定义、candidate offset 的具体集合、micro-probe 的训练协议、utility 计算方式、adaptive sampler 的伪代码、数据目录结构，以及逐文件对应到现有 `ACES/qmask` 哪些代码该删除、哪些重构、哪些新增。



---
Powered by [ChatGPT Exporter](https://www.chatgptexporter.com)