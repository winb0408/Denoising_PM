# ACES-QMask 技术思路（复核实验）

## 1. 问题：为什么原“椭球采样”在神经finder 上失败

对照 `ACES_Final_Executable_Technical_Plan.md` §3 与实测结果，原方案主链路有三处硬伤：

1. **几何提取取反了**。原方案/代码用 `r_a = max{d : Q_a(d) > tau}`（`ACES/aces/pairability.py`），在 Q 随距离单调衰减时，这恰是选“最远仍勉强合法”的配对——结构已经被平移破坏得最严重。神经finder 的 `Q(d)` 在 d=1..8 始终 ≥0.6，于是 `x=y=z=8`，退化成一个近各向同性的 8×8×8 块，完全失去各向异性校准意义。
2. **“半径”这个对象本身错位**。N2N 要求“同一内容的两份独立噪声观测”。3P Stage1 中 `N` 在 d=1 已≈1，各向异性几乎全在 `S` 上；z 轴 `S(d=1)=0.882` 已是 z 的最佳值，再加大 z 偏移只会更差。因此 z 的正确处理是“少配/不配 z”（趋近 SN2N 端点），而不是“给 z 更大半径”。
3. **采样语义与损失语义断裂**。VALID 的 4 视图是 2×2×2 共位网格的 4 个角相（同一宏体素、内容相同、只有 1px 相位与噪声不同）；原 `ellipsoid/empirical_block` 把它映射成大步长不同坐标切片（`origin → y+5`、`x+5 → z+2` 等），监督目标不再是同一内容。症状：训练 loss 最低、PSNR 最低，模型学成平滑/折中图。

## 2. QMask 修正：可用区间 + 轴参与掩码

### 2.1 几何提取
对每轴、每个间距 d，从 Stage1 `pairability_curves.csv` 读取 `N/S/Q`：

- `d_min = min{d : N_a(d) >= tau_n}`：噪声恰好解相关的最短距离。
- `d_max = max{d : S_a(d) >= tau_s}`：结构仍匹配的最长距离。
- 可用区间 `[d_min, d_max]`。

v1 的相位分裂固定为 1-voxel 偏移，因此轴参与条件为：`d_min<=1<=d_max` 且 `N(d=1)>=tau_n`、`S(d=1)>=tau_s`、`Q(d=1)>=tau_q`。阈值默认 `tau_n=0.95, tau_s=0.85, tau_q=0.60`，全部写入 config 可改。

### 2.2 参与掩码 = VALID ↔ SN2N 的连续谱
- 参与轴：在该轴 2×2 相位内分裂，block=2。
- 关闭轴：phase 固定 0，block=1（该轴原样保留，不下采样）。
- 三轴全开 ≈ VALID；z 关、xy 开 ≈ SN2N；这是同一掩码的两个端点。
- 只用 ≥2 轴参与时 4 视图共位角集才有意义；若阈值导致 <2 轴参与，按 `S(d=1)` 从高到低补足到 2 轴（记录为 promotion），极端情况回退三轴全开。

### 2.3 阈值敏感性（运行前已复核，结果见 RESULTS.md）
- 3P repeat：`tau_s` 在 0.88~0.90 之间是边界。`tau_s=0.85 → zyx`（≈VALID）；`tau_s=0.90 → .yx`（z 关，≈SN2N）。
- 神经finder proxy：`tau_n` 在 0.90~0.95 之间是边界。`tau_n=0.90 → 全开`；`tau_n=0.95 → 全关后被 promotion 成 z.x`。

这说明两类数据在默认阈值下都恰好落在判决边界上：可用度场给出的“自适应几何”对阈值敏感、且与端点重合，这是需要诚实记录的局限。

## 3. 采样实现（为什么共位相位是对的）
VALID `space_to_depth(block_size=2)` 的 channel 顺序为 `z*4+y*2+x`。QMask 用每轴 block size `1` 或 `2`，在同一个 coarse 网格上做 phase split：

- 4 视图始终是同一宏体素网格的 4 个角相（三轴全开从 VALID 8 行 `idx_pair` 固定取一行，两轴参与用全部 4 个角）。
- 输出是 `Z//bz × Y//by × X//bx`，`(input, target)` 两两仍是同一内容的互补相位。
- 与 `aces.sampling_aniso.SamplerMetadata` 兼容，直接复用 ACES 的 stride 对齐评估。

## 4. 实验设计（隔离、可考究）
- 新建 `ACES/qmask/` 子包，不改动任何 `ACES/aces/*`、原配置、原 run、顶层 md。
- 只读复用：`aces.evaluate_stage2`、`aces.io_utils`、`aces.sampling_aniso`（控制档）、VALID 的 `Network_CNR / DWT_3D / HessianConstraintLoss3D / ReadDatasets`。
- 控制档 `valid / sn2n_xy / iso3d` 与 `qmask` 同一训练/评估协议，只差采样算子。
- 训练损失与 VALID 相同：`L2neighbor + L_idt + lambda_hessian*Hessian`（bg 权重 0）。
- 评估：all/foreground/background PSNR；对齐评估（依据各档 stride）；深度分箱前景 PSNR。

## 5. 门控与 H2 判定
Pilot 默认：3 epochs × 64 batches，每 epoch 评估 1 次、9 个前景丰富 patch、5 个深度分箱。

扩量门控（三项同时满足才进完整 80ep）：
1. qmask 定性图无塌缩（可视化脚本用动态范围自检）。
2. qmask 前景 PSNR 不低于两端点中较优者（`valid`/`sn2n_xy`），margin ≥ −0.25 dB 记为持平。
3. 无前景 patch 的 `pred_mean` 与 `target_mean` 偏差 >20%。

不预设 qmask 必须超过两端点；“QMask 自动识别 z 不应配对”本身是有效产出。

## 6. 假设与局限
- 3P 用 mean9 proxy 作 GT；神经finder 无真值，用整段 temporal mean 作 proxy；不做 CIDC25。
- v1 不做深度自适应、不做 pair rejection 单独消融（只保留采样器内部退化回退）。
- v1 相位偏移固定 d=1；若 `d_min > 1` 则该轴直接关闭，而不是扩大到 d=d_min 的 block 尺寸。
- `qmask` 的三轴全开与 `valid/iso3d` 并非逐字同一实现（角集选择/随机性不同），因此数值上允许小幅差异，但语义相同。
