# PLAN.md — ACES 项目方案文档（固定版）

> **文档定位**：本项目的**唯一现行方案文档**。日后方法/协议修正时直接修改本文档并在文末变更记录中登记，不再新开并行方案文件。
> **来源**：由 `ACES_Project_Correction_Execution_Plan_v1_20260922.md`（ChatGPT 生成）经 Claude 代码级审查修正后固定（2026-09-22）。审查对原方案的修正以 ⚠️ 标注。
> **配套**：实验流水见 `PROJECT_LOG.md`；验收状态见 `project_progress_status.md`。

---

## 1. 研究问题与主线

**问题升级**：原 ACES 问"三个方向各给多大采样半径"（`r_a = max{d: Q_a(d)>τ}`，已被实验证伪）。现问：

> **对于各向异性 2P/3P 体成像，在没有 clean ground truth 的情况下，能否从数据本身估计不同 observation pairs 的风险与训练效用，并据此自动形成随方向、深度、局部信号质量自适应的采样策略？**

主线一句话：

```
Data → Risk(R_N,R_S,R_I) → Utility(micro-probe) → Policy(π(Γ|q)) → Denoising → 深层信号 SNR/CNR/SBR 恢复
```

方法名（暂定，不锁死）：**Pair-Utility Calibrated Adaptive Sampling（配对效用校准自适应采样）**。

## 2. 三层创新

1. **Pair-risk 分解**：把单一 Q=N×S 分解为噪声依赖风险 R_N、结构错配风险 R_S、恒等捷径风险 R_I——三者是不同性质的问题，不可压缩成一个 scalar。Q 降级为 legacy 诊断。
2. **Pair utility 校准**：static validity ≠ optimization utility。用极低预算 micro-probe（固定全部训练变量，只换 sampling family）的统计量估计实际训练效用 U_probe，并用 development laboratory 的 oracle utility 验证其预测力。
3. **Signal-aware 自适应采样**：π(Γ|q)（q = 深度/局部信号/噪声/各向异性状态），目标是深层低信号区域的 SNR/CNR/SBR 与可检测性，不是平均 PSNR。

## 3. 核心对象定义

- **Candidate offset** Δ=(dz,dy,dx)，同时记录 voxel 与 μm 两套单位（spacing 来自 metadata；3P 的 dz=4 μm/组为推导值，论文前须再确认，不得包装成原始标定）。
- **Sampling family** Γ={δ1..δ4}：4 个视图 offset 映射到同一 coarse grid；family 风险 = pairwise offsets 的 mean 与 worst 两种聚合。这样统一 VALID / SN2N-like / QMask / 新 adaptive family。
- **风险**：
  - R_N = |corr(N̂(r), N̂(r+Δ))|（repeat 残差；无 repeat 时用 proxy 并显式标记 noise_source）+ 可选方差一致性项；
  - R_S = 0.5·(1−ρ_LF) + 0.3·(1−SSIM_local) + 0.2·(1−ρ_grad)（权重为初始值，Phase 1 做 sensitivity）；
  - R_I = exp(−nMSE(Δ)/τ_I) · max(ρ_Y(Δ),0)；物理预期：too close→捷径↑，中间→有效区，too far→结构风险↑（U 形）。
- **R_base = λ_N R_N + λ_S R_S + λ_I R_I**（初始 0.35/0.40/0.25）仅作诊断分数，不是 utility。
- **U_probe(Γ) = Σ w_j φ_j(Γ)**，φ 为 micro-probe 训练统计量的 robust z-score 组合（初始权重 +0.25 邻域损失改善 / +0.15 恒等风险降低 / +0.15 残差独立 / +0.20 梯度保持 / +0.10 跨视图一致 / +0.05 稳定性 / +0.10 惩罚捷径）。所有权重是实验初始值，不进论文定稿。**【2026-09-25 修订】**Exp C 证明上述固定符号先验不成立（ρ=−0.31）；U_probe 定义改为 **v2：regime-conditioned 校准权重**——w_j 在目标 regime 开发数据上拟合（候选极简特征集 {loss_neighbor_mean, cross_view_consistency}，3P 单 seed 上 LOO ρ=+0.82，3-seed 均值上可分对准确率 12/13、top/bottom AUC=0.89），**权重不跨 regime 迁移**（同一特征在 3P 空间与 CIDC25 时间轴符号相反，原理性约束）。3-seed 均值复拟合权重 w=[+1.45, +4.47]。详见 §11 变更记录。
- **U_oracle**：仅在拥有参考的开发数据上计算（PSNR/CNR/深区增益），只用于验证 U_probe 预测力，**不得作为部署期 policy selector**。
- **Policy**：先 hard（argmax U），后 soft（softmax(U/T) + R_N/R_S 安全截断）；patch 级决策，不做 voxel 级；不做在线 RL/meta-learning（future）。

## 4. 数据策略

| 数据 | 定位 |
|---|---|
| 3P 9-repeat 栈 | pair-risk / utility **开发实验室**（真噪声 + 结构参考 + oracle） |
| CIDC25 | H3 已证伪的机制战场；utility 机制验证（rgrid/sgrid 协议复用） |
| Neurofinder [T,Y,X] | 时间冗余 / 机制 benchmark；**T 不是物理深度，禁止深度解释** |
| 真实 green/red/obj（AVG25） | 真实应用验证：deep SNR(z)/CNR(z)/SBR(z) + 结构保持 |

⚠️ **修正 1（原方案 §24 不可执行）**：~~原始 25 帧数据不在库中，只有 25 帧平均后的 AVG 体数据，奇偶帧 split 暂不可行。~~ **【前置条件已满足（2026-10-09 用户确认；文件 09-23 已入库并被 M1 A3-1b 使用）】** `Data/Private/thy1-yfp-{1300nm-green,1100nm-red}-raw.tif` = [6025,256,256] = 241 z × 25 repeats（plane-major，帧均值与 AVG 逐层 corr=1.0）。处置（按原第二分支执行）：
- 先查运动/漂移/bleaching，必要时配准，再 A={1,3,5..} / B={2,4,6..} 独立 split，AVG25 只做 validation/最终图参考；
- 该数据即 green/red AVG 对应的原始 25 帧，是 M3 真实数据 deep-signal 验证（Priority 6）的主战场。

预处理协议：flat-field/无效面检查 → 配准（如需）→ 独立 split → 归一化 metadata → ROI/背景定义 → patch 提取 → Calibration/Validation/Blind Test。obj 的 1.8 mm 以下伪影显式记录在 metadata（valid_depth_um + excluded_artifact_reason），不写死在模型代码。

## 5. 评估协议（P0，硬性）

- **P0-1 语义**：静态体 [Z,Y,X]、动态 [T,Y,X]、体时序 [T,Z,Y,X]；代码 metadata 必须带 axis_names。
- **P0-2**：d_min/d_max 降级为 risk field 辅助特征，不再伪装 distance-aware sampler。
- **P0-3**：Q=N×S 不再作任何 decision score。
- **P0-4 隔离**：Calibration → Validation → checkpoint/threshold/policy 选择 → Blind Test。Blind Test 禁止参与一切选择（offset/threshold/geometry/checkpoint/temperature）。
- **P0-5 指标**（intensity domain）：SNR=(μS−μB)/(σB+ε)；CNR=|μS−μB|/√(σS²+σB²)+ε；SBR=μS/(μB+ε)；dB 换算在 Methods 中写明。ROI 规则预先固定：signal ROI 来自独立 annotation/稳定分割，background ROI 用邻接环/固定解剖区/已知高背景区，红绿通道分别建立——**不得用"亮的地方"当 signal 再证明 SNR 提升**。
- **P0-6 SSIM**：正式主表只报 full SSIM；fg/bg 行不再携带重复的全 patch 值。
- **P0-7 checkpoint**：统一 Validation-selected best，结果必须记录 selected_epoch / selection_metric / selection_dataset / selection_protocol。
- **训练协议**：max 30–50 ep、每 2–5 ep 验证、patience 5–10；early stopping 只由 validation 决定。主表只保留 training-resolution-aligned + validation-selected 一个口径，full-resolution 入 supplementary。
- **统计**：正式比较 ≥3 seeds，mean±SD（+95%CI）；统计单位（volume/animal/field/patch）显式声明，防 pseudo-replication。

## 6. 实验矩阵与 Gate

| Gate | 实验 | 通过判据 |
|---|---|---|
| P0→P1 | 仓库清理（本节 §5 全项） | 协议统一、语义统一、QMask/legacy 隔离、test 不参与选择 |
| P1→P2 | **Exp B** risk field（3P + CIDC25 + nf） | R_N/R_S/R_I 规律不同；Q 无法解释；至少一个真实数据集有明显 U 形 trade-off |
| P2→P3 | **Exp C** micro-probe vs oracle | Spearman(U_probe, U_oracle) 显著 > Spearman(Q, U_oracle)；否则回炉 utility 定义，禁止进 adaptive。**【2026-09-25 判定：v1 不通过（ρ=−0.31 ns）→ 已回炉；v2（regime-conditioned）初判 LOO ρ=+0.82，待 3-seed 复核后重判】【2026-10-07 终判：有条件通过】** 3-seed 均值 oracle（10 family 全量）上：朴素 Spearman ρ=+0.467（p=0.174，不显著）——但噪声诊断表明该判据不可达：oracle 排序自身 seed 信度仅 ≈0.33（单 seed 间 ρ̄=+0.143），任何预测器对 3-seed 均值的相关上限 ≈0.58；45 对家族中仅 13 对统计可分（gap/noise>1.96），中段排序在 oracle 本身即不稳定。在可分对上 **LOO 方向准确率 12/13（p≈0.002）**、top3-vs-bottom3 判别 **LOO AUC=0.89**（adaptive 选择所需的核心能力）。判定：v2 对可靠可分的家族效用具真实预测力，M3 adaptive 以 top/bottom 判别为准入依据（非全排序）。待补：80ep 预算敏感性（运行中）、CIDC25 反向验证。 |
| P3→P4 | **Exp D** adaptive-hard | Oracle Gap G_oracle 显著小于 fixed/QMask。**【2026-10-08 最终判定：校准 regime 内成立 + 跨预算迁移失效（均为证据）】** ①**20ep 校准 regime 内**：3P（dev=s3407 拟合权重、held-out s3408/3409 评估）adaptive 选 sn2n_xy_diag **gap=+0.039dB**（双 held-out seed 一致）vs fixed-QMask +0.473 / fixed-VALID +0.526（收窄 92%）；CIDC25（9 dt × 3 seed，权重 3-seed 均值拟合）adaptive gap **+1.82±0.91** vs nearest(dt=1) **+3.45±1.51**（点估计收窄 47%，配对 t n=3 p=0.31）。②**20ep 校准 → 80ep 生产迁移失效**：80ep 3-seed 下 adaptive 臂 sn2n_xy_diag 7.39±0.18 在 4 关键家族中**垫底**（qmask_xy 7.68±0.07 最优，配对差 +0.296，3/3 seed 为正）；机制为预算饱和不对称（sn2n 20ep 已收敛 Δ+0.02dB，qmask/valid 慢热各 +0.5dB）。**结论：U_probe 校准是（数据 regime × 预算 regime）双条件函数**——与 CIDC25 反向验证共同构成 regime-conditioning 完整证据（H4）；M3 生产 pipeline 设计约束：目标预算 B 的部署须在 B 下做 oracle 校准。附：CIDC25 dt=8「内点峰」seed 脆弱（s3409 后期发散，std 5.21dB），dt=64 最稳最优（29.34±0.50）。 |
| P4→P5 | **Exp E** adaptive-soft | seed 方差/策略熵/鲁棒性有收益，否则保留 hard |
| P5→P6 | **Exp F** depth/signal-conditioned | ΔSNR_deep/ΔCNR_deep/ΔSBR_deep > 0 且结构不退化 |
| 应用 | **Exp G** 真实 2P/3P | 深层 SNR(z)/CNR(z)/SBR(z) + neuron detectability（Raw/AVG25/VALID/QMask/Proposed；各方法遵守各自数据假设） |

**H1–H6 假设**：H1 各向异性冗余存在（已有数据支持）；H2 半径不是充分表示（ellipsoid/rgrid 证据）；H3 **pair similarity ≠ training utility（核心，已获时间轴+空间轴双闭环初步证据）**；H4 风险分解改善 utility 预测；H5 utility 采样逼近 oracle；H6 depth-conditioned policy 改善深层可检测性。

## 7. 代码架构（⚠️ 修正 2：增量式，不做一次性大迁移）

新增两个包，现有 `ACES/qmask/`、`ACES/aces/` 原地保留为 baseline/legacy：

```
ACES/profiling/      # M1: candidates.py + risk.py + risk_field.py（CLI）
ACES/calibration/    # M2: micro_probe.py + utility.py（复用 t4b_rgrid 协议）
ACES/sampling/       # M3+: adaptive.py（hard→soft→depth-conditioned，Gate 后才写）
ACES/evaluation/     # 按需从 evaluate_stage2.py 演化，不抢跑
```

对应关系：`t4b_rgrid.py`/`t4b_sgrid.py` = micro-probe 素材；`qmask/geometry.py`+`sampling.py` = baseline；`shortcut_risk.py` 思路并入 `profiling/risk.py`；`evaluate.py` 深度诊断以后并入 evaluation。所有 sampler 实现统一接口 `sample(image, policy_state, seed) → (views, metadata)`，metadata 含 family_id/offsets_voxel/offsets_um/stride/policy_probability/risk/utility。

单元测试（§42）：offset 不变性、policy 可复现、相关噪声→R_N↑、平移结构→R_S↑、复制图像→R_I≈1、风险分解独立性、family 聚合正确性、test loader 不触碰 calibration API。

## 8. 预算与优先级（⚠️ 修正 3）

- **Priority 1**：candidate Δ/Γ → R_N/R_S/R_I（risk field，CPU 为主）。
- **Priority 2**：Γ → 2-epoch micro-probe → U_probe vs oracle（**全项目关键 Gate**）。
  - ⚠️ oracle 成本控制：只在 8–12 个精选 family 上算 oracle，且用 20ep pilot 协议（与 rgrid 一致），不做 M×80ep×3seeds 全矩阵。
- **Priority 3–5**：hard → soft → depth-conditioned（逐 Gate 推进）。
- **Priority 6**：真实数据 deep-signal 验证（**25 帧已到位**，thy1-yfp green/red raw；A/B split + AVG25 参考协议，2026-10-09 启动）。
  - 补记（10-11 更新，green1300 **7/7 家族全部完成**）：先定性（深层切块 + 高分辨 zoom 无塌缩/伪影）→ 再定量。**PSNR 增益随深度单调增大**（shallow −0.39~+3.30 / mid −0.01~+5.93 / **deep +3.35~+6.71 dB**），深层获益最大；单帧去噪背景高频噪声 proxy 比 25 帧平均还低 2.1~2.5×，深层 SBR 保留至 AVG-25 参考的 90~94.5%（无明显过平滑）。**双协议一致**：oracle 分带 U_oracle 与 deepsignal 分带 PSNR 在三带上家族排序逐位相同（deep 带 0 处交换）——互为交叉验证。**⚠️ 但 M1 风险场的家族排序未复现**（7/7：Spearman −0.393, p=0.383；qmask_xy 实测第 1 而 M1 判 5/15，qmask_xyz_tetra 实测第 5 而 M1 判 1/15；下半区 valid_corners/sn2n_xy_diag/axis_z_d2 三档 3/7 精确命中）——M1 只在"识别坏 family"上有效，**A3-1b 先验的 family 序是 regime 依赖的，不可跨数据/预算搬运**，与 A3-2e（U_probe 双 regime）结论同源。**新证据（M3 Exp F 动机）**：最优 family **随深度翻转**——shallow 冠军 qmask_xy（deep 带仅第 4）vs deep 冠军 sn2n_xy_diag（shallow 带倒数第 2），qmask_yz 由 shallow 第 5 升至 deep 第 2；H6 深度条件策略有同数据直接证据。详见 PROJECT_LOG A3-4b / A3-4c / A3-4d。
- **Priority 7–8**：3D Wiener 频域基线（排除"只是激进平滑"解释）；online curriculum/RL/4D（future，不做）。
- micro-probe 预算：2 ep × 100–200 batches × 64–128 patches，探索期 1 seed、正式校准 3 seeds。
- **不再投入**：τ/阈值 sweep、新固定 ellipsoid、更多固定 xyz mask、堆 epoch、test proxy 挑 checkpoint、单图证明 deep enhancement、第一版上 RL/Transformer。

⚠️ **修正 4（资源现实）**：GPU1–3 常被并行项目占用；risk field 全程 CPU 可做，micro-probe/oracle 训练按 GPU 空闲排队，长训练一律 screen 启动。

## 9. 归因与统计纪律

- 第一阶段严格固定 backbone/loss/optimizer/lr/patch/budget/augmentation，只变 sampling——Performance=F(sampling)。
- sampling 机制证明有效之后，才允许研究 deep-aware loss（L=L_ssl+λd·L_deep+λg·L_grad），且 w_deep 只能由部署期可观测信号质量导出。
- 每个正式 run 目录：config.json / metadata.json / calibration/(candidate_families, risk_field, utility, policy) / logs/(train_losses, eval_curve, policy_trace) / checkpoints / metrics / report/RESULTS.md。policy_trace 逐 epoch 记录 family histogram、策略熵、mean risk/utility（监测 policy collapse）。

## 10. 成功标准（Level 1–5）

1. 机制：pairability ≠ training utility（已有初步证据，需正式跨数据验证）；
2. 预测：U_probe → U_oracle 可重复正相关；
3. 方法：Γ̂ 稳定逼近 Γ*（G_oracle → 0）；
4. 应用：ΔSNR/CNR/SBR_deep > 0 且结构保真；
5. 生物：深层 neuron detection/reconstruction 实际改善。
Level 1–4 成立后才把完整 biological claim 写进论文主结论。

## 11. 变更记录

| 日期 | 变更 | 依据 |
|---|---|---|
| 2026-09-22 | 初版：由 ChatGPT 修正方案经审查固定；4 处修正（25 帧 split 降级为前置条件 / 重构改增量式 / oracle subset 化 / GPU 排队） | 方案审查（PROJECT_LOG §3-S2） |
| 2026-09-25 | **U_probe v1 → v2 回炉**：Exp C 判定 v1 不通过（Spearman=−0.31 ns）。归因：①固定符号先验在 3P 空间 20ep 预算下方向全反（相似度类特征实际正向预测：R_I +0.89、residual_corr +0.82、cross_view +0.84）；②**跨 regime 符号矛盾**（同一特征 3P 空间 +0.89 vs CIDC25 时间轴 −0.82）——固定权重分数原理上不可能跨 regime 预测效用（H4 核心证据）。v2 定义改为 **regime-conditioned 校准**：在目标 regime 开发数据上用极简特征（{loss_neighbor_mean, cross_view_consistency}，LOO ρ=+0.82 初判）拟合线性权重，权重不跨 regime 迁移；配合 3-seed oracle 误差棒与 80ep 预算敏感性复核。 | PROJECT_LOG A3-2d、`runs/m2_{micro_probe,oracle}_3p_seed3407/`（⚠️ 该路径位于 /data3，挂载点当前离线，恢复后复核数字） |
| 2026-10-07 | **Gate P2→P3 终判：有条件通过（conditional PASS）**。10 family × 3 seed oracle 均值上重拟合 v2：朴素 Spearman ρ=+0.467（p=0.174）不显著，但归因于判据不可达而非 probe 失效——oracle 排序自身单 seed 间 ρ̄=+0.143（信度 ≈0.33，任何预测器上限 ≈0.58），45 对家族仅 13 对统计可分，中段排序在 oracle 本身即 seed 主导。核心证据：可分对上 LOO 方向准确率 **12/13（二项 p≈0.002）**、top3-vs-bottom3 **LOO AUC=0.89**（全拟合 1.0）——adaptive 选择所需的判别能力成立。3-seed 均值权重 w=[+1.45, +4.47]（存 `u_probe_v2_3seed.json`）。仅有的可靠错判：axis_y_d1 vs diag_yz（z=8.5 最大对，v2 低估 axis_y_d1）。M3 准入以 top/bottom 判别为准。**【2026-10-08 补齐】**80ep 预算敏感性：shortcut 平台假说不成立（同 seed 排序 ρ=+1.000，相似性家族优势 4× 预算下保持）；CIDC25 反向验证：3P 权重迁移失效（ρ=+0.17 ns vs regime 内 +0.73）且会把 dt=1 捷径陷阱选为最优臂——regime-conditioned 设计闭环，**M2 全部收口**。 | PROJECT_LOG A3-2h、`runs/m2_oracle_3p_seed340{7,8,9}/`、`u_probe_v2_3seed.json`、`u_oracle_3seed_full.json`、`calibration/v2_judgment.py` |
| 2026-10-08 | **80ep 预算敏感性：shortcut 平台假说不成立**。6 代表 family × seed 3407 的 80ep oracle：①同 seed 20ep→80ep 排序 Spearman=+1.000，U_oracle 排序预算稳定；②相似性家族（valid_corners/qmask_xy）优势在 4× 预算下保持（valid_corners 80ep 7.740 为 6 家族最高；增益差 −0.157dB ≪ seed 噪声）——20ep oracle 量的是持久性质而非暂时捷径；③20ep 3-seed-mean vs 80ep ρ=+0.43 的分歧与信度上限分析一致（seed 而非预算所致）。P2→P3 有条件通过的最后一个预算侧保留意见撤销；仅余 CIDC25 反向验证。 | PROJECT_LOG A3-2i、`runs/m2_oracle_3p_80ep_seed3407/` |
| 2026-10-08 | **CIDC25 反向验证通过（M2 全部收口）**：时间轴 micro-probe（新模块 `calibration/micro_probe_temporal.py`，镜像 t4b_rgrid Δt 配对 + micro_probe 特征协议，9 个 dt × 2ep×100batch）证明 **3P 拟合权重不可跨 regime 迁移**：全 9 dt 上迁移 ρ=+0.17（p=0.67，失效）vs 时间轴 regime 内拟合 ρ=+0.73（p=0.02）；**3P 权重把 dt=1 选为最优臂（U=6.96 全场最高，由 L_N=2.32 驱动）而 dt=1 是 oracle 最差档**（26.4dB；dt=1 帧间噪声相关破坏 N2N 假设，与 M1 时间轴 R_N 高一致）——正是 H4 警告的近重复捷径陷阱。机制细化（H4 探针级证据）：cross_view_consistency 两 regime 同号为普适特征（CIDC25 +0.80, p=0.01）；loss_neighbor_mean 在 3P 正预测、在时间轴被 dt=1 捷径点反号（L_N 最高 × 效用最低）；所需权重比 cross/L_N 从 3P 的 3.1 → CIDC25 的 24.2。结论：特征可共享、**校准必须逐 regime**——regime-conditioned v2 设计的正确性闭环。保留项：CIDC25 oracle 为 rgrid 单 seed（16/4 网络），probe 为 32/8 网络（与 3P 探针对齐）。 | PROJECT_LOG A3-2j、`runs/m2_micro_probe_cidc25_seed3407/`、`calibration/micro_probe_temporal.py` |
| 2026-10-08 | **Exp D（adaptive-hard）解析版 pilot**（M3 启动）：新模块 `calibration/exp_d_gap.py` 零新训练完成 Oracle Gap 分析（3P dev/held-out seed 分离协议 + CIDC25 regime 内）。双 regime 点估计均过 Gate P3→P4 判据：3P adaptive gap 0.039dB（vs QMask 0.473，收窄 92%）、CIDC25 adaptive gap 0.756dB（vs 最近邻 3.629，收窄 79%）。正式判定补项已列（CIDC25 补 2 seed、3P 80ep 3-seed）。 | PROJECT_LOG A3-3a、`runs/m2_oracle_3p_seed3407/exp_d_gap.json` |
| 2026-10-08 | **Exp D 最终判定：校准 regime 内成立 + 跨预算迁移失效（双发现）**。补种完成后：①20ep 校准 regime 内 adaptive 有效（3P gap 0.039 双 held-out seed 一致、CIDC25 3-seed 收窄 47%）；②**20ep 校准 → 80ep 生产预算迁移失效**：adaptive 臂 sn2n_xy_diag 80ep 3-seed 7.39±0.18 在 4 关键家族垫底（qmask_xy 7.68±0.07 最优，3/3 seed 为正差 +0.296）；机制=预算饱和不对称（sn2n 20ep 已收敛 Δ+0.02，qmask/valid 慢热各 +0.5）。**U_probe 校准是（数据 regime × 预算 regime）双条件函数**——与 CIDC25 数据 regime 反向验证共同构成 regime-conditioning 完整证据（H4 升级）；M3 设计约束：目标预算 B 的部署须在 B 下校准。附带：dt=8 内点峰 seed 脆弱（std 5.21dB）须标注；20ep 同 seed 排序 ρ=+1.000 与 3-seed 均值排序反转（ρ=−0.60）并存——单 seed 排序稳定性部分是 seed 噪声对齐。 | PROJECT_LOG A3-3b/c、`m2_oracle_3p_80ep_seed340{7,8,9}/`、`stage01_cidc25_rgrid_seed340{8,9}/` |
| 2026-10-09 | **定性（视觉）对比图补齐启动**（用户要求：6 张定量图之外需直观去噪结果供人工审阅）。新脚本 `calibration/render_qualitative_cidc.py`（零新训练）：CIDC25 定性面板 `fig_q1_cidc25_qualitative.png`——3 行（F1 Validation 高亮前景 patch 64×128×128 取 z 中切片，t0=718）× 5 列（输入 F1 \| GT F0 均值 \| dt=1 \| dt=8 \| dt=64），归一化与 t4b_rgrid eval 协议一致（F1 缓存统计 mean=204.45/std=139.76），逐行 1–99.5 百分位显示窗。review 切块 `review/cidc25_row_{1,2,3}.png`（1100×220，符合读图限制）已人工可读；初步定性：dt=1（最差臂）过度平滑/结构丢失可见，dt=64（3-seed 稳健最优）最接近 GT——与定量结论一致。3P 侧定性面板（fig_q2，4 家族 vs AVG-9 伪 GT）依赖 `m2_qual_3p_seed3407` 重训练（screen `qual_3p`，GPU1 进行中，save_checkpoint=True）。README 审核索引已更新。

**同日补记（fig_q2 完成）**：`m2_qual_3p_seed3407` 重训练完成（4/4 checkpoint，U_oracle：qmask_xy +7.5310 / valid_corners +7.6738 / sn2n_xy_diag +7.0246 / axis_z_d2 +6.7451 dB，raw=12.809，n=2016 eval patch）。新脚本 `calibration/render_qualitative_3p.py` 渲染 `fig_q2_3p_qualitative.png`：3 行（repeat 栈 z 三分带各取 AVG-9 最亮 patch，z 中切片）× 6 列（输入 frame 0 | AVG-9 伪 GT | 4 家族，各列标注 U_oracle），checkpoint 内 pool_stats 归一化。review 切块 `review/3p_row_{1,2,3}.png`（1170×220）已逐张判读：4 家族跨浅/中/深 z 带均明显去噪，valid_corners/qmask_xy 结构保留最好，排序与 U_oracle 定量一致，无塌缩/伪影。**同日按用户反馈（灰度对比不明显）将两脚本统一为 turbo 伪彩 + (1.0, 99.9) 分位窗（与 `qmask/paperviz.py` 格式一致）并重渲**，6 张 review 切块重读复核结论不变。**定性图补齐两轨道全部完成，待用户人工审阅。** | PROJECT_LOG A3-3d、`/ssd1/wjb/Denoising_PM/ACES/runs/m2m3_paper_figures/`（fig_q1 + review/）、`calibration/render_qualitative_cidc.py` |
