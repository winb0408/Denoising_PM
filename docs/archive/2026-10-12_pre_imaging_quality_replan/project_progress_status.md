# 项目进度总览（project_progress_status）

> **文档定位**：面向**整体验收**的同步状态文档。逐次实验流水见 `PROJECT_LOG.md`；
> 现行方案见 `PLAN.md`（2026-09-22 起为唯一方案文档）；早期技术细节见
> `ACES_Plan_v2_20260901.md`（§8 修改小结 #1–#25，逐条可溯）。
> 本文档在每个重要节点后同步更新；结论一句话 + 证据目录，便于快速核查。
> **最后更新：2026-10-11**（**真实数据 deep-signal 验证（PLAN Priority 6）已完成**：green/red 原始 25 帧确认入库（即 AVG 对应原始帧，plane-major 241z×25，无 bleaching/漂移，A/B split 解锁）；green1300 oracle **7/7 家族全部完成**，**定性通过 + 定量显示 PSNR 增益随深度单调增大（deep +3.35~+6.71 dB，深层获益最大），深层 SBR 保留 90~94.5%，抑噪 proxy 优于 25 帧平均 2.1~2.5×**；**双协议一致**——oracle 分带 U_oracle 与 deepsignal 分带 PSNR 在三带上家族排序逐位相同；**⚠️ 但 M1 风险场家族排序未复现（7 家族 Spearman −0.393, p=0.383；qmask_xy 实测第 1 而 M1 判 5/15）**——A3-1b 先验 family 序是 regime 依赖的；**新证据**：最优 family 随深度翻转（shallow qmask_xy vs deep sn2n_xy_diag），构成 M3 Exp F/H6 直接动机。前情 10-09：定性图两轨道补齐并重渲 turbo 伪彩，待用户人工审阅）

---

## 0. 一页速览

**主线一句话（2026-09-22 升级）**：建立"**数据 → 风险分解(R_N/R_S/R_I) → micro-probe 效用 → 自适应采样策略**"的自监督荧光去噪采样框架——不再人工指定固定采样几何（也不再止步于 QMask 的逐轴掩码），而是让数据自己决定哪些 observation pairs 值得用于训练，最终目标是深层弱神经元信号的 SNR/CNR/SBR 与可检测性。

**方案文档**：`PLAN.md`（由 ChatGPT 修正方案经代码级审查固定，4 处修正：25 帧 split 降级为前置条件 / 代码重构增量式 / oracle 只在 8–12 family subset 上算 / GPU 排队）。

**创新点验证状态**：

| # | 创新点 | 验证状态 | 一句话结论 |
|---|---|---|---|
| IC1 | ~~数据驱动逐轴配对几何（QMask）~~ → **升级为 pair-risk 分解 + 效用校准** | 🔄 方案定稿（PLAN.md §2-3），QMask 降级为 baseline | 时间轴+空间轴双闭环证明：几何选择价值由捷径危害度（噪声级）决定，无统一阈值——这正是"从静态判据走向 utility 校准"的动机 |
| IC2 | 共位相位分裂（stride-1）采样 | ✅ 同骨干矩阵最优臂（保留为 baseline） | 3-seed：qmask 80ep fg 22.14±0.33；五档对比证明采样几何本身 ~2 dB |
| IC3 | 采样几何的可证伪科学检验（H3） | ✅ 已闭环——结论为证伪（可发表发现） | 双闭环：τ_s 门控与实测最优几何反相关（Spearman≈−1）；该结论直接构成新阶段核心假设 H3'（pairability≠utility）的已有证据 |
| IC4（新） | Pair-risk field（R_N/R_S/R_I 三维分解） | ✅ M1 完成（3P/red/green 空间 + CIDC25 F1-F3/nf 时间，23/23 单测） | 三风险规律跨数据集一致；固定权重 r_base 推不出 CIDC25 内部效用峰（d*=16/8/64）→ 单标量不可替代。Gate P1→P2 通过 |
| IC5（新） | Micro-probe utility → adaptive policy | 🔄 M2 oracle 已完成；**Gate P2→P3 判定：U_probe v1 不通过**（ρ=−0.31 ns），但探针特征本身信号强（+0.82/+0.84），失败根源是固定符号先验 + 跨 regime 符号矛盾（3P 空间相似度 +0.89 vs CIDC25 时间轴 −0.82，H4 直接证据）→ 回炉 utility（v2 = regime-conditioned 权重校准） |

**当前进行中**：**①真实数据 deep-signal 验证（10-11 完成，PLAN Priority 6）**：green/red 原始 25 帧已确认（thy1-yfp raw tif = AVG 对应原始帧，体检无 bleaching/漂移，A/B split 可行）；green1300 通道 7 家族 oracle（qmask_yz/qmask_xz/qmask_xyz_tetra 为 M1 先验最优 + qmask_xy/valid_corners/sn2n_xy_diag/axis_z_d2 对照）**7/7 全部完成**（screen `oracle_green`，GPU0，run `m2_oracle_green1300_seed3407`；U_oracle：qmask_xy +5.4070 > qmask_xz +4.8787 ≈ qmask_yz +4.8775 > valid_corners +4.8459 > qmask_xyz_tetra +4.3290 > sn2n_xy_diag +4.0303 > axis_z_d2 +1.0123，raw=19.871）。**7 家族全量判读已出（先定性→再定量）**：①定性——深层 patch 三行 turbo 切块仅 ~127 px/家族，不足以判细结构，故新增可复用脚本 `calibration/render_deep_zoom.py` 渲染 ~400 px/家族的高分辨 tile + 64×64 细节裁剪（全程未读 2880×960 大图）；噪声显著抑制、树突/荧光亮斑保留、无塌缩/块状拼接/网格伪影，**通过**。②定量——**PSNR 增益随深度单调增大**（shallow −0.39~+3.30 / mid −0.01~+5.93 / **deep +3.35~+6.71 dB**，input 深层 17.00 → sn2n_xy_diag 23.71），即深层获益最大；抑噪 proxy（背景 Laplacian std）深层输入 3083.7 → 去噪单帧 211–253（比 25 帧平均 535.5 还低 2.1–2.5×）；深层 SBR 保留至 AVG-25 参考的 **90.0~94.5%**（无明显过平滑）；**过平滑假设被推翻**——深层 SBR 最高的 sn2n_xy_diag 其 bg_lap 也最高（背景抑制最弱），真实权衡是"信号保留 vs 背景清洁"。③**双协议一致**：oracle 分带 U_oracle 与 deepsignal 分带 PSNR 在 shallow/mid/deep 三带上家族排序逐位相同（deep 带 0 处交换，mid 仅 xz/yz 相邻交换）——互为交叉验证。**⚠️ 但 M1 风险场家族排序未复现（7/7）**：Spearman(M1 risk, U_oracle) = **−0.393 (p=0.383)**；实测 qmask_xy **+5.4070 第 1**（M1 风险序 5/7）、qmask_xyz_tetra **第 5**（M1 风险序 1/7）；下半区精确命中 3/7（valid_corners 4=4、sn2n_xy_diag 6=6、axis_z_d2 7=7）。判读：M1 风险场能识别"坏 family"（大 support/近 no-op），不能选出"最好 family"——**A3-1b 先验 family 序是 regime 依赖的**，与 A3-2e（U_probe 双 regime）同源。**④新证据（M3 Exp F/H6 动机）**：最优 family **随深度翻转**——shallow 冠军 qmask_xy（deep 带仅第 4）vs deep 冠军 sn2n_xy_diag（shallow 带倒数第 2），qmask_yz 由 shallow 第 5 升至 deep 第 2。red 通道 SNR>150 非瓶颈，暂缓。**②定性（视觉）对比图补齐已完成，待用户人工审阅（10-09）**。两张面板已按用户反馈（灰度对比不明显）重渲为 **turbo 伪彩 + (1.0, 99.9) 分位窗**，与 `ACES/qmask/paperviz.py` 格式一致（脚本 `calibration/render_qualitative_cidc.py` / `render_qualitative_3p.py`），6 张 review 切块重读复核结论不变。①**CIDC25 定性面板 fig_q1**（`/ssd1/wjb/Denoising_PM/ACES/runs/m2m3_paper_figures/fig_q1_cidc25_qualitative.png`，人工审阅走 `review/cidc25_row_{1,2,3}.png` 切块）：3 行 F1 高亮前景 patch × 5 列（输入 | GT F0 | dt=1/8/64）；初步判读与定量一致——dt=1（最差臂/捷径陷阱）过度平滑、dt=64（稳健最优）最接近 GT。②**3P 定性面板 fig_q2 完成**（`fig_q2_3p_qualitative.png`，审阅走 `review/3p_row_{1,2,3}.png`）：`m2_qual_3p_seed3407` 重训练完成（4 家族 × 20ep，save_checkpoint=True；U_oracle：valid_corners +7.67 > qmask_xy +7.53 > sn2n_xy_diag +7.02 > axis_z_d2 +6.75 dB），面板 3 行（z 三分带 AVG-9 最亮 patch）× 6 列（输入 | AVG-9 伪 GT | 4 家族带 U 标注）；逐行判读：4 家族均明显去噪、排序与定量一致、无塌缩/伪影。前情（10-08，Exp D 最终判定/M3 首阶段收口）：①校准 regime 内（20ep）adaptive 有效（3P gap 0.039 vs QMask 0.473 收窄 92%；CIDC25 收窄 47%，n=3 显著性不足）；②**跨预算迁移失效（原理性发现）**：80ep 3-seed 下 adaptive 臂 sn2n_xy_diag 7.39±0.18 垫底（qmask_xy 7.68±0.07 最优）——U_probe 校准是（数据×预算）双 regime 条件函数，与 CIDC25 反向验证共同构成 H4 完整证据；M3 设计约束：目标预算 B 下校准。**M1/M2 已收口**；B 通道、真实数据 VALID 通道已收口。**前置条件**：red/green 原始 25 帧已入库（thy1-yfp raw tif，2026-10-09 确认为 AVG 对应原始帧，A/B split 解锁）；obj_2mm 无原始帧（单次 AVG 体）。

---

## 1. 主线与叙事链

1. **问题**：自监督去噪（Noise2Noise 族）的效果强依赖配对采样几何（哪个轴配对、配对距离多大），现有方法（VALID、SN2N）靠人工先验固定几何。
2. **主张**：配对几何可以从数据统计中自动决策，且"哪种几何最优"有可证伪的科学判据。
3. **证据链**（按论文叙事顺序）：
   - 同骨干五档对比 → 采样几何本身值 ~2 dB（T4）；
   - qmask 共位相位分裂 + 逐轴掩码 → 同骨干矩阵最优臂（IC2）；
   - 外部基线表（原仓库、原协议）→ 与已发表方法公平对比（E1/E2）；
   - H3 判据科学检验 → 原判据证伪 + 修正假说两轴验证（IC3/IC1）；
   - 几何与过平滑的机制联动（大 d 惩罚恒等捷径 = 间接抗平滑）→ B 通道定量支撑。

---

## 2. 创新点逐项验证

### IC1 数据驱动逐轴配对几何（核心方法创新）— 🔄 修订中

- **原始形式（v1）**：~~`d_a* = max{d: Q_a(d) > τ}` 单一公式选几何~~
  **→ 修订一（v2，依据 T3b/T4）**：区间化几何 `d_max = max{d: S_a(d)≥τ_s}`、`d_min = min{d: N_a(d)≥τ_n}`，参与掩码 = 连续 VALID↔SN2N 谱。
- **修订二（依据 T3b+Q-1+CIDC25 三数据集 proxy-N 失效证据）**：~~N 阈值参与决策~~ → **N 仅记录不决策**（`use_n_decision=False`，S+Q 主导）。
- **修订三（依据 T4b H3 证伪，#12/#14）**：~~结构相关性门控（S/τ_s）~~ → **shortcut-risk 判据**（捷径回避）；A1 离线分析进一步否定静态损失差形式，修正为**优化动力学假说**（恒等解是强吸引子）。
- **修订四（依据 A2 空间轴闭环，#22）**：~~统一漂移尺度阈值（nMSE0≈0.02）判据~~ → **几何选择的价值随捷径危害度（噪声级）变化，无跨档/跨轴统一阈值**。
- **当前证据**：时间轴（rgrid 9d，d*=16/8/64）+ 空间轴（sgrid 6s，最优 s=0/1/16 随噪声级递增）双闭环。
- **未完成**：A3（把修订判据写入 qmask 几何决策并单元验证）、R10（两网格 3-seed）。
- **证据目录**：
  - `/data3/wjb/Denoising_PM/ACES/runs/stage01_cidc25_rgrid_seed3407/`（时间轴网格 + `metrics/shortcut_risk_analysis.json` + `metrics/h3_closure.json`）
  - `/data3/wjb/Denoising_PM/ACES/runs/stage01_cidc25_sgrid_seed3407/`（空间轴网格 + `metrics/sgrid_closure.json`）
  - `/data3/wjb/Denoising_PM/ACES/runs/stage01_cidc25_gt_pairability_seed3407/`（第一切片 r_t 收缩曲线）
  - τ 扫描：`ACES/qmask/runs/tau_sweep_20260901/`

### IC2 共位相位分裂采样 + 同骨干归因 — ✅ 已验证（R10 部分完成）

- **验证实验**：同骨干（VALID 骨干+损失+同 eval 协议）只换采样的内部归因表。
  - 五档对比（40ep）：valid 22.02 / sn2n_xy 21.98 / iso3d 21.49 / ellipsoid 20.33 / empirical_block 19.87 fg PSNR——**采样几何本身 ~2 dB**；
  - qmask80ep × 3 seed（80ep 生产规模）：**fg 22.14±0.33，最优臂**；
  - 4 臂×3 seed 重复矩阵（pilot 规模）：qmask 21.39 vs sn2n_xy 21.23 / iso3d 21.02 / valid 20.60（差距在 R10 方差量级内，生产规模以 80ep 三 seed 为准）。
- **配套修正（Q-3）**：~~qmask 在 3p 上亏 valid 1.49 dB（疑似方法缺陷）~~ → **主因定位：行内槽位固定配对（修复 +0.80 dB）；剩余 −0.56 dB 未解释**，且该赤字在 R10 单次方差（±1.5 dB）量级内。
- **证据目录**：`/data3/wjb/Denoising_PM/ACES/runs/stage02_nf_qmask80ep_seed{3407,3408,3409}/`、`stage02_nf_matrix4_seed{3407,3408,3409}/`（矩阵）、`stage01_3p_*`（五档/3p 各档）

### IC3 几何判据的可证伪科学检验（H3）— ✅ 闭环，结论=证伪

- **第一切片**：r_t 随噪声级单调收缩（8→3→0），H3 现象首次直接观测（`stage01_cidc25_gt_pairability_seed3407/`）。
- **第二切片（闭环）**：~~H3 成立~~ → **证伪**：τ_s 门控预测的最优距离与实测 d*=16/8/64 反相关（Spearman≈−1）；机制 = 恒等/平均捷径（小 d 时 target≈input）。
- **延伸（A1/A2）**：shortcut-risk 静态判据同样否定（档内 Spearman −0.82~−0.87）；空间轴 sgrid 证明位移收益由噪声级决定、无统一阈值。
- **对外叙事**：负结果 + 修正假说（捷径危害度决定几何收益）本身是论文的可证伪支点。

### 支撑轨道（非独立创新点）

- **抗平滑与信背比（P-B）——已收口**：B0 全臂偏软实锤；B1 ✅ wr2e-5 折中最优（SSIM +0.08 / PSNR −0.16，用户目检通过）；B2 ✅ 梯度损失无收益不入主线（g0.01 22.50 / g0.1 22.51 vs ctrl 22.60，SSIM 无改善）；B3 ❌ α=4 发散留诊断样本。
- **指标体系（P-C）**：C1/C2/C3 ✅；P0 评估协议清理（SSIM region 修复、CNR 入列、checkpoint selection 元数据）随 A3 阶段进行中。
- **外部基线通道（E）**：E1/E2 ✅——外部表 3P 行 VALID 18.53 ≈ SN2N(p2p2) 18.40；nf 行 VALID 23.97 vs SN2N-3D 15.53。**论文只引外部表**（内部表含 +3.5 dB 双重选择偏置）。
- **真实数据通道（2026-09 新增）**：VALID 单方法三笔真实数据（green/red/obj2mm）管线打通；green→red 跨模态零样本迁移成立（corr 0.9475 vs 自训 0.9559，信背比 19.46 vs 24.37，输入 7.55）；matched 逐层直方图匹配数据有结构性缺陷（破坏轴向结构）。证据：`/data3/wjb/Denoising_PM/Private_eval/runs/`。

---

## 3. 阶段完成情况总表

| 阶段 | 任务 | 状态 | 一句话结论 | 证据目录/产物 |
|---|---|---|---|---|
| 复现与剖析 | T0 环境 / T1 VALID 复现 / T3 各向异性 / T3b proxy vs 金标准 | ✅ | proxy-Q 转移良好（ρ 0.93–1.00），**proxy-N 在 y/z 轴失效**（ρ 0.07~−0.26）——后续所有 N 决策降级的依据 | `runs/stage01_3p_repeat_pairability_seed3407/` 等 |
| 五档归因 | T4 五档对比（3P 40ep + nf 80ep） | ✅ | 采样几何决定 ~2 dB；下采样档有 train/eval 尺度错配（aligned 评估补救） | `runs/stage01_3p_*`、`stage02_nf_*` |
| qmask 待修 | Q-1 τ 扫描 / Q-2 N 降级 / Q-3 端点赤字 | ✅/✅/✅ | 掩码决策确实脆弱（0.0026 边距）；N 决策降级已实现；赤字主因=槽位固定配对 | `ACES/qmask/runs/tau_sweep_20260901/`、`runs/qmask_3p_q3_mosaic_seed3407/` |
| 外部基线 | E1 原 SN2N / E2 原 VALID / E3 qmask 入表 | ✅/✅/⏳ | 外部表 3P 两行齐（18.53 vs 18.40）、nf 行齐（23.97 vs 15.53）；E3 待 qmask 最终配置定稿 | `Results/review_report_e1.html`、外部 run 目录（见计划 §4-5/6） |
| H3 标定 | T4b 切片 1+2 | ✅ | **H3 证伪**：门控预测与实测反相关（Spearman≈−1），机制=恒等捷径 | `runs/stage01_cidc25_gt_pairability_seed3407/`、`stage01_cidc25_rgrid_seed3407/` |
| P-A 判据重构 | A1 / A2 / A3 / A4 | ✅(否定性) / ✅(pilot) / ⏳ / ⏳ | 静态判据否定；空间轴无统一阈值；修订判据待实现与 3-seed | `.../rgrid.../shortcut_risk_analysis.json`、`.../sgrid.../sgrid_closure.json` |
| P-B 抗平滑 | B0 / B1 / B2 / B3 | ✅ / ✅(待3seed) / 🔄 / ❌ | 全臂偏软实锤；降正则换锐度有代价（wr2e-5 折中）；B2 运行中；B3 发散待修 | `runs/stage02_nf_b1_*`、`paperviz_b1/`、`stage02_nf_b2_*`、`stage02_nf_b3_fg4_seed3407/` |
| P-C 指标扩展 | C1 / C2 / C3 | ✅/✅/✅ | 新指标三区域常规列；112 历史文件补列；paperviz 统一出表 | `aces/evaluate_stage2.py`、`runs/metrics_backfill_summary.json`、`qmask/paperviz.py` |
| 重复性纪律 | R10 | 部分 | 单次方差实测 ±1.5 dB → 定量主张必须 ≥3 seed；qmask80ep/矩阵已 3-seed，rgrid/sgrid/B1 待补 | 各 run 目录 seed{3407,3408,3409} |

---

## 4. 计划修订轨迹（依据实验的修正，前后对照）

> 样式：~~删除线 = 被证伪/废止的原计划~~ → **粗体 = 修订后**；括号内为触发证据与小结编号。

1. 几何公式：~~`d_a* = max{d: Q>τ}` 单一公式~~ → **区间化 d_max/d_min + 参与掩码**（T3b proxy-N 失效 + 8×8×8 退化证据，v2 §2.1）
2. 塌缩判据：~~mean9 偏差 >20%~~ → **ptp 动态范围塌缩 + repeat 统计一致性**（3P 上 mean9 偏差系统性偏严，v2 §2.3）
3. N 决策：~~N 阈值参与 d_min 决策~~ → **N 仅记录不决策**（T3b+Q-1+CIDC25 三数据集证据，#9 前后）
4. E1 管线：~~原 SN2N 3D 管线跑两条 leg~~ → **nf 走 2D 管线；3P 改 P2Pmode=2 单变量重跑**（用户目检 FAIL + 代码核实，#1）
5. 几何判据：~~结构相关性 S/τ_s 门控~~ → **shortcut-risk（捷径回避）**（H3 证伪，#12）→ **优化动力学假说**（A1 否定静态损失差，#14）→ **价值随捷径危害度（噪声级）变化，无统一阈值**（A2 空间轴，#22）
6. B1 消融设计：~~weight_reg/bg_reg 2×2~~ → **weight_reg 一维扫 {5e-5, 2e-5, 0}**（查证 bg_reg 缺省=0 无贡献，#15）
7. B3 前景加权：~~fg_loss_weight=4.0 提 SBR~~ → **发散终止，待降 α/平滑掩码后重设计**（ep10+旧 ckpt 回测双证系统性发散，#21）
8. 长训练执行方式：~~工具后台任务/裸 nohup~~ → **一律 screen 启动**（会话退出连带杀训练事故，#18；setsid 为本次过渡方案）

---

## 5. 待办与风险（按优先级，2026-09-22 按新方案重排）

| 项 | 内容 | 依赖 |
|---|---|---|
| M1 risk field | `ACES/profiling/` 候选库 + R_N/R_S/R_I + `plot_risk_field` 论文图 | ✅ 完成：空间（3P/red/green raw 25 帧）+ 时间（CIDC25 F1-F3/nf）；Gate P1→P2 三判据全过（2026-09-23） |
| P0 评估协议清理 | SSIM region 修复 / CNR / checkpoint selection / Z-T 语义标注 | ✅ 完成（12/12 单测） |
| M2 micro-probe + oracle | 2ep probe × M families；8–12 family subset 20ep oracle；Spearman(U_probe,U_oracle) vs Spearman(Q,U_oracle) | ✅ **全部收口（10-08）**：Gate P2→P3 有条件通过（v1 ρ=−0.31 回炉 → v2 可分对 LOO 12/13、top/bottom AUC=0.89）；80ep 预算敏感性过（排序预算稳定）；CIDC25 反向验证过（3P 权重迁移失效 ρ=+0.17 ns、regime 内 +0.73，dt=1 捷径陷阱被 3P 权重误选为最优） |
| 真实 25 帧数据 | red/green raw 已到位并已用于 M1（A3-1b）；**2026-10-09 用户确认即 AVG_green-1300/AVG_red-1100 对应原始帧（plane-major 241z×25，corr=1.0 验证）——25 帧 A/B split 可行，M3 deep-signal 主战场**；obj_2mm 为单次 AVG 体（无 repeat，不在 25 帧 split 范畴） | red/green ✅（split 解锁），obj（无原始帧）⏳ |
| E3 外部表收口 | 最终方法（adaptive）入外部表 | M3+ |
| 遗留（降级） | ~~R10 三 seed 批~~ 缩减为 micro-probe 相关部分；~~B3 修复~~ 不再投入（按 PLAN.md §8"不再投入"清单） | — |
| 风险 | GPU1–3 被 threePM 等并行项目占用（显存占用低，用户允许共享）；deep eval 的 ROI 协议需在盲测前固定 | 持续遵守 |

---

## 6. 方案文档体系（2026-09-22 起）

| 文档 | 角色 |
|---|---|
| `PLAN.md` | **唯一现行方案文档**（方案变更改此文件 + 其变更记录） |
| `PROJECT_LOG.md` | 逐次实验流水（做了什么→结论→证据目录） |
| `project_progress_status.md` | 本文档：整体验收状态 |
| `ACES_Plan_v2_20260901.md` | 历史方案（v2 时期，§8 #1–#25 仍是权威过程记录） |
| `ACES_Project_Correction_Execution_Plan_v1_20260922.md` | ChatGPT 原始修正方案（PLAN.md 的来源，留档） |
