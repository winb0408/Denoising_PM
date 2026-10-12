# 项目进度总览（project_progress_status）

> **文档定位**：面向**整体验收**的同步状态文档；现行目标、协议、阶段与图表要求以 `PLAN.md` 为准；逐次实验流水仍见 `PROJECT_LOG.md`。
> **最后更新**：2026-10-12（Asia/Shanghai），成像质量主线重构。
> **核查范围**：读取仓库代码、精选 JSON/CSV 与历史日志；本次仅更新方案与状态，**未重新训练、未重算原始体数据、未启动新实验**。
> **归档**：旧 PLAN 与旧状态表原样保存在 `docs/archive/2026-10-12_pre_imaging_quality_replan/`；旧“完成/通过”不自动等于新 Gate 通过。

---

## 0. 一页速览

**研究目标**：经自监督去噪后，深层有效结构/神经元信号更可辨、背景噪声减弱，并保持真实结构和信号。

**主线一句话**：可靠配对 → 自监督去噪/弱信号恢复 → 独立参考与固定 ROI → 结构可辨性、背景和保真的联合验证。

**工作顺序**：先证明图像变好，再解释配对贡献，最后研究是否需要自动选择。risk field/micro-probe/adaptive 是支撑或可选模块；方法指标不能代替最终成像效果。

**当前定位**：具备已有模型、配对实现和真实数据 pilot 证据；**正式成像验收尚未完成**。下一步优先 S0 独立参考/固定 ROI/选模隔离，再进入 S1 最小成像闭环。不以重跑风险矩阵作为前置任务。

| 项目 | 当前判断 |
|---|---|
| 真实深层恢复潜力 | 🟡 green1300 单 seed、7-family pilot 有 PSNR gain 和抑噪 proxy 改善 |
| 固定配对方法基础 | 🟡 QMask/VALID-like/SN2N-like 代码与历史消融可复用，正式成像贡献待验 |
| 独立参考/固定 ROI 正式协议 | ⏳ 当前 deep-signal 代码不满足，须先补齐 |
| 结构/神经元可检测性与保真 | ⏳ 无正式固定标注、误检与结构非劣验收；不能据 proxy 排除过平滑 |
| 自适应增量收益 | ⏳ 可选；现有 gap/排序结果不足以证明深层成像优势 |
| 跨体/模态应用与论文图 | ⏳ 历史素材可复用，正式图需按新计划重组织/复算 |

**状态符号**：✅ 本项在注明范围内完成并有证据；🟡 pilot/历史证据或部分完成；⏳ 待做；❌ 已见失败/不满足；➖ 可选/暂不投入。阶段只按 PLAN 的 G0–G5 验收，不按文件是否存在打勾。

## 1. 研究层级与创新点状态

| 编号 | 现行候选贡献 | 状态 | 当前证据与下一验收 |
|---|---|---|---|
| IC1 | 深层弱信号自监督恢复，联合抑噪与保真 | 🟡 | green pilot 提示恢复潜力；须 E1/E4 独立参考、固定 ROI、检测/结构与背景联合验证 |
| IC2 | 各向异性配对/共位相位分裂的恢复贡献 | 🟡 | 已有 sampling 实现和同骨干消融；须 E2 在主成像数据、同协议下复核 |
| IC3 | 深度/可观测质量条件策略 | ⏳ 可选 | 同数据浅深家族偏好不同只是动机；须优于验证选定的最佳固定策略并计入成本 |
| IC4 | 配对统计/效用的 regime 依赖边界 | 🟡 支撑 | 固定 risk 排序失效、跨预算迁移失败已留证据；只作机制解释或补充材料 |

当前不能宣称：独立盲测成像闭环已通过、普遍神经元检测改善、钙活动保真、普适 adaptive 或双模态优势。2P/3P 主张范围须按采集 metadata 和正式测试限定。

## 2. 当前证据链与重要修正

### 2.1 可核查的 green1300 pilot

- **配置**：seed 3407、20ep×100batch、patch `(64,128,128)`、7 families；来自 `ACES/runs/m2_oracle_green1300_seed3407/families/summary.json`。
- **当前参考**：`oracle.py` 与 `analyze_green_deepsignal.py` 均使用全部 repeat 均值；被评估输入也在该均值中。`oracle.py` 明示训练/评估 patch 可能重叠，不能作为严格留出测试。
- **深层指标**：`deepsignal/metrics.json` 的 deep 带 z160–241，input PSNR=17.0016 dB；此处使用旧参考/分位范围口径。

| family | deep PSNR（dB） | 相对 input gain（dB） | 范围说明 |
|---|---|---|---|
| sn2n_xy_diag | 23.7116 | +6.7100 | 单 seed、含输入的 AVG25 参考 |
| qmask_yz | 23.4970 | +6.4954 | 同上 |
| qmask_xz | 22.7888 | +5.7872 | 同上 |
| qmask_xy | 22.6186 | +5.6170 | 同上 |
| qmask_xyz_tetra | 22.2576 | +5.2560 | 同上 |
| valid_corners | 21.8150 | +4.8134 | 同上 |
| axis_z_d2 | 20.3531 | +3.3515 | 同上；浅层为负增益 |

这是**真实数据探索结果**，不能写成正式神经元信号恢复验收。该表为 legacy pilot 留档，不与未来独立参考主表混用。

### 2.2 旧结论调整

| 旧表述/做法 | 现行判断 | 证据/原因 | 下一动作 |
|---|---|---|---|
| “真实数据 deep-signal 验证已完成” | **pilot 完成，正式成像验收未完成** | 单 seed；AVG25 含 input；ROI 为每张输出自己的分位数；无正式检测/独立单位统计 | S0→S1→S3 |
| SBR 保留 90–94.5% = 真实结构保留 | **只能作为探索的分布统计** | `plane_sbr_cnr()` 为每个输出重选 top10%/bottom50%；不同方法不是同一组解剖 ROI | 固定 ROI、线剖面与检测/结构指标 |
| 背景 Laplacian 低于 AVG25 = 比平均更好 | **仅提示高频减少** | 高频抑制也可能来自平滑；`bg_lap_std()` 在 Z/Y/X 带上算前两轴差分，背景选择随输出变，非标定各向同性 3D 噪声估计 | 固定背景与独立 repeat 差分；保真联合判定 |
| “过平滑假设被推翻” | **证据不足，需保真验证** | SBR/high-frequency proxy 不能充分证明细树突、线宽和结构不损失 | E4 检测/线剖面/残差/假阳性 |
| 两协议排序一致 = 独立复现 | **同数据/模型的一致性检查** | oracle 与 deepsignal 共用来源与参考；不是独立样本重复 | 补独立体/视野和统一协议 |
| 最优 family 随深度改变 = 条件策略已有效 | **策略动机，尚非部署效果** | 浅深赢家来自同一 pilot 结果；未做冻结策略对比 | S4 验证集选择、盲测，不用测试 oracle 挑赢家 |
| M1 能可靠识别坏 family | **仅本次数据的探索现象** | 7-family 单 seed 排序不足以推普适筛选能力 | 降级诊断，不作为固定部署 Gate |

### 2.3 保留的历史机制与复现材料

| 证据 | 仓库入口 | 在新方案中的用途 |
|---|---|---|
| 配对/几何同骨干矩阵与 QMask 3-seed | `ACES/runs/stage02_nf_qmask80ep_seed{3407,3408,3409}/`、`stage02_nf_matrix4_seed*/` | E2 候选与历史消融；Neurofinder 不解释为深度 |
| risk field / temporal risk | `ACES/profiling/`；`ACES/runs/m1_risk_field_green_raw/metrics/risk_families.csv` | 机制诊断，固定 risk 不等于最终成像 selector |
| micro-probe v2 与 oracle | `ACES/calibration/`；`ACES/runs/m2_oracle_3p_seed3407/u_probe_v2_3seed.json`、`u_oracle_3seed_full.json` | 开发拟合素材；v2 full Spearman=0.4667，LOO=0.3939，不能写成普遍高相关 |
| 80ep 关键 family 3-seed | `ACES/runs/m2_oracle_3p_80ep_seed340{7,8,9}/families/summary.json` | 跨预算失效/收敛边界；不以旧校准权重直接部署 |
| CIDC25 rgrid/sgrid 闭环 | `ACES/qmask/t4b_rgrid.py`、`t4b_sgrid.py` 与历史 runs | 机制补充/负结果；保留原实验语境 |
| B1/B2/B3 | `ACES/configs/b1_*`、`b2_*`、`b3_*` 与历史日志 | 温和正则候选/失败例；不强制重启无收益或发散模块 |
| 原 VALID/SN2N 复现 | `ACES/aces/eval_valid_original*`、`eval_sn2n_original*`、`ACES/logs/` | 基线管线复用；主数据需同参考/输入/尺度下补结果 |
| 定性与 zoom | `ACES/calibration/render_deep_zoom.py`、green `deepsignal/review/`、`deepsignal/zoom/` | review 素材与绘图接口；正式图须冻结区域、窗宽、尺度 |

旧日志的精确数字与负结果不删除。来源仅在服务器的结论标记“历史记录、待源数据复核”，不将日志声明升级为当前复现。

## 3. 数据与项目框架状态

| 项 | 当前状态 | 可复用资产 | 缺口/限制 |
|---|---|---|---|
| green/red raw repeat | 🟡 历史已确认格式 | 日志 A3-4a：plane-major 241z×25，raw 均值与 AVG 一致；green 主战场 | 原始 repeat 文件不在当前执行挂载，需服务器复核并产出 manifest |
| 3P 9-repeat | 🟡 历史开发资产 | 风险/probe/oracle JSON | 正式帧/空间/独立体划分与参考隔离待做 |
| obj_2mm | 🟡 AVG 示例可展示 | `ACES/example/` 和历史应用日志 | 无 repeat，不能构造独立噪声参考；伪影/有效深度规则待固定 |
| 成像模态与物理 spacing | ⏳ 正式冻结 | 历史记录/推导值 | 原始标定待核实；未核实时只用 z index |
| 固定采样/训练 | ✅ 实现存在，🟡 正式验收 | `qmask/`、`aces/`、`calibration/family_sampler.py` | 正式 split 与 validation-selected checkpoint 接口需核查/补齐 |
| 评估框架 | 🟡 已有基础 | `aces/evaluate_stage2.py` 与 deep-signal 脚本 | 固定 ROI、独立参考、结构/检测、噪声定义统一待做 |
| 完整体推理 | 🟡 pilot 可用 | `analyze_green_deepsignal.full_inference()` | 边界 tile 可能覆盖重叠并直接覆盖结果，须验证融合、接缝与尺度 |
| 统一绘图 | 🟡 多套旧入口 | `qmask/paperviz.py`、`calibration/plot_m2m3_figures.py` | 新 Fig ID、统一 manifest 和正式指标重绘待做 |
| 条件/adaptive 部署 | ⏳ 可选 | family/风险/probe 分析素材 | 不将 `exp_d_gap.py` 的离线选择分析视为完整部署策略实现 |

新 `ACES/evaluation/` / `ACES/sampling/` 如需增加，属于后续增量任务；本次未创建这些代码包。历史 `/data3` 路径按日志迁移为 `/ssd1`，当前会话无法核实服务器文件。

## 4. 新阶段完成情况总表（与 PLAN S0–S5 对齐）

| 阶段/Gate | 状态 | 现在拥有的证据 | 未满足的验收 | 下一交付 |
|---|---|---|---|---|
| S0 / G0 协议冻结 | ⏳ 首要任务 | 轴语义/指标基础与旧体检记录 | 独立参考、固定 ROI、主终点/δ、严格 split、统一选模尚未冻结 | E0 + split/ROI manifest + 开发样例 |
| S1 / G1 最小成像闭环 | 🟡 旧 pilot 可复用 | green 7-family、深层 zoom 与 legacy 指标 | 新协议下 Raw/经典滤波/方法基线/核心方案联合比较 | E1 pilot + Fig. 2/3 草图 |
| S2 / G2 方法与消融冻结 | 🟡 候选已具备 | sampling、历史同骨干/正则证据 | green 正式同协议消融与验证选定预算/checkpoint | E2 + 冻结最终配置 |
| S3 / G3 正式成像验证 | ⏳ 未通过 | 无新协议正式验收结果 | ≥3 seeds、足够独立样本、检测/结构/背景联合 Gate | E1/E4 正式表 + Fig. 2–3 |
| S4 / G4 条件策略 | ⏳ 可选 | 深度赢家不同、旧 gap 与跨预算失败 | 对最佳固定策略的额外成像收益/成本/保真 | E3 + Fig. 4 条件部分；失败则删模块 |
| S5 / G5 应用与论文收口 | ⏳ | red/obj 历史素材、原基线管线 | 跨体/模态证据、正式论文表、失败边界与图表复算 | E5 + Fig. 5 + Table 1/S1/S2 |
| E7 动态功能 | ➖ 数据条件触发 | Neurofinder benchmark 素材 | 独立动态数据/事件验证未形成主线 | 没有证据则不作功能结论 |

S3 不以 S4 通过为前提；固定方案可靠即可完成主线。S4 无额外成像收益时停止，而不是为了完整“风险→probe→policy”叙事继续扩实验。

## 5. 下一轮执行清单（按优先级）

| 优先级/任务 | 具体执行 | 验收/产物 | 状态 |
|---|---|---|---|
| P0-1 数据/参考 | 服务器核实 raw 顺序、模态、spacing、运动/漂移/噪声相关性；input/reference 不重叠 | `metadata.json`、`split_manifest.json`，参考帧不参与输入/权重更新 | ⏳ |
| P0-2 固定 ROI 与 Gate | 独立标注/参考冻结胞体、弱树突、背景；确定主终点、假阳性阈值、结构 δ | `roi_manifest.json`；记录阈值数值及验证集依据 | ⏳ |
| P0-3 评估统一 | 实现 fixed-ROI CNR/SNR/SBR、背景残余、结构/检测与独立参考 PSNR/full SSIM | 开发数据评估样例；legacy proxy 单独标记；定义一致 | ⏳ |
| P0-4 训练/选模隔离 | validation-selected checkpoint、归一化与 patch 支持不跨 split、部署设置冻结 | selection 元数据、split 审计；不以 final oracle 冒充 best | ⏳ |
| P1-1 核心 pilot | 先两个代表固定候选 qmask_xy/sn2n_xy_diag；补 Raw、经典滤波、原方法基线 | 同位置深层 zoom+残差+联合指标，不重开 7-family 大矩阵 | ⏳ |
| P2-1 消融 | 固定 backbone/loss/步数，只改关键配对；必要时仅加一个温和损失/正则候选 | E2 表与冻结方案，无收益模块移除 | ⏳ |
| P3-1 正式验证 | 冻结配置后 3 seeds + 独立测试；统计独立体/视野，seed 方差单列 | G3 联合验收、Fig. 2/3、Table 1 草表 | ⏳ |
| P4-1 条件增强 | 最佳固定 vs 简单分带/输入质量策略；有收益再 probe-adaptive | G4 实际成像指标与成本；无收益停止 | ➖ |
| P5-1 应用/论文 | 补独立体/通道/模态，重绘图与失败例 | Fig. 5、Table S1/S2，限定适用范围 | ⏳ |

**不再优先**：新风险阈值 sweep、大量 family 排名、仅为相关性补 oracle、旧失败 B3 复启、无目标加 epoch、RL/meta-learning、新大骨干。原始数据/GPU 不在当前环境，本轮没有将上述待办标为“运行中”。

## 6. 论文实验与绘图验收跟踪

| 图表 | 当前状态 | 已有可复用素材 | 新验收要求/下一动作 |
|---|---|---|---|
| Fig. 1 最终方法与隔离流程 | ⏳ | 配对代码和旧示意 | 方法冻结后画实际流程；可选模块不画成必经链 |
| Fig. 2 浅/中/深核心对比 | 🟡 旧素材 | green `deepsignal/review/`、`zoom/` | 独立参考、同位置/窗宽/比例尺、原方法基线、普通及弱信号区域 |
| Fig. 3 深度与保真联合图 | 🟡 旧曲线 | green `deepsignal/curves.csv`、`metrics.json` | 固定 ROI 重算；检测/结构、背景残余、CI、线剖面/残差 |
| Fig. 4 配对/条件消融 | 🟡 历史机制图 | 同骨干矩阵、M2/M3 旧图 | 先配对的最终成像贡献；条件策略只有 G4 通过才入主图 |
| Fig. 5 跨数据与失败边界 | ⏳ | red/obj 历史展示 | 独立样本与模态确认；obj 无参考标注；展示伪影/高 SNR 对照 |
| Table 1 正式比较 | ⏳ | 历史外部基线表 | 不能拼旧口径；同输入、参考、尺度与选择协议，检测/噪声/成本共同报告 |
| Table S1/S2 数据与配置 | ⏳ | 日志、configs、JSON | metadata/split/ROI/预算/独立单位/校准成本可追溯 |
| Fig. S1–S4 协议/保真/鲁棒性 | ⏳ | 旧体检、正则/失败素材 | reference noise、随机区域、seed、接缝、残差与完整失败例 |
| Fig. S5–S6 机制（可选） | 🟡 历史素材 | risk/probe、CIDC25、预算迁移 | 补源数据复核；围绕最终方法解释，不牵引新大矩阵 |
| Fig. 6 功能（扩展） | ➖ | 尚无正式功能验证 | 有独立动态数据再启动，否则删图/删功能主张 |

**绘图执行纪律**：先 review 切块目检再定量；同位置、同窗宽/colormap/尺度；≥300 dpi PNG + 曲线 PDF/SVG；逐样本 CSV/JSON 与 `figure_manifest.json`；不得在绘图阶段重选 test family、阈值、ROI 或 checkpoint。ROI/深度窗口与代表区选择规则先固定，不只挑最亮块。

## 7. 主要不足、风险与处置

| 不足/风险 | 当前影响 | 处置与关闭条件 |
|---|---|---|
| input/reference 重叠 | AVG25 PSNR 与 oracle 带相关噪声，不能证明独立恢复 | 不重叠帧参考与明确训练隔离，manifest 审计通过 |
| 输出分位数 ROI | SBR/CNR 的 signal/background 随方法变，可能循环论证 | 固定解剖 ROI；统一指标与弱结构标注 |
| 高频 proxy 被当作真实噪声 | 平滑也会降高频，旧结构保真结论过强 | 背景 repeat 残余、结构检测/线剖面/残差联合验收 |
| seed/patch 伪重复 | 多 patch/多 seed 不等于独立样本 | 记录 animal/volume/field，分层统计和样本不足限制 |
| 参数/数据/预算 regime 变化 | 旧 utility 权重或 risk 序不能直接迁移 | 对目标设置验证；无额外成像收益不保留 selector |
| 单帧与平均帧比较混杂 | 平均改变采集成本且参考含残余噪声 | 单帧方法同条件比较；平均单列参考/等成本实验 |
| 物理标定与动态语义 | z 深度/μm 或功能解释可能不成立 | metadata 核查，未确认用 index；repeat 不写成钙时序 |
| 推理尺度/边界覆盖 | 视觉接缝可能被局部最亮块掩盖 | 完整体推理审计、固定随机/边界 ROI、必要时重叠融合 |
| 创新定位尚未冻结 | 成像目标本身不是新方法创新 | 固定模型后核查文献，按消融/应用证据限定候选贡献 |

## 8. 文档体系与更新规则

| 文档 | 角色 |
|---|---|
| `PLAN.md` | 唯一现行研究方案；目标、阶段、协议、论文实验和图表要求 |
| `project_progress_status.md` | 本文档：现行 Gate 状态、证据与下一动作 |
| `PROJECT_LOG.md` | 原始实验流水，本次不改写或删除历史记录 |
| `README.md` | 仓库入口与现行主线导航；旧机制证据仅作地图 |
| `docs/archive/2026-10-12_pre_imaging_quality_replan/` | 重构前 PLAN/状态表的原样快照与归档说明 |
| 其余早期方案文件 | 历史背景，不作为并行现行指令 |

每个节点记录“任务→定性→定量→Gate 判定→证据路径→下一动作”。完成代码、pilot、正式验收分别标注；没有新证据不得沿用旧日期/旧勾选。后续实验必须先映射 PLAN E0–E7 与对应 Fig/Table，再占用训练预算。

## 9. 变更记录

| 日期 | 变更 | 依据 |
|---|---|---|
| 2026-10-12 | 原样归档旧状态；按深层成像目标重排创新与 S0–S5；保留 pilot 数字和失败证据，撤销其正式验收含义；新增论文图表跟踪、优先待办与限制 | 用户授权重构与归档；PLAN 新主线；代码/仓库 JSON/历史日志核对；本轮仅文档变更 |
