# Denoising_PM — 面向深层有效信号恢复的自监督荧光去噪

> 本仓库是**项目快照**(代码 + 计划/状态文档 + 精选 run 证据),不含训练权重与大文件
> (`.pth/.tif/checkpoints` 等均不入库,见 `.gitignore`)。
> 面向审查者:先读本 README → 再读 `project_progress_status.md`(验收视角)→ 需要细节时按证据地图查 JSON。

> **现行方案更新：2026-10-12**。以算法后成像质量为最终评判；先读 `PLAN.md` → `project_progress_status.md`。风险/probe/adaptive 为支撑或可选模块，旧实验保留。

## 1. 项目一句话

利用可靠配对和自监督去噪，让深层弱神经元/精细结构更可辨、背景噪声减弱，并保持真实结构和信号。研究顺序：**先证明图像变好 → 再验证配对贡献 → 最后决定是否需要自适应选择**。

正式成功依赖独立参考、固定 ROI、结构/检测、背景与保真的联合证据；PSNR、风险排名、oracle gap 或单张视觉改善不能代替正式成像验收。

## 2. 现行贡献与验证状态

| 项目 | 状态 | 下一验收 |
|---|---|---|
| 深层弱信号自监督恢复（核心候选） | 🟡 真实 green 单 seed pilot 可复用 | 独立参考/固定 ROI 下的成像与保真联合验证 |
| 各向异性配对/共位相位分裂（方法基础） | 🟡 已有实现和历史消融 | 主成像数据上同骨干、同协议消融 |
| 深度/输入质量条件策略（可选增强） | ⏳ | 优于验证选定的最佳固定策略，计入成本 |
| risk/probe 的数据与预算依赖 | 🟡 历史支撑/失败边界 | 按最终方法需要复用，不再作主线前置 Gate |

当前处于 **S0 协议冻结 → S1 最小成像闭环** 的准备阶段。旧“完成/通过”按历史口径留档，正式成像验收尚未完成；详见状态表。

## 3. 仓库导览

```
Denoising_PM/
├── README.md                        ← 本文件
├── project_progress_status.md       ← 验收视角总览(阶段结论 + 证据目录 + 修订轨迹)
├── PLAN.md                         ← 唯一现行方案（目标、阶段、论文实验、绘图验收）
├── PROJECT_LOG.md                  ← 历史与后续实验流水
├── docs/archive/2026-10-12_pre_imaging_quality_replan/ ← 重构前方案/状态原样归档
├── ACES_Plan_v2_20260901.md         ← 历史技术计划
├── ACES/
│   ├── aces/                        ← 外部基线复现与评估(VALID/SN2N 原管线封装、指标体系)
│   ├── qmask/                       ← 本项目核心模块(见 §4)
│   ├── configs/                     ← B1/B2/B3 等 stage02 实验配置
│   ├── logs/                        ← 关键训练日志节选
│   └── runs/                        ← 精选 run 证据(仅 JSON/CSV,无权重/图像大文件)
└── (其余 *.md 为历史技术计划与团队论文分析,供背景阅读)
```

## 4. qmask 模块地图(`ACES/qmask/`)

| 模块 | 作用 |
|---|---|
| `sampling.py` | 共位相位分裂采样 + 逐轴参与掩码(配对基础实现) |
| `geometry.py` / `geometry_sweep.py` | 配对几何定义与网格扫描 |
| `tau_sweep.py` | 结构相关性 Q/τ 门控扫描(Q-1:掩码决策脆弱性证据) |
| `shortcut_risk.py` | 恒等捷径危害度分析(A1 判据重构) |
| `t4b_rgrid.py` | 时间轴位移网格(d∈{1..64})训练 + H3 闭环分析 |
| `t4b_sgrid.py` | 空间轴位移网格(s∈{0..16},含奇偶相位交换处理)训练 + 空间闭环 |
| `train.py` / `evaluate.py` | 统一训练/评估入口 |
| `evaluate_stage2.py`(在 `aces/`) | 三区域 PSNR/SSIM/SNR/SBR/std_ratio 指标 |
| `paperviz.py` | 论文级图表统一渲染 |
| `backfill_eval_metrics.py` | 历史指标文件补列 |

## 5. 历史精选证据地图(探索结论 ↔ 文件)

以下保留原探索口径；不同参考/尺度/选模协议不能拼入新正式主表。

| 结论 | 证据文件(仓库内) |
|---|---|
| qmask 80ep 最优臂 22.14±0.33(3-seed) | `ACES/runs/stage02_nf_qmask80ep_seed{3407,3408,3409}/qmask/metrics/eval_best_metrics_aligned.json` |
| 4 臂×3-seed 同骨干矩阵 | `ACES/runs/stage02_nf_matrix4_seed{3407,3408,3409}/{qmask,valid,sn2n_xy,iso3d}/metrics/` |
| 采样几何五档 ~2 dB 归因 | `ACES/runs/nf_matrix4_summary.json` |
| H3 证伪闭环(Spearman≈−1,恒等捷径机制) | `ACES/runs/stage01_cidc25_rgrid_seed3407/metrics/h3_closure.json` |
| A1 静态 shortcut-risk 判据否定 | `ACES/runs/stage01_cidc25_rgrid_seed3407/metrics/shortcut_risk_analysis.json` |
| 空间轴闭环(位移收益随噪声级变化,无统一阈值) | `ACES/runs/stage01_cidc25_sgrid_seed3407/metrics/sgrid_closure.json` |
| 第一切片:r_t 随噪声级单调收缩(8→3→0) | `ACES/runs/stage01_cidc25_gt_pairability_seed3407/metrics/` |
| 时间轴网格逐档指标(d1–d64) | `ACES/runs/stage01_cidc25_rgrid_seed3407/d{1,2,3,4,6,8,16,32,64}/metrics/eval_best_F{1,2,3}.json` |
| 空间轴网格逐档指标(s0–s16) | `ACES/runs/stage01_cidc25_sgrid_seed3407/s{0,1,2,4,8,16}/metrics/` |
| B1 降正则消融(锐度-PSNR 代价曲线) | `ACES/runs/stage02_nf_b1_wr{0,2e-5,5e-5}_seed3407/` |
| B3 前景加权发散诊断(α=4) | `ACES/runs/stage02_nf_b3_fg4_seed3407/`(保留作为诊断记录) |
| 历史指标补列 | `ACES/runs/metrics_backfill_summary.json` |

## 6. 建议阅读顺序

1. `PLAN.md` §1–2：研究目标、主线层级与候选贡献；
2. `PLAN.md` §4–7：数据/独立参考、联合验收与 S0–S5 执行；
3. `PLAN.md` §8–9：论文实验结构及必须遵守的图表要求；
4. `project_progress_status.md`：当前证据、缺口、下一动作与图表状态；
5. `PROJECT_LOG.md` 和本文件历史证据地图：核查过程及探索结果。

## 7. 复现环境

- Python 3.10+(实际环境:conda env `threePM`),PyTorch + CUDA(A6000);
- 训练入口:`PYTHONPATH=ACES python -m qmask.train --config <yaml>`,seed 固定 3407/3408/3409;
- 数据集:CIDC25(nf/3P 两通道)、EPFL 3P(外部基线);数据与权重不入库。

## 8. CIDC25 数据使用条款约束(实验设计硬约束)

Validation 通道(F0–F3)**仅允许**用于评估/模型选择;权重更新只允许来自 Training
通道(A1/B1/C2/D2)。所有已报告数字遵守该条款。

## 9. 当前优先待办（2026-10-12）

- 冻结 input/reference 与 development/validation/test 隔离、ROI/主终点/结构非劣界；
- 统一评估与 validation-selected checkpoint，旧输出分位数 ROI 指标单列 legacy；
- 在 green 主数据构建 Raw/经典滤波/原 VALID/原 SN2N/核心方案的最小成像闭环；
- 做少量关键配对消融、3 seeds 与独立样本正式验证，按 Fig. 1–5/Table 1 组织图表；
- 条件策略仅在带来额外深层成像收益时进入最终方法。具体任务及验收见现行 PLAN/状态表。
