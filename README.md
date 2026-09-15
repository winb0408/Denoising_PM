# Denoising_PM — ACES/qmask 自监督荧光去噪采样框架

> 本仓库是**项目快照**(代码 + 计划/状态文档 + 精选 run 证据),不含训练权重与大文件
> (`.pth/.tif/checkpoints` 等均不入库,见 `.gitignore`)。
> 面向审查者:先读本 README → 再读 `project_progress_status.md`(验收视角)→ 需要细节时按证据地图查 JSON。

## 1. 项目一句话

建立"**数据 → 可观测统计 → 配对几何决策**"的自监督荧光去噪采样框架(qmask):不再人工指定
VALID / SN2N 式固定采样几何,而是从原始数据的噪声/结构统计出发,逐轴决定自监督训练的
配对方式,并给出**可证伪**的判据科学检验。

## 2. 三条主张及当前验证状态

| # | 创新点 | 状态 | 一句话结论 |
|---|---|---|---|
| IC1 | 数据驱动逐轴配对几何(参与掩码 + 区间化几何) | 🔄 修订中 | 原判据(结构相关性门控/统一阈值)已被两轴闭环证伪;修订假说(几何收益由恒等捷径危害度即噪声级决定)证据累积中;判据工程实现(A3)未开始 |
| IC2 | 共位相位分裂(stride-1)采样 | ✅ 已验证 | 同骨干矩阵最优臂:qmask 80ep fg 22.14±0.33(3-seed),优于 valid/sn2n_xy/iso3d;五档对比证明采样几何本身值 ~2 dB |
| IC3 | 采样几何的可证伪科学检验(H3) | ✅ 闭环(结论=证伪) | 时间轴(rgrid)与空间轴(sgrid)双闭环:门控预测与实测最优几何反相关(Spearman≈−1);"漂移尺度阈值"不变量不跨轴迁移。负结果 + 修正假说本身是论文的可证伪支点 |

支撑轨道:抗平滑 B1 完成(降 Hessian 正则换锐度,wr2e-5 折中点)、B2 运行中、B3(前景加权 α=4)发散已终止待修;
指标体系 C1–C3 完成(PSNR/SSIM/SNR/SBR/std_ratio 三区域 + paperviz 统一渲染);
外部基线 E1/E2 完成(外部表为论文引用口径,内部表含双重选择偏置高 3.5 dB 仅作内部归因)。

## 3. 仓库导览

```
Denoising_PM/
├── README.md                        ← 本文件
├── project_progress_status.md       ← 验收视角总览(阶段结论 + 证据目录 + 修订轨迹)
├── ACES_Plan_v2_20260901.md         ← 主技术计划(§8 修改小结 #1–#24 逐条可溯)
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
| `sampling.py` | 共位相位分裂采样 + 逐轴参与掩码(IC2 的核心实现) |
| `geometry.py` / `geometry_sweep.py` | 配对几何定义与网格扫描 |
| `tau_sweep.py` | 结构相关性 Q/τ 门控扫描(Q-1:掩码决策脆弱性证据) |
| `shortcut_risk.py` | 恒等捷径危害度分析(A1 判据重构) |
| `t4b_rgrid.py` | 时间轴位移网格(d∈{1..64})训练 + H3 闭环分析 |
| `t4b_sgrid.py` | 空间轴位移网格(s∈{0..16},含奇偶相位交换处理)训练 + 空间闭环 |
| `train.py` / `evaluate.py` | 统一训练/评估入口 |
| `evaluate_stage2.py`(在 `aces/`) | 三区域 PSNR/SSIM/SNR/SBR/std_ratio 指标 |
| `paperviz.py` | 论文级图表统一渲染 |
| `backfill_eval_metrics.py` | 历史指标文件补列 |

## 5. 精选证据地图(结论 ↔ 文件)

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

## 6. 建议阅读顺序(给审查者)

1. `project_progress_status.md` §0 一页速览 —— 主线、三条创新点状态;
2. `project_progress_status.md` §2 —— 每个创新点的证据链与修订轨迹;
3. 本 README §5 证据表 —— 逐条打开 JSON 核对数字;
4. `ACES_Plan_v2_20260901.md` §8 修改小结 —— 每次实验决策的完整时间线(#1–#24);
5. 需要看方法实现时:`ACES/qmask/sampling.py`(IC2)、`t4b_rgrid.py`/`t4b_sgrid.py`(IC3 闭环)。

## 7. 复现环境

- Python 3.10+(实际环境:conda env `threePM`),PyTorch + CUDA(A6000);
- 训练入口:`PYTHONPATH=ACES python -m qmask.train --config <yaml>`,seed 固定 3407/3408/3409;
- 数据集:CIDC25(nf/3P 两通道)、EPFL 3P(外部基线);数据与权重不入库。

## 8. CIDC25 数据使用条款约束(实验设计硬约束)

Validation 通道(F0–F3)**仅允许**用于评估/模型选择;权重更新只允许来自 Training
通道(A1/B1/C2/D2)。所有已报告数字遵守该条款。

## 9. 已知遗留清单(截至 2026-09-15)

- A3:修订判据(捷径危害度)写入几何决策 + 单元验证 + 3ep pilot —— 未开始;
- R10:rgrid / sgrid / B1 wr2e-5 的 3-seed 重复 —— 待 GPU;
- B2 两臂(梯度损失)验收 —— 运行中(预计 9/16);B3 需降 α/平滑掩码后重设计;
- E3:qmask 最终配置入外部基线表 —— 待 A3 定稿。
