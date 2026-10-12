# M2/M3 论文图审核索引（2026-10-09 生成，全部由既有数据零新训练复算）

生成脚本：`ACES/calibration/plot_m2m3_figures.py`（`PYTHONPATH=ACES python -m calibration.plot_m2m3_figures`，幂等可重跑）

**显示格式（2026-10-09 按用户反馈更新）**：两张定性图已从灰度改为 **turbo 伪彩 + (1.0, 99.9) 分位窗**，与 `ACES/qmask/paperviz.py`（paperviz 系列）格式完全一致。生成脚本：`ACES/calibration/render_qualitative_cidc.py` / `render_qualitative_3p.py`。

## 定性图（去噪结果直观对比，供人工审阅）

| 图 | 内容 | 审阅方式 |
|---|---|---|
| `fig_q1_cidc25_qualitative.png` | CIDC25 定性面板：3 行（F1 Validation 高亮前景 patch，64×128×128 取 z 中切片）× 5 列（输入 F1 \| GT F0均值 \| dt=1 \| dt=8 \| dt=64）。dt=1 为最差臂可见过度平滑/结构丢失，dt=64（稳健最优）最接近 GT | 逐行读 `review/cidc25_row_{1,2,3}.png`（1100×220，符合读图限制）|
| `fig_q2_3p_qualitative.png` | 3P 定性面板：3 行（repeat 栈 z 三分带各取 AVG-9 最亮 patch，64×128×128 取 z 中切片）× 6 列（输入 frame 0 \| AVG-9 伪 GT \| qmask_xy \| valid_corners \| sn2n_xy_diag \| axis_z_d2，各列标注 U_oracle）。4 家族均明显去噪；valid_corners(+7.67)/qmask_xy(+7.53) 结构保留最好，与 U_oracle 排序一致 | 逐行读 `review/3p_row_{1,2,3}.png`（1170×220） |

| 图 | 对应创新点/假设 | 核心信息 | 数据源 |
|---|---|---|---|
| `fig_c1_probe_vs_oracle.png` | H4/H5：Exp C 主图 | U_v2(LOO) vs U_oracle 3-seed 均值（误差棒=±std/√3）；可分对 12/13、top/bottom AUC 0.89 | `m2_oracle_3p_seed3407/u_probe_v2_3seed.json` + 3 seed oracle |
| `fig_c2_feature_regime.png` | H4：跨 regime 符号矛盾 | 10 个 probe 特征与 oracle 的 Spearman，3P 空间（蓝）vs CIDC25 时间（红）并排——符号矛盾即 regime-conditioning 证据 | `m2_micro_probe_3p_seed3407/micro_probe.csv` + `m2_micro_probe_cidc25_seed3407/micro_probe_temporal.csv` |
| `fig_c3_reverse_validation.png` | H4：反向验证 | CIDC25 oracle vs dt（均值±std）+ 3P 权重迁移的 U 曲线：迁移版把 dt=1（最差臂/捷径陷阱）选为最优 | 同上 + `stage01_cidc25_rgrid_seed340{7,8,9}` |
| `fig_c4_budget_transfer.png` | H4/H5：Exp D 预算迁移失效 | 20ep vs 80ep 四家族 mean±std：sn2n(adaptive 臂) 饱和 Δ+0.02，qmask/valid 慢热 +0.5dB 反超 | `m2_oracle_3p_seed340{7,8,9}` + `m2_oracle_3p_80ep_seed340{7,8,9}` |
| `fig_d1_gap_bars.png` | H5：Exp D 主图 | 各臂 Oracle Gap 条形图（3P held-out / CIDC25 3-seed 双 panel）；adaptive 蓝、fixed 灰 | `m2_oracle_3p_seed3407/exp_d_gap.json` |
| `fig_t1_dt_seed_fragility.png` | 统计纪律 | CIDC25 per-seed PSNR vs dt：dt=8 脆弱（s3409 后期发散）、dt=64 稳健最优 | `stage01_cidc25_rgrid_seed340{7,8,9}` |

注：文件名以实际为准（`fig_c1_probe_vs_oracle.png`）。M1 图形（risk field）见各自 run 的 `figures/`。
