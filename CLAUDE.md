# Denoising_PM 工作约定

## 大图必须先切块再读（强制）

本项目生成的定性对比图（如 `figures/qualitative_comparison.png`）通常为
2000×2000+ 像素、4–7 MB，**直接 Read 会超出多模态 API 限制并卡死会话**。
已有前科：Codex 读 3p pilot 对比图（2322×2393, 6.7MB）导致对话挂起。

规则：

1. **禁止直接 Read 任何 >1200 px 或 >3.5 MB 的图片。** 读图前先用
   `ls -l` 或 PIL 检查尺寸/大小，不确定就查该目录的 `review/manifest.json`。
2. **定性分析只读 review 切块**：`figures/review/qualitative_overview.png`
   （≤1200×1200）和 `figures/review/tiles/row_*.png`（每张 ≤3.5MB/2）。
   先看 overview，再逐行看 tiles。
3. **run 目录若没有 `review/`，先跑切块再分析**（幂等，几秒钟）：
   ```
   PYTHONPATH=ACES python -m qmask.preview --run-dir <run_dir>
   ```
   `qmask.visualize` 已挂钩：`render_qualitative` 落盘大图后会自动生成 review 切块。
4. 若需要分析其他来源的大图（非 qmask run 目录），同样先切块：
   用 `qmask/preview.py` 的 `make_review_previews` 思路，或临时写脚本切成
   ≤1200 px 的行块再逐块读。
5. 对比分析流程固定为：**先定性（读切块确认无塌缩/伪影）→ 再定量（PSNR/gate 表）**。
   顺序不能反。

## 环境速查

- Python: `/data2/wjb/anaconda3/envs/threePM/bin/python`
- 运行模块（仓库根目录）：`PYTHONPATH=ACES python -m qmask.<module>` 或 `python -m aces.<module>`
- GPU: A6000，`CUDA_VISIBLE_DEVICES` 选择；seed 3407
