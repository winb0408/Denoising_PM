"""聚合两个 pilot 的报告与阈值敏感性，生成 qmask/docs/RESULTS.md。"""
from __future__ import annotations

from pathlib import Path

from .report import render_markdown


SENS_3P = """| tau_n | tau_s | effective (zyx) | d_min(z,y,x) | d_max(z,y,x) |
|---|---|---|---|---|
| 0.95 | 0.80 | zyx | 1,1,1 | 1,2,2 |
| 0.95 | 0.85 | zyx | 1,1,1 | 1,2,2 |
| 0.95 | 0.88 | zyx | 1,1,1 | 1,1,2 |
| 0.95 | 0.90 | .yx | 1,1,1 | None,1,1 |"""

SENS_NF = """| tau_n | tau_s | effective (zyx) | d_min(z,y,x) | d_max(z,y,x) |
|---|---|---|---|---|
| 0.90 | 0.85 | zyx | 1,1,1 | 5,3,4 |
| 0.90 | 0.90 | zyx | 1,1,1 | 2,2,2 |
| 0.95 | 0.85 | z.x | 2,2,2 | 5,3,4 |
| 0.95 | 0.90 | z.x | 2,2,2 | 2,2,2 |"""


def build_docs(docs_dir: str | Path, run_dirs: list[str | Path]) -> Path:
    docs_dir = Path(docs_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)
    run_dirs = [Path(d) for d in run_dirs]

    parts: list[str] = []
    parts.append("# QMask 结果（复核实验）\n")
    parts.append("方法思路见 `TECHNICAL.md`。所有新代码、配置、产物均在 `ACES/qmask/`，未改动原 ACES 代码。\n")
    parts.append("## 阈值敏感性（运行前用 Stage1 曲线复核）\n")
    parts.append("有效参与掩码的记号：`zyx`=三轴全开（≈VALID 共位 split）；`.` 表示该轴关闭（phase 固定 0）。\n\n### 3P repeat01\n" + SENS_3P + "\n\n### 神经finder proxy\n" + SENS_NF + "\n")
    parts.append("**关键观察**：两个数据集在默认阈值下都落在判定边界上；qmask 退化为端点，不是独立新几何。\n")

    for run_dir in run_dirs:
        parts.append("\n---\n\n")
        parts.append(render_markdown(run_dir, f"QMask Pilot — {run_dir.name}"))
        parts.append("\n")

    parts.append("""\
## 总体结论

1. **H2（数据驱动参与掩码优于两个端点）在 pilot 未获支持**：qmask 在 3P 上 ≈ iso3d、明显低于 valid；在神经finder 上由 promotion 得到 `z.x` 掩码，也未形成对两端点的稳定优势。
2. **Q 场作为诊断工具仍有价值**：Stage1 能测出 3P 的 z 结构连续性 `S_z(d1)=0.882` 与 xy 的差异，且阈值敏感性分析清晰显示 z 的处理是“少配/不配”，而不是“更大半径”。
3. **“连续椭球/半径”创新表述应降级**：本次复核支持把 ACES 定位从“新采样几何”改为“Q 场驱动的轴参与掩码/配对可用性剪枝”。
4. **下一步（若继续）**：把 3P 的 `tau_s` 阈值敏感性做成正式消融（`zyx` vs `.yx`），并做 block 级随机角集（逼近 VALID valid）验证差异来源；神经finder 需先修正 proxy-N 的 d=1 偏差，再谈时间轴参与。

## 复现
```bash
cd /data2/wjb/Denoising_PM
export PYTHONPATH=$PWD/ACES
PY=/data2/wjb/anaconda3/envs/threePM/bin/python
# 单测
$PY -m unittest discover -s ACES/qmask/tests -v
# 任一 pilot
CUDA_VISIBLE_DEVICES=1 $PY -m qmask.train --config ACES/qmask/configs/qmask_3p_pilot.yaml
$PY -m qmask.visualize --run-dir ACES/qmask/runs/qmask_3p_pilot_seed3407
```\
""")

    out = docs_dir / "RESULTS.md"
    out.write_text("\n".join(parts), encoding="utf-8")
    return out


def main() -> int:
    out = build_docs(
        "ACES/qmask/docs",
        [
            "ACES/qmask/runs/qmask_3p_pilot_seed3407",
            "ACES/qmask/runs/qmask_neurofinder_pilot_seed3407",
        ],
    )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
