"""Q-1 τ 扫描：参与掩码对 (tau_n, tau_s) 的稳定性图（ACES Plan v2 §2.2 Q-1）。

对每个数据集的 stage1 pairability 曲线，在 tau_n × tau_s 网格上重跑 derive_qmask，
输出：
  * 每个组合的参与掩码 {z,y,x} 与逐轴 margin（N(d=1)-tau_n, S(d=1)-tau_s）；
  * 掩码翻转边界（= 各轴 d=1 处的 N/S/Q 实测值）；
  * 默认阈值组合离最近翻转的 τ-距离。

用法（仓库根目录）：
    PYTHONPATH=ACES python -m qmask.tau_sweep
"""
from __future__ import annotations

import csv
import json
from itertools import product
from pathlib import Path

from .geometry import AXIS_ORDER, derive_qmask

TAU_N_GRID = (0.85, 0.90, 0.95, 0.97)
TAU_S_GRID = (0.80, 0.85, 0.88, 0.90, 0.92)
TAU_Q = 0.60

DATASETS = {
    "3p_repeat": {
        "stage1_run_dir": "/data2/wjb/Denoising_PM/ACES/runs/stage01_3p_repeat_pairability_seed3407",
        "region": "foreground",
        "noise_source": "repeat",
        "default": (0.95, 0.85),
    },
    "neurofinder_proxy": {
        "stage1_run_dir": "/data2/wjb/Denoising_PM/ACES/runs/stage01_neurofinder_00_00_proxy_pairability_seed3407",
        "region": "foreground",
        "noise_source": "proxy",
        "default": (0.95, 0.85),
    },
}

OUT_DIR = Path("/data2/wjb/Denoising_PM/ACES/qmask/runs/tau_sweep_20260901")


def _load_rows(stage1_run_dir: str) -> list[dict[str, str]]:
    csv_path = Path(stage1_run_dir) / "metrics" / "pairability_curves.csv"
    with csv_path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _mask_str(geo) -> str:
    # 展示提升后的有效掩码（与训练实际使用的 effective_participate 一致）
    eff = geo.effective_participate()
    return "".join("1" if eff.get(a) else "0" for a in AXIS_ORDER)


def _min_tau_distance(margins_n: dict[str, float], margins_s: dict[str, float]) -> float:
    """默认组合到最近掩码翻转的 τ-距离（各轴 N/S margin 的最小值，截到 0 以上）。"""
    vals = [abs(m) for m in list(margins_n.values()) + list(margins_s.values())]
    return min(vals) if vals else float("inf")


def sweep_dataset(name: str, spec: dict) -> dict:
    rows = _load_rows(spec["stage1_run_dir"])
    grid = {}
    table_lines = [
        f"| tau_n \\ tau_s | {' | '.join(f'{ts:.2f}' for ts in TAU_S_GRID)} |",
        "|---|" + "---:|" * len(TAU_S_GRID),
    ]
    for tau_n, tau_s in product(TAU_N_GRID, TAU_S_GRID):
        geo = derive_qmask(
            rows, region=spec["region"], noise_source=spec["noise_source"],
            tau_n=tau_n, tau_s=tau_s, tau_q=TAU_Q,
        )
        mask = _mask_str(geo)
        margins_n = {a: (geo.axes[a].n_d1 - tau_n) if geo.axes[a].n_d1 is not None else None for a in AXIS_ORDER}
        margins_s = {a: (geo.axes[a].s_d1 - tau_s) if geo.axes[a].s_d1 is not None else None for a in AXIS_ORDER}
        grid[f"tau_n={tau_n:.2f},tau_s={tau_s:.2f}"] = {
            "mask_zyx": mask,
            "margins_N_d1": margins_n,
            "margins_S_d1": margins_s,
            "reasons": {a: geo.axes[a].reason for a in AXIS_ORDER},
        }
    # 表格（在 grid 已填好后重放一次生成行）
    for tau_n in TAU_N_GRID:
        cells = [f"tau_n={tau_n:.2f}"]
        for tau_s in TAU_S_GRID:
            entry = grid[f"tau_n={tau_n:.2f},tau_s={tau_s:.2f}"]
            m = entry["mask_zyx"]
            marker = "" if (tau_n, tau_s) == spec["default"] else " "
            cells.append(f"{m}{marker}")
        table_lines.append("| " + " | ".join(cells) + " |")

    # 默认组合的逐轴 margin 与 d=1 翻转边界
    dn, ds = spec["default"]
    geo_default = derive_qmask(
        rows, region=spec["region"], noise_source=spec["noise_source"],
        tau_n=dn, tau_s=ds, tau_q=TAU_Q,
    )
    boundaries = {
        a: {"N_d1": geo_default.axes[a].n_d1, "S_d1": geo_default.axes[a].s_d1, "Q_d1": geo_default.axes[a].q_d1}
        for a in AXIS_ORDER
    }
    margins_n = {a: (boundaries[a]["N_d1"] - dn) if boundaries[a]["N_d1"] is not None else None for a in AXIS_ORDER}
    margins_s = {a: (boundaries[a]["S_d1"] - ds) if boundaries[a]["S_d1"] is not None else None for a in AXIS_ORDER}
    summary = {
        "dataset": name,
        "stage1_run_dir": spec["stage1_run_dir"],
        "noise_source": spec["noise_source"],
        "default_tau": {"tau_n": dn, "tau_s": ds, "tau_q": TAU_Q},
        "default_mask_zyx": _mask_str(geo_default),
        "default_margins_N_d1": margins_n,
        "default_margins_S_d1": margins_s,
        "min_tau_distance_to_flip": _min_tau_distance(margins_n, margins_s),
        "d1_flip_boundaries": boundaries,
        "grid": grid,
        "table_markdown": "\n".join(table_lines),
    }
    return summary


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_masks = {}
    report_lines = ["# Q-1 τ 扫描：参与掩码稳定性（2026-09-01）", ""]
    for name, spec in DATASETS.items():
        summary = sweep_dataset(name, spec)
        (OUT_DIR / f"{name}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        report_lines += [
            f"## {name}（noise_source={spec['noise_source']}）",
            "",
            f"默认 (tau_n=0.95, tau_s=0.85) → mask zyx={summary['default_mask_zyx']}，"
            f"离最近翻转 τ-距离={summary['min_tau_distance_to_flip']:.4f}",
            "",
            "掩码表（zyx，1=参与配对；`*`=默认组合）：",
            "",
            summary["table_markdown"],
            "",
            "d=1 翻转边界（掩码在该值处翻转）：",
            "",
            "| axis | N(d=1) | S(d=1) | Q(d=1) |",
            "|---|---:|---:|---:|",
        ]
        for a in AXIS_ORDER:
            b = summary["d1_flip_boundaries"][a]
            fmt = lambda v: "n/a" if v is None else f"{v:.4f}"
            report_lines.append(f"| {a} | {fmt(b['N_d1'])} | {fmt(b['S_d1'])} | {fmt(b['Q_d1'])} |")
        report_lines.append("")
        all_masks[name] = {"default_mask": summary["default_mask_zyx"],
                           "min_tau_distance": summary["min_tau_distance_to_flip"]}
    (OUT_DIR / "REPORT.md").write_text("\n".join(report_lines), encoding="utf-8")
    (OUT_DIR / "summary.json").write_text(json.dumps(all_masks, indent=2), encoding="utf-8")
    print(f"saved {OUT_DIR / 'REPORT.md'}")
    for name, m in all_masks.items():
        print(f"{name}: default mask={m['default_mask']}  min tau-distance to flip={m['min_tau_distance']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
