"""Risk field 可视化（M1，论文动机/结论图）。

输入：多个 run 目录下的 ``metrics/risk_field.csv``（由 ``profiling.risk_field``
生成），按数据集分组渲染对比图。所有图落盘 ``figures/``，供论文正文/补充材料
选用；本模块只出图不做解释（读图定性验收由用户人工完成——GLM-5.3 无多模态）。

图集：
  1. risk_curves_*.png   — R_N / R_S / R_I 随 offset 距离的曲线（按轴分色），
                           证明三个风险规律不同（单一 Q 不可替代 → 方法动机）。
  2. risk_cross_data.png — 各数据集 R_S 沿 z（或时间）轴衰减对比（跨数据一致性）。
  3. risk_families.png   — family 级 r_base 条形图（采样族风险地图）。
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

AXIS_COLORS = {"x": "#1f77b4", "y": "#2ca02c", "z": "#d62728", "diag": "#7f7f7f", "other": "#bbbbbb"}
RISK_LABELS = {"rn": r"$R_N$ (noise)", "rs": r"$R_S$ (structure)", "ri": r"$R_I$ (shortcut)"}


@dataclass
class DatasetRisk:
    """一个数据集的 risk_field 表。"""

    name: str
    rows: list[dict[str, str]]

    @classmethod
    def load(cls, run_dir: str | Path, name: str) -> "DatasetRisk":
        path = Path(run_dir) / "metrics" / "risk_field.csv"
        with open(path, encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        if not rows:
            raise ValueError(f"{path} 为空")
        return cls(name=name, rows=rows)

    def axis_of(self, row: dict[str, str]) -> str:
        z, y, x = int(row["offset_voxel_z"]), int(row["offset_voxel_y"]), int(row["offset_voxel_x"])
        if z == 0 and y == 0:
            return "x"
        if z == 0 and x == 0:
            return "y"
        if y == 0 and x == 0:
            return "z"
        if abs(z) == abs(y) == abs(x):
            return "diag"
        return "other"

    def distance_of(self, row: dict[str, str]) -> float:
        z, y, x = int(row["offset_voxel_z"]), int(row["offset_voxel_y"]), int(row["offset_voxel_x"])
        return math.sqrt(z * z + y * y + x * x)


def plot_risk_curves(ds: DatasetRisk, out_path: Path, title: str) -> None:
    """单数据集：R_N/R_S/R_I vs 体素距离（按轴分色）。"""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, key in zip(axes, ("rn", "rs", "ri")):
        for axis, color in AXIS_COLORS.items():
            pts = [
                (ds.distance_of(r), float(r[key]))
                for r in ds.rows
                if ds.axis_of(r) == axis and axis in ("x", "y", "z", "diag")
            ]
            if not pts:
                continue
            pts.sort()
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", color=color,
                    label={"x": "x (in-plane)", "y": "y (in-plane)",
                           "z": "z (depth)", "diag": "diagonal"}[axis],
                    markersize=4, linewidth=1.2)
        ax.set_xlabel("offset distance (voxel)")
        ax.set_title(RISK_LABELS[key])
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("risk")
    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


@dataclass
class TemporalRisk:
    """一个数据集的 risk_field_temporal 表（dt 扫描，由 profiling.risk_field_temporal 生成）。"""

    name: str
    rows: list[dict[str, str]]

    @classmethod
    def load(cls, run_dir: str | Path, name: str) -> "TemporalRisk":
        path = Path(run_dir) / "metrics" / "risk_field_temporal.csv"
        with open(path, encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        if not rows:
            raise ValueError(f"{path} 为空")
        return cls(name=name, rows=rows)


def _temporal_series(rows: list[dict[str, str]], key: str) -> tuple[list[float], list[float]]:
    pts = sorted((int(r["dt"]), float(r[key])) for r in rows)
    return [p[0] for p in pts], [p[1] for p in pts]


def plot_temporal_curves(ds: TemporalRisk, out_path: Path) -> None:
    """单数据集：R_N/R_S/R_I vs 时间间隔 dt（三规律分离 → 动机图）。"""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, key in zip(axes, ("rn", "rs", "ri")):
        xs, ys = _temporal_series(ds.rows, key)
        ax.plot(xs, ys, "o-", color="#4c72b0", markersize=4, linewidth=1.2)
        ax.set_xlabel("temporal offset dt (frame)")
        ax.set_title(RISK_LABELS[key])
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("risk")
    fig.suptitle(f"{ds.name}: three risks vs temporal offset", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_temporal_cross(dss: list[TemporalRisk], out_path: Path) -> None:
    """跨数据集：R_S/R_I 随 dt 演化对比（规律一致性，补充材料图）。"""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    colors = plt.cm.tab10.colors
    for idx, ds in enumerate(dss):
        for ax, key, ttl in (
            (axes[0], "rs", r"$R_S$ vs dt (structure decorrelation)"),
            (axes[1], "ri", r"$R_I$ vs dt (shortcut risk)"),
        ):
            xs, ys = _temporal_series(ds.rows, key)
            ax.plot(xs, ys, "o-", color=colors[idx % 10], label=ds.name, markersize=4)
            ax.set_title(ttl)
    for ax in axes:
        ax.set_xlabel("dt (frame)")
        ax.set_ylabel("risk")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_cross_data(dss: list[DatasetRisk], out_path: Path) -> None:
    """跨数据集：R_S 沿 z 轴衰减对比（规律一致性）。"""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    colors = plt.cm.tab10.colors
    for idx, ds in enumerate(dss):
        zs = sorted(
            (abs(int(r["offset_voxel_z"])), float(r["rs"]))
            for r in ds.rows
            if ds.axis_of(r) == "z"
        )
        if zs:
            axes[0].plot([p[0] for p in zs], [p[1] for p in zs], "o-",
                         color=colors[idx % 10], label=ds.name, markersize=4)
        ris = sorted(
            (abs(int(r["offset_voxel_z"])), float(r["ri"]))
            for r in ds.rows
            if ds.axis_of(r) == "z"
        )
        if ris:
            axes[1].plot([p[0] for p in ris], [p[1] for p in ris], "o-",
                         color=colors[idx % 10], label=ds.name, markersize=4)
    axes[0].set_title(r"$R_S$ vs |dz| (structure anisotropy)")
    axes[1].set_title(r"$R_I$ vs |dz| (shortcut risk)")
    for ax in axes:
        ax.set_xlabel("|dz| (voxel)")
        ax.set_ylabel("risk")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_families(ds: DatasetRisk, run_dir: Path) -> None:
    """family 级 r_base 条形图。"""
    path = Path(run_dir) / "metrics" / "risk_families.csv"
    if not path.exists():
        return
    with open(path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    rows.sort(key=lambda r: float(r["r_base_mean"]))
    names = [r["family_id"] for r in rows]
    means = [float(r["r_base_mean"]) for r in rows]
    maxs = [float(r["r_base_max"]) for r in rows]
    fig, ax = plt.subplots(figsize=(8, max(4, 0.3 * len(rows))))
    ypos = range(len(rows))
    ax.barh(ypos, means, color="#4c72b0", label="mean")
    ax.barh(ypos, [m - v for m, v in zip(maxs, means)], left=means, color="#c44e52",
            alpha=0.6, label="max−mean")
    ax.set_yticks(list(ypos))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel(r"$R_{base}$")
    ax.set_title(f"{ds.name}: sampling family risk landscape")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(Path(run_dir) / "figures" / "risk_families.png", dpi=150)
    plt.close(fig)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="risk field 论文图渲染")
    parser.add_argument("--datasets", nargs="+", default=[],
                        help="name=run_dir 形式，可多个（如 3P=.../m1_risk_field_3p_seed3407）")
    parser.add_argument("--temporal", nargs="+", default=[],
                        help="时间轴数据集 name=run_dir（metrics/risk_field_temporal.csv）")
    parser.add_argument("--out-dir", default=None, help="跨数据集对比图输出目录（默认第一个 run 的 figures/）")
    args = parser.parse_args()
    if not args.datasets and not args.temporal:
        parser.error("至少提供 --datasets 或 --temporal 之一")

    dss: list[DatasetRisk] = []
    for spec in args.datasets:
        name, run_dir = spec.split("=", 1)
        ds = DatasetRisk.load(run_dir, name)
        dss.append(ds)
        fig_dir = Path(run_dir) / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        plot_risk_curves(ds, fig_dir / "risk_curves.png", f"{name}: three risks vs offset distance")
        plot_families(ds, Path(run_dir))
        print(f"[plot] {name}: risk_curves.png + risk_families.png -> {fig_dir}")

    tdss: list[TemporalRisk] = []
    for spec in args.temporal:
        name, run_dir = spec.split("=", 1)
        ds = TemporalRisk.load(run_dir, name)
        tdss.append(ds)
        fig_dir = Path(run_dir) / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        plot_temporal_curves(ds, fig_dir / "risk_curves_temporal.png")
        print(f"[plot] {name}: risk_curves_temporal.png -> {fig_dir}")

    out_dir = Path(args.out_dir) if args.out_dir else Path(
        (args.datasets or args.temporal)[0].split("=", 1)[1]
    ) / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    if dss:
        plot_cross_data(dss, out_dir / "risk_cross_data.png")
        print(f"[plot] cross-data -> {out_dir / 'risk_cross_data.png'}")
    if tdss:
        plot_temporal_cross(tdss, out_dir / "risk_cross_data_temporal.png")
        print(f"[plot] temporal cross-data -> {out_dir / 'risk_cross_data_temporal.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
