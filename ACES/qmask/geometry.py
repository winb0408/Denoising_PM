"""QMask 几何引擎：从 Stage1 N/S/Q 曲线导出轴参与掩码。

核心修正（对照 ACES 原方案 §3.3）：
    - 原方案取最远仍合法的距离 ``r_a = max{d : Q_a(d) > tau}``，在 Q 随距离
      单调衰减时反而选中结构退化最严重的配对；
    - QMask 改为 ``d_min = min{d : N_a(d) >= tau_n}``（噪声恰好解相关的最短
      距离）与 ``d_max = max{d : S_a(d) >= tau_s}``（结构仍匹配的最长距离），
      并把 v1 的相位分裂固定为 1-voxel 偏移，因此轴参与 = 在 d=1 处同时满足
      N/S/Q 三个阈值。无法参与的轴在采样中被固定 phase=0（即“保留该轴”），
      这样 VALID(全轴分裂) <-> SN2N(XY 分裂、Z 保留) 是同一掩码的两个端点。

只读现有 ``pairability_curves.csv``，不重跑 Stage1。
"""
from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


AXIS_ORDER = ("z", "y", "x")


@dataclass
class AxisGeometry:
    axis: str
    participate: bool
    reason: str
    d_min: int | None
    d_max: int | None
    n_d1: float | None
    s_d1: float | None
    q_d1: float | None
    distances: list[int] = field(default_factory=list)


@dataclass
class QMaskGeometry:
    source: str
    region: str
    noise_source: str
    tau_n: float
    tau_s: float
    tau_q: float
    axes: dict[str, AxisGeometry] = field(default_factory=dict)
    fallback_order: list[str] = field(default_factory=list)
    promote_reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["participate"] = {axis: self.axes[axis].participate for axis in AXIS_ORDER if axis in self.axes}
        payload["effective_participate"] = self.effective_participate()
        return payload

    def effective_participate(self) -> dict[str, bool]:
        """返回保证至少两轴参与的参与掩码（采样器要求 4 视图共位角集）。"""
        participate = {axis: self.axes[axis].participate for axis in AXIS_ORDER if axis in self.axes}
        off = [axis for axis, on in participate.items() if not on]
        on = [axis for axis, on in participate.items() if on]
        for axis in self.fallback_order:
            if len(on) >= 2:
                break
            if axis in off:
                participate[axis] = True
                off.remove(axis)
                on.append(axis)
        return participate


def _axis_rows(rows: list[dict[str, str]], axis: str, region: str, noise_source: str) -> list[dict[str, str]]:
    return [
        row for row in rows
        if row.get("depth_bin") == "global"
        and row.get("axis") == axis
        and row.get("region") == region
        and row.get("noise_source") == noise_source
    ]


def _f(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except ValueError:
        return None
    return None if out != out else out  # NaN guard


def derive_qmask(
    rows: list[dict[str, str]],
    *,
    region: str,
    noise_source: str,
    tau_n: float,
    tau_s: float,
    tau_q: float,
    source: str = "",
    use_n_decision: bool = True,
) -> QMaskGeometry:
    """由 pairability 曲线推导参与掩码。

    use_n_decision=False（Plan v2 Q-2）：N 阈值不参与决策，仅记录 d_min。
    背景：T3b 实测 proxy-N 与 9-repeat 金标准在 y/z 轴不相关（Spearman 0.07~0.36、
    z 轴为负），proxy-only 数据上由 N 单独否决轴参与不可靠（neurofinder 的 x 轴
    参与决策曾悬在 0.0026 的 proxy-N 边距上）。此时参与由 S 区间 + Q 门主导。
    """
    geometry = QMaskGeometry(
        source=source,
        region=region,
        noise_source=noise_source,
        tau_n=tau_n,
        tau_s=tau_s,
        tau_q=tau_q,
    )
    axis_s1: dict[str, float] = {}
    n_d_min_reference: dict[str, int] = {}  # Q-2：use_n_decision=False 时被弃用的 N 派生 d_min
    for axis in AXIS_ORDER:
        rows_axis = _axis_rows(rows, axis, region, noise_source)
        if not rows_axis:
            geometry.axes[axis] = AxisGeometry(
                axis=axis,
                participate=False,
                reason=f"no rows for axis {axis}",
                d_min=None,
                d_max=None,
                n_d1=None,
                s_d1=None,
                q_d1=None,
            )
            continue

        by_distance: dict[int, dict[str, float | None]] = {}
        for row in rows_axis:
            try:
                distance = int(float(row["distance_px"]))
            except (KeyError, ValueError):
                continue
            by_distance[distance] = {
                "N": _f(row.get("N")),
                "S": _f(row.get("S")),
                "Q": _f(row.get("Q")),
            }

        distances = sorted(by_distance)
        n_pair = by_distance[distances[0]]

        d_min_candidates = [d for d in distances if by_distance[d]["N"] is not None and by_distance[d]["N"] >= tau_n]
        d_max_candidates = [d for d in distances if by_distance[d]["S"] is not None and by_distance[d]["S"] >= tau_s]
        d_min = min(d_min_candidates) if d_min_candidates else None
        d_max = max(d_max_candidates) if d_max_candidates else None

        d1 = by_distance.get(1)
        n1 = d1["N"] if d1 else None
        s1 = d1["S"] if d1 else None
        q1 = d1["Q"] if d1 else None

        reason_parts: list[str] = []
        if d1 is None:
            participate = False
            reason_parts.append("d=1 missing")
        else:
            checks = {
                "N(d=1)>=tau_n": (n1 is not None and n1 >= tau_n),
                "S(d=1)>=tau_s": (s1 is not None and s1 >= tau_s),
                "Q(d=1)>=tau_q": (q1 is not None and q1 >= tau_q),
            }
            if not use_n_decision:
                # Q-2：N 阈值仅记录不决策（proxy-N 不可靠时使用）。
                # 注意：N 派生的 d_min 一并弃用（否则 N(d=1)<τ_n 会给出 d_min=2，
                # 区间 [2,d_max] 不覆盖 d=1，轴被关——第一版实现因此形同未改，
                # neurofinder 上 effective mask 仍为提升后的 101）。
                checks["N(d=1)>=tau_n"] = True
                if d_min is not None and d_min != 1:
                    n_d_min_reference[axis] = d_min
                d_min = 1  # 区间下界退化为 d=1 本身（S/Q 已门控）
            interval_ok = d_min is not None and d_max is not None and d_min <= 1 <= d_max
            participate = all(checks.values()) and interval_ok
            if not interval_ok:
                reason_parts.append("no legal [d_min,d_max] interval covering d=1")
            for name, ok in checks.items():
                if not ok:
                    reason_parts.append(name)
        reason = "; ".join(reason_parts) if reason_parts else "legal d=1 pair"

        geometry.axes[axis] = AxisGeometry(
            axis=axis,
            participate=participate,
            reason=reason,
            d_min=d_min,
            d_max=d_max,
            n_d1=n1,
            s_d1=s1,
            q_d1=q1,
            distances=distances,
        )
        if s1 is not None:
            axis_s1[axis] = s1

    geometry.fallback_order = sorted(axis_s1, key=lambda a: axis_s1[a], reverse=True)
    geometry.n_d_min_reference = n_d_min_reference
    return geometry


def geometry_from_stage1_run(
    stage1_run_dir: str | Path,
    *,
    region: str,
    noise_source: str,
    tau_n: float,
    tau_s: float,
    tau_q: float,
    use_n_decision: bool = True,
) -> QMaskGeometry:
    csv_path = Path(stage1_run_dir) / "metrics" / "pairability_curves.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Stage1 pairability CSV not found: {csv_path}")
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return derive_qmask(
        rows,
        region=region,
        noise_source=noise_source,
        tau_n=tau_n,
        tau_s=tau_s,
        tau_q=tau_q,
        source=str(csv_path),
        use_n_decision=use_n_decision,
    )


def preview_text(geometry: QMaskGeometry) -> str:
    lines = [
        f"QMask geometry  source={geometry.source}",
        f"  region={geometry.region} noise_source={geometry.noise_source}",
        f"  thresholds tau_n={geometry.tau_n} tau_s={geometry.tau_s} tau_q={geometry.tau_q}",
    ]
    for axis in AXIS_ORDER:
        if axis not in geometry.axes:
            continue
        g = geometry.axes[axis]
        lines.append(
            f"  {axis}: participate={g.participate}  "
            f"d_min={g.d_min} d_max={g.d_max}  "
            f"N(d1)={g.n_d1 if g.n_d1 is None else round(g.n_d1, 4)} "
            f"S(d1)={g.s_d1 if g.s_d1 is None else round(g.s_d1, 4)} "
            f"Q(d1)={g.q_d1 if g.q_d1 is None else round(g.q_d1, 4)}  "
            f"reason={g.reason}"
        )
    effective = geometry.effective_participate()
    lines.append(f"  effective_participate={effective}")
    lines.append(f"  fallback_order={geometry.fallback_order}")
    return "\n".join(lines)


def write_geometry(path: str | Path, geometry: QMaskGeometry) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(geometry.as_dict(), indent=2, ensure_ascii=False), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Derive QMask axis-participation mask from a Stage1 pairability CSV.")
    parser.add_argument("--stage1-run-dir", required=True)
    parser.add_argument("--region", default="foreground")
    parser.add_argument("--noise-source", required=True)
    parser.add_argument("--tau-n", type=float, default=0.95)
    parser.add_argument("--tau-s", type=float, default=0.85)
    parser.add_argument("--tau-q", type=float, default=0.60)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    geometry = geometry_from_stage1_run(
        args.stage1_run_dir,
        region=args.region,
        noise_source=args.noise_source,
        tau_n=args.tau_n,
        tau_s=args.tau_s,
        tau_q=args.tau_q,
    )
    print(preview_text(geometry))
    write_geometry(args.out, geometry)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
