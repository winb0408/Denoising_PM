"""QMask pilot 报告生成：读 run 产物，输出门控判定与 Markdown。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _f(value: Any, digits: int = 2) -> str:
    if value is None or value != value or value in (float("inf"), float("-inf")):
        return "n/a"
    return f"{float(value):.{digits}f}"


def _mode_metrics(run_dir: Path, mode: str) -> dict[str, Any]:
    summary = _load(run_dir / mode / "logs" / "summary.json")
    full = summary.get("eval_regions") or {}
    aligned = summary.get("eval_regions_aligned")
    depth = summary.get("depth_binned") or {}
    best_fg = summary.get("best_fg_psnr_db")
    best_epoch = summary.get("best_epoch")
    return {
        "mode": mode,
        "best_epoch": best_epoch,
        "best_fg_full": best_fg,
        "final_fg_full": full.get("foreground", {}).get("psnr_db"),
        "final_all_full": full.get("all", {}).get("psnr_db"),
        "final_fg_aligned": (aligned or {}).get("foreground", {}).get("psnr_db") if aligned else None,
        "depth_fg_overall": depth.get("overall_foreground_psnr_db"),
        "final_loss": summary.get("final_loss"),
        "train_batches": summary.get("train_batches"),
        "aligned_stride": summary.get("aligned_stride"),
    }


def _collapse_audit(run_dir: Path, mode: str) -> list[dict[str, Any]]:
    """Q-4 修订后的塌缩审计（Plan v2 §2.3）。

    判据 1（动态范围塌缩）：前景 patch 输出 std 相对 target std 的比值 < 0.2，
    即输出方差几乎消失——这是真正的模型塌缩信号，与参考目标的选择无关。
    判据 2（强度漂移）：pred_mean/target_mean 偏差 > 20% 保留，但只在伴随
    std 比值异常（<0.5，输出被过度抹平）时才计为塌缩——单帧去噪输出对
    mean9 proxy 本来就存在前景强度差，单独的均值偏差不构成塌缩证据。
    """
    path = run_dir / mode / "metrics" / "eval_metrics.json"
    if not path.exists():
        return []
    data = _load(path)
    out = []
    for row in data.get("per_patch", []):
        if row.get("region") != "foreground":
            continue
        pred = row.get("pred_mean")
        target = row.get("target_mean")
        pred_std = row.get("pred_std")
        target_std = row.get("target_std")
        if pred is None or target is None or target == 0:
            continue
        dev = abs(pred - target) / abs(target)
        std_ratio = (pred_std / target_std) if (pred_std is not None and target_std and target_std > 0) else None
        range_collapsed = std_ratio is not None and std_ratio < 0.2
        drift_with_flattening = dev > 0.20 and std_ratio is not None and std_ratio < 0.5
        if range_collapsed or drift_with_flattening:
            out.append({
                "patch_index": row.get("patch_index"),
                "pred_mean": pred,
                "target_mean": target,
                "pred_std": pred_std,
                "target_std": target_std,
                "std_ratio": std_ratio,
                "mean_deviation_reference_only": dev,
                "reason": "range_collapse" if range_collapsed else "drift_with_flattening",
            })
    return out


def write_gate(run_dir: Path, modes: list[str], *, gate_db: float = -0.25) -> dict[str, Any]:
    metrics = {m: _mode_metrics(run_dir, m) for m in modes}
    q = metrics.get("qmask")
    endpoints = [m for m in ("valid", "sn2n_xy") if m in metrics]
    endpoint_best = max((metrics[m]["final_fg_aligned"] for m in endpoints if metrics[m]["final_fg_aligned"] is not None), default=None)

    def pick_psnr(row: dict[str, Any]) -> float | None:
        return row["final_fg_aligned"] if row["final_fg_aligned"] is not None else row["final_fg_full"]

    q_psnr = pick_psnr(q) if q else None
    ref_psnr = None
    ref_label = "aligned"
    if q_psnr is not None and endpoints:
        ref_values = [pick_psnr(metrics[m]) for m in endpoints if pick_psnr(metrics[m]) is not None]
        if ref_values:
            ref_psnr = max(ref_values)
            if all(metrics[m]["final_fg_aligned"] is not None for m in endpoints):
                ref_label = "aligned"
            else:
                ref_label = "full (aligned missing)"
    else:
        ref_label = "full"

    collapse_count = len(_collapse_audit(run_dir, "qmask")) if q else None
    margin = (q_psnr - ref_psnr) if (q_psnr is not None and ref_psnr is not None) else None
    q_win = margin is not None and margin >= gate_db
    no_collapse = collapse_count == 0
    verdict = "PASS (eligible to scale)" if (q_win and no_collapse) else "REVIEW (do not auto-scale)"

    return {
        "metrics": metrics,
        "gate": {
            "qmask_fg_psnr_db": q_psnr,
            "endpoint_best_fg_psnr_db": ref_psnr,
            "reference": ref_label,
            "gate_db": gate_db,
            "margin_db": margin,
            "qmask_meets_psnr_gate": q_win,
            "qmask_collapsed_patches": collapse_count,
            "no_collapse": no_collapse,
            "verdict": verdict,
        },
    }


def render_markdown(run_dir: Path, title: str, extra_lines: list[str] | None = None) -> str:
    cfg = _load(run_dir / "config.json")
    modes = [m for m in ("valid", "sn2n_xy", "iso3d", "qmask") if (run_dir / m).exists()]
    result = write_gate(run_dir, modes)
    metrics = result["metrics"]
    gate = result["gate"]
    geo = _load(run_dir / "qmask_geometry.json") if (run_dir / "qmask_geometry.json").exists() else {}
    eff = geo.get("effective_participate", {})

    lines = [f"# {title}", ""]
    lines.append(f"- Run dir: `{run_dir}`")
    lines.append(f"- Modes: `{modes}`")
    lines.append(f"- Geometry thresholds: tau_n={geo.get('tau_n')}, tau_s={geo.get('tau_s')}, tau_q={geo.get('tau_q')}")
    lines.append(f"- Effective participation mask (zyx): `{eff}`")
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append("| mode | best epoch | best fg PSNR (full) | final fg PSNR (full) | final fg PSNR (aligned) | final all PSNR (full) | depth-bin fg PSNR | final loss | batches |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for mode in modes:
        m = metrics[mode]
        lines.append(
            f"| {m['mode']} | {m['best_epoch']} | {_f(m['best_fg_full'])} | {_f(m['final_fg_full'])} | "
            f"{_f(m['final_fg_aligned'])} | {_f(m['final_all_full'])} | {_f(m['depth_fg_overall'])} | "
            f"{_f(m['final_loss'], 4)} | {m['train_batches']} |"
        )
    lines.append("")
    lines.append("## Gate")
    lines.append("")
    lines.append(f"- qmask foreground PSNR: {_f(gate['qmask_fg_psnr_db'])} dB")
    lines.append(f"- best endpoint foreground PSNR ({gate['reference']}): {_f(gate['endpoint_best_fg_psnr_db'])} dB")
    lines.append(f"- margin: {_f(gate['margin_db'])} dB (gate {gate['gate_db']} dB)")
    lines.append(f"- qmask collapsed foreground patches (Q-4 判据: 输出 std 比<0.2，或 mean 漂移>20% 伴随 std 比<0.5): {gate['qmask_collapsed_patches']}")
    lines.append(f"- **verdict: {gate['verdict']}**")
    if extra_lines:
        lines.append("")
        lines.extend(extra_lines)
    lines.append("")
    return "\n".join(lines)


def write_run_report(run_dir: str | Path) -> Path:
    run_dir = Path(run_dir)
    title = f"QMask Pilot Report — {run_dir.name}"
    md = render_markdown(run_dir, title)
    out = run_dir / "report" / "RESULTS.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--extra-note", default=None)
    args = parser.parse_args(argv)
    extra = [f"- Note: {args.extra_note}"] if args.extra_note else None
    run_dir = Path(args.run_dir)
    print(render_markdown(run_dir, f"QMask Pilot Report — {run_dir.name}", extra))
    out = write_run_report(run_dir)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
