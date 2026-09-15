"""几何阈值敏感性表：只读 Stage1 曲线，批量打印各 (tau_n, tau_s) 下的参与掩码。"""
from __future__ import annotations

import argparse

from .geometry import geometry_from_stage1_run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1-run-dir", required=True)
    parser.add_argument("--noise-source", required=True)
    parser.add_argument("--region", default="foreground")
    parser.add_argument("--tau-q", type=float, default=0.60)
    parser.add_argument("--tau-n-list", default="0.90,0.93,0.95")
    parser.add_argument("--tau-s-list", default="0.80,0.85,0.88,0.90")
    args = parser.parse_args(argv)

    tau_ns = [float(x) for x in args.tau_n_list.split(",")]
    tau_ss = [float(x) for x in args.tau_s_list.split(",")]
    print(f"{'tau_n':>6} {'tau_s':>6}  {'effective_participate':<38} {'d_min(z,y,x)':<16} {'d_max(z,y,x)':<16}")
    for tau_n in tau_ns:
        for tau_s in tau_ss:
            g = geometry_from_stage1_run(
                args.stage1_run_dir,
                region=args.region,
                noise_source=args.noise_source,
                tau_n=tau_n,
                tau_s=tau_s,
                tau_q=args.tau_q,
            )
            dmin = ",".join(str(g.axes[a].d_min) for a in ("z", "y", "x"))
            dmax = ",".join(str(g.axes[a].d_max) for a in ("z", "y", "x"))
            eff = g.effective_participate()
            eff_str = "".join(a if eff[a] else "." for a in ("z", "y", "x")) or "()"
            print(f"{tau_n:>6.2f} {tau_s:>6.2f}  {eff_str:<38} {dmin:<16} {dmax:<16}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
