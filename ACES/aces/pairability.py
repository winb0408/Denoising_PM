from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .paths import DEFAULT_RUN_DIR


@dataclass(frozen=True)
class PairabilityGeometry:
    radii_px: dict[str, int]
    q_values: dict[str, dict[int, float]]
    threshold: float
    region: str
    noise_source: str
    metric: str


def load_pairability_geometry(
    csv_path: str | Path,
    *,
    threshold: float = 0.6,
    region: str = "foreground",
    noise_source: str = "repeat",
    metric: str = "Q",
    min_radius: int = 1,
) -> PairabilityGeometry:
    rows: list[dict[str, Any]] = []
    with Path(csv_path).open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if (
                row.get("depth_bin") == "global"
                and row.get("region") == region
                and row.get("noise_source") == noise_source
            ):
                rows.append(row)
    if not rows:
        raise ValueError(f"No pairability rows for region={region!r}, noise_source={noise_source!r}")

    q_values: dict[str, dict[int, float]] = {"x": {}, "y": {}, "z": {}}
    radii: dict[str, int] = {}
    for axis in ("x", "y", "z"):
        axis_rows = sorted((row for row in rows if row["axis"] == axis), key=lambda row: int(row["distance_px"]))
        if not axis_rows:
            raise ValueError(f"No pairability rows for axis={axis!r}")
        for row in axis_rows:
            q_values[axis][int(row["distance_px"])] = float(row[metric])
        valid_distances = [distance for distance, value in q_values[axis].items() if value >= threshold]
        radii[axis] = max(max(valid_distances) if valid_distances else min_radius, min_radius)

    return PairabilityGeometry(
        radii_px=radii,
        q_values=q_values,
        threshold=threshold,
        region=region,
        noise_source=noise_source,
        metric=metric,
    )


def geometry_from_stage1_run(
    run_dir: str | Path = DEFAULT_RUN_DIR,
    *,
    threshold: float = 0.6,
    region: str = "foreground",
    noise_source: str = "repeat",
    metric: str = "Q",
) -> PairabilityGeometry:
    return load_pairability_geometry(
        Path(run_dir) / "metrics" / "pairability_curves.csv",
        threshold=threshold,
        region=region,
        noise_source=noise_source,
        metric=metric,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Derive ACES sampling geometry from Stage 1 pairability curves.")
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--region", default="foreground")
    args = parser.parse_args(argv)
    geometry = geometry_from_stage1_run(args.run_dir, threshold=args.threshold, region=args.region)
    print(json.dumps({
        "radii_px": geometry.radii_px,
        "threshold": geometry.threshold,
        "region": geometry.region,
        "noise_source": geometry.noise_source,
        "metric": geometry.metric,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
