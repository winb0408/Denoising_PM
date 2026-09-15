from __future__ import annotations

import argparse
import json

import torch

from .pairability import geometry_from_stage1_run
from .sampling_aniso import VALID_MODES, metadata_to_dict, sample_four_views


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Static checks for ACES Stage 2 samplers.")
    parser.add_argument("--stage1-run-dir", required=True)
    parser.add_argument("--noise-source", default="repeat")
    parser.add_argument("--region", default="foreground")
    parser.add_argument("--metric", default="Q")
    args = parser.parse_args(argv)
    geometry = geometry_from_stage1_run(
        args.stage1_run_dir,
        noise_source=args.noise_source,
        region=args.region,
        metric=args.metric,
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    img = torch.arange(1 * 16 * 64 * 64, dtype=torch.float32, device=device).reshape(1, 16, 64, 64)
    result = {"geometry": {"radii_px": geometry.radii_px, "threshold": geometry.threshold}, "samplers": {}}
    for mode in VALID_MODES:
        views, metadata = sample_four_views(img, mode, geometry=geometry, seed=123)
        again, _ = sample_four_views(img, mode, geometry=geometry, seed=123)
        if len(views) != 4:
            raise AssertionError(f"{mode}: expected four views")
        if any(tuple(view.shape) != tuple(views[0].shape) for view in views):
            raise AssertionError(f"{mode}: view shapes differ")
        if any(not torch.equal(left, right) for left, right in zip(views, again)):
            raise AssertionError(f"{mode}: not reproducible with fixed seed")
        try:
            sample_four_views(img[:, :1], mode, geometry=geometry, seed=123)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{mode}: collapsed z dimension did not fail fast")
        result["samplers"][mode] = metadata_to_dict(metadata)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
