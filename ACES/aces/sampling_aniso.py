from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from .pairability import PairabilityGeometry, geometry_from_stage1_run
from .paths import VALID_ROOT


VALID_MODES = ("valid", "sn2n_xy", "iso3d", "empirical_block", "ellipsoid")


@dataclass(frozen=True)
class SamplerMetadata:
    mode: str
    input_shape_nzyx: list[int]
    output_shape_nzyx: list[int]
    offsets_zyx: list[list[int]]
    radii_px: dict[str, int] | None
    pairability_threshold: float | None
    note: str


def _require_4d(img: torch.Tensor) -> None:
    if img.dim() != 4:
        raise ValueError(f"Sampler input must be [N,Z,Y,X], got {tuple(img.shape)}")
    if min(img.shape[1:]) < 2:
        raise ValueError(f"3D fail-fast: sampler input collapsed, got {tuple(img.shape)}")


def _valid_sampler(img: torch.Tensor, seed: int | None) -> tuple[list[torch.Tensor], SamplerMetadata]:
    sys.path.insert(0, str(VALID_ROOT))
    from datasets import sampling as valid_sampling
    from datasets.sampling import generate_mask_pair, generate_subimages

    if seed is not None:
        valid_sampling.operation_seed_counter = int(seed)
        random.seed(int(seed))
    masks = generate_mask_pair(img)
    sub = [generate_subimages(img, mask) for mask in masks]
    return sub, SamplerMetadata(
        mode="valid",
        input_shape_nzyx=list(img.shape),
        output_shape_nzyx=list(sub[0].shape),
        offsets_zyx=[],
        radii_px=None,
        pairability_threshold=None,
        note="Original VALID Tetris sampling functions, unmodified.",
    )


def _phase_views(img: torch.Tensor, offsets: list[tuple[int, int, int]]) -> list[torch.Tensor]:
    max_z = max(offset[0] for offset in offsets)
    max_y = max(offset[1] for offset in offsets)
    max_x = max(offset[2] for offset in offsets)
    out_z = img.shape[1] - max_z
    out_y = img.shape[2] - max_y
    out_x = img.shape[3] - max_x
    out_z -= out_z % 2
    out_y -= out_y % 2
    out_x -= out_x % 2
    if min(out_z, out_y, out_x) < 2:
        raise ValueError(f"Offsets {offsets} too large for input shape {tuple(img.shape)}")
    return [
        img[:, dz : dz + out_z, dy : dy + out_y, dx : dx + out_x]
        for dz, dy, dx in offsets
    ]


def _sn2n_xy_sampler(img: torch.Tensor) -> tuple[list[torch.Tensor], SamplerMetadata]:
    if img.shape[2] < 4 or img.shape[3] < 4:
        raise ValueError(f"sn2n_xy requires Y/X >= 4, got {tuple(img.shape)}")
    sub = [
        img[:, :, 0::2, 0::2],
        img[:, :, 0::2, 1::2],
        img[:, :, 1::2, 0::2],
        img[:, :, 1::2, 1::2],
    ]
    min_y = min(view.shape[2] for view in sub)
    min_x = min(view.shape[3] for view in sub)
    sub = [view[:, :, :min_y, :min_x] for view in sub]
    return sub, SamplerMetadata(
        mode="sn2n_xy",
        input_shape_nzyx=list(img.shape),
        output_shape_nzyx=list(sub[0].shape),
        offsets_zyx=[[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1]],
        radii_px=None,
        pairability_threshold=None,
        note="XY-only four-phase sampling; z is preserved. This is SN2N-like, not an equivalence claim.",
    )


def _iso3d_sampler(img: torch.Tensor) -> tuple[list[torch.Tensor], SamplerMetadata]:
    sub = [
        img[:, 0::2, 0::2, 0::2],
        img[:, 0::2, 1::2, 1::2],
        img[:, 1::2, 0::2, 1::2],
        img[:, 1::2, 1::2, 0::2],
    ]
    min_z = min(view.shape[1] for view in sub)
    min_y = min(view.shape[2] for view in sub)
    min_x = min(view.shape[3] for view in sub)
    sub = [view[:, :min_z, :min_y, :min_x] for view in sub]
    return sub, SamplerMetadata(
        mode="iso3d",
        input_shape_nzyx=list(img.shape),
        output_shape_nzyx=list(sub[0].shape),
        offsets_zyx=[[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]],
        radii_px=None,
        pairability_threshold=None,
        note="Deterministic isotropic 2x2x2 tetrahedral corner sampling.",
    )


def _empirical_offsets(geometry: PairabilityGeometry) -> list[tuple[int, int, int]]:
    rz = max(1, geometry.radii_px["z"])
    ry = max(1, geometry.radii_px["y"])
    rx = max(1, geometry.radii_px["x"])
    candidates = [(0, 0, 0), (0, ry, rx), (rz, 0, rx), (rz, ry, 0)]
    return candidates


def _ellipsoid_offsets(geometry: PairabilityGeometry) -> list[tuple[int, int, int]]:
    # 椭球度量的四个主轴端点：原点 + 三个半轴端点 (rx, ry, rz)。
    # 这些点恰好落在各向异性椭球 d_m = sqrt((dz/rz)^2+(dy/ry)^2+(dx/rx)^2) = 1 的面上，
    # 是覆盖椭球主方向的最小四视图采样集合。四个点已满足四视图需求，无需再从
    # 椭球内部按 Q 加权补点，因此这里直接返回主轴端点。
    rz = max(1, geometry.radii_px["z"])
    ry = max(1, geometry.radii_px["y"])
    rx = max(1, geometry.radii_px["x"])
    return [(0, 0, 0), (0, 0, rx), (0, ry, 0), (rz, 0, 0)]


def sample_four_views(
    img: torch.Tensor,
    mode: str,
    *,
    geometry: PairabilityGeometry | None = None,
    seed: int | None = None,
) -> tuple[list[torch.Tensor], SamplerMetadata]:
    _require_4d(img)
    if mode not in VALID_MODES:
        raise ValueError(f"Unknown sampler mode {mode!r}; expected one of {VALID_MODES}")
    if mode == "valid":
        return _valid_sampler(img, seed)
    if mode == "sn2n_xy":
        return _sn2n_xy_sampler(img)
    if mode == "iso3d":
        return _iso3d_sampler(img)
    if geometry is None:
        raise ValueError(f"Sampler mode {mode!r} requires pairability geometry")
    offsets = _empirical_offsets(geometry) if mode == "empirical_block" else _ellipsoid_offsets(geometry)
    sub = _phase_views(img, offsets)
    return sub, SamplerMetadata(
        mode=mode,
        input_shape_nzyx=list(img.shape),
        output_shape_nzyx=list(sub[0].shape),
        offsets_zyx=[list(offset) for offset in offsets],
        radii_px=dict(geometry.radii_px),
        pairability_threshold=geometry.threshold,
        note=(
            "Integer anisotropic block corner sampling from Stage 1 Q geometry."
            if mode == "empirical_block"
            else "Continuous ellipsoid metric candidate sampling from Stage 1 Q geometry."
        ),
    )


def metadata_to_dict(metadata: SamplerMetadata) -> dict[str, Any]:
    return asdict(metadata)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run ACES sampler shape checks.")
    parser.add_argument("--stage1-run-dir", default=None)
    args = parser.parse_args(argv)
    geometry = geometry_from_stage1_run(args.stage1_run_dir) if args.stage1_run_dir else None
    img = torch.arange(1 * 16 * 32 * 32, dtype=torch.float32, device="cuda" if torch.cuda.is_available() else "cpu").reshape(1, 16, 32, 32)
    out: dict[str, Any] = {}
    for mode in VALID_MODES:
        views, metadata = sample_four_views(img, mode, geometry=geometry, seed=123)
        again, _ = sample_four_views(img, mode, geometry=geometry, seed=123)
        if len(views) != 4 or any(view.shape != views[0].shape for view in views):
            raise AssertionError(f"{mode} did not return four matching views")
        if any(not torch.equal(a, b) for a, b in zip(views, again)):
            raise AssertionError(f"{mode} is not deterministic")
        out[mode] = metadata_to_dict(metadata)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
