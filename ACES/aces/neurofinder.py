from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import tifffile

from .io_utils import write_json


def _image_files(root: Path) -> list[Path]:
    files = sorted((root / "images").glob("*.tiff"))
    if not files:
        raise FileNotFoundError(f"No neurofinder TIFF frames found under {root / 'images'}")
    return files


def _page_tags(path: Path) -> dict[str, Any]:
    with tifffile.TiffFile(path) as tif:
        page = tif.pages[0]
        keys = ("ImageDescription", "XResolution", "YResolution", "ResolutionUnit", "Software", "DateTime")
        return {
            "shape": list(page.shape),
            "dtype": str(page.dtype),
            "tags": {key: str(page.tags[key].value) for key in keys if key in page.tags},
        }


def load_roi_mask(regions_path: str | Path, shape_yx: tuple[int, int]) -> np.ndarray:
    regions = json.loads(Path(regions_path).read_text(encoding="utf-8"))
    mask = np.zeros(shape_yx, dtype=bool)
    for region in regions:
        for x, y in region.get("coordinates", []):
            if 0 <= int(y) < shape_yx[0] and 0 <= int(x) < shape_yx[1]:
                mask[int(y), int(x)] = True
    return mask


def inspect_dataset(root: str | Path) -> dict[str, Any]:
    dataset_root = Path(root).resolve()
    files = _image_files(dataset_root)
    first = _page_tags(files[0])
    last = _page_tags(files[-1])
    regions_path = dataset_root / "regions" / "regions.json"
    regions = json.loads(regions_path.read_text(encoding="utf-8")) if regions_path.exists() else []
    calibration_tags = first["tags"]
    has_xy_calibration = all(key in calibration_tags for key in ("XResolution", "YResolution", "ResolutionUnit"))
    return {
        "event": "aces_neurofinder_source_audit",
        "dataset_root": str(dataset_root),
        "frame_count": len(files),
        "first_frame": str(files[0]),
        "last_frame": str(files[-1]),
        "first_frame_tiff": first,
        "last_frame_tiff": last,
        "roi_region_count": len(regions),
        "has_tiff_xy_physical_calibration": bool(has_xy_calibration),
        "physical_calibration_um_per_px": None,
        "calibration_note": (
            "No XResolution/YResolution/ResolutionUnit tags were present in the local TIFF frames; "
            "ACES treats x/y as pixel units and z as frame/time units for this dataset."
            if not has_xy_calibration
            else "TIFF resolution tags are present and should be converted before physical-distance reporting."
        ),
    }


def prepare_dataset(root: str | Path, out_dir: str | Path, *, overwrite: bool = False) -> dict[str, Any]:
    dataset_root = Path(root).resolve()
    output_root = Path(out_dir).resolve()
    train_dir = output_root / "train"
    proxy_dir = output_root / "proxy"
    train_dir.mkdir(parents=True, exist_ok=True)
    proxy_dir.mkdir(parents=True, exist_ok=True)

    files = _image_files(dataset_root)
    audit = inspect_dataset(dataset_root)
    shape_yx = tuple(audit["first_frame_tiff"]["shape"])
    stack_path = train_dir / "neurofinder_00_00_stack.tif"
    mean_path = proxy_dir / "neurofinder_00_00_temporal_mean.tif"
    roi_path = proxy_dir / "neurofinder_00_00_roi_mask.tif"

    if overwrite or not stack_path.exists():
        stack = tifffile.memmap(stack_path, shape=(len(files), *shape_yx), dtype=np.uint16, bigtiff=True)
        running_sum = np.zeros(shape_yx, dtype=np.float64)
        for index, path in enumerate(files):
            frame = tifffile.imread(path)
            if frame.shape != shape_yx:
                raise ValueError(f"Frame shape changed at {path}: {frame.shape} != {shape_yx}")
            stack[index] = frame
            running_sum += frame.astype(np.float64, copy=False)
        stack.flush()
        mean = (running_sum / float(len(files))).astype(np.float32)
        tifffile.imwrite(mean_path, mean)
    elif not mean_path.exists():
        stack = tifffile.imread(stack_path)
        tifffile.imwrite(mean_path, stack.mean(axis=0).astype(np.float32))

    regions_path = dataset_root / "regions" / "regions.json"
    if regions_path.exists() and (overwrite or not roi_path.exists()):
        roi = load_roi_mask(regions_path, shape_yx)
        tifffile.imwrite(roi_path, roi.astype(np.uint8))

    prepared = {
        **audit,
        "prepared_root": str(output_root),
        "train_folder": str(train_dir),
        "stack_path": str(stack_path),
        "mean_proxy_path": str(mean_path),
        "foreground_mask_path": str(roi_path) if roi_path.exists() else None,
        "layout": "[frame,y,x]; ACES Stage 2 z-axis is frame/time, not optical depth.",
    }
    write_json(output_root / "metadata.json", prepared)
    return prepared


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit and prepare neurofinder.00.00 for ACES Stage 2.")
    parser.add_argument("--root", default="Methods/FAST-main/data/neurofinder.00.00")
    parser.add_argument("--out-dir", default="ACES/data/neurofinder.00.00")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    result = prepare_dataset(args.root, args.out_dir, overwrite=args.overwrite)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
