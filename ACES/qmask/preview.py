"""Create bounded-size previews for qualitative comparison figures.

Model outputs and research figures should be inspected before quantitative
comparison, but full-resolution qualitative grids are often too large to send
to a multimodal API. This utility produces two review artifacts:

* ``qualitative_overview.png``: a single compact overview image.
* ``review_tiles/``: individual method rows for closer inspection.

All outputs are enforced to be PNG-compatible RGB(A) images with bounded pixel
dimensions and an optional file-size ceiling.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def _rgb_copy(image: Image.Image) -> Image.Image:
    if image.mode in ("RGB", "RGBA"):
        return image.copy()
    if "transparency" in image.info:
        return image.convert("RGBA")
    return image.convert("RGB")


def _resize_within(image: Image.Image, max_width: int, max_height: int) -> Image.Image:
    scale = min(max_width / image.width, max_height / image.height, 1.0)
    if scale == 1.0:
        return image
    width = max(1, round(image.width * scale))
    height = max(1, round(image.height * scale))
    return image.resize((width, height), Image.Resampling.LANCZOS)


def _save_bounded_png(
    image: Image.Image,
    path: Path,
    max_bytes: int,
    max_width: int,
    max_height: int,
) -> dict[str, int | str]:
    current = _resize_within(image, max_width, max_height)
    while True:
        path.parent.mkdir(parents=True, exist_ok=True)
        current.save(path, format="PNG", optimize=True)
        size = path.stat().st_size
        if size <= max_bytes or min(current.size) <= 128:
            break
        current = _resize_within(current, round(current.width * 0.85), round(current.height * 0.85))
    return {
        "width": current.width,
        "height": current.height,
        "bytes": size,
        "path": str(path),
    }


def make_review_previews(
    run_dir: str | Path,
    source_name: str = "qualitative_comparison.png",
    max_width: int = 1200,
    max_height: int = 1200,
    max_bytes: int = 3_500_000,
) -> dict[str, object]:
    run_dir = Path(run_dir)
    source_path = run_dir / "figures" / source_name
    if not source_path.exists():
        raise FileNotFoundError(
            f"missing qualitative figure: {source_path}; run `python -m qmask.visualize --run-dir {run_dir}` first"
        )

    with Image.open(source_path) as opened:
        opened.load()
        source = _rgb_copy(opened)

    manifest: dict[str, object] = {
        "source": str(source_path),
        "source_width": source.width,
        "source_height": source.height,
        "source_bytes": source_path.stat().st_size,
        "limits": {"max_width": max_width, "max_height": max_height, "max_bytes": max_bytes},
    }

    overview_path = run_dir / "figures" / "review" / "qualitative_overview.png"
    manifest["overview"] = _save_bounded_png(source, overview_path, max_bytes, max_width, max_height)

    overview_width, overview_height = source.size
    row_height = 280
    tile_paths: list[dict[str, int | str]] = []
    for row_index, top in enumerate(range(0, overview_height, row_height)):
        tile = source.crop((0, top, overview_width, min(top + row_height, overview_height)))
        tile_path = run_dir / "figures" / "review" / "tiles" / f"row_{row_index + 1:02d}.png"
        tile_paths.append(
            _save_bounded_png(
                tile,
                tile_path,
                max_bytes // 2,
                max_width,
                row_height,
            )
        )
    manifest["tiles"] = tile_paths
    manifest["review_order"] = [
        "Inspect overview first; then inspect tiles one row at a time.",
        "Compare input / best / final / target within each method row.",
        "Reject conclusions if any method shows collapsed range, shifted intensity, or grid/block artifacts.",
    ]
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate bounded-size review previews for QMask figures.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--max-width", type=int, default=1200)
    parser.add_argument("--max-height", type=int, default=1200)
    parser.add_argument("--max-bytes", type=int, default=3_500_000)
    args = parser.parse_args(argv)
    manifest = make_review_previews(
        args.run_dir,
        max_width=args.max_width,
        max_height=args.max_height,
        max_bytes=args.max_bytes,
    )
    out = Path(args.run_dir) / "figures" / "review" / "manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(__import__("json").dumps(manifest, indent=2), encoding="utf-8")
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
