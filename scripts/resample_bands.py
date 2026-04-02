from __future__ import annotations

import argparse
from pathlib import Path

import rasterio
from rasterio.enums import Resampling

from kwdet_common import ensure_dir


def resample_to_reference(source_path: Path, reference_path: Path, output_path: Path) -> None:
    ensure_dir(output_path.parent)
    with rasterio.open(reference_path) as ref, rasterio.open(source_path) as src:
        data = src.read(
            out_shape=(src.count, ref.height, ref.width),
            resampling=Resampling.bilinear,
        )
        profile = src.profile.copy()
        profile.update(
            width=ref.width,
            height=ref.height,
            transform=ref.transform,
            crs=ref.crs,
        )
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(data)


def resample_product(
    product_dir: Path,
    output_dir: Path,
    reference_band: str = "B08",
    bands: tuple[str, ...] = ("B8A", "B11"),
    overwrite: bool = False,
) -> list[Path]:
    reference_path = product_dir / f"{reference_band}.tif"
    if not reference_path.exists():
        raise FileNotFoundError(f"Missing reference band: {reference_path}")

    outputs: list[Path] = []
    for band in bands:
        source_path = product_dir / f"{band}.tif"
        if not source_path.exists():
            raise FileNotFoundError(f"Missing source band: {source_path}")
        target_dir = ensure_dir(output_dir / product_dir.name)
        output_path = target_dir / f"{band}_10m.tif"
        if output_path.exists() and not overwrite:
            outputs.append(output_path)
            continue
        resample_to_reference(source_path, reference_path, output_path)
        outputs.append(output_path)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline resampling helper for Sentinel-2 20 m bands to 10 m.")
    parser.add_argument("--source", type=Path, help="Single input raster to upsample")
    parser.add_argument("--reference", type=Path, help="Reference 10 m raster")
    parser.add_argument("--output", type=Path, help="Output raster path")
    parser.add_argument("--product-dir", type=Path, help="Product directory containing B08/B8A/B11")
    parser.add_argument("--output-dir", type=Path, help="Output root for resampled product bands")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.source and args.reference and args.output:
        resample_to_reference(args.source, args.reference, args.output)
        print(f"Resampled {args.source} -> {args.output}")
        return

    if args.product_dir and args.output_dir:
        outputs = resample_product(args.product_dir, args.output_dir, overwrite=args.overwrite)
        print(f"Resampled {len(outputs)} band(s) for {args.product_dir.name}")
        for output in outputs:
            print(output)
        return

    parser.error("Provide either --source/--reference/--output or --product-dir/--output-dir.")


if __name__ == "__main__":
    main()
