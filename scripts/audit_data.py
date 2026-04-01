from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from kwdet_common import (
    FINLAND_MANIFEST_PATH,
    S2SHIPS_NPY_DIR,
    S2SHIPS_WATER_DIR,
    ensure_dir,
    list_s2ships_samples,
    load_s2ships_sample,
    product_asset_path,
    read_json,
    scene_dimensions_from_label,
    write_json,
)


def audit_s2ships() -> dict:
    scenes: dict[str, dict] = {}
    total_instances = 0
    total_labeled_pixels = 0

    for sample in list_s2ships_samples():
        payload = load_s2ships_sample(sample["npy_path"])
        data = payload["data"]
        label = payload["label"]

        if data.shape != (938, 1783, 12):
            raise ValueError(f"Unexpected data shape for {sample['npy_path'].name}: {data.shape}")
        if label.shape != (938, 1783, 1):
            raise ValueError(f"Unexpected label shape for {sample['npy_path'].name}: {label.shape}")
        if data.dtype != np.float64:
            raise ValueError(f"Unexpected data dtype for {sample['npy_path'].name}: {data.dtype}")

        label_2d = label[:, :, 0]
        _, n_instances = ndi.label(label_2d > 0)
        total_instances += int(n_instances)
        total_labeled_pixels += int((label_2d > 0).sum())
        nan_count = int(np.isnan(data).sum())

        water_path = S2SHIPS_WATER_DIR / f"{sample['scene']}_water.tif"
        water_exists = water_path.exists()
        height, width = scene_dimensions_from_label(label_2d)

        scenes[sample["scene"]] = {
            "source_npy": str(sample["npy_path"].relative_to(sample["npy_path"].parents[2])),
            "shape": [height, width, int(data.shape[2])],
            "label_shape": list(label.shape),
            "dtype": str(data.dtype),
            "instances": int(n_instances),
            "labeled_pixels": int((label_2d > 0).sum()),
            "data_nan": nan_count,
            "data_min": float(np.nanmin(data)),
            "data_max": float(np.nanmax(data)),
            "water_mask_exists": water_exists,
        }

    return {
        "dataset": "S2SHIPS",
        "scene_count": len(scenes),
        "scenes": scenes,
        "total_instances": total_instances,
        "total_labeled_pixels": total_labeled_pixels,
    }


def audit_finland(manifest_path: Path) -> dict:
    manifest = read_json(manifest_path)
    products: list[dict] = []
    missing_assets: list[str] = []

    for product in manifest:
        product_dir = manifest_path.parent / Path(product["product_dir"]).name
        if not product_dir.exists():
            raise FileNotFoundError(f"Missing Finland product directory: {product_dir}")

        assets: list[dict] = []
        for asset in product["assets"]:
            asset_path = product_asset_path(product_dir, asset["filename"])
            exists = asset_path.exists()
            size_bytes = asset_path.stat().st_size if exists else 0
            asset_entry = {
                "key": asset["key"],
                "filename": asset["filename"],
                "exists": exists,
                "size_bytes": size_bytes,
            }
            assets.append(asset_entry)
            if not exists or size_bytes <= 0:
                missing_assets.append(str(asset_path))

        products.append(
            {
                "item_id": product["item_id"],
                "tile": product["tile"],
                "layer": product["layer"],
                "date": product["date"],
                "product_dir": str(product_dir.relative_to(manifest_path.parents[1])),
                "asset_count": len(assets),
                "assets": assets,
            }
        )

    totals = {
        "product_count": len(products),
        "expected_assets_per_product": sorted({product["asset_count"] for product in products}),
        "missing_asset_count": len(missing_assets),
    }
    return {
        "dataset": "Finland 2025 L2A",
        "manifest_path": str(manifest_path.relative_to(manifest_path.parents[1])),
        "products": products,
        "totals": totals,
        "missing_assets": missing_assets,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit S2SHIPS and Finland dataset completeness.")
    parser.add_argument("--s2-output", type=Path, default=Path("data/audit_s2ships.json"))
    parser.add_argument("--fin-output", type=Path, default=Path("data/audit_finland.json"))
    parser.add_argument("--manifest", type=Path, default=FINLAND_MANIFEST_PATH)
    args = parser.parse_args()

    s2_report = audit_s2ships()
    fin_report = audit_finland(args.manifest)

    ensure_dir(args.s2_output.parent)
    write_json(args.s2_output, s2_report)
    write_json(args.fin_output, fin_report)

    print(
        f"S2SHIPS audited: {s2_report['scene_count']} scenes, "
        f"{s2_report['total_instances']} connected components."
    )
    print(
        f"Finland audited: {fin_report['totals']['product_count']} products, "
        f"missing assets={fin_report['totals']['missing_asset_count']}."
    )


if __name__ == "__main__":
    main()
