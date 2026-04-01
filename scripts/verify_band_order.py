from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio

from kwdet_common import BAND_ORDER, ensure_dir, find_band_tif, list_s2ships_samples, load_s2ships_data, write_json


def verify_band_order() -> dict:
    scene_results: dict[str, dict] = {}
    canonical_mapping: dict[str, int] | None = None

    for sample in list_s2ships_samples():
        scene = sample["scene"]
        scene_data = load_s2ships_data(scene)
        per_scene_mapping: dict[str, int] = {}

        for band in BAND_ORDER:
            tif_path = find_band_tif(scene, band)
            with rasterio.open(tif_path) as src:
                tif = src.read(1)

            matches = [idx for idx in range(scene_data.shape[2]) if np.array_equal(scene_data[:, :, idx], tif)]
            if len(matches) != 1:
                raise ValueError(
                    f"Expected exactly one exact match for scene={scene}, band={band}; found {matches}"
                )
            per_scene_mapping[band] = int(matches[0])

        if canonical_mapping is None:
            canonical_mapping = per_scene_mapping
        elif per_scene_mapping != canonical_mapping:
            raise ValueError(f"Inconsistent band order for scene {scene}: {per_scene_mapping}")

        scene_results[scene] = {
            "mapping": per_scene_mapping,
            "verified_bands": BAND_ORDER,
        }

    return {
        "verified": True,
        "canonical_mapping": canonical_mapping,
        "scenes": scene_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify S2SHIPS dataset_npy channel order.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/audit/band_order_verification.json"),
        help="Where to write the verification report.",
    )
    args = parser.parse_args()

    report = verify_band_order()
    ensure_dir(args.output.parent)
    write_json(args.output, report)
    print("Verified band order:")
    for band, idx in report["canonical_mapping"].items():
        print(f"  {band} -> channel {idx}")


if __name__ == "__main__":
    main()
