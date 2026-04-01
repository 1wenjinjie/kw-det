from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np

from kwdet_common import FINLAND_ANN_DIR, ensure_dir, list_gpkg_layers, write_json


TEST_TILES = {"34VEN", "34VER"}


def analyze_finland_annotations(annotation_dir: Path) -> tuple[dict, str]:
    rows: list[dict] = []

    for gpkg_path in sorted(annotation_dir.glob("*.gpkg")):
        tile = gpkg_path.stem
        for layer in list_gpkg_layers(gpkg_path):
            gdf = gpd.read_file(gpkg_path, layer=layer)
            if gdf.empty:
                continue

            areas_m2 = gdf.geometry.area.to_numpy(dtype=float)
            side_px = np.sqrt(areas_m2) / 10.0

            rows.append(
                {
                    "tile": tile,
                    "layer": str(layer),
                    "count": int(len(gdf)),
                    "median_area_m2": float(np.median(areas_m2)),
                    "mean_area_m2": float(np.mean(areas_m2)),
                    "median_side_px": float(np.median(side_px)),
                    "mean_side_px": float(np.mean(side_px)),
                    "lt_3px_count": int((side_px < 3.0).sum()),
                    "lt_3px_ratio": float((side_px < 3.0).mean()),
                    "ge_4px_count": int((side_px >= 4.0).sum()),
                    "crs": str(gdf.crs),
                    "is_official_test_tile": tile in TEST_TILES,
                }
            )

    rows.sort(key=lambda item: (item["tile"], item["layer"]))
    if not rows:
        raise RuntimeError(f"No Finland annotation rows found in {annotation_dir}")

    all_counts = sum(row["count"] for row in rows)
    weighted_lt_3 = sum(row["lt_3px_count"] for row in rows) / max(all_counts, 1)
    test_rows = [row for row in rows if row["is_official_test_tile"]]
    test_counts = sum(row["count"] for row in test_rows)
    weighted_test_lt_3 = (
        sum(row["lt_3px_count"] for row in test_rows) / max(test_counts, 1) if test_rows else 0.0
    )

    summary = {
        "annotation_dir": str(annotation_dir),
        "layer_count": len(rows),
        "total_annotations": all_counts,
        "overall_lt_3px_ratio": weighted_lt_3,
        "official_test_tiles": sorted(TEST_TILES),
        "official_test_annotation_count": test_counts,
        "official_test_lt_3px_ratio": weighted_test_lt_3,
        "rows": rows,
    }

    decision_lines = []
    if weighted_lt_3 > 0.30:
        decision_lines.append(
            "Overall `<3 px` ratio exceeds 30%, so zero-shot reporting must include a detectable subset (`>= 3 px`)."
        )
    else:
        decision_lines.append(
            "Overall `<3 px` ratio stays under 30%, so full-set zero-shot metrics remain interpretable."
        )
    if weighted_test_lt_3 > 0.30:
        decision_lines.append(
            "Official test tiles (`34VEN`, `34VER`) are especially tiny-heavy, so size-binned reporting is mandatory there."
        )
    else:
        decision_lines.append(
            "Official test tiles (`34VEN`, `34VER`) are not dominated by `<3 px` objects, but size bins should still be reported."
        )

    md_lines = [
        "# Finland Size Report",
        "",
        "## Summary",
        f"- Total layers: {len(rows)}",
        f"- Total annotations: {all_counts}",
        f"- Overall `<3 px` ratio: {weighted_lt_3:.3f}",
        f"- Official test tiles (`34VEN`, `34VER`) annotations: {test_counts}",
        f"- Official test tiles `<3 px` ratio: {weighted_test_lt_3:.3f}",
        "",
        "## Decision Notes",
    ]
    md_lines.extend(f"- {line}" for line in decision_lines)
    md_lines.extend(
        [
            "",
            "## Per-Layer Statistics",
            "",
            "| Tile | Layer | Count | Median side (px) | Mean side (px) | `<3 px` count | `<3 px` ratio | Official test tile |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in rows:
        md_lines.append(
            f"| {row['tile']} | {row['layer']} | {row['count']} | {row['median_side_px']:.2f} | "
            f"{row['mean_side_px']:.2f} | {row['lt_3px_count']} | {row['lt_3px_ratio']:.3f} | "
            f"{'yes' if row['is_official_test_tile'] else 'no'} |"
        )
    md_lines.append("")

    return summary, "\n".join(md_lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Finland annotation sizes in pixel-equivalent units.")
    parser.add_argument("--annotation-dir", type=Path, default=FINLAND_ANN_DIR)
    parser.add_argument("--output-md", type=Path, default=Path("data/finland_size_report.md"))
    parser.add_argument("--output-json", type=Path, default=Path("data/finland_size_report.json"))
    args = parser.parse_args()

    summary, markdown = analyze_finland_annotations(args.annotation_dir)
    ensure_dir(args.output_md.parent)
    args.output_md.write_text(markdown, encoding="utf-8")
    write_json(args.output_json, summary)
    print(f"Wrote {args.output_md}")
    print(f"Wrote {args.output_json}")


if __name__ == "__main__":
    main()
