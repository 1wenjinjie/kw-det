from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from imageio.v3 import imwrite

from kwdet_common import BAND_TO_INDEX, AUDIT_DIR, ensure_dir, load_s2ships_data, load_water_mask, read_json


SCORE_BINS = [(0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]


def normalize_scene_band(band: np.ndarray, water_mask: np.ndarray) -> np.ndarray:
    water = water_mask > 0
    values = band[water]
    if values.size == 0:
        return np.zeros_like(band, dtype=np.float32)
    band = band.astype(np.float32)
    band_min = float(values.min())
    band_max = float(values.max())
    return np.clip((band - band_min) / max(band_max - band_min, 1e-6), 0.0, 1.0)


def draw_rectangle(image: np.ndarray, bbox: list[float], color: tuple[int, int, int], thickness: int = 2) -> None:
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    x1 = max(x1, 0)
    y1 = max(y1, 0)
    x2 = min(x2, image.shape[1] - 1)
    y2 = min(y2, image.shape[0] - 1)
    for t in range(thickness):
        image[max(y1 - t, 0) : min(y1 - t + 1, image.shape[0]), x1 : x2 + 1] = color
        image[min(y2 + t, image.shape[0] - 1) : min(y2 + t + 1, image.shape[0]), x1 : x2 + 1] = color
        image[y1 : y2 + 1, max(x1 - t, 0) : min(x1 - t + 1, image.shape[1])] = color
        image[y1 : y2 + 1, min(x2 + t, image.shape[1] - 1) : min(x2 + t + 1, image.shape[1])] = color


def crop_rgb(rgb: np.ndarray, cx: float, cy: float, half_size: int = 128) -> tuple[np.ndarray, tuple[int, int]]:
    y0 = max(int(round(cy)) - half_size, 0)
    y1 = min(int(round(cy)) + half_size, rgb.shape[0])
    x0 = max(int(round(cx)) - half_size, 0)
    x1 = min(int(round(cx)) + half_size, rgb.shape[1])
    return rgb[y0:y1, x0:x1].copy(), (x0, y0)


def load_instance_map(enhanced: dict) -> dict[tuple[str, int], dict]:
    mapping: dict[tuple[str, int], dict] = {}
    for scene_entry in enhanced["scenes"]:
        for instance in scene_entry["instances"]:
            mapping[(scene_entry["scene"], int(instance["id"]))] = instance
    return mapping


def stratified_sample(wakes: list[dict], per_bin: int) -> list[dict]:
    selected: list[dict] = []
    for low, high in SCORE_BINS:
        bucket = [wake for wake in wakes if low <= wake["s_wake"] < high]
        bucket.sort(key=lambda item: (-item["s_wake"], item["id"]))
        selected.extend(bucket[: min(per_bin, len(bucket))])
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Create wake audit crops and CSV template.")
    parser.add_argument("--wake-json", type=Path, default=Path("data/labels/wake_pseudolabels.json"))
    parser.add_argument("--enhanced-json", type=Path, default=Path("data/labels/s2ships_enhanced.json"))
    parser.add_argument("--output-dir", type=Path, default=AUDIT_DIR / "wake_audit_samples")
    parser.add_argument("--csv-output", type=Path, default=AUDIT_DIR / "wake_audit_labels.csv")
    parser.add_argument("--report-output", type=Path, default=AUDIT_DIR / "wake_quality_report.md")
    parser.add_argument("--per-bin", type=int, default=80)
    args = parser.parse_args()

    wake_payload = read_json(args.wake_json)
    enhanced = read_json(args.enhanced_json)
    instance_map = load_instance_map(enhanced)
    ensure_dir(args.output_dir)
    ensure_dir(args.csv_output.parent)

    all_wakes = [wake for scene_entry in wake_payload["scenes"] for wake in scene_entry["wakes"]]
    sampled_wakes = stratified_sample(all_wakes, args.per_bin)

    rows: list[dict[str, str]] = []
    scene_cache: dict[str, np.ndarray] = {}

    for wake in sampled_wakes:
        scene = wake["scene"]
        if scene not in scene_cache:
            data = load_s2ships_data(scene)
            water = load_water_mask(scene, shape=data.shape[:2])
            b08_norm = normalize_scene_band(data[:, :, BAND_TO_INDEX["B08"]], water)
            gray = (b08_norm * 255.0).clip(0, 255).astype(np.uint8)
            scene_cache[scene] = np.repeat(gray[:, :, None], 3, axis=2)

        instance = instance_map[(scene, int(wake["ship_instance_id"]))]
        rgb_crop, (x0, y0) = crop_rgb(scene_cache[scene], *wake["centroid_xy"])
        wake_bbox = [
            wake["bbox_xyxy"][0] - x0,
            wake["bbox_xyxy"][1] - y0,
            wake["bbox_xyxy"][2] - x0,
            wake["bbox_xyxy"][3] - y0,
        ]
        ship_bbox = [
            instance["bbox_xyxy"][0] - x0,
            instance["bbox_xyxy"][1] - y0,
            instance["bbox_xyxy"][2] - x0,
            instance["bbox_xyxy"][3] - y0,
        ]
        draw_rectangle(rgb_crop, ship_bbox, (0, 255, 0), thickness=2)
        draw_rectangle(rgb_crop, wake_bbox, (255, 0, 0), thickness=2)

        image_path = args.output_dir / f"{wake['id']}.png"
        imwrite(image_path, rgb_crop)

        bin_name = next(
            f"{low:.1f}-{high:.1f}" for low, high in SCORE_BINS if low <= wake["s_wake"] < high
        )
        rows.append(
            {
                "wake_id": wake["id"],
                "scene": scene,
                "ship_instance_id": str(wake["ship_instance_id"]),
                "s_wake": f"{wake['s_wake']:.3f}",
                "score_bin": bin_name,
                "judgment": "",
                "image_path": str(image_path.relative_to(Path.cwd())),
            }
        )

    with args.csv_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["wake_id", "scene", "ship_instance_id", "s_wake", "score_bin", "judgment", "image_path"],
        )
        writer.writeheader()
        writer.writerows(rows)

    report_text = "\n".join(
        [
            "# Wake Quality Report",
            "",
            "## Status",
            "- Pending manual audit.",
            "",
            "## Audit Protocol",
            "- Fill `data/audit/wake_audit_labels.csv` with one of: `correct`, `ambiguous`, `wrong`.",
            "- After manual review, replace this pending note with a precision summary and one of:",
            "  - `hard supervision`",
            "  - `attention prior`",
            "  - `disable wake supervision`",
            "",
            "## Auto-generated Sample Counts",
            f"- Sampled wakes: {len(rows)}",
            f"- Requested per score bin: {args.per_bin}",
            "",
            "## Decision",
            "- Pending manual audit.",
        ]
    )
    args.report_output.write_text(report_text, encoding="utf-8")

    print(f"Generated {len(rows)} audit samples in {args.output_dir}")
    print(f"Wrote CSV template to {args.csv_output}")


if __name__ == "__main__":
    main()
