from __future__ import annotations

import argparse
from pathlib import Path

from pycocotools.coco import COCO

from kwdet_common import OBB_DIR, read_json


def validate_coco(path: Path, min_annotations: int) -> tuple[int, int]:
    coco = COCO(str(path))
    n_images = len(coco.imgs)
    n_annotations = len(coco.anns)
    if n_annotations < min_annotations:
        raise ValueError(f"{path} only has {n_annotations} annotations; expected >= {min_annotations}")
    return n_images, n_annotations


def validate_obb_dir(obb_dir: Path) -> int:
    total_lines = 0
    txt_files = sorted(obb_dir.glob("*.txt"))
    if not txt_files:
        raise ValueError(f"No OBB txt files found in {obb_dir}")
    for txt_path in txt_files:
        for line in txt_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            vals = list(map(float, stripped.split()))
            if len(vals) != 6:
                raise ValueError(f"Bad OBB line in {txt_path}: {line}")
            if not (0.0 <= vals[1] <= 1.0 and 0.0 <= vals[2] <= 1.0):
                raise ValueError(f"OBB center out of range in {txt_path}: {line}")
            total_lines += 1
    return total_lines


def validate_enhanced(path: Path) -> set[tuple[str, int]]:
    payload = read_json(path)
    instance_keys: set[tuple[str, int]] = set()
    total_instances = 0
    for scene_entry in payload["scenes"]:
        for instance in scene_entry["instances"]:
            key = (scene_entry["scene"], int(instance["id"]))
            instance_keys.add(key)
            total_instances += 1
    if total_instances <= 0:
        raise ValueError(f"No instances found in {path}")
    return instance_keys


def validate_wakes(path: Path, instance_keys: set[tuple[str, int]]) -> int:
    payload = read_json(path)
    total_wakes = 0
    for scene_entry in payload["scenes"]:
        scene = scene_entry["scene"]
        for wake in scene_entry["wakes"]:
            for field in ["theta_wake", "len_wake", "s_wake", "ship_instance_id", "bbox_xyxy", "centroid_xy"]:
                if field not in wake:
                    raise ValueError(f"Missing field '{field}' in wake {wake.get('id')}")
            key = (scene, int(wake["ship_instance_id"]))
            if key not in instance_keys:
                raise ValueError(f"Wake {wake.get('id')} references unknown ship instance {key}")
            total_wakes += 1
    return total_wakes


def validate_report(report_path: Path) -> None:
    if not report_path.exists():
        raise FileNotFoundError(f"Missing wake quality report: {report_path}")
    text = report_path.read_text(encoding="utf-8").lower()
    if "pending manual audit" in text:
        raise ValueError(f"Wake quality report is still pending: {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Stage 1 COCO, OBB, and wake label outputs.")
    parser.add_argument("--train", type=Path, default=Path("data/labels/s2ships_coco_train.json"))
    parser.add_argument("--val", type=Path, default=Path("data/labels/s2ships_coco_val.json"))
    parser.add_argument("--enhanced", type=Path, default=Path("data/labels/s2ships_enhanced.json"))
    parser.add_argument("--wakes", type=Path, default=Path("data/labels/wake_pseudolabels.json"))
    parser.add_argument("--obb-dir", type=Path, default=OBB_DIR)
    parser.add_argument("--report", type=Path, default=Path("data/audit/wake_quality_report.md"))
    parser.add_argument("--require-wake-audit", action="store_true")
    args = parser.parse_args()

    train_images, train_annotations = validate_coco(args.train, min_annotations=800)
    val_images, val_annotations = validate_coco(args.val, min_annotations=150)
    obb_lines = validate_obb_dir(args.obb_dir)
    instance_keys = validate_enhanced(args.enhanced)
    total_wakes = 0
    if args.wakes.exists():
        total_wakes = validate_wakes(args.wakes, instance_keys)
    if args.require_wake_audit:
        validate_report(args.report)

    print(
        f"Validation passed: train_images={train_images}, train_annotations={train_annotations}, "
        f"val_images={val_images}, val_annotations={val_annotations}, obb_lines={obb_lines}, wakes={total_wakes}"
    )


if __name__ == "__main__":
    main()
