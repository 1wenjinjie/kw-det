from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from kwdet_common import (
    BAND_ORDER,
    LABEL_DIR,
    OBB_DIR,
    bbox_from_coords,
    ensure_dir,
    float_round,
    list_s2ships_samples,
    load_split_from_protocol,
    load_s2ships_label,
    pca_obb_from_coords,
    write_json,
)


def build_instance_records(scene: str, label_mask: np.ndarray) -> list[dict]:
    labeled, n_components = ndi.label(label_mask > 0)
    slices = ndi.find_objects(labeled)
    instances: list[dict] = []

    for component_id in range(1, n_components + 1):
        component_slice = slices[component_id - 1]
        if component_slice is None:
            continue
        component_mask = labeled[component_slice] == component_id
        area = int(component_mask.sum())
        if area < 4:
            continue

        ys_local, xs_local = np.nonzero(component_mask)
        xs = xs_local + int(component_slice[1].start)
        ys = ys_local + int(component_slice[0].start)
        coords = np.column_stack([xs, ys])
        x_min, y_min, x_max, y_max = bbox_from_coords(coords)
        width = (x_max - x_min) + 1.0
        height = (y_max - y_min) + 1.0
        obb = pca_obb_from_coords(coords)

        instance = {
            "id": len(instances) + 1,
            "global_id": f"{scene}:{len(instances) + 1}",
            "bbox_xyxy": [
                float_round(x_min),
                float_round(y_min),
                float_round(x_max),
                float_round(y_max),
            ],
            "bbox_xywh": [
                float_round(x_min),
                float_round(y_min),
                float_round(width),
                float_round(height),
            ],
            "obb_cxcywha": [
                float_round(obb["cx"]),
                float_round(obb["cy"]),
                float_round(obb["w"]),
                float_round(obb["h"]),
                float_round(obb["theta"]),
            ],
            "area_px": area,
            "large_vessel": area > 50000,
            "suspicious": obb["aspect_ratio"] > 30.0,
            "aspect_ratio": float_round(obb["aspect_ratio"]),
            "centroid_xy": [float_round(obb["cx"]), float_round(obb["cy"])],
            "wake_id": None,
        }
        instances.append(instance)

    return instances


def coco_image_entry(image_id: int, scene: str, width: int, height: int, split: str) -> dict:
    return {
        "id": image_id,
        "file_name": f"{scene}.npy",
        "width": width,
        "height": height,
        "scene": scene,
        "split": split,
    }


def instance_to_coco(annotation_id: int, image_id: int, instance: dict) -> dict:
    return {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": 1,
        "bbox": instance["bbox_xywh"],
        "area": instance["area_px"],
        "iscrowd": 0,
        "segmentation": [],
        "attributes": {
            "scene_instance_id": instance["id"],
            "global_instance_id": instance["global_id"],
            "suspicious": instance["suspicious"],
            "large_vessel": instance["large_vessel"],
        },
        "obb_cxcywha": instance["obb_cxcywha"],
    }


def instance_to_obb_line(instance: dict, width: int, height: int) -> str:
    cx, cy, obb_w, obb_h, theta = instance["obb_cxcywha"]
    return (
        f"0 {cx / width:.6f} {cy / height:.6f} "
        f"{obb_w / width:.6f} {obb_h / height:.6f} {theta:.6f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild scene-level S2SHIPS COCO and OBB labels from masks.")
    parser.add_argument("--split", type=Path, default=Path("plan/protocol_split.md"))
    parser.add_argument("--train-output", type=Path, default=Path("data/labels/s2ships_coco_train.json"))
    parser.add_argument("--val-output", type=Path, default=Path("data/labels/s2ships_coco_val.json"))
    parser.add_argument("--enhanced-output", type=Path, default=Path("data/labels/s2ships_enhanced.json"))
    parser.add_argument("--obb-dir", type=Path, default=OBB_DIR)
    args = parser.parse_args()

    split = load_split_from_protocol(args.split)
    train_set = set(split["train"])
    val_set = set(split["val"])

    categories = [{"id": 1, "name": "ship"}]
    coco_train = {"images": [], "annotations": [], "categories": categories}
    coco_val = {"images": [], "annotations": [], "categories": categories}
    enhanced = {
        "metadata": {
            "source": "S2SHIPS dataset_npy label masks",
            "band_order": BAND_ORDER,
            "train_scenes": split["train"],
            "val_scenes": split["val"],
        },
        "scenes": [],
    }

    ensure_dir(args.train_output.parent)
    ensure_dir(args.obb_dir)

    annotation_id = 1
    image_id = 1

    for sample in list_s2ships_samples():
        scene = sample["scene"]
        if scene in train_set:
            split_name = "train"
            coco_target = coco_train
        elif scene in val_set:
            split_name = "val"
            coco_target = coco_val
        else:
            raise ValueError(f"Scene {scene} is not present in protocol split")

        label_mask = load_s2ships_label(scene)
        height, width = label_mask.shape
        instances = build_instance_records(scene, label_mask)

        coco_target["images"].append(coco_image_entry(image_id, scene, width, height, split_name))
        obb_lines: list[str] = []
        for instance in instances:
            coco_target["annotations"].append(instance_to_coco(annotation_id, image_id, instance))
            annotation_id += 1
            obb_lines.append(instance_to_obb_line(instance, width, height))

        (args.obb_dir / f"{scene}.txt").write_text("\n".join(obb_lines) + ("\n" if obb_lines else ""), encoding="utf-8")
        enhanced["scenes"].append(
            {
                "scene": scene,
                "split": split_name,
                "width": width,
                "height": height,
                "source_npy": str(sample["npy_path"].relative_to(Path.cwd())),
                "instances": instances,
            }
        )

        image_id += 1

    write_json(args.train_output, coco_train)
    write_json(args.val_output, coco_val)
    write_json(args.enhanced_output, enhanced)

    print(
        f"Rebuilt labels: train annotations={len(coco_train['annotations'])}, "
        f"val annotations={len(coco_val['annotations'])}."
    )
    print(f"OBB files written to {args.obb_dir}")


if __name__ == "__main__":
    main()
