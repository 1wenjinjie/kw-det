from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from kwdet_common import full_coverage_starts, list_s2ships_samples, load_split_from_protocol, load_s2ships_label, write_json


def count_positive_windows(mask: np.ndarray, chip: int, stride: int) -> tuple[int, int]:
    h, w = mask.shape
    row_starts = full_coverage_starts(h, chip, stride)
    col_starts = full_coverage_starts(w, chip, stride)
    total = 0
    positive = 0
    for y0 in row_starts:
        for x0 in col_starts:
            total += 1
            if mask[y0 : y0 + chip, x0 : x0 + chip].any():
                positive += 1
    return total, positive


def estimate(split_path: Path, chip: int, strides: list[int]) -> dict:
    split = load_split_from_protocol(split_path)
    train_scenes = set(split["train"])
    scene_results: dict[str, dict] = {}
    aggregate: dict[str, dict] = {}

    for stride in strides:
        aggregate[str(stride)] = {"total_windows": 0, "positive_windows": 0}

    for sample in list_s2ships_samples():
        scene = sample["scene"]
        if scene not in train_scenes:
            continue
        mask = load_s2ships_label(scene) > 0
        h, w = mask.shape
        per_stride: dict[str, dict] = {}
        for stride in strides:
            total, positive = count_positive_windows(mask, chip, stride)
            per_stride[str(stride)] = {
                "total_windows": int(total),
                "positive_windows": int(positive),
            }
            aggregate[str(stride)]["total_windows"] += int(total)
            aggregate[str(stride)]["positive_windows"] += int(positive)

        scene_results[scene] = {"shape": [int(h), int(w)], "strides": per_stride}

    for stride in strides:
        agg = aggregate[str(stride)]
        agg["meets_gate_ge_800_before_aug"] = agg["positive_windows"] >= 800

    return {
        "chip_size": chip,
        "train_scenes": split["train"],
        "evaluated_strides": strides,
        "scene_results": scene_results,
        "aggregate": aggregate,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate dense chip counts for S2SHIPS train scenes.")
    parser.add_argument("--split", type=Path, default=Path("plan/protocol_split.md"))
    parser.add_argument("--chip", type=int, default=640)
    parser.add_argument("--strides", type=int, nargs="+", default=[512, 256, 128])
    parser.add_argument("--output", type=Path, default=Path("data/chip_estimate.json"))
    args = parser.parse_args()

    report = estimate(args.split, args.chip, args.strides)
    write_json(args.output, report)

    print(f"Chip size: {report['chip_size']}")
    for stride in args.strides:
        agg = report["aggregate"][str(stride)]
        print(
            f"stride={stride}: total_windows={agg['total_windows']}, "
            f"positive_windows={agg['positive_windows']}, "
            f"gate_ge_800={agg['meets_gate_ge_800_before_aug']}"
        )


if __name__ == "__main__":
    main()
