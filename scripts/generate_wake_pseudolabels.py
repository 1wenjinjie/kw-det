from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from skimage.transform import radon

from kwdet_common import (
    BAND_TO_INDEX,
    angle_diff_deg,
    ensure_dir,
    float_round,
    load_s2ships_data,
    load_water_mask,
    normalize_scene_band,
    pca_obb_from_coords,
    read_json,
    write_json,
)


ANGLE_SET = np.arange(0, 180, 12, dtype=float)


def build_oriented_kernel(angle_deg: float, size: int = 21, sigma_long: float = 8.0, sigma_short: float = 1.2) -> np.ndarray:
    radius = size // 2
    yy, xx = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    theta = math.radians(angle_deg)
    xr = xx * math.cos(theta) + yy * math.sin(theta)
    yr = -xx * math.sin(theta) + yy * math.cos(theta)
    kernel = np.exp(-0.5 * ((xr / sigma_long) ** 2 + (yr / sigma_short) ** 2))
    kernel = kernel - kernel.mean()
    kernel /= np.abs(kernel).sum() + 1e-6
    return kernel.astype(np.float32)


KERNELS = {float(angle): build_oriented_kernel(float(angle)) for angle in ANGLE_SET}


def extract_patch(arr: np.ndarray, cx: float, cy: float, half_size: int) -> tuple[np.ndarray, tuple[int, int]]:
    y0 = max(int(round(cy)) - half_size, 0)
    y1 = min(int(round(cy)) + half_size, arr.shape[0])
    x0 = max(int(round(cx)) - half_size, 0)
    x1 = min(int(round(cx)) + half_size, arr.shape[1])
    return arr[y0:y1, x0:x1], (x0, y0)


def best_wake_for_instance(
    b08_norm: np.ndarray,
    water_mask: np.ndarray,
    coast_distance: np.ndarray,
    instance: dict,
) -> dict | None:
    ship_cx, ship_cy, ship_w, ship_h, ship_theta = instance["obb_cxcywha"]
    ship_len = max(ship_w, ship_h)
    half_size = int(np.clip(max(64.0, ship_len * 3.0), 64.0, 160.0))
    patch, (x0, y0) = extract_patch(b08_norm, ship_cx, ship_cy, half_size)
    water_patch, _ = extract_patch(water_mask, ship_cx, ship_cy, half_size)
    if patch.size == 0 or water_patch.sum() == 0:
        return None

    patch = patch.astype(np.float32)
    water_patch = water_patch > 0
    patch_hp = patch - ndi.gaussian_filter(patch, sigma=3.0)
    patch_hp *= water_patch.astype(np.float32)

    response_stack = np.stack(
        [ndi.convolve(patch_hp, kernel, mode="nearest") for kernel in KERNELS.values()],
        axis=0,
    )
    best_response = response_stack.max(axis=0)
    response_values = best_response[water_patch]
    if response_values.size == 0:
        return None
    threshold = float(np.percentile(response_values, 99.0))
    candidate_mask = (best_response > threshold) & water_patch

    ship_bbox = instance["bbox_xyxy"]
    sx1 = max(int(math.floor(ship_bbox[0])) - x0 - 2, 0)
    sy1 = max(int(math.floor(ship_bbox[1])) - y0 - 2, 0)
    sx2 = min(int(math.ceil(ship_bbox[2])) - x0 + 3, candidate_mask.shape[1])
    sy2 = min(int(math.ceil(ship_bbox[3])) - y0 + 3, candidate_mask.shape[0])
    candidate_mask[sy1:sy2, sx1:sx2] = False

    if candidate_mask.sum() == 0:
        return None

    # Exclude ship body from Radon patch so the peak encodes wake direction,
    # not the (stronger) ship hull response.
    radon_patch = np.clip(best_response, 0.0, None)
    radon_patch[sy1:sy2, sx1:sx2] = 0.0
    theta_candidates = np.arange(0, 180, 1, dtype=float)
    sinogram = radon(radon_patch, theta=theta_candidates, circle=False)
    radon_idx = int(np.argmax(sinogram.max(axis=0)))
    theta_radon = float((theta_candidates[radon_idx] + 90.0) % 180.0)

    labeled, n_components = ndi.label(candidate_mask)
    slices = ndi.find_objects(labeled)
    best_candidate: dict | None = None
    best_score = -1.0

    for component_id in range(1, n_components + 1):
        component_slice = slices[component_id - 1]
        if component_slice is None:
            continue
        component_mask = labeled[component_slice] == component_id
        area = int(component_mask.sum())
        if area < 4:
            continue

        ys_local, xs_local = np.nonzero(component_mask)
        xs_patch = xs_local + int(component_slice[1].start)
        ys_patch = ys_local + int(component_slice[0].start)
        xs = xs_patch + x0
        ys = ys_patch + y0
        coords = np.column_stack([xs, ys])
        obb = pca_obb_from_coords(coords)
        len_wake = max(obb["w"], obb["h"])
        if len_wake <= 0.5 * ship_len:
            continue

        centroid_x = obb["cx"]
        centroid_y = obb["cy"]
        closest_dist = float(np.min(np.hypot(xs - ship_cx, ys - ship_cy)))
        if closest_dist >= 2.0 * ship_len:
            continue

        coast_min = float(coast_distance[ys, xs].min())
        if coast_min < 5.0:
            continue

        theta_wake_deg = float(math.degrees(obb["theta"]) % 180.0)
        diff_deg = angle_diff_deg(theta_wake_deg, theta_radon)
        if diff_deg > 20.0:
            continue

        component_response = best_response[ys_patch, xs_patch]
        response_strength = float(component_response.max() / max(response_values.max(), 1e-6))
        linearity_score = float(np.clip((obb["aspect_ratio"] - 1.0) / 9.0, 0.0, 1.0))
        radon_agreement = float(np.clip(1.0 - diff_deg / 20.0, 0.0, 1.0))
        score = 0.4 * response_strength + 0.3 * linearity_score + 0.3 * radon_agreement
        if score <= 0.7 or score <= best_score:
            continue

        x_min = float(xs.min())
        x_max = float(xs.max())
        y_min = float(ys.min())
        y_max = float(ys.max())
        best_candidate = {
            "theta_wake": float_round(obb["theta"]),
            "len_wake": float_round(len_wake),
            "s_wake": float_round(score),
            "centroid_xy": [float_round(centroid_x), float_round(centroid_y)],
            "bbox_xyxy": [
                float_round(x_min),
                float_round(y_min),
                float_round(x_max),
                float_round(y_max),
            ],
            "response_strength": float_round(response_strength),
            "linearity_score": float_round(linearity_score),
            "radon_theta_deg": float_round(theta_radon),
            "radon_agreement": float_round(radon_agreement),
        }
        best_score = score

    return best_candidate


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate wake pseudolabels from S2SHIPS B08.")
    parser.add_argument("--enhanced", type=Path, default=Path("data/labels/s2ships_enhanced.json"))
    parser.add_argument("--output", type=Path, default=Path("data/labels/wake_pseudolabels.json"))
    args = parser.parse_args()

    enhanced = read_json(args.enhanced)
    wake_scenes: list[dict] = []

    for scene_entry in enhanced["scenes"]:
        scene = scene_entry["scene"]
        data = load_s2ships_data(scene)
        b08 = data[:, :, BAND_TO_INDEX["B08"]]
        water_mask = load_water_mask(scene, shape=b08.shape)
        b08_norm = normalize_scene_band(b08, water_mask)
        coast_distance = ndi.distance_transform_edt(water_mask > 0)

        wakes: list[dict] = []
        for instance in scene_entry["instances"]:
            candidate = best_wake_for_instance(b08_norm, water_mask, coast_distance, instance)
            if candidate is None:
                continue

            wake_id = f"{scene}_w{len(wakes) + 1:04d}"
            candidate["id"] = wake_id
            candidate["scene"] = scene
            candidate["ship_instance_id"] = instance["id"]
            candidate["ship_global_id"] = instance["global_id"]
            wakes.append(candidate)
            instance["wake_id"] = wake_id

        wake_scenes.append({"scene": scene, "wakes": wakes})

    output_payload = {
        "metadata": {
            "source_band": "B08",
            "score_threshold": 0.7,
            "angle_grid_deg": ANGLE_SET.tolist(),
        },
        "scenes": wake_scenes,
    }

    ensure_dir(args.output.parent)
    write_json(args.output, output_payload)
    write_json(args.enhanced, enhanced)

    total_wakes = sum(len(scene["wakes"]) for scene in wake_scenes)
    print(f"Generated {total_wakes} wake pseudolabels across {len(wake_scenes)} scenes.")


if __name__ == "__main__":
    main()
