from __future__ import annotations

import json
import math
import re
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
PLAN_DIR = ROOT / "plan"
SCRIPTS_DIR = ROOT / "scripts"

S2SHIPS_DIR = DATA_DIR / "S2SHIPS"
S2SHIPS_NPY_DIR = S2SHIPS_DIR / "dataset_npy"
S2SHIPS_TIF_DIR = S2SHIPS_DIR / "dataset_tif"
S2SHIPS_WATER_DIR = S2SHIPS_DIR / "water_mask"

FINLAND_ANN_DIR = DATA_DIR / "finland_2025_annotations"
FINLAND_IMG_DIR = DATA_DIR / "finland_2025_imagery" / "l2a"
FINLAND_MANIFEST_PATH = FINLAND_IMG_DIR / "finland_l2a_manifest.json"

LABEL_DIR = DATA_DIR / "labels"
OBB_DIR = LABEL_DIR / "obb"
AUDIT_DIR = DATA_DIR / "audit"

PROTOCOL_SPLIT_PATH = PLAN_DIR / "protocol_split.md"

BAND_ORDER = [
    "B01",
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B09",
    "B11",
    "B12",
    "B8A",
]
BAND_TO_INDEX = {band: idx for idx, band in enumerate(BAND_ORDER)}

DEFAULT_TRAIN_SCENES = [
    "rome",
    "suez1",
    "suez2",
    "suez3",
    "suez4",
    "suez5",
    "brest1",
    "toulon",
    "marseille",
    "rotterdam1",
    "rotterdam2",
    "rotterdam3",
    "southampton",
]
DEFAULT_VAL_SCENES = ["portsmouth", "panama", "suez6"]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)


def scene_name_from_npy_name(name: str) -> str:
    stem = Path(name).stem
    match = re.match(r"^\d+_mask_(.+)$", stem)
    if match:
        return match.group(1)
    return stem


def list_s2ships_samples() -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for npy_path in sorted(S2SHIPS_NPY_DIR.glob("*.npy")):
        scene = scene_name_from_npy_name(npy_path.name)
        samples.append({"scene": scene, "npy_path": npy_path, "stem": npy_path.stem})
    return samples


def find_s2ships_npy(scene: str) -> Path:
    for sample in list_s2ships_samples():
        if sample["scene"] == scene:
            return sample["npy_path"]
    raise FileNotFoundError(f"Could not find S2SHIPS npy for scene '{scene}'")


def load_s2ships_sample(scene_or_path: str | Path) -> dict[str, Any]:
    path = Path(scene_or_path)
    if not path.exists():
        path = find_s2ships_npy(str(scene_or_path))
    sample = np.load(path, allow_pickle=True).item()
    return sample


def load_s2ships_data(scene: str) -> np.ndarray:
    return load_s2ships_sample(scene)["data"]


def load_s2ships_label(scene: str) -> np.ndarray:
    sample = load_s2ships_sample(scene)
    return sample["label"][:, :, 0]


def load_water_mask(scene: str, shape: tuple[int, int] | None = None) -> np.ndarray:
    with rasterio.open(S2SHIPS_WATER_DIR / f"{scene}_water.tif") as src:
        water = src.read(1)
    if shape is not None:
        if water.shape[0] < shape[0] or water.shape[1] < shape[1]:
            raise ValueError(
                f"Water mask for '{scene}' is {water.shape}, "
                f"smaller than requested shape {shape}. "
                "Cannot safely crop to target dimensions."
            )
        water = water[: shape[0], : shape[1]]
    return water


def find_band_tif(scene: str, band: str) -> Path:
    scene_dir = S2SHIPS_TIF_DIR / scene
    matches = [
        path
        for path in sorted(scene_dir.glob("*.tiff"))
        if band in path.name and "Raw" in path.name and "True_color" not in path.name and "NDWI" not in path.name
    ]
    if not matches:
        raise FileNotFoundError(f"Missing band {band} for scene {scene} in {scene_dir}")
    return matches[0]


def list_gpkg_layers(gpkg_path: Path) -> list[str]:
    with sqlite3.connect(gpkg_path) as conn:
        rows = conn.execute(
            "SELECT table_name FROM gpkg_contents WHERE data_type IN ('features', 'attributes')"
        ).fetchall()
    return [row[0] for row in rows]


def normalize_angle_half_pi(angle_rad: float) -> float:
    while angle_rad < -math.pi / 2:
        angle_rad += math.pi
    while angle_rad >= math.pi / 2:
        angle_rad -= math.pi
    return angle_rad


def angle_diff_deg(angle_a_deg: float, angle_b_deg: float) -> float:
    diff = abs(((angle_a_deg - angle_b_deg + 90.0) % 180.0) - 90.0)
    return diff


def full_coverage_starts(length: int, chip: int, stride: int) -> list[int]:
    if length <= chip:
        return [0]
    starts = list(range(0, max(length - chip, 0) + 1, stride))
    last_start = length - chip
    if starts[-1] != last_start:
        starts.append(last_start)
    return sorted(set(starts))


def component_slices(mask: np.ndarray) -> tuple[np.ndarray, list[slice | tuple[slice, slice] | None]]:
    labeled, _ = ndi.label(mask > 0)
    return labeled, list(ndi.find_objects(labeled))


def bbox_from_coords(coords_xy: np.ndarray) -> tuple[float, float, float, float]:
    x_min = float(coords_xy[:, 0].min())
    x_max = float(coords_xy[:, 0].max())
    y_min = float(coords_xy[:, 1].min())
    y_max = float(coords_xy[:, 1].max())
    return x_min, y_min, x_max, y_max


def pca_obb_from_coords(coords_xy: np.ndarray) -> dict[str, float]:
    if coords_xy.shape[0] == 0:
        raise ValueError("Cannot compute OBB on empty coordinates")

    coords_xy = coords_xy.astype(np.float64)
    centroid = coords_xy.mean(axis=0)

    if coords_xy.shape[0] == 1:
        return {
            "cx": float(coords_xy[0, 0]),
            "cy": float(coords_xy[0, 1]),
            "w": 1.0,
            "h": 1.0,
            "theta": 0.0,
            "aspect_ratio": 1.0,
        }

    centered = coords_xy - centroid
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]
    proj = centered @ eigvecs
    min_proj = proj.min(axis=0)
    max_proj = proj.max(axis=0)
    sizes = max_proj - min_proj + 1.0
    center_local = (min_proj + max_proj) / 2.0
    center_xy = centroid + center_local @ eigvecs.T

    width = float(sizes[0])
    height = float(sizes[1])
    angle = math.atan2(eigvecs[1, 0], eigvecs[0, 0])
    if width < height:
        width, height = height, width
        angle += math.pi / 2
    angle = normalize_angle_half_pi(angle)
    aspect_ratio = float(width / max(height, 1e-6))

    return {
        "cx": float(center_xy[0]),
        "cy": float(center_xy[1]),
        "w": width,
        "h": height,
        "theta": float(angle),
        "aspect_ratio": aspect_ratio,
    }


def scene_dimensions_from_label(label: np.ndarray) -> tuple[int, int]:
    return int(label.shape[0]), int(label.shape[1])


def load_split_from_protocol(path: Path | None = None) -> dict[str, list[str]]:
    path = path or PROTOCOL_SPLIT_PATH
    if not path.exists():
        return {"train": DEFAULT_TRAIN_SCENES.copy(), "val": DEFAULT_VAL_SCENES.copy()}

    train_scenes: list[str] | None = None
    val_scenes: list[str] | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if text.lower().startswith("- train scenes:"):
            train_scenes = [item.strip() for item in text.split(":", 1)[1].split(",") if item.strip()]
        if text.lower().startswith("- val scenes:"):
            val_scenes = [item.strip() for item in text.split(":", 1)[1].split(",") if item.strip()]

    if train_scenes is None or val_scenes is None:
        raise ValueError(f"Could not parse train/val scenes from {path}")
    return {"train": train_scenes, "val": val_scenes}


def product_asset_path(product_dir: Path, filename: str) -> Path:
    return product_dir / filename


def float_round(value: float, digits: int = 6) -> float:
    return round(float(value), digits)


def normalize_scene_band(band: np.ndarray, water_mask: np.ndarray) -> np.ndarray:
    """Normalize a single spectral band to [0, 1] using water-pixel statistics."""
    water = water_mask > 0
    values = band[water]
    if values.size == 0:
        return np.zeros_like(band, dtype=np.float32)
    band = band.astype(np.float32)
    band_min = float(values.min())
    band_max = float(values.max())
    return np.clip((band - band_min) / max(band_max - band_min, 1e-6), 0.0, 1.0)
