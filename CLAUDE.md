# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

**KW-Det** — *Kinematic Wake-guided Detector for Multispectral Ship Detection in Sentinel-2 Imagery*

Research hypothesis: Sentinel-2 multispectral bands (especially NIR/SWIR) provide physically-grounded priors for ship detection under thin cloud, high sea state, and tiny-target conditions that outperform pure-RGB approaches when combined with wake-guided fusion and geometry-consistency constraints.

## Stack

- **Language:** Python 3
- **Deep Learning:** PyTorch + Ultralytics (YOLO-based)
- **Geospatial:** rasterio / GDAL, geopandas
- **Data source:** AWS Element84 Earth Search STAC API (Sentinel-2 L2A)

Environment dependencies (to be captured in `requirements.txt` during Stage 0):
`torch`, `ultralytics`, `rasterio`, `gdal`, `geopandas`, `pycocotools`, `requests`

## Data Scripts

```bash
# Download Finland L2A imagery matching annotation GPKGs
python scripts/download_finland_l2a.py \
  --annotations-dir data/finland_2025_annotations \
  --output-dir data/finland_2025_imagery/l2a \
  --workers 4 \
  --timeout 60
```

The script performs resume-safe downloads (`.part` files + size check) with concurrent workers.

## Dataset Structure

```
data/
├── S2SHIPS/                    # Primary training dataset (5.13 GB, 16 scenes)
│   ├── dataset_npy/            # NumPy: data=(938,1783,12), label=(938,1783,1)
│   ├── dataset_tif/            # GeoTIFF originals
│   ├── s2ships_labels_mask/    # Ground-truth masks — TRUE label source
│   ├── water_mask/
│   └── pretrained_backbones/
├── finland_2025_annotations/   # External validation (5 GPKGs, 8,866 boat polygons)
│   └── *.gpkg                  # One file per MGRS tile (34VEM/34VEN/34VER/34WFT/35VLG)
└── finland_2025_imagery/l2a/   # Downloaded Sentinel-2 L2A products
    ├── finland_l2a_manifest.json
    └── S2*_<TILE>_<DATE>_*/   # Per-product dirs with band TIFFs + item.json
```

## Critical Data Decisions

**Do NOT use the pre-built COCO files** (`coco-1622202804.029818.json`, `coco-s2ships.json`) — they are incomplete exports with inconsistent class definitions. The true label source is `s2ships_labels_mask` / `dataset_npy` labels.

**Label reconstruction protocol:**
- S2SHIPS: reconstruct instance-level bounding boxes from masks (connected components → axis-aligned box + OBB with main-axis angle)
- Finland: export axis-aligned boxes from GPKG polygons; Finland annotations intentionally include part of the wake — do not attempt to align them to S2SHIPS tight-hull style

**Band protocol** (main experiments use 6 bands):
| Stream | Bands | Resolution | Purpose |
|--------|-------|-----------|---------|
| Stream A (RGB) | B02, B03, B04 | 10 m | Ship hull texture |
| Stream B (Wake) | B08 | 10 m | Primary wake cue |
| Stream B (Wake) | B8A, B11 | 20 m | NIR/SWIR water surface |

All multi-resolution fusion must explicitly record the resampling strategy used.

## Architecture Overview

Two-stream detection network built on YOLO/Ultralytics backbone:

```
Stream A (B02/B03/B04) ─────────────────────────────┐
                                                      ├─ Wake-guided Fusion ─ Neck/FPN ─ Head
Stream B (B08/B8A/B11) ─ shallow large-RF module ───┘
                            (no shared weights with A)

F_fused = F_rgb ⊗ (1 + α · M_wake)   # α learnable, init 0.1
```

**Geometry losses** (added after main-task convergence, λ warmed up 0.1→1.0):
- `L_axis = 1 - |cos(θ_ship − θ_wake)|` — main-axis consistency (not bow direction, avoids 180° ambiguity)
- `L_scale = 𝟙(s_wake > τ) · max(0, r − len_wake/len_ship)` — masked soft scale prior, high-confidence wakes only

## Evaluation Protocol

**S2SHIPS (internal):** mAP, Recall, small-object recall, hard-case recall

**Finland zero-shot (external, no fine-tuning):** mAP, Recall, center-hit rate, relaxed IoU, per-tile breakdown. Primary official test tiles: `34VEN` + `34VER`.

Finland zero-shot uses relaxed metrics because Finland annotations include wakes while S2SHIPS labels target hull only — single mAP comparison would be misleading.

## Project Stages & Timeline

| Stage | Dates | Goal |
|-------|-------|------|
| 0 | Mar 30 – Apr 12 | Env setup, Finland imagery, protocol unification |
| 1 | Apr 13 – Apr 26 | S2SHIPS label reconstruction + wake pseudo-label audit |
| 2 | Apr 27 – May 10 | Strong baseline suite (RGB / RGB+B08 / all-band / OBB) |
| 3 | May 11 – May 24 | Two-stream + wake-guided fusion |
| 4 | May 25 – Jun 7 | Geometry consistency + joint training |
| 5 | Jun 8 – Jun 14 | Hard-case robustness, ablation, Finland zero-shot |
| 6 | Jun 15 – Jun 21 | Visualization + paper writing |

## Wake Pseudo-label Generation (Stage 1)

Input: B08 (primary), optionally B08+B8A+B11. Pipeline: directional Gaussian filter → line structure enhancement → Radon transform → connected-component + direction filtering.

Each candidate wake produces: `theta_wake`, `len_wake`, `s_wake`. Retain only `s_wake > 0.7` and wakes adjacent to a ship instance.

**Decision gate:** if high-confidence wake precision < 0.8 on audited sample (200–300 instances), use wakes only as attention prior, not hard supervision.
