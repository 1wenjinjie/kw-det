# Protocol: Bands and Resampling

## Locked Scope

- Authority: `plan/stage0_stage1_detail.md`
- Effective date: `2026-04-01`
- This protocol is locked for Stage 0 and Stage 1.

## Verified S2SHIPS Band Order

The `dataset_npy/*.npy` tensors were verified against the raw GeoTIFF bands by exact array equality.

| Channel index | Band |
| --- | --- |
| 0 | B01 |
| 1 | B02 |
| 2 | B03 |
| 3 | B04 |
| 4 | B05 |
| 5 | B06 |
| 6 | B07 |
| 7 | B08 |
| 8 | B09 |
| 9 | B11 |
| 10 | B12 |
| 11 | B8A |

## Stream Assignment

- Stream A: `B02`, `B03`, `B04`
- Stream B primary: `B08`
- Stream B auxiliary: `B8A`, `B11`

## Resampling Rule

- Main rule: `offline resample`
- Target resolution: `10 m`
- Applied to: `B8A`, `B11`
- Reference band: `B08`
- Interpolation: `rasterio.enums.Resampling.bilinear`
- Label masks and water masks are never bilinearly resampled

## Normalization Rule

- Source tensors are treated as raw numeric bands and kept in their original scale during data staging
- Model-stage normalization will use per-band `mean/std` statistics computed on the S2SHIPS train split
- RGB-only color-space augmentation is forbidden for multispectral training

## CLI Reference

Single raster:

```powershell
python scripts/resample_bands.py `
  --source path\\to\\B8A.tif `
  --reference path\\to\\B08.tif `
  --output path\\to\\B8A_10m.tif
```

Whole product:

```powershell
python scripts/resample_bands.py `
  --product-dir data\\finland_2025_imagery\\l2a\\S2A_34VEM_20220515_0_L2A `
  --output-dir data\\finland_2025_imagery\\l2a_resampled
```
