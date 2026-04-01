# Protocol: Chip Strategy

## Locked Scope

- Authority: `plan/stage0_stage1_detail.md`
- This protocol describes the Stage 0 locked chip and augmentation policy.

## Locked Parameters

- Chip size: `640`
- Dense-grid stride: `128`
- Purpose: maximize positive coverage on the small 16-scene S2SHIPS corpus

## Estimation Rule

- `scripts/estimate_chips.py` must report dense-grid counts for `stride = 512 / 256 / 128`
- The locked extraction setting for downstream training preparation is `chip=640, stride=128`

## Allowed Augmentations

- `RandomFlip` (horizontal / vertical) with synchronized transforms across all bands and labels
- `RandomRotate90` with synchronized transforms across all bands and labels
- Overlapping random crops derived from the locked dense-grid policy
- Per-band intensity scaling in a conservative range such as `[0.8, 1.2]`
- Mild Gaussian noise to simulate sea-state variation

## Forbidden Augmentations

- RGB-only color jitter or HSV/Lab perturbation
- Direct mosaic across scenes before band normalization is defined
- Any augmentation that desynchronizes multispectral bands and labels

## Notes

- Stage 0 only estimates chip counts; it does not materialize a training chip dataset yet.
- If the pre-augmentation positive-chip count remains below the `>= 800` gate, that result must be recorded explicitly before Stage 2 begins.
