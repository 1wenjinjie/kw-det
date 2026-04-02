# Protocol: Scene Split

## Locked Scope

- Authority: `plan/stage0_stage1_detail.md`
- This split is fixed across Stage 0 and Stage 1.
- Finland is excluded from all train and validation operations in Stage 0 and Stage 1.

## Locked Split

- Train scenes: rome, suez1, suez2, suez3, suez4, suez5, brest1, toulon, marseille, rotterdam1, rotterdam2, rotterdam3, southampton
- Val scenes: portsmouth, panama, suez6

## Rationale

- `portsmouth`, `panama`, and `suez6` provide geographic spread and reduce same-port leakage.
- All later audits, label exports, and experiments must consume this exact split.
