## 1. Orientation fix

- [x] 1.1 Change `equiarea._apply_axis` to `(x, y, z) → (x, y, -z)` (= `transform.B[:3,:3]`) and make `_apply_axis_inv` identical (B is an involution); document why it must match the geometry basis.

## 2. Tests

- [x] 2.1 Regression test pinning `_apply_axis == B[:3,:3]` over random directions, plus the `+y`-preserved / `+z`-flipped axis checks.
- [x] 2.2 Test that `_apply_axis` and `_apply_axis_inv` coincide and round-trip to identity (involution).
- [x] 2.3 Existing equiarea/env-wiring tests stay green; full pbrt non-gpu suite green.

## 3. Render verification (manual A/B)

- [x] 3.1 Re-render `sss_dragon_small.pbrt`: sky hue b/g 1.21 == pbrt 1.21, ground 1.06→1.23 (pbrt 1.22), scene mean 0.039→0.081 (~2×). Captured pbrt | before | after side-by-side.

## 4. Validate

- [x] 4.1 `openspec validate pbrt-env-equiarea-orientation --strict`; ruff clean.
