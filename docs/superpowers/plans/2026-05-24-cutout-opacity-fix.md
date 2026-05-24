# Cutout-Opacity Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore binary alpha-cutout transparency for UsdPreviewSurface materials whose `opacity` is texture-driven with `opacityThreshold > 0`, fixing the sunflower quads in `assets/assets-main/full_assets/McUsd/McUsd.usda`.

**Architecture:** Single-function edit in `src/skinny/shaders/materials/flat/flat_shading.slang::fetchFlatHitData`. Split opacity-texture handling into two explicit modes: cutout (texture + threshold > 0 → force `opacity = 1.0` because `isCutoutTransparent` already drops transparent hits before sample()) vs. alpha-blend (texture, threshold == 0 → existing texture-driven opacity). Removes dead `snap-to-zero` branch.

**Tech Stack:** Slang (compiled to SPIR-V via `slangc` at runtime by `vk_compute.ComputePipeline`), Python pytest for packing regression test.

**Spec:** `docs/superpowers/specs/2026-05-24-cutout-opacity-fix-design.md`

---

## File Structure

- **Modify** `src/skinny/shaders/materials/flat/flat_shading.slang` — replace the opacity block inside `fetchFlatHitData` (current lines ~153–165). One function, one block.
- **Modify** `tests/test_struct_layout.py` — add a regression test that exercises `pack_flat_material` with a cutout configuration (opacity texture idx + threshold + channel mask `a`) and asserts the bytes land at the expected offsets so the packing side cannot silently regress again.

No other files change. The renderer, USD loader, scene graph, integrators, and `FlatMaterialParams` layout are already correct.

---

### Task 1: Pin packing for cutout opacity in tests

**Files:**
- Modify: `tests/test_struct_layout.py` (append a new test inside `TestPythonPackingSizes`)

- [ ] **Step 1: Write the failing test**

Append the following test method to the existing `TestPythonPackingSizes` class in `tests/test_struct_layout.py` (after `test_flat_material_pack_size`, keep indentation at 4 spaces to match the class):

```python
    def test_flat_material_cutout_packing(self):
        """Cutout opacity config: threshold, opacity texture idx, and
        channelMask opacity=a must land at the documented offsets.
        Guards the data-flow side of the cutout-opacity fix (design doc:
        docs/superpowers/specs/2026-05-24-cutout-opacity-fix-design.md).
        """
        from types import SimpleNamespace
        from skinny.renderer import pack_flat_material, _encode_channel_mask

        material = SimpleNamespace(
            parameter_overrides={"opacityThreshold": 0.5},
        )
        channel_mask = _encode_channel_mask({"opacity": "a"})
        data = pack_flat_material(
            material,
            opacity_texture_idx=7,
            channel_mask=channel_mask,
        )

        assert len(data) == 128
        # opacityTextureIdx at byte 76 (uint)
        assert struct.unpack_from("I", data, 76)[0] == 7
        # opacityThreshold at byte 92 (float)
        assert abs(struct.unpack_from("f", data, 92)[0] - 0.5) < 1e-6
        # channelMask at byte 108 (uint); opacity slot is bits 12..16
        mask = struct.unpack_from("I", data, 108)[0]
        assert ((mask >> 12) & 0xF) == 4  # "a" code
```

- [ ] **Step 2: Run test to verify it passes (data flow is already correct, this pins it)**

Run: `.venv/bin/pytest tests/test_struct_layout.py::TestPythonPackingSizes::test_flat_material_cutout_packing -v`

Expected: PASS. (This test is a regression guard for the data side; the bug being fixed is shader-side and not covered here.)

If the test fails, the renderer's packing has drifted from the layout documented in `pack_flat_material`. Stop and reconcile before continuing — the shader fix assumes these offsets are correct.

- [ ] **Step 3: Commit**

```bash
git add tests/test_struct_layout.py
git commit -m "test: pin cutout-opacity packing offsets in flat material" 
```

---

### Task 2: Apply the shader fix

**Files:**
- Modify: `src/skinny/shaders/materials/flat/flat_shading.slang` (the opacity block inside `fetchFlatHitData`, currently approximately lines 153–165)

- [ ] **Step 1: Locate the current opacity block**

Open `src/skinny/shaders/materials/flat/flat_shading.slang` and find the block inside `fetchFlatHitData`. It currently reads:

```slang
    out_.mat.specular = clamp(p.specular, 0.0, 1.0);
    out_.mat.ior = max(p.ior, 1.0);
    out_.mat.opacity = clamp(p.opacity, 0.0, 1.0);
    if (p.opacityTextureIdx != 0xFFFFFFFFu)
    {
        float4 s = flatMaterialTextures[NonUniformResourceIndex(p.opacityTextureIdx)]
                       .SampleLevel(h.uv, 0.0);
        out_.mat.opacity = pickChannel(s, channelFor(p.channelMask, 12u));
    }
    if (p.opacityThreshold > 0.0 && out_.mat.opacity < p.opacityThreshold)
    {
        out_.mat.opacity = 0.0;
    }
```

- [ ] **Step 2: Replace with the mode-split logic**

Replace the entire opacity-related block (from `out_.mat.opacity = clamp(p.opacity, 0.0, 1.0);` through and including the `if (p.opacityThreshold > 0.0 && ...)` block) with:

```slang
    out_.mat.specular = clamp(p.specular, 0.0, 1.0);
    out_.mat.ior = max(p.ior, 1.0);
    // Two opacity modes:
    //   Cutout (opacity texture + threshold > 0): the path/bdpt integrators
    //     already drop `alpha < threshold` hits via isCutoutTransparent, so
    //     any hit that reaches here has alpha >= threshold and the BSDF
    //     must see opacity = 1.0. Skip the texture fetch — it cannot
    //     influence the result.
    //   Alpha-blend (opacity texture, threshold == 0, OR constant opacity):
    //     sample the texture (or use the constant) and let FlatMaterial.sample
    //     route (1 - opacity) of evaluations through delta transmission /
    //     dielectric refraction.
    out_.mat.opacity = clamp(p.opacity, 0.0, 1.0);
    bool cutoutMode = (p.opacityTextureIdx != 0xFFFFFFFFu) && (p.opacityThreshold > 0.0);
    if (p.opacityTextureIdx != 0xFFFFFFFFu && !cutoutMode)
    {
        float4 s = flatMaterialTextures[NonUniformResourceIndex(p.opacityTextureIdx)]
                       .SampleLevel(h.uv, 0.0);
        out_.mat.opacity = pickChannel(s, channelFor(p.channelMask, 12u));
    }
    else if (cutoutMode)
    {
        out_.mat.opacity = 1.0;
    }
```

The two-line keep of `specular`/`ior` above the block is for orientation only — do not delete those lines.

- [ ] **Step 3: Sanity-check the file compiles via slangc**

Run: `slangc src/skinny/shaders/materials/flat/flat_shading.slang -target spirv -stage compute -entry mainImage -I src/skinny/shaders -o /tmp/skinny_check.spv 2>&1 | head -40`

Expected: either a clean exit (no output before `head` truncates) or, if `mainImage` is not defined in this module directly, a "no entry point" error — that is fine. **What you must NOT see** are syntax errors referencing `flat_shading.slang`. If the entry-point error blocks the check, instead compile the full main pass:

Run: `slangc src/skinny/shaders/main_pass.slang -target spirv -entry mainImage -stage compute -I src/skinny/shaders -o /tmp/skinny_check.spv 2>&1 | tail -20`

Expected: clean exit (file written to `/tmp/skinny_check.spv`). Any error pointing into `flat_shading.slang` means the edit broke syntax — re-read Step 2.

- [ ] **Step 4: Run the full test suite**

Run: `.venv/bin/pytest -q -k "not gpu"`

Expected: all non-GPU tests pass. The shader edit doesn't touch Python code, so behavior should be unchanged for the Python test surface.

- [ ] **Step 5: Lint**

Run: `.venv/bin/ruff check src/`

Expected: clean (the edit was Slang-only; this is just a no-regression check).

- [ ] **Step 6: Commit**

```bash
git add src/skinny/shaders/materials/flat/flat_shading.slang
git commit -m "fix(shader): split cutout vs alpha-blend opacity in fetchFlatHitData

Cutout-mode hits (opacity texture + opacityThreshold > 0) now force
m.opacity = 1.0 instead of inheriting the texture's alpha. Transparent
side is already handled out-of-band by the integrators'
isCutoutTransparent skip loop, so the BSDF only ever sees the opaque
side and must report fully opaque to avoid the dielectric-refraction
branch in FlatMaterial.sample.

Restores correct rendering of UsdPreviewSurface cutout materials such
as the sunflower quads in McUsd.usda."
```

---

### Task 3: Visual verification with the failing asset

**Files:** none (manual run).

This task is required because the bug is visual and the unit tests cannot exercise it. Do not skip.

- [ ] **Step 1: Render the sunflower asset**

Run skinny against the failing scene. From a separate terminal (so the GUI stays usable):

```bash
.venv/bin/skinny  # or: .venv/bin/python -m skinny.app
```

Then load `assets/assets-main/full_assets/McUsd/McUsd.usda` through the UI's scene/model picker.

Frame the sunflower geometry (asset name contains "sunflower" — back/front/bottom/top quads).

- [ ] **Step 2: Confirm the fix**

You should see:
- Hard alpha-tested silhouette: petal interior is fully opaque, the area outside the petals shows whatever is behind the quad.
- No refractive bend or partial-transparency tint across the opaque petal interior.

If the petals still look washed out or you see the scene behind warping through them, the shader edit didn't land. Re-check Task 2 Step 2.

- [ ] **Step 3: Verify the threshold parameter has an effect**

Find the sunflower material in the UI's material editor (or wherever `opacityThreshold` is exposed for live editing — typically the StandardSurface / PreviewSurface inputs panel for the bound material). Move it. You should see the silhouette tighten (threshold up) or expand (threshold down). If the parameter has no visible effect, the renderer is not re-uploading the FlatMaterialParams buffer on edit — that is a separate bug, capture it as a follow-up and continue.

- [ ] **Step 4: Smoke-test glass refraction (regression check)**

Load a model that uses constant `opacity < 1` for refraction (the glass material added in commit `ad98a9f`, e.g. one of the bottle / glass assets if present). Confirm refraction still bends the background and that the refractive surface is not now binary-opaque. This protects the alpha-blend path that the fix deliberately left alone.

If you cannot find a glass asset locally, render any test scene that previously used `opacity = 0.3` constant material and visually compare with the pre-fix HEAD. Any obvious change to refraction means the conditional in Task 2 Step 2 is mis-keyed — most likely `cutoutMode` is firing for constant opacity (it shouldn't, because `opacityTextureIdx` is the sentinel for constants).

- [ ] **Step 5: Capture before/after (optional but recommended)**

Save a screenshot of the sunflower-correctly-rendered frame to `docs/screenshots/2026-05-24-cutout-opacity-fixed.png` for the PR description. Not required for plan completion.

---

### Task 4: Final integration check

**Files:** none (verification only).

- [ ] **Step 1: Run lint + non-GPU tests one more time**

```bash
.venv/bin/ruff check src/
.venv/bin/pytest -q -k "not gpu"
```

Expected: both clean.

- [ ] **Step 2: Confirm git state is what you expect**

Run: `git log --oneline -5`

Expected: top two commits are the cutout-opacity fix and the regression test (order depends on which task you finished last). The pre-existing uncommitted material-overhaul diff in `src/skinny/renderer.py`, `src/skinny/scene.py`, `src/skinny/usd_loader.py`, `src/skinny/vk_compute.py`, and `tests/test_struct_layout.py` is **not** owned by this plan — leave it alone unless the user has explicitly asked for it to be folded in. The new test added in Task 1 is the only addition to `tests/test_struct_layout.py` from this plan.

- [ ] **Step 3: Done**

Report completion to the user with the two new commit SHAs and a one-line confirmation that the sunflower renders correctly.
