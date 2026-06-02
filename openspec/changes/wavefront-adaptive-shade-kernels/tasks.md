## 1. Bin-packer (pure, CPU)

- [ ] 1.1 Add `src/skinny/wavefront_shade_packer.py`: `pack_shade_groups(members, sizes, threshold, pin_alone)` → ordered list of groups (member lists), first-fit-decreasing, `skin` pinned to its own group. Pure function, no GPU. Constants `MAX_SLOTS = 32`, `SHADE_SIZE_THRESHOLD ≈ 2.4 MB` (documented relative to the ~2.83 MB MoltenVK danger line).
- [ ] 1.2 Unit tests (`tests/test_shade_packer.py`, no GPU): packs members so every group Σsize ≤ threshold; skin always isolated; deterministic FFD order; `MAX_SLOTS` headroom (representative member set stays well under 32 groups); flat-only set → zero non-flat groups.

## 2. Slot count + buffers (dynamic via MAX_SLOTS)

- [ ] 2.1 `wf_shade_common.slang`: `WF_NUM_SLOTS` (static `2u`) → `MAX_SLOTS` (`32`); clear-counts / build-args / classify loop bounds use it. Confirm group-count/offset/cursor/indirect-args indexing stays correct at the larger bound.
- [ ] 2.2 `wavefront_layout.queue_buffer_sizes`: size `material_count` / `material_offset` / `indirect_args` to `MAX_SLOTS` (not the live material count). Keep the per-stream queue buffers stream-sized.

## 3. Classifier: materialId → slot

- [ ] 3.1 `wf_shade_common.slang`: replace `wfSlotForType(matType)` with `wfSlotForMaterial(matId)` reading a new `materialSlot[]` set-1 storage binding (one `uint`/material; flat + graph → 0). Update `wavefront_path.slang` classify (`wfLaneSlot[i] = wfSlotForMaterial(hit.materialId)`).
- [ ] 3.2 `vk_wavefront.WavefrontPathPass`: own + bind the `materialSlot` buffer; upload from the grouping (renderer supplies the array).

## 4. Grouped shade pipelines

- [ ] 4.0 **Prereq — extract per-type bounce evaluators** (mirrors `flat_bounce`): `skin_bounce.slang` (`evaluateSkinBounce`, imports only `materials.skin.skin_material`) and a python evaluator (`evaluatePythonBounce(pyId)`, imports only `python_materials_dispatcher`), each a verbatim copy of the matching `evaluateBounce` switch case. `evaluateBounce` stays for the megakernel. This is what lets a per-member shade kernel import only its own subtree (and be measured in PASS 1).
- [ ] 4.1 `WavefrontPathPass`: constructor takes a `grouping` (ordered groups + `materialSlot` array) instead of `build_catchall: bool`. Build one shade pipeline per group; emit per-group shade entries that import only their members (single-member → straight-line; multi-member → switch on type / `pythonMaterialId`). Slot 0 (flat/graph) unchanged.
- [ ] 4.2 `WavefrontPathPass` dispatch loop: record exactly `len(groups)` indirect shade dispatches (`vkCmdDispatchIndirect` per group over its slot queue), one barrier after all shades. Empty/unused slots are never dispatched.

## 5. Renderer wiring + measurement

- [ ] 5.1 `_compile_shade` / `WavefrontPathPass`: PASS 1 — compile each non-flat member's shade isolated and read the `.spv` byte size (file size of the per-entry spv; reuse the content-hash cache so repeats are free).
- [ ] 5.2 `renderer._ensure_wavefront_path_pass`: build the grouping via `pack_shade_groups` from `_material_types` (+ python ids), upload `materialSlot[]`, and key the pass rebuild on the grouping signature (unchanged material set → reuse). Flat-only scenes build slot 0 only (no isolate-compiles), matching today.

## 6. Tests + verification

- [ ] 6.1 A/B parity (`test_wavefront_path_ab`): add a forced-split scene (flat + skin + ≥1 python material) — assert wavefront matches megakernel within the noise floor AND that >1 shade group was built.
- [ ] 6.2 Size regression (the bug guard): every compiled path-shade `.spv` ≤ `SHADE_SIZE_THRESHOLD`; surface skin's isolated size explicitly (xfail/skip-with-message if skin alone exceeds the MoltenVK limit — separate effort).
- [ ] 6.3 Slot-map / classify test: `materialSlot[]` routes each material to a group containing its code; per-slot counting-sort counts match a CPU reference for a mixed scene.
- [ ] 6.4 Flat-only regression: a flat/graph-only scene compiles exactly slot 0 (no isolate-compiles, no extra pipelines), image unchanged from today.
- [ ] 6.5 `ruff check src/` clean (no new errors vs `main`); GPU suite under the 3.13 build venv; `main_pass.spv` byte-unchanged (megakernel untouched).
