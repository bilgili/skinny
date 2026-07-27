# Design review — renderer-pure-core-extraction

Adversarial review, 2026-07-27, against the tree at `8247148`. Recorded rather
than folded. **Fold before implementing.**

**Verdict: the goal is right and there is a ~2-line way to get it.** The premise
holds — `renderer.py:16` is a hard module-scope `import vulkan as vk` — but the
proposal buys it with ~1,330 lines of movement, 6 new modules and ~50 test
repoints, and most of the supporting evidence is inflated or wrong.

## MAJOR

**M1 — A 2-line fix achieves the stated Why, and is neither adopted nor
rejected.** All 465 `vk.` references are inside method bodies — first at
`renderer.py:2280`; none at class-body, decorator, default-arg or `except`
scope (`grep -n "^    [^ ].*\bvk\." → 0`; `grep -n "def .*vk\.\|except .*vk\." → 0`).
The only other module-scope Vulkan barrier is `renderer.py:59`
`from skinny.vk_context import VulkanContext`, and `VulkanContext` is used
**only** as an annotation at `:1446` — inert under the file's existing
`from __future__ import annotations` (`:3`). Every other module-scope import is
transitively vulkan-free.

```python
class _LazyVk:
    def __getattr__(self, name):
        import vulkan
        globals()["vk"] = vulkan          # self-replaces on first touch
        return getattr(vulkan, name)
vk = _LazyVk()
```

Move line 59 into the existing `if TYPE_CHECKING:` block at `:66`, replace line
16 with the proxy, and `import skinny.renderer` succeeds with no SDK — unskipping
**every** currently-skipping test, including the ones extraction can never
unblock (M2). The proposal must adopt this or state why a 1,330-line move beats
it. "Making `Renderer` importable without vulkan … is the backend adapter's job"
is asserted, not argued — and it is the cheaper goal.

**M2 — Extraction does not close the silent-skip failure mode it is sold on.**
Several currently-skipping "pure" tests need the `Renderer` class itself and keep
their skip after the move: `tests/test_metal_megakernel_tiling.py:38` calls
`R.Renderer._metal_megakernel_bands(stub)` on a `SimpleNamespace`;
`tests/test_camera_placement.py:171,180,189,206,260,312` import `Renderer` and
the `needs_renderer` marker (`:32`) survives;
`tests/pbrt/test_named_spectra.py:244` and `tests/test_mlt_host.py:223` skip on
`(ImportError, OSError)` for symbols the proposal never lists (M4). The spec
scenario "they execute rather than skip" is unsatisfiable for the file that most
conspicuously demonstrates the problem.

**M3 — D1's justification ("each cluster has a real consumer already") is false
for 3 of 6, and 2 of its 3 named examples are wrong.**
- **`_hashable_value` is not in `renderer.py`.** It is `params.py:213`;
  `renderer.py:49` is already an explicit re-export with a comment naming
  `ui/qt/windows/bxdf.py` as the consumer. The "small helpers" cluster's flagship
  symbol does not exist to move — `bxdf.py:22` needs a one-line repoint to
  `skinny.params`, no new module.
- "the parity harness wants film IO" — `pbrt/parity.py:174` mentions
  `FilmParameters` **in a comment only**; parity imports `skinny.headless`, not
  `skinny.renderer` (`:119-120`). `_write_exr`/`_write_hdr_rgbe` have exactly one
  caller: `renderer.py:11196`. `FilmParameters` has one: `:1849`.
- `TexturePool` has exactly one consumer: `renderer.py:3985`. No test, no other
  module.

Only camera is as described. Collapse to what has a consumer: one `camera.py`,
one `material_pack.py`, and leave the rest.

**M4 — The six clusters do not cover the range the proposal claims to empty.**
Unlisted: `_spectral_analytic_proposal_token` (`:104-121`); the FrameConstants
field tables `_FC_SCALAR_FIELDS` / `_FC_SCALAR_FIELDS_MLT` / `_FC_MLT_FIELDS` /
`_TILE_ORIGIN_Y_OFFSET` (`:190-201`, read by
`tests/test_metal_megakernel_tiling.py:23-29` and `tests/test_mlt_host.py`);
`_VK_UNIFORM_BUFFER_BYTES` **and its import-time `assert`** (`:247-251`); the
Metal band constants (`:210-215`, `:226`, `:236`); `_accum_hash_resolvers` plus
the mutable `_ACCUM_HASH_RESOLVERS` global (`:325-350`); `_instance_local_basis`
(`:292`); `_light_value_to_vec3` (`:353`); `MATERIAL_TYPE_*` (`:362-391`);
`MEDIUM_*` (`:393-395`); `TOOL_MODE_*` (`:442-443`); `MAX_LENS_ELEMENTS`
(`:256`); `WORKGROUP_SIZE` / `MAX_FRAMES_IN_FLIGHT` (`:69-70`); the SPPM/MLT
defaults (`:84-101`); `_CONDUCTOR_METAL_ID` / `_SPECTRAL_METAL_ORDER`
(`:600-607`, read by `tests/pbrt/test_named_spectra.py:246+`).

Notably `MATERIAL_TYPE_FLAT` and `_METAL_WAVEFRONT_HEAVY_EYE_BAND_LANES` are
imported **lazily inside function bodies** by `vk_wavefront.py:1968` and
`metal_wavefront.py:1347,1406,1437` — the only non-`debug_viewport` src
consumers of the pure range, and a live cycle workaround the proposal does not
notice. A `renderer_consts.py` would let those become module-level imports.

**M5 — The "35 references across 9 files" cost claim is inflated ~6×.**
35 raw hits ✓, but across **7** files, and only **3** contain an actual import
(`tests/test_metal_flat_material_layout.py:36`,
`tests/test_mtlx_synthesis.py:521`, `tests/test_struct_layout.py:44,211,239,268`)
— **6 import statements**. The other four are comments/docstrings;
`tests/test_scene_graph_material_props.py:11` explicitly says it *"never imports
`skinny.renderer`"*. The repointing work is ~6 lines, not 35 across 9 files.
This is the change's headline cost justification.

## MINOR

- **D2 is safe for a reason the design does not record, and has one real hazard.**
  `src/skinny/__init__.py` contains only a docstring and `__version__` — it does
  not import `renderer`, so the re-export is genuinely load-bearing. But
  `_ACCUM_HASH_RESOLVERS` (`:325`) is a **rebindable** module global mutated via
  `global` at `:340`; `from newmod import _ACCUM_HASH_RESOLVERS` binds `None`
  forever. Add to D2: re-export functions and immutable constants only; a
  rebindable global stays put or is accessed through its owning module.
- **D4's "it does not import them" is not quite true.** `TexturePool.filled_slots`
  (`:589`) is annotated `list[tuple[int, SampledImage]]`, satisfied by the
  `TYPE_CHECKING` import at `:66-67` — inert only because of the future-import.
  Make carrying both an explicit task; note `typing.get_type_hints` on that
  method still raises without the SDK.
- **D3 proposes more machinery than its own precedent.**
  `tests/test_render_session_module.py:16-31` does not make PySide6
  unimportable; it imports in a subprocess and asserts `'PySide6' not in
  sys.modules` (`:20`). Same assertion works for `vulkan`; no import-blocker shim
  needed.
- **The "constant read at import time" risk is a non-risk in the stated
  direction.** There are **zero** module-scope definitions below line 1440 —
  `class Renderer:` at `:1435` is the last column-0 statement in the file. The
  genuine import-time item is the `assert` at `:248-251`, which must keep
  executing wherever `_VK_UNIFORM_BUFFER_BYTES` lands.
- **Import-cycle risk is nil for every proposed module**; drop task 3.4.
  camera→`LensSystem` is a string annotation only (`:1187`, `:1349`);
  packing→`slang_layout` (pure) + `skinny.pbrt.data`, whose package init is a
  PEP-562 lazy `__getattr__`; helpers→`skinny.params`. One caveat: `_write_exr`
  imports `OpenEXR` *inside* the function (`:1373`) — do not hoist it, or the
  hostless gate fails on a box without the bindings.
- Line-range labels are wrong: `:104-291` is not all SPPM math (that is
  `:124-174`); the rest is FrameConstants blob derivation, the Vulkan UBO size +
  assert, Metal band budgets and the instance/flat strides.
- The spec delta normatively mandates a "small shared helpers" module that,
  per M3, would be empty. Drop it, or name the helpers that do exist
  (`_instance_local_basis`, `_light_value_to_vec3`, `_encode_channel_mask`,
  `_override_float`, `_override_color3`, `_material_is_subsurface`,
  `_material_is_volume`) — most of which belong with material packing.
- Task 2.6 should repoint `bxdf.py:22` to `skinny.params`, and delete
  `renderer.py:49`'s re-export line and its comment in the same edit.

## Confirmed as claimed

`renderer.py:16` module-scope import ✓ · first `vk.` use at `:2280`, so
104–1434 is import-time device-free ✓ · `debug_viewport.py:30` imports the
camera symbols ✓ · `bxdf.py:22` imports `_hashable_value` from renderer ✓ (but
it originates in `params.py`) · `tests/test_camera_placement.py:20-33`
`_have_renderer()` skip with the stated reason ✓ · 35 raw `pack_flat_material`
hits ✓ · the no-toolkit subprocess gate exists ✓ · camera / writers /
`FilmParameters` / `SkinParameters` ranges ✓ (`SkinParameters` starts at the
`@dataclass` on `:988`).
