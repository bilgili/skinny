## Why

Next-event estimation (NEE) on emissive **meshes** (area lights with more than a
couple of triangles) is weak in two independent ways, so imported interiors
render **dark and noisy** while small quad lights look fine:

1. **Silent truncation (correctness / energy loss).** `_upload_emissive_triangles`
   (`renderer.py:5070-5123`) caps the emissive-triangle buffer at
   `EMISSIVE_TRI_CAPACITY = 256` (`n = min(len(records), 256)`), with no warning.
   A high-poly emissive mesh loses every triangle past 256, so its light energy
   is dropped and the scene is biased **dark**.
2. **Uniform-by-index selection (variance).** `allLightsNEE` (`nee.slang:90-95`)
   picks one emissive triangle uniformly by index (`selectionPdf = 1/N` in
   `emissive_triangle_light.slang`). It is unbiased but high-variance when
   triangles vary in area or brightness, or when one bright light is split into
   many triangles among dim stray emissives — most samples are wasted.

Small lights already work: the `diffuse_arealight` corpus (a 2-triangle quad)
passes parity at FLIP 0.020. The failure is purely at **scale** — e.g. the
contemporary-bathroom test scene (5 emissive meshes, dark room). This was flagged
as the out-of-importer-scope follow-up from `pbrt-v4-scene-import`.

## What Changes

- **Power-weighted importance sampling** of emissive triangles. The host builds a
  cumulative-power CDF over all emissive triangles, weighted by
  `w_i = area_i × luminance(emission_i)` (Rec.709 luminance), and the shader
  selects a triangle by a binary search over that CDF instead of a uniform index.
  The CDF is packed **inline** into the existing emissive-triangle buffer (binding
  18) — the same one-slot trick the **environment** distribution uses to ride a
  single buffer — so no new descriptor binding is added (which would exceed
  native-Metal's 31-buffer argument-table cap that the neural/records wavefront
  builds already sit at).
- **Remove the 256 truncation.** Size the (single) emissive-triangle buffer
  dynamically to the actual triangle count (grow-and-rebind, like
  `material_capacity`); no silent energy loss. Log the emissive-triangle count.
- **Consistent pdf / MIS.** The per-triangle selection probability becomes
  `p_i = w_i / Σw`; `selectionPdf = p_i` and `pdfArea = p_i / triArea_i`. The
  downstream solid-angle pdf and the MIS power heuristic are unchanged in form —
  only the selection probability changes — so NEE stays unbiased and the BSDF-hit
  emission gate (no double-count) is untouched.
- **ReSTIR uses the same distribution.** `light_ris.slang` reports its emissive
  source pdf through the shared `light.samplePoint` / `selectionPdf` seam, so its
  candidate *index draw* switches to the same shared power-CDF helper (one line)
  to stay matched with that pdf — the reservoir / RIS / GRIS reuse code is
  untouched. The new distribution additionally helps every **secondary** bounce,
  which ReSTIR (primary-hit only) never touches.

## Capabilities

### New Capabilities
- `emissive-light-sampling`: The renderer SHALL sample emissive-triangle (mesh)
  area lights with a power-weighted (area × luminance) importance distribution
  and SHALL include every emissive triangle in the scene (no fixed cap), so
  emissive-mesh NEE is unbiased and low-variance at scale.

### Modified Capabilities
<!-- None. integrator-convergence governs the specular→emitter BSDF-hit rule;
     this change only alters the NEE light-selection distribution, which no
     existing spec governs. -->

## Impact

- **Host:** `renderer.py` — `_upload_emissive_triangles` (inline CDF pack +
  dynamic sizing + grow/rebind of binding 18), count logging, a test-only
  uniform-selection toggle; `skinny.pbrt.parity.render_linear` grows an
  `emissive_uniform=` parameter for the power-vs-uniform A/B.
- **Shaders:** `common.slang` (`cw`/`pi` properties on `EmissiveTriangle`),
  `scene_lights.slang` (shared `sampleEmissiveTriangle` helper),
  `nee.slang` + `materials/skin/skin_direct.slang` + `restir/light_ris.slang`
  (use the helper for the index draw), `emissive_triangle_light.slang`
  (`selectionPdf = tri.pi`). No new binding. Shaders recompiled by the runtime
  (native Metal in-process; Vulkan `main_pass.spv` regenerated).
- **Reuses unchanged:** the BSDF-hit emission gate, the MIS power heuristic, the
  ReSTIR reservoir / RIS / GRIS reuse code.
- **Tests:** new headless tests reusing `skinny.pbrt.parity.render_linear` + the
  corpus (native Metal via `select_backend`): a >256-triangle emissive-mesh
  correctness/energy test, a multi/uneven-emitter equal-spp variance test, and a
  `diffuse_arealight` no-regression check.
- **Docs:** `docs/Architecture.md` binding map, `docs/Megakernel.md` NEE note,
  `CHANGELOG.md`.
- **Out of scope:** alias-table sampling (CDF chosen), light BVH / spatial
  clustering, env+emissive joint-MIS rebalance, ReSTIR changes beyond inheriting
  the distribution.
