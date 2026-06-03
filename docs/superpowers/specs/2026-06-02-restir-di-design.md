# ReSTIR DI — reservoir spatiotemporal resampling for direct lighting

**Date:** 2026-06-02
**Status:** Approved design

## Problem

The pluggable scene-sampling seam (merged, `decbbb9`) reserved a **reuse hook**
around direct + indirect lighting — a `ReusePlugin` socket with a `reuseMode`
uniform, `IdentityReuse` baseline, and pass-structural switching — but ships
only the identity (stock NEE) implementation. This change lands the first real
reuse plugin: **ReSTIR DI**, reservoir-based spatiotemporal resampling of direct
lighting at the primary visible point. It accelerates direct-lighting
convergence (and, with the env proposal, complements the bounce seam) by reusing
the best light sample per pixel across neighbors and accumulation frames.

skinny is a **progressive accumulator** with **no temporal infrastructure** (no
motion vectors, reprojection, G-buffer, or history; `_current_state_hash`
changes — including camera moves — reset accumulation to 0 spp). That reshapes
ReSTIR away from the game-style 1-spp-while-moving regime toward
convergence-acceleration on a static camera, and makes the heavy reprojection
machinery an explicitly separate phase.

## Decisions (locked during brainstorm)

1. **Regimes — all three selectable, nested.** `spatial` (on/off) × `temporal`
   (off / progressive / reprojected). Spatial-only and progressive-temporal fit
   skinny's progressive model with **no new infra**; reprojected temporal needs
   a motion-vector + prev-frame G-buffer subsystem and is **deferred to its own
   follow-on change** (the mode is reserved in the config enum).
2. **Scope — primary-hit, screen-space.** Per-pixel reservoirs at the primary
   visible point (the main visual for a skin renderer). Secondary path vertices
   have no screen-space pixel ⇒ indirect bounces keep stock NEE. (World-space /
   secondary-vertex ReSTIR is out of scope.)
3. **Light domain — unified, all light types.** One reservoir RIS's over
   directional + sphere + emissive-triangle + env; ReSTIR wholesale-replaces
   `allLightsNEE` at the primary hit. A reservoir sample is
   `(lightType, lightId, point-on-light)`.
4. **Bias — unbiased default + biased toggle.** Default to the unbiased
   combination (GRIS-lite: per-neighbor MIS weights + the `1/Z` domain-aware
   normalization + the reconnection Jacobian); expose `biased` as a faster
   toggle. Matches the renderer's unbiasedness gates (furnace mode, the parity
   tests).
5. **Integration — canonical, RIS owns primary direct.** Candidate generation
   mixes **light-sampled AND BSDF-sampled** candidates into the RIS; the primary
   bounce **skips its own sphere-light / env-miss MIS term** (ReSTIR already
   counted it). The proposal mixture still drives the bounce *direction* for
   indirect; `depth ≥ 1` vertices use stock NEE.
6. **Backend — wavefront-only.** Multi-pass reuse can't live in the megakernel;
   selecting ReSTIR + megakernel falls back to identity (capability gate, like
   wavefront-bdpt).

## Architecture

ReSTIR DI is `reuseMode = 1` (`RESTIR_DI`), realized by `RestirDiReuse(ReusePlugin)`.
It owns a stack of **wavefront compute passes** + persistent **per-pixel**
buffers (reservoirs, G-buffer) + their descriptor bindings — the pass-structural
socket the seam reserved. Selecting it rebuilds the wavefront pipeline (like
`--execution-mode`). It runs **once per pixel at the primary hit**, before the
path continues for indirect.

### Reservoir + G-buffer (per pixel, screen-space)

```hlsl
struct LightSampleRef {       // unified over all 4 light types
    uint   packed;            // type:2 | lightId:30
    float2 uv;                // point-on-light / env-dir parameterization
};
struct Reservoir {
    LightSampleRef y;         // surviving sample
    float wSum;               // Σ RIS weights seen
    float W;                  // unbiased contribution weight = wSum / (M · p̂(y))
    uint  M;                  // sample count (capped for temporal)
    float pHat;               // cached target fn p̂(y) = luminance(f·Le·G), UNSHADOWED
};
```

A **G-buffer** per pixel (world position, shading normal, materialId, wo) backs
spatial-neighbor rejection and re-evaluating a neighbor's `p̂` at this shading
point. The reservoir buffer is **double-buffered** (prev/curr) for temporal
reuse. All buffers are `width × height`, owned + sized by the plugin.

### Pass pipeline (wavefront, primary hit)

```
primary intersect ─► [G-buffer fill]   (pos / normal / matId / wo per pixel)
   │
   ├─► [initial RIS]    M_light light candidates + M_bsdf BSDF candidates
   │                    → weighted-reservoir-sample to 1   (p̂ UNSHADOWED — no shadow rays yet)
   ├─► [temporal reuse] merge prev-frame reservoir   (progressive: same pixel)   · M-cap
   ├─► [spatial reuse]  merge k neighbors in radius if G-buffer-similar
   │                    (unbiased: m_i + 1/Z + Jacobian; biased: sum)
   └─► [resolve/shade]  ONE shadow ray for surviving y → direct = f(y)·V(y)·W → path radiance
                        ↓
   indirect: primary bounce samples a direction via the PROPOSAL mixture and continues as a
             normal path — its own sphere-light / env-miss MIS term SKIPPED (ReSTIR owned
             primary direct); depth ≥ 1 vertices use stock NEE.
```

**Deferred visibility** is the ReSTIR win: candidates use the cheap *unshadowed*
`p̂`; only the single survivor pays a shadow ray.

### Unbiased combination (merge math)

- **RIS estimate:** `W = wSum / (M · p̂(y))`; final direct = `f(y)·V(y)·W` (full
  shaded `f`, single shadow ray `V`).
- **Spatial/temporal merge (unbiased):** each merged neighbor contributes with
  an MIS weight `m_i` (generalized balance heuristic) that **only counts
  neighbors whose domain could have produced `y`** — the `1/Z` normalization.
  Reconnecting a neighbor's light sample to *this* shading point needs a
  **Jacobian** (DI shift = same light point, different shading point ⇒ ratio of
  geometry terms `cosθ / d²`) plus a horizon/visibility domain check.
- **Biased toggle** drops the `m_i` + Jacobian (sum reservoirs, normalize by
  `ΣM`) → discontinuity darkening, but cheaper.
- **Target `p̂`** = `luminance(f · Le · G)` unshadowed, cached so neighbor
  re-evaluation is one BSDF eval.

### Selectable regimes + config

`reuse_modes` → `["None", "ReSTIR DI"]`; the data-driven `_disc` selector
surfaces it across all front-ends + settings persistence for free (same as the
proposal presets). The ReSTIR sub-config lives in a **small ReSTIR
push-constant/UBO the plugin owns** (not bloating `FrameConstants`):

| field | meaning |
| --- | --- |
| `restir_spatial` | spatial reuse on/off |
| `restir_temporal` | off / progressive / reprojected (reprojected = P3) |
| `M_light`, `M_bsdf` | initial candidate counts |
| `spatial_k`, `spatial_radius` | neighbor count + screen radius |
| `M_cap` | temporal history cap |
| `biased` | unbiased(0) / biased(1) toggle |

Changing any resets accumulation (folded into `_current_state_hash`).

### Integration with the bounce loop

When `reuseMode == RESTIR_DI`, the bounce gains a **"primary direct owned by
reuse"** gate: at `depth == 0` it skips `allLightsNEE` and the sphere-light /
env-miss MIS term (ReSTIR supplies primary direct via its resolve pass). The
proposal mixture still selects the bounce *direction* for the indirect ray.
`depth ≥ 1` is unchanged (stock NEE + proposal mixture). Identity reuse keeps
today's behavior exactly.

## Phasing

The reprojected regime is a motion-vector + prev-G-buffer subsystem skinny lacks
— too large to bundle. **This spec implements P1, P2, P4; P3 is a follow-on
change** (the mode stays reserved in the enum / selector, falling back until it
lands):

- **P1** — reservoir core + initial RIS + spatial reuse + unbiased + shade,
  static camera (no temporal). Ships ReSTIR DI working.
- **P2** — progressive temporal: carry the prev-frame reservoir across
  accumulation iterations (reprojection = identity since the camera is static).
  The convergence win.
- **P4** — biased toggle + tuning defaults.
- **P3 (follow-on, own spec)** — reprojected temporal: motion vectors +
  prev-frame G-buffer + disocclusion rejection for a moving camera.

## Changes

### Slang (new)

- `shaders/sampling/reuse.slang` — the reuse-seam interface + `identityReuseDirect`
  (the module the scene-sampling spec reserved), now realized with a second
  implementation to switch against.
- `shaders/restir/reservoir.slang` — `Reservoir`, `LightSampleRef`, weighted-
  reservoir-update, the target function `p̂`, pack/unpack.
- `shaders/restir/initial.slang` — initial RIS (light + BSDF candidates over the
  unified light set).
- `shaders/restir/temporal.slang` — progressive temporal merge + M-cap.
- `shaders/restir/spatial.slang` — k-neighbor merge, G-buffer similarity,
  unbiased `m_i` + Jacobian (biased path).
- `shaders/restir/resolve.slang` — shadow ray + `f·V·W` → radiance.

### Slang (edit)

- The wavefront primary path + `integrators/path.slang` / `wf_shade_common.slang`:
  the depth-0 "primary direct owned by reuse" gate.
- `common.slang`: the G-buffer record layout (if shared) / `RESTIR_DI` constant.

### Host (Python)

- `sampling/reuse.py` — `RestirDiReuse(ReusePlugin)` filling `passes()` /
  `bindings()` / lifecycle; owns the reservoir (×2) + G-buffer buffers, the
  ReSTIR UBO, and the regime/tuning config.
- `sampling/registry.py` — register `RestirDiReuse`; `reuse_modes` gains
  `"ReSTIR DI"`.
- `vk_wavefront.py` (or a new `vk_restir.py`) — build + schedule the ReSTIR pass
  set in wavefront mode when active; the reuse-mode switch triggers a pass
  rebuild (the seam's pass-structural contract).
- `renderer.py` — allocate/resize the per-pixel buffers on framebuffer resize;
  fold the ReSTIR config into `_current_state_hash`; capability-gate ReSTIR to
  wavefront (fall back to identity on megakernel/Metal).
- `params.py` — the ReSTIR sub-config params (gated visible when ReSTIR active).

## Testing

`tests/test_restir_di.py`, reusing the `test_sampling_parity::_accumulate`
deterministic-accumulation harness (pinned `frame_index`/`accum_frame`).

1. **Converge-to-reference (the unbiased gate):** ReSTIR DI vs stock NEE
   (`reuse=none`), both high-spp on an emissive-triangle / area-light scene —
   image-mean (integrated radiance) matches within tolerance (same direct
   integral). The primary correctness test.
2. **Variance reduction:** ReSTIR error < stock-NEE error at equal low spp vs a
   converged reference (strongest with many emissive triangles).
3. **Temporal helps:** progressive-temporal error < spatial-only error.
4. **Biased bounded:** biased mode darkening within a stated bound (allowed to
   differ from reference; must not explode).
5. **Capability gate:** `reuse=ReSTIR` + megakernel → identity fallback (no
   reservoir passes built); wavefront builds them.
6. **Furnace + determinism:** furnace test still passes; pinned-seed runs
   reproducible.
7. `ruff` + `slangc` recompile (`main_pass` + wavefront variants) + `py_compile`.

## Out of scope

World-space / secondary-vertex ReSTIR · ReSTIR GI / PT (path reuse) · the
reprojection subsystem (P3, its own change) · denoising · motion vectors.

## Implementation-planning transition

Per the project workflow, planning goes through an **OpenSpec change proposal**
(`openspec/changes/restir-di/`) — a new `restir-di` capability that depends on
the `scene-sampling` reuse hook — not a generic plan doc.

## Implementation outcome (2026-06-03)

Deviations from the plan, found while completing the spec:

- **Canonical integration (Decision 5) shipped.** The RIS owns primary direct:
  candidate generation mixes light- and BSDF-sampled candidates with the
  UNWEIGHTED target `p̂ = lum(f·Le)` and a balance-heuristic mixture source pdf
  `p_mix = (M_light·p_light + M_bsdf·p_bsdf)/M`; the path tracer's depth-0
  BSDF-hits-sphere (`wf_shade_common`) and env-miss (`wavefront_path`) terms are
  gated off. BSDF candidates target **sphere + env only** (sphere hits recover a
  reproducible uv, env stores an octahedral direction); emissive triangles are
  NEE-only in the stock renderer (no BSDF-tri MIS term) so they are light-technique
  only — still unbiased (any unbiased estimator converges to the same integral).
  (An intermediate "option B" — light-only RIS composing with shade's still-active
  BSDF half, no gate — was the merged starting point; it was replaced by Decision 5.)
- **Unbiased combination = GRIS, not the bare 1/Z.** Spatial/temporal reuse uses
  the generalized balance heuristic `m_s = M_s·p̂_s(z_s)/Σ_j M_j·p̂_j(z_s)`,
  re-evaluating the survivor's target in every source's own domain (the source
  lane's material re-loaded from `wfHits[j]` — no fat G-buffer needed). DI reuses
  the same world light point, so the reconnection shift is identity (Jacobian 1).
  The naive biased ΣM combination over-brightened glossy surfaces via
  spatial→temporal feedback (up to ~48% vs path tracing on a glossy material); the
  GRIS m_i bounds it. The biased ΣM path is kept as a faster toggle.
- **Default regime = Spatial only.** On the progressive accumulator, temporal reuse
  double-counts correlated history (bias ∝ M_cap, glossy-specific) — see the
  Variance-reduction spec amendment. Spatial reuse (GRIS) is unbiased and reduces
  variance; progressive-temporal / temporal-only stay selectable but limited;
  reprojected temporal is the P3 follow-on.
- **Verification.** Converges to stock NEE on cornell_box_sphere / cornell_box_
  emissive / three_materials (glossy), A/B-verified against megakernel-PT, BDPT and
  wavefront-NEE (all agree). Variance demo: `assets/restir_variance_demo.usda` +
  `tests/test_restir_variance.py` (~30% lower RMSE than NEE at equal low spp).
