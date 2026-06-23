## Why

`assets/bathroom.usda` (imported from the pbrt `contemporary-bathroom`) is flagged
`known_divergent` in the parity corpus: at 256²/path the render scores relMSE≈233
and FLIP≈0.39 against the pbrt v4 reference EXR, and BDPT diverges from the path
anchor at relMSE≈6612. Both numbers looked like catastrophic material/lighting
errors.

Decomposing the metric proved otherwise. **The divergence is entirely a handful of
firefly pixels**, not a structural error:

- path vs pbrt: the **top 9 pixels carry 99.7 %** of the relMSE mass. skinny's
  image maxes at 7985 where the reference maxes at 2120 and clips at ~51.
- bdpt vs path: the firefly pixels carry **100 %** of relMSE 6612, while **FLIP is
  only 0.085** — i.e. bdpt is structurally identical to the path tracer apart from
  fireflies.

The reference is clipped at ~51 because the pbrt scene sets
`Film "rgb" "float maxcomponentvalue" 50` — pbrt's per-sample radiance clamp
(`RGBFilm::AddSample` scales any sample whose largest RGB component exceeds the
threshold down to it). The scene is lit by a window area light (`scale 10`) and
four light-bulb filament area lights at `scale 7000`; those tiny, ultra-bright
emitters are textbook firefly sources, and pbrt tames them with the clamp.

skinny imports and already honours the film **`iso`** exposure (baked into
emitters as `imagingRatio`), but it has **no equivalent of `maxcomponentvalue`** —
so the fireflies survive and blow up both gates. Confirmed: clamping skinny's
output offline collapses relMSE 232.8 → 10.9 (and the residual is itself the same
fireflies under a coarser final-pixel clamp, which a true per-sample clamp removes).

This is a film/sensor parity gap, the direct analogue of the `iso` bake already in
the importer — not a material bug. The recent `mtlx-graph-texcoord-uv` work, the
window light, and the env handling are all ruled out (wall/wood/sink colours match
the reference).

## What Changes

- **New film capability: a per-sample radiance clamp (`maxComponentValue`).** When
  the imported pbrt film sets `maxcomponentvalue`, the renderer clamps each path
  sample's radiance proportionally (preserving hue: scale so the max RGB component
  equals the threshold) before it is accumulated — matching pbrt
  `RGBFilm::AddSample`. The default (no clamp / `+∞`) keeps every other render
  byte-identical.
- **Importer** already preserves the whole pbrt film verbatim in
  `customLayerData.pbrt.film.params` (so `maxcomponentvalue` is present today, beside
  the already-baked `iso` exposure) — no emit change needed.
- **Loader → renderer → `FrameConstants`** reads `film.params.maxcomponentvalue` and
  plumbs the threshold as a scalar tail
  field (`filmMaxComponent`, `0` ⇒ disabled), included in the accumulation
  state-hash so changing it resets accumulation.
- **Shaders** apply the clamp at every sample-accumulation site that feeds the film:
  the megakernel path/BDPT camera contribution, the wavefront path/BDPT/SPPM
  accumulation, and the BDPT light-path splat. Gated on `filmMaxComponent > 0` so
  the disabled path is unchanged; Metal adaptations (if any) stay behind
  `#if defined(SKINNY_METAL)` so Vulkan SPIR-V is byte-stable when disabled.
- **Parity matrix**: the clamp removes the firefly catastrophe — bathroom
  path-vs-pbrt drops **232.8 → ~0.36** and BDPT-vs-path drops **6612 → 0.36**
  (MSE 0.017 — structurally identical, the documented BDPT divergence is gone).
  The change therefore **lowers the recorded bathroom baselines ~500×** (the
  harness-first workflow: a fix lowers a baseline, never raises it) but **keeps
  `known_divergent: true`**, because two real residuals remain (each a separate
  follow-up):
  - The bathroom **reference is regenerated with pbrt's `path` integrator**
    (`regen_refs.py --integrator path`) so the pbrt-truth gate compares skinny's
    path anchor against a like-for-like reference (the authored `sppm` EXR is
    unclamped — max ≈ 2120, bypassing `maxcomponentvalue` — and pbrt's own
    path-vs-sppm differ by relMSE ≈ 0.63 on this caustic scene). The residual
    skinny-path vs pbrt-path ≈ 0.34 is an RGB-vs-spectral / blackbody-emitter
    mismatch (follow-up).
  - The **self-consistency** relMSE vs the path anchor stays large on this noisy
    heavy scene (dark-pixel `/b²` amplification: BDPT MSE 0.017 but relMSE 0.36;
    sppm genuinely differs at MSE 10.4). That is a metric-robustness / sppm-vs-
    path follow-up, not loosened here — so the scene stays flagged.
  - `megakernel ≡ wavefront` is reconfirmed exactly (path mega-vs-wave relMSE
    `0.0000`).

## Impact

- Affected specs: **film-sensor** (new). The `render-parity-matrix` spec is
  unchanged — bathroom stays `known_divergent` (its existing recorded-baseline
  requirement covers lowering the baseline).
- Affected code: `usd_loader.py`, `renderer.py`, `scene.py`,
  `shaders/common.slang` + the path/bdpt/sppm accumulation sites, recompiled
  `main_pass.spv` (+ wavefront SPIR-V for Vulkan), `tests/pbrt/regen_refs.py`
  (integrator override), `tests/pbrt/corpus/manifest.json` (lowered baselines +
  pbrt-path reference), `tests/pbrt/corpus/refs/bathroom.exr` (regenerated).
- No new CLI flag, no new descriptor binding, no importer (`emit`) change — the
  film already round-trips in `customLayerData.pbrt.film.params`. The clamp is
  data-driven from the scene; renders of scenes without `maxcomponentvalue` are
  byte-identical.
