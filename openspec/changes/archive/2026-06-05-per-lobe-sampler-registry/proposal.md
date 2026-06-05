## Why

The merged `unify-flat-bsdf-on-lobes` change put both `FlatMaterial.sample()`
and `FlatMaterial.evaluate()` on one `{coat, spec, diffuse}` lobe set and shipped
a per-lobe `samplerId` seam — but left it **unpopulated** (native strategies
only). The dispatch indirection exists and does nothing. This change populates
the seam so HOW each lobe draws directions becomes a runtime-selectable,
GUI-switchable strategy, turning the unified BSDF into the composable per-lobe
importance-sampling framework it was structured for — and proves the seam carries
a real alternative without regressing the Change-1 unbiasedness / firefly
guarantees.

## What Changes

- Add a host-side **per-lobe sampler registry** mirroring the proposal seam
  (`sampling/proposals.py` + `ProposalPlugin`): a `LobeSamplerStrategy`
  descriptor (name, valid-lobe mask, shader id, CLI token) + a registry list.
- Register the **first alternate strategies**: the Heitz-2018 basis-form VNDF for
  the GGX coat and spec lobes (the native GGX sampler is already the 2023
  spherical-cap / bounded-VNDF form, so the basis-form is the genuinely-different
  warp of the *same* visible-normal distribution); uniform-hemisphere for the
  Lambert diffuse lobe. Native (2023 spherical-cap VNDF / cosine) stays the
  default.
- Transport the per-lobe selection through a single new `FrameConstants`
  field (`flatLobeSamplers`, 8 bits per lobe) — **no new descriptor bindings**,
  exactly like the env-importance proposal reused the env-CDF.
- Make `flatSampleLobe` / `flatLobePdf` honor `samplerId` (they currently ignore
  the parameter). `sample()` and `evaluate()` read the same `fc` per-lobe id, so
  the draw density and the weight/pdf density stay equal for **any** registered
  strategy. `flatBsdfResponse` (= `f·cos`, physical, sampler-invariant) is
  unchanged.
- Surface three per-lobe selectors in the GUI (`flat.coat_sampler`,
  `flat.spec_sampler`, `flat.diffuse_sampler`) and a CLI override
  (`--lobe-samplers`), parallel to `--proposals`. Changing any selection resets
  progressive accumulation.

Non-goals (explicitly out of scope): no new lobes, no neural / learned samplers,
no per-material selection (the GUI selection is global across flat materials).

## Capabilities

### New Capabilities

(none — this populates the seam and transport defined by existing capabilities)

### Modified Capabilities

- `flat-bsdf-lobes`: the existing "per-lobe runtime-pluggable sampler seam"
  requirement moves from native-only to a **populated registry**, and gains the
  contract that the renderer-selected per-lobe strategy is used by **both**
  `sample()` and `evaluate()` so the pdf agreement (hence unbiasedness and
  bounded weight) holds for every registered strategy, not just the native one.
- `scene-sampling`: the command-line / GUI / persisted-selection requirement
  extends to the per-lobe sampler selection, transported through `FrameConstants`
  with no new GPU buffers or descriptor bindings (mirroring the analytic-proposal
  "contributes no GPU state" rule).

## Impact

- **Shaders**: `shaders/common.slang` (FrameConstants `flatLobeSamplers` field);
  `shaders/materials/flat/flat_lobes.slang` (samplerId dispatch cases, new
  Heitz-2018 basis-form VNDF + uniform-hemisphere strategies, samplerId-aware
  per-lobe bounded weight); `shaders/materials/flat/flat_material.slang` (unpack `fc`
  per-lobe id, thread it into the lobe helpers). `main_pass.spv` recompiles at
  runtime via the vk_compute source-hash cache (not git-tracked).
- **Host**: new `sampling/lobe_samplers.py` (registry); `renderer.py`
  (`_pack_uniforms` packs `flatLobeSamplers` in std140 lockstep, three selection
  indices, `_current_state_hash` includes them); `app.py` (`ALL_PARAMS`
  selectors + `--lobe-samplers` CLI parsing).
- **Tests**: `tests/test_sampling_parity.py` gains a per-sampler parity case
  (coat/spec = Heitz-2018 basis-form VNDF converges identically, megakernel ≡
  wavefront, PT ≡ BDPT) and a diffuse native-vs-uniform variance A/B; the ReSTIR suite and
  the all-native default-preset pixel-identity check must stay green.
- **Docs**: `docs/Architecture.md` (FrameConstants field), the flat-BSDF section
  of `docs/SkinRendering.md`, `README.md` (`--lobe-samplers` flag).
