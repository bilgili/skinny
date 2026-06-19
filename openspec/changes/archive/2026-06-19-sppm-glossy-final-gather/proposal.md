## Why

Surface SPPM (PM-1) stores one visible point at the first non-perfectly-specular
hit and reconstructs its radiance by gathering photons within a search radius.
On glossy reflectors — polished metals (`metalness=1`, low roughness) — the BRDF
lobe is sharply peaked, so photon density inside the lobe is far too sparse to
rebuild a sharp inter-object reflection. A pure metal has no diffuse term, so its
whole appearance *is* that reflection: under SPPM it washes out. Seen on
`three_materials_demo` — the brass sphere fails to reflect the wood/marble
spheres (path/bdpt resolve it; SPPM does not). The eye-walk continue-vs-gather
test is `bs.pdf <= 0` (perfect-delta only), so glossy lobes that are visually
near-mirror fall on the gather side and lose their reflection.

## What Changes

- **Glossy continuation in the SPPM eye walk.** When the sampled lobe is below a
  roughness threshold (near-specular), follow the BSDF-sampled direction one
  bounce — exactly as the existing delta caustic-carrier branch does — instead of
  storing a visible point on the glossy surface. The visible point then lands on
  the next non-glossy hit, so the glossy reflection is reconstructed sharply and
  accumulated across progressive passes. This is the cheap, high-impact fix and
  fits the current carrier branch.
- **Roughness threshold** `sppmGlossyContinueRoughness` (FrameConstants tail +
  `--sppm-glossy-roughness` CLI), tuned so polished metals continue and matte /
  rough surfaces still gather. `0` reproduces today's behavior (delta-only).
- **Photon deposit unchanged** — still deposits at non-specular vertices after
  ≥1 bounce; a glossy-continued vertex is treated like a specular vertex (no
  deposit), keeping direct/indirect disjoint.
- Both backends, wavefront, **flat materials only** — same gating as PM-1.
- A full **final-gather** variant (one BSDF-sampled gather ray per glossy VP that
  reads the photon estimate at its next hit) gives lower variance for
  *mid*-roughness but needs an extra wavefront ray pass; recorded as a deferred
  direction in `design.md`, not built here.

## Capabilities

### Modified Capabilities
- `photon-mapping`: adds a requirement that glossy / near-specular reflectors are
  reconstructed by following the lobe (not lost to the photon gather), with a
  roughness threshold separating continue from gather. PM-1's flat-only,
  wavefront-only, both-backend, no-double-count, and caustic-parity requirements
  are unchanged.

## Impact

- **Shaders:** `shaders/integrators/wavefront_sppm.slang` (eye-walk continue
  branch + threshold; photon walk treats glossy-continue as specular),
  `shaders/common.slang` (FrameConstants tail `sppmGlossyContinueRoughness`).
- **Renderer/host:** `renderer.py` (pack the threshold into the fc tail +
  `_FC_SCALAR_FIELDS` + the MSL fc pin), `cli_common.py`
  (`--sppm-glossy-roughness`), front-end persistence.
- **Docs:** `docs/PhotonMapping.md` (glossy-continuation section + the limitation
  it lifts), `README.md` (new flag), `CHANGELOG.md`.
- **Tests:** three_materials_demo glossy-reflection A/B (brass reflection present
  + trends toward the path reference); PM-1 caustic parity + energy-ratio gates
  stay green; path/bdpt/skin/volume byte-unaffected.
- **No breaking changes.** Threshold default is opt-in-shaped; `0` == today.
