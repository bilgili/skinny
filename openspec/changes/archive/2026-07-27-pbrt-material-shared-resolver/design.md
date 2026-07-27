# Design — pbrt-material-shared-resolver

## Context

`src/skinny/pbrt/materials.py` maps a pbrt material to two authoring targets.
Read of the full file gives this duplication map:

**Already shared (single implementation, both pipelines call it):**

- `resolve_texture` (:37) — imagemap/scale texture resolution.
- `ParamValue` (:64) + `get_float_texture` (:81) / `get_spectrum_texture`
  (:144) — pbrt-style promoting accessors (constant / texture / named
  spectrum), incl. `_named_spectrum_scalar` (:121) and the `_IOR_PARAM_NAMES`
  guard (:118) that keeps a glass IOR out of roughness lanes.
- `pbrt_roughness_to_alpha` (:182) / `alpha_to_usd_roughness` (:188) — the
  parity-critical roughness calibration chain.
- `_conductor_basecolor` (:222) — named-metal / eta+k / copper-fallback
  reflectance (reads both `eta` and `conductor.eta` spellings).
- `material_spectral_overrides` (:254) — named-conductor / dispersive-glass
  identity for spectral mode.
- `_subsurface_overrides` (:334) — pbrt subsurface → σ_a/σ_s/g/eta medium
  coefficients (explicitly documented "emitted by both mappers so the two
  import paths produce identical coefficients").

**Twin / duplicated (copy-paste, the target of this change):**

- `_resolve_roughness` (:193) vs `_resolve_roughness_mtlx` (:295) — identical
  remap flag read, identical triple `get_float_texture` calls, identical
  texture-fallback (`ParamValue(0.5, tex)` + same note string), identical
  isotropic chain. They differ **only** in anisotropic representation.
- `map_material` (:553) vs `map_material_mtlx` (:375) — the same 11-branch
  `mtype` dispatch (`""`/`none`, `interface`, `diffuse`, `conductor`,
  `dielectric`/`thindielectric`, `coateddiffuse`, `coatedconductor`,
  `diffusetransmission`, `subsurface`, unknown) with the same param reads,
  same defaults, same notes, same `_subsurface_overrides` merge, plus
  duplicated local closures (`put`/`reflectance`/`roughness`/`scalar`) and a
  duplicated verbatim APPROX-on-unresolved-texture postlude.

**Genuinely target-specific (must stay in the adapters):**

- Input vocabulary: `diffuseColor`/`roughness`/`metallic`/`opacity` vs
  `base_color`/`specular_roughness`/`metalness`/`transmission`/`coat_*`.
- Anisotropy: USD collapses to isotropic geometric mean (+note);
  standard_surface represents it as `specular_roughness` mean +
  `specular_anisotropy`.
- Transmission: USD `opacity = 0.0` gate vs mtlx `transmission = 1.0` +
  `transmission_color` (+ `thin_walled` for thindielectric).
- Emission: USD `emissiveColor` vs mtlx `emission = 1.0` + `emission_color`
  (the unit weight is load-bearing for the round-trip; comment at :536).
- Richness: mtlx additionally authors `specular_color` (conductors),
  `subsurface_color`/`subsurface_radius`, `transmission_color` from
  `transmittance`, `coat_*`; USD drops these (subset target).
- tex_inputs `value_type` derivation: USD from `_USD_INPUT_KIND`; mtlx from
  the constant's Python type.

**Accidental copy drift (real divergence, not target-driven):**

1. `coatedconductor` base-metal roughness: mtlx reads `conductor.roughness`
   (:493, correct pbrt-v4 spelling); USD reads top-level `roughness` (:646).
   The only *live* conflicting-read drift. (Historical evidence of the drift
   class: the `coateddiffuse` coat-roughness source diverged once and was
   fixed — comment at :480; both paths now consistently read top-level
   `roughness` there.)

**One-sided pbrt-param reads (exhaustive inventory — every read that happens
in only one pipeline, so the other emits neither value, note, nor status
escalation for it):**

- `diffusetransmission` `transmittance` — mtlx only (:509); USD never reads it.
- `subsurface` `reflectance` → `subsurface_color` — mtlx only (:519); the USD
  branch (:654–664) reads no reflectance at all.
- `subsurface` `radius` — mtlx only (:523).
- `coateddiffuse` `interface.eta` → `coat_IOR` — mtlx only (:479); USD branch
  (:635–642) never reads it.
- `coatedconductor` `interface.eta` → `coat_IOR` — mtlx only (:504); USD
  branch (:643–648) never reads it.

These matter beyond value-plumbing because reads have **side effects**: the
promoting accessors append notes on unresolved textures, `scalar()` appends
target-worded degradation notes (":437 standard_surface" vs ":603 USD"), and
EXACT→APPROX escalation keys on note *content* (`any("unresolved/unsupported"
in n)` at :547/:674). An unconditional resolver read of, say, a texture-bound
`interface.eta` on `coateddiffuse` would add a note (and possibly flip status)
on the USD path that today emits neither — breaking the report/byte-identity
gate.

Callers (whole blast radius): `api.py:_author_material` (:375) and
`_author_material_mtlx` (:427) only; `mtlx_emit.py` consumes the mtlx return
shape. Both mapper signatures/return shapes are public seams to preserve.

## Goals / Non-Goals

**Goals**

- One resolver: pbrt params → target-agnostic `ResolvedMaterial`; each mapper
  becomes a thin emit adapter over it.
- New pbrt param wired exactly once.
- Hostless resolver tests; existing output-level tests pass unmodified.
- Byte-identical importer output (both plain-USD and `-mtlx` paths).

**Non-Goals**

- No behavior change — the three copy drifts above are **preserved** (encoded
  explicitly in the resolved form / adapters) and filed as follow-ups; fixing
  drift 1 here would change committed fixture output.
- No change to mapper signatures, return shapes, notes text, or report
  statuses (notes strings are asserted by tests and surfaced in reports).
- No new pbrt material types, no shader/renderer changes, no changes to
  `api.py` / `mtlx_emit.py`.

## Decisions

**D1 — Resolved-intermediate shape.** A plain dataclass `ResolvedMaterial`
(same module; no new file unless materials.py stays unwieldy):

- `lobes: dict[str, ParamValue]` keyed by a small target-neutral vocabulary
  (`base_color`, `roughness`, `metallic`, `ior`, `transmission`,
  `transmission_color`, `coat`, `coat_ior`, `coat_roughness`, `subsurface_*`,
  `emission_rgb`, …). `ParamValue` is reused as-is — it already carries
  const + texture.
- `anisotropy: RoughnessAniso | None` — the *unreduced* per-axis roughness
  `(ru, rv)` so each adapter applies its own reduction (geometric-mean
  collapse vs mean+anisotropy). This is where the twin roughness resolvers
  merge: one resolver returns texture / iso / per-axis; the collapse is
  adapter policy.
- `overrides: dict` — `_subsurface_overrides` result (already shared).
- `status`, `notes: list[str]` — per-flavor: the resolver is invoked with a
  `flavor` and produces exactly the notes/status that flavor's pipeline emits
  today, in today's **order** (accessor notes interleave with branch notes in
  read order; a shared note list computed flavor-blind cannot reproduce
  either path). Target-worded notes (e.g. `scalar()`'s ":437
  standard_surface" vs ":603 USD" degradation wording) take their wording
  from the flavor. Adapter-appended notes (e.g. the USD "coat roughness
  texture not connected" note) stay in the adapter at the same sequence
  point. Notes and statuses feed the import report, which is part of the
  task-1.1 hash-diff gate — they are output, not diagnostics.
- Alternative considered: a per-material-type union of typed records — more
  structure, zero extra safety here (adapters immediately flatten to dicts);
  rejected as over-modeling.

**D2 — Where real divergences live.** Adapters own only: vocabulary renaming,
anisotropy reduction, transmission encoding, emission encoding, value_type
derivation, and *dropping* lobes their target can't express (USD drops
`specular_color`/`subsurface_color` etc. by simply not mapping those keys).
Adapters never read `ParamSet` — which forces the resolver to own every read,
including the one-sided ones. Because reads have side effects (notes, status
escalation — see Context), **every flavor-divergent or one-sided read is
flavor-gated in the resolver**, exhaustively:

- drift 1: `coatedconductor` base roughness — `conductor.roughness` under
  `mtlx`, top-level `roughness` under `usd`;
- one-sided, performed only under `mtlx`: `diffusetransmission`
  `transmittance`; `subsurface` `reflectance` and `radius`; `interface.eta`
  on `coateddiffuse` and on `coatedconductor`.

Under `usd` these reads simply do not happen — no value, no note, no status
flip — reproducing today's USD branches (:635–648, :654–664) exactly. Each
gate carries a comment naming its follow-up change. This inventory is closed
by construction: task 3.3 deletes the twin mappers, and the grep gate (spec
scenario) proves no `ParamSet` read survives outside `resolve_material`, so a
missed one-sided read cannot hide — it either appears flavor-gated in the
resolver or the byte-identity diff catches it. Alternatives rejected:
resolve both readings always and let adapters choose (doubles reads, and the
extra reads' notes/status side effects break byte-identity — precisely the
failure mode above); lazy thunks per lobe (defers the same side effects to an
adapter-controlled order, more machinery than a flag check).

**D3 — Mappers stay the public API.** `map_material` / `map_material_mtlx`
keep their exact signatures and `(inputs, tex_inputs, status, notes)` return;
internally each is `resolve_material(...)` + its emit table. No caller edits.

**D4 — Tests.** New hostless `tests/pbrt/test_material_resolve.py` asserting
resolved forms per material type (incl. named-spectra, texture bindings,
anisotropy, subsurface precedence). The byte-identity lock is a diff over the
suite/corpus imported `.usda` before vs after (see Migration Plan), plus the
untouched output-level tests.

## Risks / Trade-offs

- **[Risk] `-mtlx` subsurface radius vec3≠color3 + sigma-input malformations
  regress** (recorded bug, mtlx-subsurface-plugin-roundtrip). → Mitigation:
  `subsurface_radius` stays a plain list in the resolved form; the mtlx emit
  path (`mtlx_emit.py`) is untouched; `test_mtlx_roundtrip.py` +
  `test_materials_mtlx.py` run unmodified.
- **[Risk] transmission→opacity bridge fires on authored opacity (glass goes
  opaque)** (recorded bug, glass-transmission-opacity-gate — lives on the
  MaterialX *intake* side). → Mitigation: no intake changes at all; the mtlx
  adapter must keep authoring `transmission` without any `opacity`, exactly as
  today; round-trip tests lock it.
- **[Risk] skinnyOverrides SSS merge order changes** (recorded bug,
  pbrt-mtlx-roundtrip-fix: overrides merge in `api.py`). → Mitigation:
  `api.py` untouched; `_subsurface_overrides` keys and the
  `inputs.update(...)` placement preserved verbatim in the resolved form.
- **[Risk] constant-spectrum achromatic [v,v,v] handling drifts** (recorded
  bug, constant-spectrum-achromatic-rgb). → Mitigation: `spectra.param_to_rgb`
  and `get_spectrum_texture` are shared already and not restructured;
  `test_spectra.py` / `test_named_spectra.py` unmodified.
- **[Risk] named conductor/glass spectra (7 metals / 7 glasses) resolution
  changes** — d-line IOR substitution, `_IOR_PARAM_NAMES` guard,
  `conductor.eta` spelling, copper fallback + note text. → Mitigation:
  `_conductor_basecolor`, `_named_spectrum_scalar`,
  `material_spectral_overrides` move (at most) untouched; note strings are
  byte-preserved; `test_named_spectra.py` asserts them.
- **[Risk] note/status text drifts** (reports and tests assert exact
  strings). → Mitigation: notes emitted from single shared literals; the
  output-level tests are the tripwire.
- **[Risk] "while I'm here" temptation to fix drift 1 (`coatedconductor`
  roughness)** silently changes fixtures. → Mitigation: explicitly frozen via
  the `flavor` parameterization; follow-up change filed instead.
- **[Risk] a one-sided read is missed in the D2 inventory** and gets resolved
  unconditionally, emitting a note/status on the path that never read it. →
  Mitigation: the inventory in Context is exhaustive by grep; the grep gate
  forbids `ParamSet` reads outside the resolver; the synthetic all-mtypes
  scene (Migration Plan step 1) puts every branch — including
  `coatedconductor` and `diffusetransmission`, absent from suite+corpus —
  under the byte-identity diff, reports included.
- **Trade-off:** the `flavor` argument is a wart — a pure resolver would be
  flavor-free. It exists only to freeze recorded drift and one-sided reads
  (with their note/status side effects); each gate is commented and removable
  by the follow-up fix changes.

## Migration Plan

Single-step refactor gated on byte-identity:

1. Before touching code, snapshot importer output: run the importer over the
   confirming-suite scenes + corpus `.pbrt` sources (both plain and `-mtlx`)
   into a scratch dir; record hashes of the `.usda`/`.mtlx` documents **and
   the import reports** (notes/statuses are output too). Coverage gap: no
   suite or corpus scene uses `coatedconductor` or `diffusetransmission`, so
   the suite+corpus diff alone never exercises drift 1 or the `transmittance`
   read at document level — add one synthetic hostless `.pbrt` exercising all
   11 material-type branches (incl. textured and named-spectra variants) to
   this sweep.
2. Land the resolver + adapters.
3. Re-run the same imports; **diff must be empty** (byte-identical `.usda` and
   `.mtlx`). Any diff = bug in the refactor, not a regen candidate — the
   committed `.usda` fixtures under `tests/` are importer output and the
   target is *no regeneration* (pbrt-named-spectra convention: if output ever
   legitimately changed it would need regen + diff-gate, which this change
   forbids for itself).
4. Hostless: full `tests/pbrt/` non-gpu sweep + new resolver tests.
5. GPU: confirming-suite authoring-equivalence gate
   (`tests/pbrt/test_suite.py`, plain-USD ≡ MaterialX) green on Metal — the
   behavioral proof.
6. No rollback machinery needed: single module, callers untouched; revert is
   `git revert`.

## Open Questions

- Same module or new `material_resolve.py`? Default: same module (677 lines
  shrinks after dedup); split only if the resolver + adapters exceed roughly
  today's size.
- Should drift 1 (`coatedconductor` USD base roughness reads `roughness` not
  `conductor.roughness`) be fixed in an immediate follow-up change with fixture
  regen + baseline diff, or batched with drift 3? (Out of scope here either
  way.)
- Is a resolver-level golden test (serialized `ResolvedMaterial` per corpus
  material) worth keeping after landing, or is it scaffolding to delete once
  byte-identity is proven? Default: keep the per-type unit tests, drop any
  golden snapshots.
