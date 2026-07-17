## Context

The spectral GPU path is already built and shipped: named-conductor Fresnel binds
vendored eta/k at binding 48 (Group 6.2), authored illuminant SPDs bind at binding 50
(Group 6.3), and hero-λ glass dispersion consumes a Cauchy `(A, B)` pair packed into
`FlatMaterialParams` (Group 6.4). The import side already carries authored spectra to
the renderer over the `skinnyOverrides` customData side-channel
(`materials.material_spectral_overrides` → `conductor_metal` / `glass_dispersion`;
`lights._preserve_spectral` → `spectral` payload, consumed for distant lights only).

What is missing is **coverage and resolution**, not architecture:

| Surface | Today | pbrt-v4 |
|---|---|---|
| `spectral_tables._GLASS_CAUCHY` | `{"default", "bk7"}`, both = the same BK7 fit | 7 named glasses |
| `_extract_pbrt_spectra._METALS` | Ag, Al, Au, Cu | + CuZn, MgO, TiO2 |
| `stdillum-*` | unhandled anywhere | 16 SPDs (A, D50, D65, F1–F12, acesD60) |

Three failure modes follow, all silent. `spectra.named_glass_key` maps **any**
unrecognised glass name to `"default"`, which is the BK7 fit — `glass-LASF9` (n_d≈1.850)
renders at n≈1.504. `materials._conductor_reflectance` falls back to **copper** for an
unvendored metal. And `usd_loader._extract_light_spd` early-returns unless the payload
`kind` is `spectrum_samples`, so the `spectrum_named` payload an importer already writes
for `stdillum-A` is dropped on the floor.

Constraint: this must stay additive. An RGB-only scene must import byte-identically, and
`glass-BK7` plus the four existing metals must keep their current keys.

## Goals / Non-Goals

**Goals:**

- Import every pbrt-v4 named glass, metal, and standard illuminant at pbrt's own data.
- Resolve named illuminants to both an RGB chromaticity and a bindable 95-sample SPD.
- Preserve authored spectra wherever a consumer exists (lights). *Inline spectra on
  materials were a goal originally and were cut — see Non-Goals.*
- Make an unrecognised name produce a visible import note naming its fallback.
- Keep the identity surviving import → `.usda` → render on the plain-USD and `-mtlx`
  paths alike.

**Non-Goals:**

- **No USD→pbrt writer.** "Export" here means round-tripping the authored identity
  through skinny's own USD/MaterialX output, not emitting `.pbrt` files.
- **No descriptor-binding or `FlatMaterialParams` layout change.** Every new datum is
  expressed in the formats the GPU already consumes (Cauchy `(A, B)`, eta/k curves,
  95-sample SPDs).

  **CORRECTED TWICE during implementation — this was originally "no shader change", and
  that was wrong.** The first pass verified `bindings.slang:136 namedMetalEtaK` indexes
  `(metalId-1) * SPECTRAL_METAL_STRIDE` generically and concluded no `.slang` edit was
  needed. That check was too narrow: a *second* gate,
  `spectral_flat_common.slang:53`, hard-coded `c.isConductor = (metalId >= 1u && metalId
  <= 4u)`. So CuZn/MgO/TiO2 would upload at correct offsets and then silently render with
  RGB Schlick instead of their vendored eta/k — the exact class of silent-wrong-material
  bug this change exists to kill. The gate now reads a new
  `SPECTRAL_METAL_COUNT` constant, pinned to `len(renderer._SPECTRAL_METAL_ORDER)` by a
  hostless test.

  Lesson: "does the GPU consume this generically?" must be answered by grepping **every**
  use of the id, not just the indexing helper. The RGB SPIR-V is verified byte-identical
  to main, so the blast radius stays inside the spectral build.
- **No camera-response curves.** pbrt's `canon_*` / `ilford_*` named spectra are film
  sensor responses, not scene spectra; they belong to a film-sensor change.
- **No RGB-reflectance-spectrum fidelity work.** Reflectance still reduces under an
  equal-energy whitepoint (the existing documented simplification).
- **No inline-spectrum preservation on materials.** *Cut during implementation* (it was
  originally in scope). The payload had nowhere to go: `skinnyOverrides["spectral"]` is read
  only by `usd_loader._extract_light_spd`, and only for distant **light** prims. There is no
  per-material SPD field, buffer, or shader path, so writing it would serialize data nothing
  reads — a dead override that reads as a working feature (the same trap as D9's stale
  assets). Materials keep upsampling from their RGB reduction. Doing it properly needs a
  loader field + packer + binding + shader change, i.e. its own change; the honest version of
  "inline spectra are covered" is: preserved **on lights**, where they are consumed.
- **No widening of spectral SPD consumption.** Only distant lights bind an illuminant SPD
  today; point/spot/infinite/area lights spectrally upsample from RGB. A named illuminant
  gets the correct chromaticity on every light type, but an SPD only where one is already
  consumed (D4).

## Decisions

### D1: Fit a 2-term Cauchy per glass rather than vendor pbrt's tabulated eta

pbrt stores each glass as a `PiecewiseLinearSpectrum` (~24 samples, 300–1100 nm). The GPU
wants `n(λ) = A + B/λ_µm²`. Fitting per glass by least-squares over skinny's 360–830 nm
grid keeps the packing, the shader, and the binding map untouched — the entire change stays
in Python and one `.npz`.

Measured max |Δn| over the visible, fit against pbrt's tables:

| Glass | A | B | max resid | n_d |
|---|---|---|---|---|
| BK7 | 1.50431 | 0.004267 | 3.0e-04 | 1.51673 |
| BAF10 | 1.64775 | 0.007720 | 1.5e-03 | 1.66988 |
| FK51A | 1.47768 | 0.003035 | 1.7e-04 | 1.48651 |
| LASF9 | 1.80852 | 0.014634 | 4.6e-03 | 1.85004 |
| F5 | 1.63949 | 0.011655 | 4.0e-03 | 1.67254 |
| F10 | 1.68848 | 0.013994 | 5.9e-03 | 1.72806 |
| F11 | 1.73547 | 0.017399 | 7.5e-03 | 1.78448 |

*Alternative considered — a 3rd Cauchy term* (`+ C/λ⁴`): measured, and it does **not** pay.
On the worst glass (F11) the residual only moves 7.5e-03 → 6.1e-03, and on LASF9 it gets
*worse* (4.6e-03 → 4.9e-03) — the residual is piecewise-linear interpolation error in
pbrt's own sparse table, not a missing Cauchy order. A third term would cost a
`FlatMaterialParams` layout change and a shader edit to buy nothing.

*Alternative considered — vendoring the tabulated curve and interpolating on the GPU*:
correct to the last digit, but costs a new binding, a texture upload, and a shader change
for a ≤0.4% IOR error on the two densest flints. Rejected as out of proportion; the
per-glass residual is recorded as a tolerance in a hostless test so a regression is caught.

### D2: Refit BK7 from pbrt's table too, rather than pinning the current constants

The existing `bk7` entry `(1.5046, 0.00420)` came from published catalogue coefficients;
the refit is `(1.50431, 0.004267)`. They differ by |Δn| ≈ 3e-4. Refitting everything from
one source of truth (pbrt's table) is what makes "match pbrt" checkable, and one fit path
for all seven glasses is less code than a special case that exempts BK7.

**Consequence (spectral)**: BK7 dispersion shifts by ~3e-4 in IOR, so this is *not*
bit-identical for existing BK7 spectral scenes — the `dispersion_prism` showcase asset must
be re-measured. The shift moves *toward* pbrt, so per the standing rule the baseline must
come out equal or lower; if it rises, the fit is wrong and the change is blocked, not the
baseline raised.

**Consequence (RGB) — larger, and the more important one.** The Cauchy-A→`ior` substitution
at `renderer.py:691` is explicitly **spectral-only**: the comment at `renderer.py:688-690`
records that the RGB build leaves `ior` at its authored/fallback value, which for a named
glass is the generic `scalar("eta", 1.5)` default (`materials.py:563`) — because
`get_float_texture` has no scalar for a named spectrum. Giving the importer the per-glass
d-line IOR (the fix this change wants) therefore moves the **RGB** build too:
BK7 `1.5 → 1.51673` (|Δn| ≈ 1.7e-2, ~50× the spectral refit shift) and LASF9 `1.5 → 1.85`.

So the blast radius is *both* pipelines, not just spectral, and the "spectral-only"
invariant recorded in that comment stops being true for named-glass materials. Two
follow-ons: RGB-mode named-glass baselines need re-measuring alongside the spectral ones,
and the stale comment must be corrected rather than left to mislead the next reader. This
is still the right fix — an RGB `glass-LASF9` rendering at n=1.5 is simply wrong — but it
must be declared, not discovered at gate time.

**Coordination**: the `spectral-dispersion-showcase-asset` change is in flight against the
same prism asset. Sequence after it lands rather than racing it.

### D3: `"default"` stops meaning BK7; unknown names report

Today `named_glass_key` returns `"default"` for any unrecognised name and `"default"` *is*
BK7 — so a typo and a genuinely-unsupported glass both silently render as BK7. Split those:
a recognised name resolves to its own key; an unrecognised one still falls back (the importer
is a best-effort translator and must not hard-fail a scene) but records an APPROX note naming
the substitution, matching how `get_float_texture` already reports an unresolved texture. Same
treatment for unknown `metal-*` (which currently defaults to copper with a note — keep the
note, widen the vendored set so it fires far less) and unknown `stdillum-*`.

**`default` keeps the BK7 refit as its value.** The note is what removes the silence; a
separate invented "generic crown" constant would be unsourced — contradicting D2's
one-source-of-truth rule — and the value is load-bearing (`named_glass_cauchy("default")`
feeds `renderer.py:695`). So `default` is BK7's coefficients *and says so*, rather than
`default` being an undefined third thing.

**An unmatched name may be a file path, not a typo.** pbrt falls back to
`readSpectrumFromFile` on a `GetNamedSpectrum` miss (`paramdict.cpp:444-450`); skinny has no
spectrum-file reader, so `spectra.py:217` would hand a `.spd` path to the glass matcher and
"substitute crown glass" for it. The note must distinguish a path-like string ("spectrum file
not read") from an unknown name, or it will confidently mis-report.

### D4: Named illuminants resolve into the existing payload shape, adding no renderer concept

The `.usda` preserves the **name** (`spectrum_named`), and the **loader** resolves it to the
vendored 95-sample SPD — the shape `_extract_light_spd`, `LightDir.spectral_spd`, and binding
50 already consume. The renderer learns nothing new; the only edit is teaching
`_extract_light_spd` to look a name up. Preserving the name (rather than baking samples into
the `.usda` at import) is what the payload requirement asks for and keeps the authored
identity legible in the file.

**Unit-luminance normalisation is right, with better provenance than "blackbody does it".**
pbrt normalises named illuminants to luminance 1 at load (`spectrum.cpp:2600-2630`
`FromInterleaved(…, true)` → `spectrum.cpp:153-155` scales by
`CIE_Y_integral / InnerProduct(spec, Y)`), and its film divides by `CIE_Y_integral` — so
pbrt's own film luminance for `"spectrum L" "stdillum-A"` is exactly 1, which is what
`xyz/xyz[1]` reproduces. (The blackbody analogy is mechanically right but coincidental:
pbrt's `BlackbodySpectrum` normalises to *peak* 1, not luminance 1.) Measured:
`stdillum-D65` → `[0.99940, 1.00018, 0.99995]`, max deviation 6.0e-4, so "≈ neutral" is
testable at ~1e-3.

**Scope limit — SPD consumption is distant-light-only, and this change does not widen it.**
`_extract_light_spd` is called from exactly one site (`usd_loader.py:1226`, the `LightDir`
construction), and binding 50 is documented per-distant-light at `renderer.py:383`. The other
light types have no SPD path: area lights read `emissive_spectral` but accept **only**
`kind == "blackbody"` (`renderer.py:5910`, `renderer.py:6282`), packing a float2
`(temperature, scale)` into binding 49 — there is nowhere to put an SPD.

Meanwhile `lights._preserve_spectral` already writes payloads for distant (`:87`), point/spot
(`:100`) **and** infinite (`:144`) lights, so today an inline `spectrum L` on a point light is
preserved to USD and then silently ignored. That is pre-existing, not introduced here.

So a named illuminant lands as: **distant** → correct chromaticity *and* SPD; **point / spot /
infinite / area** → correct chromaticity, no SPD (spectrally upsampled from RGB), exactly as an
inline `spectrum L` behaves on those lights today. That is the honest envelope and the spec
says so. Extending SPD consumption to the other light types means new bindings and a widened
binding-49 record — a separate change, not a rider on this one.

*Alternative considered — widen binding 49 to carry an SPD slot for area lights*: it would make
`stdillum-*` area lights fully spectral, but costs a record-layout change, shader edits, and a
Metal argument-table review, for a light type where the RGB chromaticity fix already captures
most of the visible error. Rejected as scope creep; recorded as the follow-up.

The RGB reduction follows the path `blackbody` already established: integrate the SPD to
XYZ → linear sRGB, normalise to unit luminance so only chromaticity comes from the name and
magnitude keeps coming from the light's `L`/`scale`.

### D5: pbrt's glass names do not match its array names

`"glass-F5"` / `"glass-F10"` / `"glass-F11"` read from the C++ arrays `GlassSF5_eta` /
`GlassSF10_eta` / `GlassSF11_eta` (SF = dense flint). The extraction tool must map the
public pbrt *name* to the internal symbol, not assume they agree — an easy source of a
silently-wrong-glass bug.

### D6: Reuse the existing extraction tool and `.npz`

`_extract_pbrt_spectra.py` already vendors pbrt data verbatim into `spectral_curves.npz`
against a local checkout (`~/projects/pbrt-v4`, the same pinned build the parity harness
uses). Extend its `_METALS` tuple and add glass/illuminant extraction; existing arrays keep
their names and values, so the `.npz` grows additively.

### D7: Extended metals need three host-side edits held together by an ordering invariant

The metals axis is not import-only — `renderer.py` hard-codes the four-metal set twice, and
the two sites must agree:

- `renderer.py:570` — `_CONDUCTOR_METAL_ID = {"au": 1, "ag": 2, "al": 3, "cu": 4}`, packed
  into `_specularColorPad.w` and read by the shader as `conductorMetalId`
  (`common.slang:133`). Needs `cuzn`/`mgo`/`tio2` → 5/6/7.
- `renderer.py:4004` — `for name in ("au", "ag", "al", "cu")` builds the `spectralMetals`
  upload. Needs the same three appended.
- `spectra._CONDUCTOR_CANON` — the import-side gate that decides a name is a known metal.

The load-bearing constraint is stated in the comment at `renderer.py:568-569`: the id **is**
the upload index + 1 (`namedMetalEtaK` does `(metalId-1) * SPECTRAL_METAL_STRIDE`). Appending
the new metals at the end of both, in the same order, keeps ids 1–4 and every existing
`spectralMetals` byte offset unchanged — so existing Au/Ag/Al/Cu scenes stay bit-identical.
Inserting alphabetically instead would silently renumber the existing metals and swap
materials in every checked-in scene; don't.

**Revised during implementation:** this decision originally said to keep the two lists and
let a test assert they agree, on the assumption that unifying them meant refactoring the
upload path. It didn't — `_SPECTRAL_METAL_ORDER` derives from `_CONDUCTOR_METAL_ID` by
sorting on the id, and the upload site changes by exactly one line
(`for name in ("au","ag","al","cu")` → `for name in _SPECTRAL_METAL_ORDER`). Deriving is
both the smaller diff and the structurally safer one, so it wins over the stated plan; the
invariant is now unbreakable by construction rather than by test. The alignment test stays,
narrowed to what still can regress: that au/ag/al/cu keep ids 1–4.

### D8: Fix the illuminant branch's missing `_Y_INTEGRAL` division rather than ship a third convention

Adding a named-illuminant RGB path forces a scale convention, which exposed a pre-existing
discontinuity in `spectra.sampled_spectrum_to_rgb`: `spectra.py:120-122` divides by
`_Y_INTEGRAL` (106.92) **only** on the reflectance branch. Measured against the live code:

| input | result |
|---|---|
| `[400 10 700 10]` illuminant (constant shortcut) | `[10, 10, 10]` |
| `[400 10 700 10.000001]` illuminant (projection) | `[1283, 1015, 971]` — **~107×** |
| `[400 0.2 700 0.2]` reflectance | `[0.2, 0.2, 0.2]` |
| `[400 0.2 700 0.2000001]` reflectance | `[0.240, 0.190, 0.182]` — 0.95×, whitepoint tint only |

A 1e-6 nudge swings a constant illuminant by two orders of magnitude. The constant shortcut is
the established-correct anchor (`spectra.py:110-112`, GPU-validated by
`constant-spectrum-achromatic-rgb`), and pbrt divides by `CIE_Y_integral` for *all* spectra
(`SpectrumToXYZ`), so the colored illuminant branch is simply wrong — the reflectance branch
is right.

Unit-luminance named illuminants agree with the constant anchor and with pbrt. Leaving the
inline branch unfixed would make named ≈107× dimmer than an inline SPD of the same shape, so
the spec's "exactly as an inline sampled SPD would" would be false by two orders of magnitude.

Fix it: one line, `xyz = xyz / _Y_INTEGRAL` on the illuminant path too. **Blast radius is
zero** — verified by grep, no checked-in scene under `tests/` or `assets/` authors a
non-constant inline `spectrum L`/`I`, and the only `illuminant=True` callers are the four
light sites (`api.py:255`, `lights.py:85/98/143`). It does modify the existing
"Colored sampled spectra keep the CIE projection" requirement, whose scenario currently
promises bit-identity with the pre-change projection — hence the MODIFIED delta.

*Alternative considered — defer it and non-goal the inconsistency*: defensible only if written
down, and it leaves three mutually-disagreeing conventions in one function for the next
reader to trip over. The fix is one line with no scenes affected; deferring costs more than
doing it.

### D9: Regenerating the committed assets is the gate, not the metric

The importer is not what renders. `spec_prism.usda:161`, `spec_prism_mtlx.mtlx:27`, and
`prism_dispersion.usda:123` are **git-tracked importer output** that hard-code
`ior = 1.5`, produced by `tests/assets/suite/_gen/build.py:312/319`. Fixing
`get_float_texture` and unit-testing it changes nothing those files say, so a baseline
re-measure over stale assets would correctly report "no change" — and the change would ship
green over a fix that never reached a pixel.

So asset regeneration comes **before** any measurement, and the gate is a **diff** showing
`1.5 → 1.51673` actually appears in the regenerated `.usda`/`.mtlx` — not a metric that can
pass by doing nothing. `assets/glass_machines.usda` (untracked, 7 `glass-BK7` dielectrics) has
no `_gen` and must be re-imported by hand.

## Risks / Trade-offs

- **BK7 refit is not bit-identical, and the d-line IOR moves the RGB build far more than
  the refit moves the spectral one (D2)** → Re-measure both the RGB-mode and spectral-mode
  baselines for named-glass scenes (`dispersion_prism` + the spectral suite); require each
  pbrt-truth metric to hold or improve. If one worsens, the fit is wrong — fix the fit,
  never raise the baseline. Correct the now-stale "spectral-only" comment at
  `renderer.py:688-690`.
- **The metal-id ↔ upload-order invariant is implicit across two files (D7)** → Append-only
  ordering plus a hostless test asserting `_CONDUCTOR_METAL_ID` agrees with the upload list.
  Renumbering existing ids would silently swap conductors in every checked-in scene.
- **The fix can silently fail to reach a render (D9)** → The committed `.usda`/`.mtlx` are
  importer output; a measurement over stale assets passes by doing nothing. Gate on the
  regenerated-asset diff *before* measuring.
- **`coatedconductor` keeps the copper bug and would render two different metals per mode** →
  `_conductor_basecolor` (`materials.py:190`) reads only `eta`, while
  `material_spectral_overrides` (`materials.py:225`) reads `eta or conductor.eta`. So
  `"spectrum conductor.eta" "metal-CuZn-eta"` gives an RGB copper base with a `cuzn` spectral
  override. Widening the metal set does not fix it — this is the exact silent-copper failure
  the change exists to kill, surviving on one material type. Read `conductor.eta` in
  `_conductor_basecolor` too (hits both the `-mtlx` and plain paths via `materials.py:422`
  and `:578`).
- **Hand-written metal RGB constants fight the vendor-verbatim premise** → `NAMED_METAL_IOR`
  is hand-approximated (`data/__init__.py:6-9`). Derive the three new entries by sampling
  `named_metal_spectrum(key)` at the sRGB primaries through `fresnel_conductor_rgb` rather
  than typing new numbers. Do **not** retrofit Au/Ag/Al/Cu — that would move existing RGB
  baselines for no reason here.
- **2-term Cauchy carries ≤7.5e-3 IOR error on dense flints (D1)** → Recorded per-glass as
  a hostless test tolerance and documented in `docs/Spectral.md`; upgrade path (tabulated
  GPU curve) is written down, not built.
- **Widening `_CONDUCTOR_CANON` changes materials that currently fall back to copper** →
  That fallback is a bug, so brass/MgO/TiO2 scenes will legitimately change appearance.
  This is the intended fix; call it out in `CHANGELOG.md` rather than hiding it.
- **Named illuminants change light chromaticity where they were previously white** →
  Same shape of intended fix; no RGB-only scene is touched because the name path only
  fires on a `spectrum`-typed param.
- **Regenerating `spectral_curves.npz` requires the pbrt-v4 checkout** → The tool is a dev
  tool run once and the `.npz` is checked in; the hostless validation test pins the values
  so a bad regeneration is caught without pbrt present.

## Open Questions

None blocking. The one sequencing item (D2: land after `spectral-dispersion-showcase-asset`
to avoid racing the prism baseline) is a scheduling constraint, not an unknown.
