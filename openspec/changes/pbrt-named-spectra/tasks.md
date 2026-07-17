## 1. Vendor pbrt's named-spectrum data

- [x] 1.1 Extend `_extract_pbrt_spectra.py` `_METALS` to `(Ag, Al, Au, Cu, CuZn, MgO, TiO2)`; confirm each `metal-<X>-eta`/`-k` array extracts onto the 360–830 nm grid
- [x] 1.2 Add glass extraction to the tool with an explicit pbrt-name → C++-symbol map (D5: `glass-F5`→`GlassSF5_eta`, `glass-F10`→`GlassSF10_eta`, `glass-F11`→`GlassSF11_eta`; BK7/BAF10/FK51A/LASF9 map directly)
- [x] 1.3 In the tool, least-squares fit `n(λ)=A+B/λ_µm²` per glass over 360–830 nm and record `(A, B)`, the d-line (589.3 nm) IOR, and the max fit residual (D1)
- [x] 1.4 Add named-illuminant SPD extraction — the 15 `stdillum-*` (A, D50, D65, F1–F12) **plus `illum-acesD60`**, which is scene-addressable in pbrt's `namedSpectra` map (`spectrum.cpp:2701`), unlike the `canon_*`/`ilford_*` sensor curves the design excludes (m3). Keep the existing `d65` array name/values intact, and make `named_illuminant_spectrum("stdillum-D65")` alias it explicitly rather than adding a second D65 (m6)
- [x] 1.4b Record pbrt's per-spectrum `normalize` flag handling: it is `true` for illuminants and `false` for metals/glasses (`spectrum.cpp:2600-2630` vs `2632-2659`), while `_resample_interleaved` ignores it. Harmless end-to-end (`_spectral_light_spd_scaled` rescales any SPD to the RGB luminance, so absolute scale cancels) — document it in the tool so nobody "fixes" it later by adding a second normalisation (m6)
- [x] 1.5 Regenerate `spectral_curves.npz` against `~/projects/pbrt-v4`; verify existing arrays are unchanged (additive `.npz` growth only)

## 2. Table lookup layer (`spectral_tables.py`)

- [x] 2.1 Replace the 2-entry `_GLASS_CAUCHY` with the per-glass fit table (7 glasses + a generic-crown `default`); `default` is no longer an alias for BK7 (D3)
- [x] 2.2 Add a per-glass d-line scalar IOR lookup
- [x] 2.3 Make `named_glass_cauchy` / `named_glass_ior` resolve each recognised glass to its own coefficients, and report unrecognised names to the caller rather than silently returning `default` (D3)
- [x] 2.4 Add `named_illuminant_spectrum(name)` → vendored 95-sample SPD, or `None` for an unrecognised name
- [x] 2.5 Confirm `named_metal_spectrum` resolves the three new metals through the existing normalisation/alias path

## 3. Import resolution (`spectra.py`, `materials.py`, `lights.py`)

- [x] 3.1 Widen `_CONDUCTOR_CANON` / `NAMED_METAL_IOR` to the seven metals; `named_conductor_key` returns the new keys. **Derive** the three new RGB entries by sampling `named_metal_spectrum(key)` at the sRGB primaries (630/532/465 nm) through `fresnel_conductor_rgb` rather than hand-typing constants — the data is already in the `.npz` and `NAMED_METAL_IOR` is hand-approximated (`data/__init__.py:6-9`). Do **not** retrofit au/ag/al/cu; that would move existing RGB baselines for no reason here (m4)
- [x] 3.2 Rework `named_glass_key` to return the per-glass key and to distinguish "recognised" from "fell back", so the caller can note it (D3)
- [x] 3.3 Add named-illuminant reduction to `param_to_rgb`: SPD → XYZ → linear sRGB, unit-luminance normalised (mirrors `blackbody_rgb`)
- [x] 3.4 ~~Preserve an inline non-constant material `spectrum`~~ — **CUT**: nothing consumes a material SPD, so the override would be dead data. Recorded as a non-goal + follow-up; a test pins that no override is authored
- [x] 3.5 Give `get_float_texture`'s named-spectrum branch (`materials.py:102-105`) the per-glass d-line IOR instead of the generic default. This single fix covers **both** authoring paths — `-mtlx` `specular_IOR` (`materials.py:398`) and plain `ior` (`materials.py:563`) both route through it; `mtlx_emit.py` authors no IOR of its own (n2), so no second task
- [x] 3.6 Emit APPROX report notes naming the unrecognised spectrum and its substituted fallback for unknown glass / metal / illuminant names; distinguish a **path-like** string ("spectrum file not read", pbrt's file fallback) from an unknown name (m2)
- [x] 3.7 Verify both `api.py` override sites (plain-USD ~line 364 and `-mtlx` ~line 416) carry the widened overrides identically
- [x] 3.8 **M3**: add the missing `xyz = xyz / _Y_INTEGRAL` to the illuminant branch of `sampled_spectrum_to_rgb` (`spectra.py:120-122`) so colored illuminants are continuous with the constant shortcut and match pbrt (D8). Verified zero blast radius: no checked-in scene authors a non-constant inline `spectrum L`/`I`
- [x] 3.9 **M4**: make `_conductor_basecolor` (`materials.py:190`) read `params.get("eta") or params.get("conductor.eta")` (same for `k`), so a `coatedconductor` stops falling back to copper in RGB while its spectral override says another metal
- [x] 3.10 Gate the named-illuminant lookup in `param_to_rgb` on `illuminant=True` (n1), so `"spectrum sigma_a" "stdillum-A"` can't reduce a medium coefficient to an illuminant chromaticity via `media.py:73-74`
- [x] 3.11 Route the subsurface `eta` read (`materials.py:291`, `p.floats("eta", [ETA_DEFAULT])`) through a named-spectrum guard — it is the one `eta` reader not using `get_float_texture` and currently raises `ValueError` on a named eta (n4)

## 4. Loader and renderer host-side (`usd_loader.py`, `renderer.py`)

- [x] 4.1 Teach `_extract_light_spd` to accept a `spectrum_named` payload by resolving it via `named_illuminant_spectrum`, keeping the existing `spectrum_samples` path and the 95-sample length guard unchanged
- [x] 4.2 Extend `_CONDUCTOR_METAL_ID` (`renderer.py:570`) with `cuzn`/`mgo`/`tio2` → 5/6/7 — **append only**, never renumber au/ag/al/cu (D7)
- [x] 4.3 Extend the `spectralMetals` upload loop (`renderer.py:4004`) with the same three metals in the **same order**; the id is the upload index + 1 (`namedMetalEtaK` does `(metalId-1)*stride`)
- [x] 4.4 Correct the now-stale "SPECTRAL-ONLY … RGB build renders byte-identical" comment at `renderer.py:688-690` — the d-line IOR now moves the RGB build for named glasses (D2)
- [x] 4.5 ~~Confirm no `.slang` edit is needed~~ — **it was needed** (codex): `spectral_flat_common.slang:53` hard-gated `metalId <= 4u`, so CuZn/MgO/TiO2 would upload correctly then render with RGB Schlick. Design non-goal corrected
- [x] 4.6 Add `SPECTRAL_METAL_COUNT` to `bindings.slang`, gate `spectral_flat_common.slang` on it, and pin it to `len(_SPECTRAL_METAL_ORDER)` with a hostless test
- [x] 4.7 Verify the RGB SPIR-V is byte-identical to main (shader edit is spectral-only)
- [x] 4.8 Restrict the named-glass scalar to IOR-bearing params (`eta`, `interface.eta`) — codex: `"spectrum roughness" "glass-BK7"` silently returned 1.51673 into a roughness lane

## 5. Hostless tests

- [x] 5.1 Table validation: each glass's `(A, B)` reproduces pbrt's tabulated eta within the recorded per-glass residual (≤ 8e-3); LASF9 ≠ BK7; d-line IORs match pbrt
- [x] 5.2 Extended-metal curves and RGB reflectances resolve for CuZn/MgO/TiO2; the four pre-existing metals are value-unchanged
- [x] 5.3 `stdillum-A` reduces to a warm chromaticity (red > blue) at unit luminance; `stdillum-D65` reduces ≈ neutral — measured `[0.99940, 1.00018, 0.99995]`, so assert at ~1e-3
- [x] 5.4 Import: `"spectrum eta" "glass-LASF9"` → scalar IOR ≈ 1.850 and a LASF9 `glass_dispersion` key; `glass-BK7`'s key is unchanged
- [x] 5.5 Import: a `stdillum-A` **distant** light round-trips to a 95-sample SPD through `_extract_light_spd`
- [x] 5.5b Import: a `stdillum-A` **area** light gets the warm chromaticity and no SPD, raising no payload-shape error (D4 envelope — asserts the limit, so a later widening has to update this test deliberately)
- [x] 5.6 Import: an inline material `spectrum` authors NO dead override; `-mtlx` and plain-USD paths record the same identity
- [x] 5.7 Unknown-name notes: unrecognised glass/metal/illuminant each produce an APPROX note naming the fallback; a recognised-only scene produces none
- [x] 5.8 Regression: an RGB-only scene imports byte-identically (no new overrides authored)
- [x] 5.9 Alignment: `_CONDUCTOR_METAL_ID` agrees with the `spectralMetals` upload order (id == index+1) and au/ag/al/cu still map to 1–4 (D7)
- [x] 5.10 **M3/D8**: an illuminant `[400 10 700 10.000001]` lands within ~5% of the constant `[400 10 700 10]` → `[10,10,10]` (continuity), and colored **reflectance** results stay bit-identical to pre-change
- [x] 5.11 **M4**: a `coatedconductor` with `"spectrum conductor.eta" "metal-CuZn-eta"` gets a CuZn RGB base colour (not copper) and a matching `conductor_metal` override — the two modes agree on the metal
- [x] 5.12 A spectrum **file** reference produces the "file not read" note, not an unknown-glass substitution note (m2)

## 6. Regenerate the committed assets (D9 — do this BEFORE any measurement)

*The renderer loads the `.usda`/`.mtlx`, not the `.pbrt`. These are git-tracked importer
output hard-coding `ior = 1.5`; measuring before regenerating them would report "no change"
and ship green over a fix that never reached a pixel.*

- [x] 6.1 Re-run `tests/assets/suite/_gen/build.py` for `spec_prism`; re-import `assets/prism_dispersion/prism_dispersion.pbrt` and `assets/glass_machines.usda` (untracked, 7 `glass-BK7` dielectrics, no `_gen` — re-import by hand)
- [x] 6.2 **Gate**: diff the regenerated `spec_prism.usda:161`, `spec_prism_mtlx.mtlx:27`, `prism_dispersion.usda:123` and confirm `1.5 → 1.51673` actually appears. If the diff is empty the importer fix did not land — stop, do not proceed to measurement

## 7. Baselines and GPU verification

*Sequence this group after `spectral-dispersion-showcase-asset` lands (D2 coordination).*

- [x] 7.1 Re-measure `spec_prism` + `prism_dispersion` + the spectral suite pbrt-truth metrics under the BK7 refit (Metal backend, headless env per CLAUDE.md); metrics must hold or improve — if one worsens, fix the fit, do not raise the baseline
- [x] 7.2 Re-measure the **RGB-mode** baselines for named-glass scenes — the d-line IOR moves RGB more than the refit moves spectral (BK7 1.5→1.51673, LASF9 1.5→1.85); same hold-or-improve rule (D2)
- [x] 7.3 Confirm no checked-in scene uses a CuZn/MgO/TiO2 conductor (grep says none, so this is expected to be a no-op — record that rather than leaving an unbounded corpus sweep). If one exists, re-measure it: the copper fallback was a bug, so it legitimately changes
- [x] 7.4 Re-verify megakernel ≡ wavefront self-consistency on all affected scenes
- [x] 7.5 Update the manifest `measured` values to the new measurements

## 8. Docs and gates

- [x] 8.1 `docs/Spectral.md`: named-spectrum coverage table (7 glasses / 7 metals / 16 illuminants), the 2-term Cauchy residual limit + tabulated-GPU-curve upgrade path, and the D4 envelope (SPD bound on distant lights only; other light types get chromaticity and upsample from RGB)
- [x] 8.2 `CHANGELOG.md`: note that brass/MgO/TiO2 conductors, `coatedconductor` (M4), and named-illuminant lights legitimately change appearance (they previously fell back to copper / neutral white), plus the illuminant-projection fix (D8)
- [x] 8.3 `.venv/bin/ruff check src/` and `.venv/bin/pytest` (hostless) green
- [x] 8.4 `openspec validate pbrt-named-spectra --strict`
- [x] 8.5 Codex pre-merge review (`codex:rescue`); fold findings back in, or fall back to a review subagent and say so if codex is unavailable
