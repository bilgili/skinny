# Tasks — spectral-dispersion-showcase-asset

## 1. SF11 Cauchy entry

- [ ] 1.1 Compute the SF11 Cauchy fit `(A, B)` by least-squares over the Schott
      Sellmeier curve sampled 400–700 nm (expect A ≈ 1.737, B ≈ 0.0166 µm²);
      record the fit + provenance in the `_GLASS_CAUCHY` docstring
- [ ] 1.2 Add `"sf11"` to `_GLASS_CAUCHY` in
      `src/skinny/pbrt/data/spectral_tables.py`; confirm `named_glass_cauchy`
      and `named_glass_ior` resolve it and unknown names still hit `"default"`
- [ ] 1.3 Sanity-check the numpy mirror: `cauchy_ior(A, B, λ)` monotone
      decreasing over 400→700 nm for the new fit

## 2. Demo asset

- [ ] 2.1 Author `assets/dispersion_prism.usda`: triangular SF11 prism (60°
      apex near minimum deviation; drop to ~45° apex if blue-end TIR bites),
      slit = narrow white **emissive mesh quad** (NOT a UsdLux light — camera
      rays and BDPT light tracing both need emissive geometry), matte white
      screen 2–3 prism-heights past the exit face, near-black room, camera
      seeing both the through-prism slit image and the screen fan;
      `skinnyOverrides.glass_dispersion = "sf11"` on the prism material
      (pattern: `assets/glass_caustics_spectral.usda`); authored `ior 1.78`
      (SF11 d-line) with a comment that spectral overwrites it with Cauchy A
      and 1.78 keeps the RGB baseline's mean bending matched; no external file
      references
- [ ] 2.2 Headless Metal render, `--spectral --integrator bdpt`: confirm a
      clearly separated rainbow fan on the screen AND that the blue band exits
      the prism (no exit-face TIR dropout); tune slit width / screen throw /
      camera until bands are distinct; show the render
- [ ] 2.3 A/B at shared tonemap: spectral vs RGB same scene (bdpt) — the RGB
      render keeps the authored ior 1.78 so the A/B isolates dispersion — plus
      a spectral path-integrator render (through-prism cue must be clean);
      show the labelled side-by-side grid

## 3. Tests

- [ ] 3.1 Hostless test module: `named_glass_cauchy("sf11")[1] >
      named_glass_cauchy("bk7")[1]`; monotone `cauchy_ior`; fallback for
      unknown name unchanged
- [ ] 3.2 Hostless asset-integrity test: `assets/dispersion_prism.usda` loads
      via the USD loader, prism material override is `"sf11"`, no external
      asset paths referenced

## 4. Docs + validation

- [ ] 4.1 `docs/Spectral.md` + `README.md`: demo asset section with the
      one-liner render command; note SPPM accepts `--spectral` but renders
      without a rainbow (v1 has no SPPM dispersion — not refused)
- [ ] 4.2 `CHANGELOG.md` entry
- [ ] 4.3 `ruff check src/`, full hostless pytest, `openspec validate
      spectral-dispersion-showcase-asset`
