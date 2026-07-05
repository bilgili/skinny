## ADDED Requirements

### Requirement: Wavefront BDPT and SPPM shade terminal non-flat first hits via the path tracer

The wavefront BDPT and SPPM integrators SHALL shade a camera (eye) first hit on a
**terminal** non-flat material — `MATERIAL_TYPE_SUBSURFACE` and
`MATERIAL_TYPE_SKIN` — by falling back to the path integrator for that lane,
producing the same radiance the megakernel produces for that pixel. They SHALL NOT
leave such a lane's radiance at zero (rendering the object black). This mirrors the
megakernel, where `main_pass` gates BDPT on `MATERIAL_TYPE_FLAT` and routes every
non-flat first hit to `PathTracer.estimateRadiance`; the BDPT integrator's own
non-flat "bail to black" branch is unreachable there.

"Terminal" means `evaluateBounce` returns a complete radiance and sets no BSDF
continuation, so `PathTracer.estimateRadiance` evaluates exactly one vertex and
stops — the same bounded dispatch shape as the wavefront path catch-all shade
(`wfPathShade`), keeping the change within the Metal dispatch-hygiene watchdog
envelope. The non-terminal non-flat types — `MATERIAL_TYPE_VOLUME` and
`MATERIAL_TYPE_PYTHON` — continue bouncing inside `estimateRadiance`; running that
multi-bounce loop in the single once-per-frame wavefront command buffer (no
band-tiling, unlike the megakernel) would be an unbounded command buffer, so those
first hits SHALL retain the black bail under wavefront BDPT/SPPM (unchanged from
before; still rendered by the megakernel path). Lifting that is a follow-up gated
on a bounded dispatch for the fallback.

For wavefront BDPT the fallback lane SHALL build no eye subpath and no light
subpath of its own (matching a megakernel non-flat pixel, which runs no BDPT), and
SHALL still receive s=1 light-tracer splats contributed by other lanes. For
wavefront SPPM the fallback lane SHALL store no photon visible point and SHALL add
the path-traced radiance (weighted by the accumulated specular-chain throughput)
to the pixel. Flat-material lanes SHALL be unchanged, so scenes with no non-flat
material are byte-identical.

#### Scenario: subsurface dragon renders under wavefront BDPT

- **WHEN** the subsurface `sss_dragon` scene is rendered with
  `--execution-mode wavefront --integrator bdpt`
- **THEN** the dragon shades as a translucent solid (not a black silhouette),
  within tolerance of the megakernel-BDPT and path anchors for the same scene

#### Scenario: subsurface dragon renders under wavefront SPPM

- **WHEN** the subsurface `sss_dragon` scene is rendered with `--integrator sppm`
  (wavefront)
- **THEN** the dragon shades (not black), the subsurface pixels carrying
  path-traced radiance rather than an empty photon estimate

#### Scenario: flat-only scenes are unchanged

- **WHEN** a scene containing only `MATERIAL_TYPE_FLAT` materials is rendered under
  wavefront BDPT or SPPM
- **THEN** every lane takes the flat path exactly as before this change, and the
  output is byte-identical
