## MODIFIED Requirements

### Requirement: Wavefront BDPT and SPPM shade terminal non-flat first hits via the path tracer

The wavefront BDPT and SPPM integrators SHALL shade a camera (eye) first hit on
**any** non-flat material — `MATERIAL_TYPE_SUBSURFACE`, `MATERIAL_TYPE_SKIN`,
`MATERIAL_TYPE_VOLUME`, `MATERIAL_TYPE_PYTHON` — by falling back to the path
integrator for that lane, producing the same radiance the megakernel produces for
that pixel. They SHALL NOT leave such a lane's radiance at zero (rendering the
object black). This mirrors the megakernel, where `main_pass` gates BDPT on
`MATERIAL_TYPE_FLAT` and routes every non-flat first hit to
`PathTracer.estimateRadiance`.

The fallback's `PathTracer.estimateRadiance` is a full multi-bounce path. On the
Metal backend the renderer SHALL bound it so no single committed command buffer
runs the multi-bounce fallback over more than one watchdog-safe band: when the
scene contains a non-terminal non-flat material (`MATERIAL_TYPE_VOLUME` or
`MATERIAL_TYPE_PYTHON`), (a) the wavefront BDPT/SPPM eye stage SHALL submit and
drain per eye tile (the row-band discipline of
`metal-megakernel-watchdog-tiling`), **and** (b) the eye `stream_size` (tile lane
count) SHALL be capped to the megakernel BDPT band budget so the tile itself — not
only tile accumulation — stays within the macOS GPU watchdog (a full-frame
`1<<20`-lane SPPM eye tile of multi-bounce volume walks would otherwise trip the
watchdog even with per-tile submit). Scenes with no non-terminal non-flat material
SHALL keep the single-submit path and full `stream_size` unchanged, and the Vulkan
backend (no watchdog) SHALL be behaviourally and byte-for-byte unchanged.

For wavefront BDPT the fallback lane SHALL build no eye subpath and no light
subpath of its own (matching a megakernel non-flat pixel, which runs no BDPT), and
SHALL still receive s=1 light-tracer splats contributed by other lanes. For
wavefront SPPM the fallback lane SHALL store no photon visible point and SHALL add
the path-traced radiance (weighted by the accumulated specular-chain throughput)
to the pixel. Flat-material lanes SHALL be unchanged, so scenes with no non-flat
material are byte-identical.

A `MATERIAL_TYPE_VOLUME` first hit under BDPT/SPPM shades via this path fallback,
but the bidirectional connection and photon strategies remain volume-blind (the
recorded medium-transport exclusion is unchanged).

#### Scenario: python-material object renders under wavefront BDPT and SPPM

- **WHEN** a scene with a `MATERIAL_TYPE_PYTHON` material
  (`cornell_box_python_material.usda`) is rendered with `--execution-mode
  wavefront` and `--integrator bdpt` or `--integrator sppm`
- **THEN** the python-material object shades (not black), within tolerance of the
  wavefront path anchor for the same scene

#### Scenario: volume first hit renders under wavefront BDPT/SPPM

- **WHEN** a scene with a `MATERIAL_TYPE_VOLUME` first hit is rendered under
  wavefront BDPT or SPPM
- **THEN** the eye-visible volume pixels shade via the path fallback (not black),
  and the render completes without wedging the GPU

#### Scenario: heavy-fallback frames stay within the GPU watchdog

- **WHEN** a non-terminal non-flat scene is rendered under wavefront BDPT/SPPM on
  Metal and the process is killed mid-render
- **THEN** the GPU remains usable afterwards (kill harness), because each committed
  command buffer bounded the multi-bounce fallback to one eye tile

#### Scenario: flat-only scenes are unchanged

- **WHEN** a scene containing only `MATERIAL_TYPE_FLAT` materials is rendered under
  wavefront BDPT or SPPM
- **THEN** every lane takes the flat path exactly as before this change, no
  per-tile flush is inserted, and the output is byte-identical
