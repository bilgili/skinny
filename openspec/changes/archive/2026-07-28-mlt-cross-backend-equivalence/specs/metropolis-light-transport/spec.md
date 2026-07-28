# Metropolis light transport — cross-backend equivalence delta

## MODIFIED Requirements

### Requirement: MLT runs on both backends under dispatch hygiene

MLT SHALL run on native Metal and Vulkan at parity. Chain-mutation dispatches
SHALL respect the `metal-dispatch-hygiene` capability: under `SKINNY_METAL` the
per-frame chain work SHALL be committed as bounded sub-batches (breadth-tiled
like the SPPM photon phase) so no single command buffer can exceed the watchdog
budget, with the tiling bit-identical to an untiled dispatch. Any change to
dispatch shape SHALL pass the GPU kill harness (`tests/test_metal_cleanup.py`).

"At parity" for MLT means the equivalence class below, and nothing stronger.
Each backend SHALL be bit-reproducible with itself: two processes on the same
backend, at the same budget and the same scene, SHALL produce a pixel-identical
image. The parity gate depends on this property.

Vulkan and native Metal SHALL NOT be assumed bit-identical to each other. Slang
compiles one source through two back ends, SPIR-V and MSL, which contract
multiply-add pairs and implement transcendental functions differently. The
resulting last-bit differences reach the image through the Q24.8 truncation in
`mltFilmSplat`, so the cross-backend difference SHALL be an integer count of
film-splat quanta `q = b / (256 × mpp_actual × frames)` and SHALL stay orders of
magnitude below the scene's own parity tolerance. The difference SHALL NOT be
Markov-noise-sized; a difference at the scale of the recorded self-consistency
tolerance means the chains diverged, which is a defect.

Documentation SHALL NOT state that MLT is bit-identical across backends. A
cross-backend measurement SHALL record the budget it was taken at, because the
quantum `q` scales with the budget and an identity measured at one budget does
not hold at another.

#### Scenario: Metal chain dispatch is watchdog-bounded
- **WHEN** a frame's mutation budget exceeds the per-dispatch cap under
  `SKINNY_METAL`
- **THEN** the work is split across multiple committed command buffers and the
  accumulated result is identical to a single dispatch

#### Scenario: One backend repeats its own image exactly
- **WHEN** the same scene, budget, and backend render in two separate processes
- **THEN** the two linear-HDR images are pixel-identical

#### Scenario: Metal and Vulkan agree within the splat quantum
- **WHEN** the same scene and budget render under MLT on native Metal and on
  Vulkan
- **THEN** the two images differ by an integer number of film-splat quanta, and
  the difference stays orders of magnitude below the scene's recorded parity
  tolerance — measured on `int_caustic` at 128×128, 512 spp, 16384 chains:
  relMSE 5.302e-10 (RGB) and 5.150e-08 (spectral), against the scene's 0.12
  relMSE gate
