## ADDED Requirements

### Requirement: NIS method in the equal-time/equal-variance sweep
The guiding variance harness SHALL include `nis` as a selectable method, measured
under the same metrics as `neural`: equal-time variance, equal-variance time, and
`1/(var·t)` efficiency, with both operating-point renders (equal-time and
equal-variance) retained. The renderer-scene comparison SHALL report
{baseline, flow, nis, restir-di, flow+restir} on the same scenes at chart `V1`.

#### Scenario: NIS sweeps with the same metrics
- **WHEN** the variance sweep is run with `nis` in the proposal set
- **THEN** the harness emits equal-time variance, equal-variance time, and
  efficiency for the NIS cells against the same reference as the flow cells, and
  retains both operating-point renders

### Requirement: Single comparison-scene catalog
The harness SHALL resolve every comparison scene from one catalog (a single
source of truth) that records, per scene: stable name, source/provenance
(Bitterli Rendering Resources / ORCA / in-repo), license, static-or-animated,
resolution, and the transport regime it exercises (direct, indirect multi-bounce,
glossy, animated). Sweep cells SHALL reference scenes by catalog key, not by ad
hoc paths, so the paper's benchmark-scene section and the data-availability
statement transcribe from one place.

#### Scenario: Scenes referenced by catalog key
- **WHEN** a sweep cell names a comparison scene
- **THEN** that scene resolves to a catalog entry carrying its source, license,
  static/animated flag, and the regime it tests

#### Scenario: Catalog covers every referenced scene
- **WHEN** the comparison-scene catalog is read
- **THEN** it lists every scene used by any reported comparison (Cornell,
  three-materials, Veach Ajar static, NVIDIA Emerald Square animated, and any
  ReSTIR scene), with no referenced scene missing

### Requirement: NIS-style image-error convergence analysis
The renderer-scene comparison SHALL report the analysis axes of NIS's evaluation:
image-error metrics (relative-L2 and MAPE vs a high-spp reference) per scene,
convergence curves of error vs sample count and error vs wall-clock time per
method, an equal-sample-count slice alongside the equal-time and equal-variance
results, and a training-cost amortization statement separating one-time training
cost from per-frame inference cost.

#### Scenario: Image-error convergence reported
- **WHEN** a renderer-scene comparison result is read
- **THEN** it includes relative-L2 / MAPE vs a high-spp reference and convergence
  curves over both sample count and wall-clock time for each method

#### Scenario: Amortization crossover stated
- **WHEN** the cost analysis for a scene is read
- **THEN** the one-time training cost and per-frame inference cost are reported
  separately, with the frame budget at which the flow amortizes its inference
  overhead
