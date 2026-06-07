## ADDED Requirements

### Requirement: CUDA real-cost re-run of the BRDF size×precision grid
The study SHALL re-run the rejected BRDF-only neural sampler's size×precision grid on
real NVIDIA CUDA hardware, measuring inference cost with on-device GPU timing rather than
the indicative Apple-MPS wall-clock of the prior offline study, across the full set of
network sizes already covered by that study crossed with the extended precision axis.

#### Scenario: The grid runs on CUDA and emits a real-cost table
- **WHEN** the grid driver is run on the CUDA device
- **THEN** for each evaluated cell it produces the quality measurements and a GPU-timed
  inference cost, and a record of exactly which cells were run

#### Scenario: A non-CUDA or misconfigured device is refused, not silently downgraded
- **WHEN** CUDA is unavailable or the PyTorch build cannot use the GPU
- **THEN** the study fails with a clear message rather than silently measuring CPU cost as
  if it were GPU cost

### Requirement: Extended floating-point precision axis with a fixed-precision spline core
The precision axis SHALL include the NVIDIA TensorCore floating-point formats supported by
the target GPU — 16-bit fp16 and bf16, the tf32 math mode, and 8-bit fp8 in both e4m3 and
e5m2 — in addition to full precision, while the rational-quadratic spline evaluation and
the returned solid-angle pdf SHALL remain at full precision in every mode, preserving the
weight-storage / GEMM-accumulate boundary established for the renderer's neural flow.

#### Scenario: Each precision mode is selectable and runs
- **WHEN** a precision mode in the extended axis is selected
- **THEN** the linear-layer GEMM evaluates in that precision while the spline core and the
  pdf stay full precision, and the sampler produces a valid sample and a finite positive pdf

#### Scenario: Full precision is unchanged
- **WHEN** the full-precision mode is selected
- **THEN** inference behaves identically to the pre-change full-precision path, with the
  tf32 math mode disabled

#### Scenario: 8-bit precision uses scaled fp8 matmul with graceful skip
- **WHEN** an fp8 mode (e4m3 or e5m2) is selected
- **THEN** the GEMM runs through the scaled fp8 matmul path with dynamic per-tensor scaling
  on a supporting device, and on an unsupporting device the cell is skipped and logged
  rather than failing the whole run

### Requirement: Honest GPU cost reporting
The study SHALL report inference cost as on-device GPU-event time measured with warmup and
repeated timed iterations at a realistic inference batch, together with the speedup of each
reduced-precision mode relative to full precision, and SHALL indicate whether a cell is
compute-bound or launch/bandwidth-bound rather than asserting a TensorCore speedup that was
not measured.

#### Scenario: Cost is reported with the precision speedup ratio
- **WHEN** a cell's inference is timed
- **THEN** the result includes the GPU-event milliseconds and the full-precision→reduced-
  precision speedup ratio for that cell

#### Scenario: A precision that does not speed up the tiny network is reported as such
- **WHEN** a reduced-precision mode shows no real speedup because the network is
  launch/bandwidth-bound
- **THEN** the study reports the measured ratio and labels the cell accordingly rather than
  claiming a TensorCore win

### Requirement: Per-precision unbiasedness and firefly reporting
Every grid cell SHALL be verified unbiased against the independent hemisphere-quadrature
reference and SHALL have its pdf normalization checked, and the study SHALL report the
per-roughness firefly tail and the per-precision pdf-parity drift as results — including
when a reduced-precision mode worsens the low-roughness specular tail — rather than hiding
the drift or reducing the failure axis to a whole-scene median.

#### Scenario: Each cell is checked unbiased and normalized
- **WHEN** a cell is evaluated
- **THEN** it is confirmed unbiased against the quadrature reference within tolerance and
  its flow pdf normalizes within the expected bound

#### Scenario: Reduced-precision drift on the sharp lobe is reported, not hidden
- **WHEN** a reduced-precision mode increases the low-roughness firefly tail or the
  pdf-parity drift relative to full precision
- **THEN** the measured per-roughness firefly and parity drift are reported as study
  results

### Requirement: Real-hardware verdict on the equal-time outcome
The study SHALL state, from the real-hardware measurements, whether any (size, precision)
cell reaches both a bounded low-roughness firefly and equal-time efficiency at least that of
the analytic MIS baseline at low roughness, and SHALL cross-reference the prior Apple-MPS
study so the precision leg of the rejection rests on measured GPU cost rather than an
indicative one.

#### Scenario: The verdict is grounded in measured GPU cost
- **WHEN** the results document is produced
- **THEN** it states whether the rejection is re-sealed on real hardware or a precision cell
  survived the low-roughness firefly and equal-time bar, and it links the corresponding
  Apple-MPS study for comparison
