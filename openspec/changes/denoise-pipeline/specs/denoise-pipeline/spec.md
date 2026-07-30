## ADDED Requirements

### Requirement: One backend-neutral denoiser interface

The renderer SHALL reach every denoiser through one `Denoiser` protocol declared
in `denoise.py`. The protocol SHALL declare a `name`, a `required_aovs` set, and
the operations `resize`, `reset`, `denoise`, and `destroy`. The renderer SHALL
hold at most one denoiser at a time, and SHALL NOT name a vendor, a graphics API,
or a specific denoiser anywhere outside `denoise.py` and the implementation
modules it registers.

A denoiser implementation SHALL own its device interop, its history state, and
its command submission. The renderer SHALL supply only the linear colour image,
the auxiliary images the implementation declared, and a per-frame record holding
the jitter offset, the exposure, the camera matrices, and the frame index.

#### Scenario: The renderer holds no vendor name

- **WHEN** the source tree outside `denoise.py` and the registered
  implementation modules is searched for the identifiers `MetalFX`, `MTLFX`,
  `NRD`, `OptiX`, or `OIDN`
- **THEN** no match is found

#### Scenario: A second implementation needs no renderer change

- **WHEN** a new denoiser class is added to the registry and declares auxiliary
  images already in the registry
- **THEN** it is selectable by name with no edit to `renderer.py`

#### Scenario: The protocol is hostless-importable

- **WHEN** `skinny.denoise` is imported in a subprocess with `vulkan` and
  `slangpy` blocked at the meta path
- **THEN** the import succeeds and no GPU package appears in `sys.modules`

### Requirement: Auxiliary images are a named registry and a denoiser declares what it needs

`denoise.py` SHALL declare every auxiliary image once, each with a name, a pixel
format, and its contents. A denoiser SHALL expose `required_aovs` as a subset of
those names. The renderer SHALL allocate, and the auxiliary pass SHALL write,
exactly the images the active denoiser declared — no more and no fewer.

A `required_aovs` entry that names no registered auxiliary image SHALL be
refused at denoiser construction, never ignored.

The registry SHALL contain at least `diffuse_albedo`, `specular_albedo`,
`normal_depth`, and `motion`.

#### Scenario: Undeclared auxiliary images are not allocated

- **WHEN** the active denoiser declares `{"normal_depth", "motion"}`
- **THEN** the resource set holds no `diffuse_albedo` and no `specular_albedo`
  image, and the auxiliary pass writes neither

#### Scenario: No denoiser means no auxiliary images

- **WHEN** the renderer runs with `--denoiser none`
- **THEN** no auxiliary image is allocated and the auxiliary pass never
  dispatches

#### Scenario: An unknown auxiliary-image name is refused

- **WHEN** a denoiser class declares an auxiliary-image name that the registry
  does not hold
- **THEN** construction raises, naming the unknown auxiliary image

### Requirement: The denoiser never writes the accumulation image

The denoiser SHALL read the linear accumulation image and SHALL write a separate
denoised image. It SHALL NOT write, clear, or otherwise modify the accumulation
image. A denoised value SHALL NOT re-enter progressive accumulation.

The parity harness SHALL keep reading the raw accumulation image, so the
pbrt-truth gate and the self-consistency gate SHALL be unaffected by any
denoiser.

#### Scenario: Accumulation is unchanged by denoising

- **WHEN** the same scene is rendered to the same sample count twice — once with
  `--denoiser none` and once with a denoiser active and `--denoise-scale 1.0` —
  and the raw accumulation image is read from both
- **THEN** the two accumulation images agree to the renderer's own reproducibility
  tolerance, and the denoised image is a separate image

#### Scenario: The parity matrix has no denoiser axis

- **WHEN** the parity matrix is enumerated
- **THEN** no combination sets a denoiser, and the recorded baselines and
  self-consistency tolerances are unchanged

### Requirement: Auxiliary images come from one dedicated primary-ray pass

The auxiliary images SHALL be produced by one dedicated compute pass that traces
one primary ray per pixel. No integrator kernel and no wavefront stage SHALL be
edited to produce them.

The pass SHALL run at render resolution, once per frame, while a denoiser is
active. It SHALL NOT dispatch while no denoiser is active.

The motion vector SHALL be produced by reprojecting the first-hit world position
with the previous frame's view-projection matrix.

#### Scenario: No existing kernel changes

- **WHEN** the compiled SPIR-V of the megakernel and of every wavefront kernel is
  compared before and after this change, with no denoiser active
- **THEN** every `.spv` file is byte-identical

#### Scenario: The auxiliary pass is integrator-independent

- **WHEN** the auxiliary pass runs under each of the `path`, `bdpt`, `sppm`, and
  `mlt` integrators
- **THEN** it produces the same auxiliary images for the same camera and scene

### Requirement: A standalone display pass owns display while a denoiser runs

While a denoiser is active, a standalone compute pass SHALL apply exposure,
tonemap, sRGB encoding, the HUD overlay, and the gizmo overlay over the denoised
image, at output resolution, and SHALL write the display image.

The pass SHALL reuse the single existing definition of those operators. The
display tails inside the megakernel and inside the wavefront resolve kernels
SHALL NOT be edited.

The megakernel's focus-plane overlay and its furnace over-energy tint need the
primary ray and the scene hit, which the display pass does not carry. Both SHALL
be inactive while a denoiser runs, and this SHALL be documented as a known limit.

#### Scenario: Display is denoised end to end

- **WHEN** a frame is rendered with a denoiser active and the display image is
  read
- **THEN** the image is the tonemapped denoised result, not the tonemapped
  accumulation

#### Scenario: One tonemap definition

- **WHEN** the shader sources are searched for a tonemap or sRGB-encoding
  implementation
- **THEN** exactly one definition of each operator exists, and the display pass
  calls it

### Requirement: Render resolution is separable from output resolution

The renderer SHALL carry a render extent and an output extent. One pure function
SHALL derive the render extent from the output extent and the denoise scale, and
SHALL be the only place that derivation happens.

With no denoiser active the denoise scale SHALL be 1.0 and the two extents SHALL
be equal.

Each size-dependent GPU resource SHALL declare which extent it sizes on. The
accumulation image, the auxiliary images, the light-splat buffer, the wavefront
record buffers, and the ReSTIR reservoirs SHALL size on the render extent. The
display image, the denoised image, the HUD overlay, and the swapchain SHALL size
on the output extent.

`FrameConstants` SHALL carry the output extent in addition to the render extent,
so the display pass can index output pixels.

#### Scenario: Scale 1.0 reproduces today's sizing

- **WHEN** the renderer runs with no denoiser
- **THEN** every size-dependent resource has the extent it had before this
  change, and the render extent equals the output extent

#### Scenario: A fractional scale shrinks only the render-extent resources

- **WHEN** the renderer runs at an output extent of 800×600 with a denoise scale
  of 0.5
- **THEN** the accumulation image and the auxiliary images are 400×300, and the
  display image, the HUD overlay, and the swapchain are 800×600

#### Scenario: Screen-space input maps between extents at one site

- **WHEN** a tool pick is performed at an output pixel while the denoise scale is
  below 1.0
- **THEN** the pick resolves to the correct render pixel, and the mapping is
  performed at exactly one call site

### Requirement: Reported jitter while a temporal denoiser runs

While a temporal denoiser is active, the primary ray SHALL take its sub-pixel
offset from a deterministic low-discrepancy sequence, and the renderer SHALL
report that same offset to the denoiser for the frame it was rendered with.

The selection SHALL be carried by a frame uniform, so the sampling with no
denoiser active is unchanged.

#### Scenario: The reported offset is the rendered offset

- **WHEN** a frame is rendered with a temporal denoiser active
- **THEN** the offset the renderer reports for that frame equals the offset the
  primary ray used

#### Scenario: No denoiser leaves sampling unchanged

- **WHEN** a scene is rendered with no denoiser, before and after this change
- **THEN** the accumulation image is bit-identical

### Requirement: Denoiser reset shares the accumulation-reset owner

The renderer SHALL call `reset()` on the active denoiser from the same place it
resets progressive accumulation. A camera move, a scene edit, a resolution
change, and a transport parameter change SHALL each reset the denoiser history.

Changing a post-process control — tonemap, exposure, the denoise toggle, or the
denoise strength — SHALL NOT reset accumulation and SHALL NOT reset the denoiser
history.

#### Scenario: A camera move drops stale history

- **WHEN** the camera moves
- **THEN** progressive accumulation resets and the denoiser history resets in the
  same step

#### Scenario: Toggling the denoiser keeps the accumulated samples

- **WHEN** the denoise toggle is switched while a scene is accumulating
- **THEN** the accumulation frame counter is unchanged and no sample is discarded

### Requirement: File output emits the denoised image

Every file-output path SHALL emit the denoised result while a denoiser is
active — `render_headless()`, `save_screenshot()`, the EXR writer, and the
Radiance writer. One accessor SHALL decide the source image, and every
file-output path SHALL read it.

A separate accessor SHALL return the raw accumulation image, SHALL never return
the denoised image, and SHALL be the one the parity harness reads.

#### Scenario: A headless render is denoised

- **WHEN** `skinny-render` runs with a denoiser active
- **THEN** the written image is the denoised result

#### Scenario: An EXR carries denoised linear values

- **WHEN** an EXR is written with a denoiser active
- **THEN** it holds the denoised linear values at output resolution

#### Scenario: The parity accessor is never denoised

- **WHEN** the parity harness reads the accumulation image with a denoiser active
- **THEN** it receives the raw accumulation values

### Requirement: Denoiser shader families refuse the axes they cannot carry

The auxiliary pass and the display pass SHALL each be a shader-variant family in
the shader-variant key. Neither SHALL carry the spectral, MLT, or neural axes. A
key that sets an axis a family cannot carry SHALL be refused at construction,
never accepted and then dropped, so the cache token can never name a variant that
differs from the one compiled.

#### Scenario: An illegal axis is refused

- **WHEN** a shader-variant key is built for the display pass with the spectral
  axis set
- **THEN** construction raises

#### Scenario: The cache token matches the compiled flags

- **WHEN** the cache token and the compiler flags are produced for every legal
  key of the two new families
- **THEN** each token corresponds to exactly one flag tuple
