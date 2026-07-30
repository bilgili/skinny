## ADDED Requirements

### Requirement: MetalFX is the Metal-backend denoiser implementation

`denoise_metalfx.py` SHALL implement the `Denoiser` protocol with Apple MetalFX
temporal denoised upscaling. It SHALL be registered under the name `metalfx`.

The implementation SHALL run only on the native Metal backend. A `metalfx`
denoiser requested on the Vulkan backend SHALL be refused at startup with a
message naming the backend, and SHALL NOT fall back to no denoiser silently.

#### Scenario: MetalFX is selectable on a Metal host

- **WHEN** `--backend metal --denoiser metalfx` is launched on an Apple-Silicon
  host with the optional dependency installed
- **THEN** the renderer starts and the active denoiser reports the name `metalfx`

#### Scenario: MetalFX on Vulkan is refused

- **WHEN** `--backend vulkan --denoiser metalfx` is launched
- **THEN** the launch is refused at startup with a message naming the backend,
  and no GPU device is constructed for the renderer

### Requirement: MetalFX reaches Metal objects through slang-rhi native handles

The implementation SHALL obtain the Metal device from the slang-rhi device's
native handles and each Metal texture from the corresponding slang-rhi texture's
native handle. It SHALL NOT create a second Metal device and SHALL NOT copy image
data through the host to reach MetalFX.

The implementation SHALL submit MetalFX work on its own Metal command queue,
built from that same device, and the renderer's existing device-idle wait SHALL
be the ordering point between the render passes, the denoiser, and the display
pass.

#### Scenario: One device is shared

- **WHEN** the MetalFX denoiser is constructed
- **THEN** the Metal device it uses is the same device the renderer's context
  owns

#### Scenario: No host round trip

- **WHEN** a frame is denoised
- **THEN** no image data is read back to the host and re-uploaded

### Requirement: MetalFX inputs are recorded from the device, not assumed

Before any renderer code consumes MetalFX, a spike SHALL record, from a real
device on this host, the inputs the scaler requires, the texture formats it
accepts, and the input-to-output size ratios it accepts. The recorded result
SHALL be checked in, and the auxiliary images the implementation declares SHALL
match it.

If the recorded formats exclude the accumulation image's format, the
implementation SHALL declare an input image in an accepted format, and the
renderer SHALL populate it with one copy pass that does not touch transport.

#### Scenario: The recorded contract is checked in

- **WHEN** the change is reviewed
- **THEN** a checked-in record states the required MetalFX inputs, the accepted
  texture formats, and the accepted size ratios, produced by running against a
  real device

#### Scenario: Declared auxiliary images match the record

- **WHEN** the implementation's `required_aovs` is compared with the recorded
  MetalFX input list
- **THEN** every required MetalFX input is covered and nothing else is declared

### Requirement: MetalFX upscales from the render extent to the output extent

The implementation SHALL take the render extent as its input size and the output
extent as its output size. A denoise scale below 1.0 SHALL therefore upscale.

If the recorded contract does not permit an input size equal to the output size,
`--denoise-scale 1.0` with `metalfx` SHALL be refused at startup with a message
naming the permitted range, and SHALL NOT be silently upscaled or silently
ignored.

#### Scenario: A fractional scale upscales

- **WHEN** the renderer runs at an output extent of 800×600 with
  `--denoiser metalfx --denoise-scale 0.5`
- **THEN** the path tracer renders 400×300 and the display image is 800×600

#### Scenario: An unsupported ratio is refused, not ignored

- **WHEN** a denoise scale outside the recorded permitted range is requested
- **THEN** the launch is refused with a message naming the permitted range

### Requirement: The MetalFX dependency is optional and its absence is a startup refusal

The Objective-C bridge MetalFX needs SHALL be installed by an optional extra. A
default install SHALL gain no new dependency.

`--denoiser metalfx` without the extra installed SHALL be refused at startup with
a message that names the install command, in the same shape the MCP flag already
uses. It SHALL NOT fail later at the first frame.

#### Scenario: A default install is unchanged

- **WHEN** the package is installed without extras
- **THEN** no Objective-C bridge package is installed and `--denoiser none`
  works

#### Scenario: The missing extra is named at startup

- **WHEN** `--denoiser metalfx` is launched without the extra installed
- **THEN** the launch is refused before any frame is rendered, with a message
  naming the install command

### Requirement: MetalFX obeys Metal dispatch hygiene

The scaler and its command queue SHALL be destroyed in the renderer's teardown
sequence, before the device closes, and the teardown SHALL be idempotent.

MetalFX work SHALL be bounded by construction: one dispatch per frame over one
image pair. The Metal cleanup harness SHALL pass before this change merges,
because the change adds GPU work and changes context lifecycle.

#### Scenario: Teardown releases the scaler

- **WHEN** the renderer is destroyed with a MetalFX denoiser active
- **THEN** the scaler and its command queue are released before the device
  closes, and a repeated destroy is a no-op

#### Scenario: The cleanup harness passes

- **WHEN** the Metal cleanup harness runs with a MetalFX denoiser active
- **THEN** the clean-exit probe, the interrupted-render probe, and the
  teardown probe all pass, and a fresh device constructs afterwards
