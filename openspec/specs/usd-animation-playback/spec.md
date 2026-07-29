# usd-animation-playback Specification

## Purpose
TBD - created by archiving change usd-animation-playback. Update Purpose after archive.
## Requirements
### Requirement: Playback clock drives USD time

The renderer SHALL maintain a playback clock that maps wall-clock delta time to
a USD time code within the loaded stage's authored time range. When playing, the
clock SHALL advance `current_time_code` by `dt * playback_fps` (expressed in time
code units) each frame and SHALL loop back to the start time code after passing
the end time code. The clock SHALL initialize its time range and default fps
from the stage's `startTimeCode`, `endTimeCode`, and `timeCodesPerSecond`
metadata, falling back to a fixed default fps when `timeCodesPerSecond` is
unauthored.

#### Scenario: Time advances while playing

- **WHEN** the clock is playing and `advance(dt)` is called
- **THEN** `current_time_code` increases by `dt * playback_fps` in time code units

#### Scenario: Playback loops at the end

- **WHEN** `advance(dt)` would move `current_time_code` past `end_time_code` while looping is enabled
- **THEN** the clock wraps `current_time_code` back into `[start_time_code, end_time_code]`

#### Scenario: Paused clock does not advance

- **WHEN** the clock is paused and `advance(dt)` is called
- **THEN** `current_time_code` is unchanged

#### Scenario: Stage with no animation reports inert

- **WHEN** the loaded stage has no authored time samples on any prim
- **THEN** the clock reports `has_animation` as false and `advance(dt)` never changes `current_time_code`

### Requirement: Animated prims are indexed at load

At load time the renderer SHALL scan the stage and record which prims have
authored time samples relevant to playback: transform tracks, light attributes,
and the camera. The index SHALL be used to restrict per-frame re-evaluation to
the animated set, and SHALL determine whether the stage `has_animation`.

#### Scenario: Animated transform is detected

- **WHEN** a prim has authored time samples on its transform stack
- **THEN** that prim's path is included in the animated-transform index and `has_animation` is true

#### Scenario: Static prim is excluded

- **WHEN** a prim has no authored time samples
- **THEN** that prim is excluded from every animation index and is not re-evaluated during playback

### Requirement: Per-frame re-evaluation of cheap animated prims

The renderer SHALL, while playing on any frame where `current_time_code`
changed, re-evaluate only the indexed animated prims at the current time code
and apply the results without rebaking mesh geometry or rebuilding the BVH. Animated
transforms SHALL recompute each affected instance's world matrix and re-upload
only that instance's TLAS `instance_buffer` record. Animated lights SHALL
re-extract their parameters at the current time code and update the
corresponding light buffers.

Per-frame light re-extraction SHALL cover DistantLight and SphereLight. It SHALL
NOT cover DomeLight, RectLight or DiskLight: a dome would re-decode its HDR
texture each frame, and a rect or disk light is carried as emissive geometry
whose refresh needs the mesh rebake and BVH rebuild that this requirement
excludes. Those three types keep the values read when the stage was extracted.
Animating them is outside this capability.

#### Scenario: Animated object moves over time

- **WHEN** playback advances across time codes where an indexed prim's transform changes
- **THEN** the corresponding TLAS instance transform is re-uploaded and the object appears at its time-correct pose, with no mesh rebake or BVH rebuild

#### Scenario: Animated light updates over time

- **WHEN** playback advances across time codes where an indexed DistantLight's or SphereLight's attributes change
- **THEN** the light buffer is updated so the rendered illumination reflects the time-correct light parameters

#### Scenario: A dome or area light carries time samples

- **WHEN** playback advances and a DomeLight, RectLight or DiskLight holds time-sampled emission attributes
- **THEN** the renderer keeps the values read at stage extraction, and does not rebake geometry or re-decode a texture to follow them

### Requirement: USD camera mode follows an animated camera

The renderer SHALL provide a `usd` camera mode that drives the viewport from the
stage's USD camera evaluated at `current_time_code`. This mode SHALL be offered
only when the loaded stage contains a USD camera. Selecting Orbit or Free SHALL
return camera control to the user. While in `usd` mode, camera motion SHALL be
reflected in the accumulation state so progressive accumulation resets as the
camera moves.

#### Scenario: Viewport follows the USD camera during playback

- **WHEN** `camera_mode` is `usd` and playback advances across an animated camera track
- **THEN** the viewport view/projection matrices follow the USD camera at the current time code

#### Scenario: USD mode unavailable without a USD camera

- **WHEN** the loaded stage contains no USD camera
- **THEN** the `usd` camera mode is not offered as a selectable option

#### Scenario: User regains control by switching mode

- **WHEN** the user selects Orbit or Free while in `usd` mode
- **THEN** the viewport is driven by the user-controlled camera and no longer follows the USD camera track

### Requirement: Built-in transport controls across front-ends

The renderer SHALL expose play/pause, a normalized 0–1 time scrubber, and an fps
control through the shared parameter/spec tree so they appear in the Qt panel,
the web panel, and the debug viewport. The time scrubber SHALL map its 0–1 value
onto `[start_time_code, end_time_code]`. The animation controls SHALL be shown
only when the loaded stage `has_animation`.

#### Scenario: Transport appears for an animated scene

- **WHEN** a stage with authored animation is loaded
- **THEN** play/pause, time scrubber, and fps controls are present in the Qt panel, web panel, and debug viewport

#### Scenario: Scrubber maps normalized value to time code

- **WHEN** the user sets the time scrubber to a normalized value `t` in `[0, 1]`
- **THEN** `current_time_code` is set to `start_time_code + t * (end_time_code - start_time_code)`

#### Scenario: Transport hidden for static scenes

- **WHEN** a stage with no animation is loaded
- **THEN** no transport controls are shown

### Requirement: Accumulation resets during playback

`current_time_code` SHALL be part of the renderer's accumulation state hash so
that progressive accumulation resets whenever the time code changes. While
playing, each advanced frame SHALL render at one sample per pixel; while paused,
the renderer SHALL accumulate samples normally.

#### Scenario: Playing resets accumulation each frame

- **WHEN** the clock is playing and `current_time_code` changes between frames
- **THEN** `accum_frame` resets to 0 so each displayed frame is freshly sampled

#### Scenario: Paused frame accumulates

- **WHEN** the clock is paused and no other state changes
- **THEN** `current_time_code` is stable and progressive accumulation continues to converge

### Requirement: Light emission attributes are read at the evaluation time code

Light extraction SHALL read every emission attribute at the time code it
evaluates the light at. The emission attributes are `inputs:color`,
`inputs:intensity` and `inputs:exposure`. This SHALL hold for each supported
light type: DistantLight, SphereLight, DomeLight, RectLight and DiskLight.

A read that omits the time code SHALL NOT occur. USD resolves a time-code-free
read at the default time code. An attribute that holds only time samples has no
value there, so USD returns the schema fallback — 50000 for a
`UsdLuxDistantLight` intensity. The light then renders at the fallback instead
of its authored value.

A stage read that receives no explicit time code SHALL evaluate at the stage's
start time code, not at the default time code. A caller that forwards an absent
time code SHALL forward its absence, and SHALL NOT substitute
`Usd.TimeCode.Default()` for it. `Usd.TimeCode.Default()` is a specific time code
at which a time-sampled attribute has no value, so substituting it discards the
information the stage read needs to choose the start time code. The playback clock also starts at
the stage's start time code, so the loaded values agree with the first rendered
frame. This matters most for the light types that are never re-extracted: for
them the value read at load is the value that renders for the whole session.

This requirement governs the read itself. It does not state how often a light is
re-extracted; the per-frame requirement below states that.

#### Scenario: A stage loads with no explicit time code

- **WHEN** a stage is read without an explicit time code and a light authors its emission attributes as time samples only
- **THEN** the load evaluates that light at the stage's start time code, so the loaded radiance is the authored sample there and not the schema fallback

#### Scenario: A headless render requests no particular time

- **WHEN** a headless render runs with its time option unset
- **THEN** the absent time code reaches the stage read as absent, and the render uses the stage's start time code

#### Scenario: A live-state resync follows a control edit

- **WHEN** a `usd:` control edits an attribute and the renderer re-reads live scene state while a time-sampled light is loaded
- **THEN** the resync re-extracts that light at the playback clock's current time code, and does not reset it to the value at another time code

#### Scenario: Intensity has time samples and no default value

- **WHEN** a light authors `inputs:intensity` as time samples only, with no default value
- **THEN** extraction at each time code returns the value interpolated from those samples, and never the schema fallback

#### Scenario: A second read of the same attribute uses the same time code

- **WHEN** extraction reads an emission attribute again to record it separately from the combined radiance, as SphereLight does for its authored colour and intensity
- **THEN** that read uses the same time code, so the recorded value and the combined radiance agree

#### Scenario: Emission attribute has no time samples

- **WHEN** a light authors `inputs:color`, `inputs:intensity` or `inputs:exposure` as a default value, or authors none of them
- **THEN** extraction at any time code returns the same value that a time-code-free read returns

