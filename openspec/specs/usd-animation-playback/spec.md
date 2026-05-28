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

#### Scenario: Animated object moves over time

- **WHEN** playback advances across time codes where an indexed prim's transform changes
- **THEN** the corresponding TLAS instance transform is re-uploaded and the object appears at its time-correct pose, with no mesh rebake or BVH rebuild

#### Scenario: Animated light updates over time

- **WHEN** playback advances across time codes where an indexed light's attributes change
- **THEN** the light buffer is updated so the rendered illumination reflects the time-correct light parameters

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

