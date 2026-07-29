## ADDED Requirements

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

## MODIFIED Requirements

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
