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

This requirement governs the read itself. It does not state how often a light is
re-extracted; the per-frame requirement below states that.

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
