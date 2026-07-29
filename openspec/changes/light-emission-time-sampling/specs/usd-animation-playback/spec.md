## MODIFIED Requirements

### Requirement: Per-frame re-evaluation of cheap animated prims

The renderer SHALL, while playing on any frame where `current_time_code`
changed, re-evaluate only the indexed animated prims at the current time code
and apply the results without rebaking mesh geometry or rebuilding the BVH. Animated
transforms SHALL recompute each affected instance's world matrix and re-upload
only that instance's TLAS `instance_buffer` record. Animated lights SHALL
re-extract their parameters at the current time code and update the
corresponding light buffers.

Light re-extraction SHALL read every emission attribute at that time code. The
emission attributes are `inputs:color`, `inputs:intensity` and
`inputs:exposure`. This SHALL hold for each supported light type: DistantLight,
SphereLight, DomeLight, RectLight and DiskLight. A read that omits the time code
SHALL NOT occur, because USD then returns the schema fallback for an attribute
that holds only time samples.

#### Scenario: Animated object moves over time

- **WHEN** playback advances across time codes where an indexed prim's transform changes
- **THEN** the corresponding TLAS instance transform is re-uploaded and the object appears at its time-correct pose, with no mesh rebake or BVH rebuild

#### Scenario: Animated light updates over time

- **WHEN** playback advances across time codes where an indexed light's attributes change
- **THEN** the light buffer is updated so the rendered illumination reflects the time-correct light parameters

#### Scenario: Light intensity has time samples and no default value

- **WHEN** a light authors `inputs:intensity` as time samples only, with no default value
- **THEN** extraction at each time code returns the value interpolated from those samples, and never the schema fallback

#### Scenario: Light emission attribute has no time samples

- **WHEN** a light authors `inputs:color`, `inputs:intensity` or `inputs:exposure` as a default value, or authors none of them
- **THEN** extraction at any time code returns the same value that a time-code-free read returns
