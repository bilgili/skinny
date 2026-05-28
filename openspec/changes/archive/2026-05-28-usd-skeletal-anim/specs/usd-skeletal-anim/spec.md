## ADDED Requirements

### Requirement: Skinned meshes are detected at load

The loader SHALL detect UsdSkel bindings at load — discovering each SkelRoot's
Skeleton, its SkelAnimation source, and the skinned Meshes bound via
`UsdSkelBindingAPI` — and SHALL record each skinned mesh as a skinned instance
with its per-vertex joint influences (`jointIndices`, `jointWeights`). A stage
that contains at least one skinned mesh SHALL report `has_animation` so the
existing playback transport is surfaced.

#### Scenario: Skinned stage enables the transport

- **WHEN** a stage containing a SkelRoot with a skinned mesh and a SkelAnimation is loaded
- **THEN** the stage reports `has_animation` and the playback transport controls appear

#### Scenario: Influences captured per skinned mesh

- **WHEN** a skinned mesh with N influences per vertex is loaded
- **THEN** its rest positions, rest normals, and per-vertex joint indices and weights are recorded for that instance

#### Scenario: Non-skinned stage unaffected

- **WHEN** a stage with no UsdSkel bindings is loaded
- **THEN** no skinned instances are recorded and skeletal animation contributes nothing to `has_animation`

### Requirement: Bind-pose bake for skinned meshes

Each skinned mesh SHALL be baked at load into its authored bind-pose geometry as
its BLAS and SHALL retain the placement transform the loader assigns (prim world
transform with up-axis correction), so that a stage which is never played still
renders the mesh correctly in bind pose. Per-frame deformed points are produced
in that same authored-points space, so the placement transform applies unchanged.

#### Scenario: Unplayed skinned scene renders bind pose

- **WHEN** a skinned stage is loaded and playback never starts
- **THEN** the skinned mesh renders in its bind pose at the correct world placement

### Requirement: Per-frame joint matrices computed on the CPU

While playing on a frame where the time code changed, the renderer SHALL compute
per-joint skinning matrices for each skeleton at the current time code using USD's
skinning evaluation, fold in the up-axis correction and the SkelRoot world
transform to produce skel→world matrices, and upload them to a GPU joint-matrix
buffer for the skinning pass.

#### Scenario: Joint matrices track the time code

- **WHEN** playback advances to a new time code
- **THEN** the joint-matrix buffer is updated with skel→world skinning matrices evaluated at that time code

#### Scenario: Matrices match a CPU skinning reference

- **WHEN** the joint matrices are applied to a skinned mesh's rest vertices via linear blend skinning
- **THEN** the resulting positions match USD's reference skinned points for the same time code within a small tolerance

### Requirement: GPU linear-blend skinning into the vertex buffer

The renderer SHALL run a GPU compute pass that linear-blend-skins each skinned
mesh's rest positions and normals by its per-vertex influences and the joint
matrices, writing the deformed positions and normals (in the authored-points
space) into that mesh's BLAS region of the vertex buffer. The pass SHALL NOT
require a GPU-to-CPU readback and SHALL NOT rebake mesh topology.

#### Scenario: Vertices deform on the GPU

- **WHEN** the skinning pass runs for a time code where joint transforms differ from bind pose
- **THEN** the mesh's vertex-buffer positions are the world-space linear-blend-skinned positions, with no readback to the CPU

### Requirement: GPU BVH refit over deformed geometry

After skinning, the renderer SHALL refit each skinned mesh's BVH on the GPU by
recomputing leaf node AABBs from the deformed triangles and propagating inner node
AABBs up the bind-pose tree topology, so the path tracer intersects the deformed
geometry without rebuilding the tree or reading geometry back to the CPU. The
skinning, refit, and main render passes SHALL be ordered by GPU barriers.

#### Scenario: BVH bounds follow the deformation

- **WHEN** the refit pass runs after skinning
- **THEN** every BVH node's AABB contains its deformed geometry and the rendered image shows the mesh intersected correctly (no missing or stretched-through surfaces)

#### Scenario: Passes are correctly ordered

- **WHEN** a frame skins and refits a mesh before rendering
- **THEN** the main render pass observes the deformed vertices and refitted BVH for that frame

### Requirement: Skinned playback resets accumulation

While playing, a change in time code SHALL reset progressive accumulation so each
displayed skinned frame is freshly sampled; while paused, accumulation SHALL
continue to converge on the static deformed pose.

#### Scenario: Playing skinned animation resets accumulation

- **WHEN** the clock is playing and the time code changes between frames
- **THEN** accumulation resets so the deformed frame is sampled fresh

#### Scenario: Paused skinned pose converges

- **WHEN** the clock is paused on a deformed pose
- **THEN** the time code is stable and progressive accumulation converges
