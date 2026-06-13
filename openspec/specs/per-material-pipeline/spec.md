# per-material-pipeline Specification

## Purpose

Under the `wavefront` execution mode, compile each MaterialX graph into its own
shading compute pipeline instead of a single megakernel `switch`, so the shade
stage dispatches the per-material pipeline for each material's queued hits and
adding a material compiles exactly one kernel rather than recompiling a combined
megakernel. A single material code generator emits both the megakernel and
wavefront forms from one per-graph fragment source.
## Requirements
### Requirement: Each material is its own compute pipeline under wavefront

In `wavefront` mode the renderer SHALL compile each MaterialX graph into its own
shading compute pipeline rather than stitching all graphs into a single megakernel
`switch`. The shade stage SHALL dispatch the per-material pipeline corresponding to
each material's queued hits.

#### Scenario: Materials map to distinct pipelines

- **WHEN** a scene with multiple distinct material graphs is loaded in `wavefront`
  mode
- **THEN** each graph has its own shading compute pipeline, and a hit on a given
  material is shaded by that material's pipeline

### Requirement: Adding a material compiles one kernel, not the megakernel

Introducing a new material graph in `wavefront` mode SHALL compile only that
material's shading pipeline and register it for dispatch (for example when a model
that references a new material is added). It SHALL NOT recompile a combined
megakernel and SHALL NOT recreate the pipelines of materials that already exist.
Materials whose graphs are unchanged SHALL be unaffected.

#### Scenario: New material adds one pipeline

- **WHEN** a model introducing a previously-unseen material graph is added in
  `wavefront` mode
- **THEN** exactly the new material's shading pipeline is compiled and registered,
  and existing materials' pipelines are reused unchanged

#### Scenario: Reusing an existing material compiles nothing

- **WHEN** a model is added in `wavefront` mode whose material graphs already exist
  in the scene
- **THEN** no shading pipeline is compiled and rendering continues with the
  existing pipelines

### Requirement: Material codegen serves megakernel and wavefront from one source

The material code generator SHALL emit from the same per-graph fragment source
both the megakernel form (graphs stitched into the `evalSceneGraph` switch) and the
wavefront form (a per-material shading compute entry and a BSDF-evaluation entry
usable for bidirectional connection events). A material feature added to the
fragment source SHALL appear in both forms without separate authoring.

#### Scenario: One fragment feeds both emitters

- **WHEN** the material code generator runs for a given graph
- **THEN** it can produce the megakernel switch case and the wavefront shading and
  BSDF-evaluation entries from that one graph fragment

#### Scenario: Connection events evaluate the material BSDF

- **WHEN** the wavefront bdpt connection stage connects a camera-subpath vertex to
  a light-subpath vertex
- **THEN** it evaluates each endpoint material's BSDF via that material's
  BSDF-evaluation entry produced by the wavefront emitter

### Requirement: Graph params share one combined param buffer

The material code generator SHALL pack every scene MaterialX nodegraph's param
block into a single combined param buffer bound at one descriptor slot
(`GRAPH_BINDING_BASE`), addressed by a compile-time per-graph byte offset plus
`matId * stride`, rather than emitting one buffer binding per graph. The combined
layout SHALL preserve graph-to-region order and each graph's matId-indexed
addressing, so the shaded result is identical to the per-graph-buffer form. The
per-graph stride encoded for addressing SHALL equal the host param-pack stride
used to fill the buffer; a divergence SHALL fail loudly rather than render
corrupt params.

#### Scenario: Multiple graphs bind one buffer

- **WHEN** a scene with two or more distinct material graphs is loaded
- **THEN** the generated shaders declare exactly one graph-param buffer at
  `GRAPH_BINDING_BASE`, each graph reads its params from that buffer at its
  emitted offset, and the converged image matches the per-graph-buffer form

#### Scenario: Stride drift is caught

- **WHEN** the reflected stride of a graph's param struct disagrees with the
  emitted addressing stride
- **THEN** the renderer raises a clear error naming the graph rather than reading
  misaligned params

#### Scenario: Zero-graph scene unaffected

- **WHEN** a scene carrying no MaterialX graphs is loaded
- **THEN** codegen and binding behave as before (a stub buffer the accessors
  never reference), with no regression to non-graph materials

