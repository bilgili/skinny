# per-material-pipeline — Delta (combine-graph-param-buffers)

## ADDED Requirements

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
