## MODIFIED Requirements

### Requirement: Each material is its own compute pipeline under wavefront

In `wavefront` mode the renderer SHALL partition path-tracer shade work into
compute pipelines such that each material's queued hits are shaded by a pipeline
that contains that material's code, rather than stitching all materials into a
single megakernel `switch`. Light materials MAY share a pipeline when their
combined code stays within the size bound (see the size-bounded-shade
requirement); heavy materials SHALL be isolated. The shade stage SHALL dispatch,
for each material's queued hits, the pipeline containing that material.

#### Scenario: Materials map to pipelines containing their code

- **WHEN** a scene with multiple distinct materials is loaded in `wavefront` mode
- **THEN** each material's hits are shaded by a compute pipeline that includes
  that material's shading code, and no material is shaded by a megakernel switch
  over all materials

## ADDED Requirements

### Requirement: Wavefront shade kernels stay within the Metal compile size bound

In `wavefront` mode the renderer SHALL keep every path-tracer shade compute
pipeline's compiled size below a safety threshold under the platform's Metal
compile danger line (MoltenVK ~2.83 MB), so no shade kernel triggers the flaky
large-kernel compile. The renderer SHALL decide the grouping from the **measured**
isolated size of each material's shade kernel (compiling each member once and
reading its SPIR-V byte size, reusing the content-hash cache), then bin-pack
members into groups whose summed isolated sizes do not exceed the threshold. A
monolithic material (for example skin) SHALL occupy its own group. Because a
group's shared imports are counted once, packing by the summed member sizes SHALL
never under-split past the threshold.

The grouping SHALL be recomputed when the material set changes (scene load, a new
material, or a live material edit) and otherwise reused. The flat / MaterialX-graph
shade kernel SHALL remain a single group, unaffected.

#### Scenario: No shade kernel approaches the compile limit

- **WHEN** a scene whose materials would otherwise combine into a single
  over-limit shade kernel (for example skin plus several python materials) is
  loaded in `wavefront` mode
- **THEN** the shade work is split into multiple size-bounded compute pipelines,
  each compiling below the safety threshold, and the rendered image is unchanged
  from the equivalent megakernel render within tolerance

#### Scenario: Flat-only scene compiles a single shade kernel

- **WHEN** a scene with only flat / MaterialX-graph materials is loaded in
  `wavefront` mode
- **THEN** only the flat/graph shade kernel is compiled, with no per-material
  isolate-compiles and no additional shade pipelines

#### Scenario: Unchanged material set reuses the grouping

- **WHEN** a frame is rendered after a previous frame with the same material set
  in `wavefront` mode
- **THEN** the shade-kernel grouping and its compiled pipelines are reused with
  no recompilation
