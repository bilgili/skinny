# geometry-suballocation Specification

## Purpose

Manage the shared vertex, index, and BVH buffers as per-mesh slabs with stable
offsets so that runtime model add/remove touches only the changed mesh — baking
and uploading just the new slab on add, freeing a slab to a free-list on remove,
and reclaiming fragmentation by compaction that rewrites referencing instance
records — without re-concatenating or re-uploading resident geometry. The
suballocator serves both the `megakernel` and `wavefront` execution modes.

## Requirements

### Requirement: Per-mesh slabs with stable offsets

The shared vertex, index, and BVH buffers SHALL be managed as per-mesh slabs
rather than a single whole-scene concatenation. Each baked mesh SHALL occupy a slab
whose vertex/index/BVH offsets remain stable for the lifetime of that slab, so the
per-BLAS offsets stored in instance/TLAS records remain valid without reindexing
when other meshes are added or removed. Growing a buffer to make room SHALL
preserve the offsets of existing slabs.

#### Scenario: Offsets survive unrelated add/remove

- **WHEN** a mesh is added or another mesh is removed
- **THEN** the vertex/index/BVH offsets of every other resident mesh are unchanged,
  and their instance records continue to reference valid geometry

### Requirement: Incremental add touches only the new mesh

Adding a model at runtime SHALL bake only the new geometry (reusing the
content-hash cache for unchanged geometry), write only the new mesh's slab to the
shared buffers, and add only its instance record(s). It SHALL NOT re-concatenate or
re-upload geometry for meshes already resident.

#### Scenario: Add uploads only the new slab

- **WHEN** a model is added to a scene that already contains other meshes
- **THEN** only the new mesh's slab and its instance record(s) are written to the
  GPU, and resident meshes' geometry is not re-uploaded

### Requirement: Incremental remove frees a slab without re-upload

Removing a model SHALL free its slab to a free-list and drop its instance
record(s) without re-concatenating or re-uploading the remaining scene. Freed slab
space SHALL be available for reuse by subsequent adds.

#### Scenario: Remove does not re-upload survivors

- **WHEN** a model is removed from a multi-mesh scene
- **THEN** its slab is freed and its instances dropped, and the remaining meshes'
  geometry is not re-uploaded

#### Scenario: Freed space is reused

- **WHEN** a model is removed and then a model whose geometry fits the freed space
  is added
- **THEN** the new mesh may occupy the freed slab space

### Requirement: Compaction preserves rendered output

When slab fragmentation is reclaimed by compaction, moving a slab SHALL rewrite
every instance/TLAS record that references it so that the rendered output is
unchanged across the compaction. Compaction SHALL be safe to skip (fragmentation is
tolerated via the free-list).

#### Scenario: Output unchanged across compaction

- **WHEN** the geometry buffers are compacted
- **THEN** all moved slabs' referencing instance records are updated and the
  rendered image is unchanged within tolerance

### Requirement: Suballocation is independent of execution mode

The geometry suballocator SHALL serve both the `megakernel` and `wavefront`
execution modes, since both read the shared vertex/index/BVH buffers. Incremental
add/remove behavior SHALL not depend on the active execution mode.

#### Scenario: Same incremental behavior in either mode

- **WHEN** a model is added or removed in `megakernel` mode and the same operation
  is performed in `wavefront` mode
- **THEN** both touch only the changed mesh's slab and instance record(s)
