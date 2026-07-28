# renderer-module-structure (delta)

## ADDED Requirements

### Requirement: GPU resource allocation, binding and destruction live in one module with paired declarations

The renderer's GPU resource inventory SHALL live in a dedicated module outside
`renderer.py`, in which each resource is declared once and that single
declaration carries its allocation inputs, its binding identity on both
backends (Vulkan descriptor binding number, Metal shader-global name, either
optionally absent), and its destruction. The module SHALL absorb `_init_gpu`,
`_create_descriptors`, the five `_rebind_*_descriptors` methods,
`_rewrite_size_dependent_descriptors`, `_ensure_mesh_buffer_capacity`,
`_build_metal_binds`, and the resource-destroy body of `cleanup`. Backend
divergence SHALL be confined to one binding step consuming the shared
declaration list — the per-method `is_metal` / `descriptor_sets is None`
early-returns MUST NOT survive the move. The resulting GPU state MUST be
identical to the pre-change renderer: same resources, same sizes and formats,
same binding numbers, and the same descriptor-write order.

#### Scenario: Every allocated resource is destroyed

- **WHEN** the resource set is constructed against a recording context and
  then closed
- **THEN** the set of resources destroyed equals the set allocated, with no
  resource allocated twice and none left undestroyed

#### Scenario: One declaration feeds both backends

- **WHEN** the set is bound for the Vulkan target and for the Metal target
- **THEN** both bindings are derived from the same declaration list, the
  Vulkan binding numbers are unique, the Metal global names are unique, and
  the two cover the same declarations modulo declarations explicitly marked as
  absent on one target

#### Scenario: Inventory matches the pre-change renderer

- **WHEN** the resource set's declarations are compared against the recorded
  pre-change inventory fixture — name, kind, size inputs, format, binding
  number and descriptor-write order captured from `_init_gpu` and
  `_create_descriptors` before the move
- **THEN** they match entry for entry, including order

#### Scenario: Growth reflows bindings through the set

- **WHEN** a resource is reallocated because its capacity grew — mesh buffers,
  the volume grid, the bindless texture pool, or a size-dependent image after
  a viewport resize
- **THEN** the rebind that follows is performed by the resource set from the
  same declaration, and no call site outside the set rewrites a descriptor
