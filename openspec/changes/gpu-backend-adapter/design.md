# Design: gpu-backend-adapter

## Context

`backend_select.resource_module(ctx)` returns one of two modules by
`getattr(ctx, "is_metal", False)`. Consumers treat the result as a uniform
handle (`self._gpu`) while simultaneously branching on backend 40 times,
importing private symbols from one specific module, and probing for attributes
that do not mean what they appear to mean.

Two facts shape the design:

1. **The seam is real** — two adapters exist and both are shipped. The problem
   is the interface, not the seam's placement.
2. **The interface is untestable today.** No third adapter exists, so the two
   real ones are compared only by `test_metal_vulkan_structural_parity.py` and
   `test_metal_vulkan_shaded_parity.py`, which need both devices on one host
   and a guarded runner. `test_scene_bindings_spectral.py` is the only test
   that parametrizes over both modules, and it checks a single property.

## Goals / Non-Goals

**Goals**
- One name per concept across the adapters.
- Backend divergence expressed as declared capabilities, read by name.
- A recording adapter, making dispatch order and binding coverage assertable
  with no device.
- A conformance test that fails on drift instead of a comment that asks for
  agreement.

**Non-Goals**
- Merging `vk_wavefront` and `metal_wavefront` pass classes. That duplication
  is real but separate; this change gives it a target to converge on later.
- Removing the Metal-only tuning constants (`SKINNY_METAL_MEGAKERNEL_BANDS`
  and friends) — watchdog tiling is a genuine Metal-only concern and stays a
  declared capability, not a hidden branch.
- Writing an MSL skinning kernel. `vk_skinning.py` has no Metal counterpart
  and the CPU fallback stays; it becomes a declared missing capability.

## Decisions

### D1 — Capabilities are data, not `isinstance`

The interface exposes a capability record: `has_descriptor_sets`,
`has_external_memory`, `has_indirect_dispatch`, `has_shared_in_place_write`,
`has_gpu_skinning`, `bindless_texture_capacity`, `needs_watchdog_tiling`.
Every current branch maps to one of these or to a genuine two-implementation
split. Rejected: keeping `is_metal` and merely centralising it — that names
the vendor, not the reason, and the 40 branches are about six distinct
reasons.

### D2 — `descriptor_sets is None` dies; `_backend_render_ready` is the model

`renderer._backend_render_ready` (`:2172`) is the one leak that was already
absorbed successfully — front-ends stopped asking about `descriptor_sets` and
started asking whether the backend can render. Every remaining sentinel gets
the same treatment: a named question with a backend-neutral answer.

### D3 — `hasattr(ctx, "compute_queue")` is a bug, fixed here

`MetalContext.compute_queue = None` makes the probe always true. Three
`vk_wavefront` factories are protected only by the caller. Replacing it with a
capability read is a behaviour fix, not just a rename, and must be called out
in the change notes rather than folded silently into a refactor.

### D4 — The recording adapter records, it does not simulate

It captures the sequence of calls — allocations, bindings, dispatches with
their group counts, readbacks — and returns zero-filled data. It does not
attempt to produce pixels. That keeps it honest: it verifies *ordering and
coverage*, which is exactly what the 40-branch sprawl endangers, and never
pretends to verify radiometry, which the parity matrix owns.

### D5 — One-sided members are declared, never discovered

`ExternalTimelineSemaphore` (Vulkan) and `MetalFrameEncoder` (Metal) are real
and stay. They go in an explicit one-sided table, exactly like
`METAL_ONLY_DEFINES` in `shader_variants`. The conformance test asserts the
adapters agree *modulo that table*, so adding a one-sided member is a
deliberate edit.

### D6 — Staged landing

This is too large for one change to land safely. Stages, each independently
green:

1. Capability record + the two broken probes replaced (smallest, fixes a bug).
2. Naming unification + argument-domain unification.
3. Recording adapter + conformance test.
4. Renderer branch migration, region by region, ordered by the region's
   existing test coverage.

Stage 4 is where the 15 Metal-only renderer methods either become adapter
implementations or stay as declared capability-gated paths; the decision is
per method and recorded in tasks.

## Risks / Trade-offs

- **Risk: a "harmless" rename changes a compiled artifact.** The
  `BINDLESS_TEXTURE_CAPACITY` fold touches a shader define. Gate: the Vulkan
  `.spv` bytes and the `build/spv_cache` keys must be unchanged, checked the
  same way `shader-variant-key-module` checked them.
- **Risk: replacing the always-true `compute_queue` probe changes which code
  path runs on Metal.** That is the point, but it must be measured: run the
  wavefront Metal suite before and after and diff images, not just exit codes.
- **Trade-off: capability records can grow into a config blob.** Bound it —
  a capability earns its place only by replacing at least one existing live
  branch. No speculative capabilities.
- **Interaction:** `renderer-gpu-resource-set` removes ~10 branches; do it
  first so stage 4 is smaller. `frame-plan-split` consumes the recording
  adapter; do it after stage 3.

## Open Questions

- Should `wavefront_driver.WavefrontRecorder` (the existing backend-neutral
  protocol) become part of this interface, or stay a separate protocol?
  Leaning: stay separate but drop `flush_heavy_eye` from it — a macOS
  watchdog concept currently sits in the backend-neutral driver protocol as a
  documented no-op on Vulkan, which belongs in the capability record instead.
- Does the recording adapter belong in `src/` or in `tests/`? Leaning `src/`,
  because the parity harness and future tooling will want it too — but only
  once a second consumer exists.
