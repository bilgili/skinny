# Metal kernel beacon (which kernel hung)

## Why

macOS cannot cancel a committed Metal command buffer. macOS cannot reset another
process's GPU. An infinite or over-long compute kernel wedges the GPU until the
machine reboots. skinny already makes cleanup structural through
`metal-dispatch-hygiene`: command buffers stay watchdog-bounded, `[loop]` bodies
carry trip-count guards, GPU work runs in a child process, and a timeout sends
SIGTERM (never SIGKILL) so `MetalContext.destroy()` runs.

One thing is still missing. When a dispatch hangs, the operator cannot tell
**which** kernel was stuck. skinny dispatches the megakernel, 35 wavefront
kernels, the preview kernel, and the debug-raster kernel. A wedge today reports
only "a Metal dispatch did not return". The operator must guess which kernel to
inspect. This change removes the guess: it reports the kernel identity of a hung
dispatch by name.

The native Apple path for this — `MTLCommandBuffer` with
`errorOptions = .encoderExecutionStatus` plus a completion handler that reads
`MTLFunctionLog` for the function name and the shader `file:line` — is not
reachable. skinny dispatches through SlangPy and slang-rhi, which hide the
`MTLCommandBuffer`. Python code sees only
`create_command_encoder → begin_compute_pass → dispatch → submit_command_buffer →
wait_for_idle`. No command-buffer status, no completion handler, no
`MTLFunctionLog`, no `errorOptions` cross the slang-rhi boundary. The native path
is a documented non-goal until slang-rhi exposes the raw command buffer.

## What Changes

This change adds **Layer A — a GPU progress beacon** that is portable, headless,
CI-usable, and needs no Xcode.

- A single small shared-storage beacon buffer (256 bytes) is bound to every
  compute kernel: the megakernel, the 35 wavefront kernels, the preview kernel,
  and the debug-raster kernel.
- A shader helper writes the kernel identity into the beacon buffer. The helper
  lives behind a new build gate `SKINNY_METAL_TRACE`. The gate is OFF by default,
  so production Metal shader binaries and Vulkan SPIR-V stay byte-identical. The
  gate is a new Metal-only axis on `shader_variants.py` (that module stays the
  owner of the define — this change references it, it does not implement it).
- The host confirms the kernel identity in a memory-mapped file. GPU work already
  runs in a child process for the timeout pattern. The child stamps the kernel
  identity into the mmap cell **before** it submits each dispatch, then flushes.
  The parent polls the mmap cell. On a wall-clock timeout the parent sends
  SIGTERM to the child (the chained handler runs `destroy()`), waits the grace
  period, then reads the last stamped cell and reports
  `kernel_id=N (name)`.
- A static kernel-identity table maps each integer id to its entry-point name.
  The wavefront entry names already live in `vk_wavefront.py`.

The stamp lands before the dispatch that may hang. `wait_for_idle()` never
returns for an infinite kernel, so the child cannot flush after the wedge. The
pre-dispatch host stamp is the load-bearing record; the GPU beacon write confirms
the kernel actually started on the device and reserves the byte slots for the
per-loop trip counts that come later.

## Scope (minimal first cut)

- ONE shared beacon buffer, 256 bytes, fixed layout.
- The shader helper writes the kernel id at each kernel's FIRST line only. No
  per-loop trip counts yet.
- The child stamps the kernel id into the mmap cell before each dispatch.
- The parent polls the mmap cell and, on a SIGTERM timeout, reports the last
  stamped kernel id and name.

## Non-goals

- **Per-loop trip counts.** The 256-byte layout reserves a `phase` slot and a
  `trip` slot, but this change writes 0 into both. See Follow-ups.
- **Layer B — operator GPU tooling.** Xcode GPU capture, Instruments Metal System
  Trace, the Shader Profiler, and richer slang-rhi encoder labels are operator
  tooling. They are documented separately, not in this change.
- **The native `MTLFunctionLog` / `os_log`-in-shader path.** slang-rhi hides the
  `MTLCommandBuffer`, so this path is unreachable. It stays a non-goal until
  slang-rhi exposes the raw command buffer.
- **The Vulkan backend.** The gate is Metal-only. Vulkan SPIR-V stays
  byte-identical because the gate defaults off and lives on a Metal-only axis.

## Follow-ups (named, not in this change)

- **Per-loop trip counts.** Add `beacon.store(KERNEL_ID, phase, tripCount)` inside
  the long `[loop]` bodies that actually hang: the skin subsurface walk, the
  volume march, and the MLT mutation. The trip count doubles as the mandated
  `[loop]` guard — break at the cap, write the sentinel, and report the count. The
  byte layout already carries the slots, so this follow-up adds no interface
  change.
- **Layer B operator tooling** documentation.

## Capabilities

### New Capabilities

- `metal-kernel-beacon`: the beacon buffer shape and binding, the
  `SKINNY_METAL_TRACE`-gated shader helper, the child-stamps / parent-polls /
  SIGTERM-reads protocol, the kernel-identity table, and the safety guarantee
  that the parent never reports a kernel the child never reached.

### Modified Capabilities

- `shader-variant-key`: a new Metal-only `metal_trace` axis carrying the
  `SKINNY_METAL_TRACE` define, default off, participating in the cache token so a
  traced variant never collides with a production variant on disk.
- `metal-dispatch-hygiene`: the SIGTERM timeout path now reads the beacon and
  reports the hung kernel's identity, so the operator no longer guesses which
  kernel wedged.

## Impact

- **Shaders**: a new beacon helper module included by every compute entry point,
  all writes gated `#if defined(SKINNY_METAL_TRACE)` so the default binaries are
  byte-unchanged.
- **Host**: a new `metal_beacon.py` module (writer, reader, report, kernel-id
  table); the child dispatch wrapper stamps before each submit; the parent
  timeout path reads and reports. `shader_variants.py` gains the `metal_trace`
  axis.
- **Argument table**: the beacon adds ONE buffer slot, and only under
  `SKINNY_METAL_TRACE` (a debug build). The production Metal argument-table budget
  (128 slots) is unchanged.
- **Tests**: hostless tests for the byte layout, the seqlock torn-read guard, the
  kernel-id table append-only contract, and the reader error semantics; a
  gpu-marked test that a stalled child is reported by kernel name (respecting the
  one-guarded-Metal-process rule).
- **No new Python or C++ dependencies.**
