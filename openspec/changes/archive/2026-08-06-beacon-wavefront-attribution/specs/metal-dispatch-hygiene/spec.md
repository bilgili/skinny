# Metal dispatch hygiene — traced per-kernel submit delta

## ADDED Requirements

### Requirement: Traced wavefront dispatch submits one kernel per command buffer

Under the `SKINNY_METAL_TRACE` gate, the wavefront `MetalFrameEncoder` SHALL
submit and drain each dispatch as its own bounded command buffer, so a wedge
isolates the one kernel that hung. Each traced dispatch SHALL stamp the beacon
cell, submit that one kernel, and drain (`wait_for_idle`) before the next kernel
is encoded. This aligns with the watchdog-bounded rule: each dispatch finishes or
hangs alone, so the operator reads the exact in-flight kernel from the beacon.

With the gate off (the production default), the wavefront encoder SHALL keep its
batched single-submit behavior unchanged. The per-kernel submit SHALL apply only
under trace, so production performance and the committed command-buffer shape are
unchanged. The SIGTERM-first, never-SIGKILL-first timeout order SHALL be
unchanged.

#### Scenario: a traced wavefront wedge isolates one kernel
- **WHEN** a wavefront frame runs under `SKINNY_METAL_TRACE` and one stage kernel
  hangs
- **THEN** only that kernel's command buffer is in flight, the beacon cell names
  that kernel, and the parent SIGTERMs and reports it by name

#### Scenario: production wavefront dispatch is unchanged
- **WHEN** a wavefront frame runs with `SKINNY_METAL_TRACE` off
- **THEN** the encoder batches the frame's stages into one command buffer and
  submits once, exactly as before this change
