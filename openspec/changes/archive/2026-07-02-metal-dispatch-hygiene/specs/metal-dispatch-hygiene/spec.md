# metal-dispatch-hygiene

Bounded Metal compute dispatches and guaranteed GPU cleanup: no kernel outlives its owning
process, no command buffer exceeds the macOS GPU watchdog, teardown runs on every exit path.

## ADDED Requirements

### Requirement: Metal command buffers are watchdog-bounded

On the Metal backend, no single committed command buffer SHALL contain unbounded per-pixel loop
work. Long marches (volume delta tracking, subsurface walks) SHALL be capped per dispatch
(`SKINNY_METAL`-gated step/bounce caps, continued across accumulation frames) so each command
buffer completes within a watchdog-safe budget. Caps SHALL be tuned so the standing parity gates
still pass (a cap that shifts a gated metric is re-measured, not hidden).

#### Scenario: volume render never trips the watchdog

- **WHEN** `disney_cloud` renders on Metal (megakernel and wavefront) for a full gate-length
  accumulation
- **THEN** every command buffer completes without a GPU fault / watchdog kill, and the run
  produces the gated image

### Requirement: Metal context teardown runs on every exit path

`MetalContext` SHALL guarantee idempotent teardown (`wait_for_idle` + device close) on: normal
exit, exception propagation, interpreter shutdown (`atexit`), and SIGINT/SIGTERM. It SHALL support
the context-manager protocol, and the headless render entry points and app SHALL acquire it through
a scope that triggers teardown. Repeated `destroy()` calls SHALL be safe no-ops.

#### Scenario: exception between dispatches still tears down

- **WHEN** a headless render raises after dispatch N but before dispatch N+1
- **THEN** the context's teardown drains the queue and closes the device before the process exits
  (observable via the teardown hook having run)

#### Scenario: SIGINT tears down

- **WHEN** a rendering process receives SIGINT mid-accumulation
- **THEN** the signal path drains and closes the device before exit, and a subsequent process can
  construct a Metal device and dispatch normally

### Requirement: abandoned GPU work does not wedge the device

A kill-harness regression test SHALL prove GPU recovery: a subprocess that is SIGKILLed
mid-render (no chance to run teardown) SHALL leave the GPU usable — a fresh probe subprocess
SHALL construct a Metal device and complete a trivial dispatch within a fixed time budget. The
harness SHALL be gpu-marked and respect the one-guarded-Metal-process rule.

#### Scenario: SIGKILL mid-render leaves GPU usable

- **WHEN** a gpu-marked test SIGKILLs a child mid volume render and then runs a probe dispatch in
  a new subprocess
- **THEN** the probe completes within its time budget, proving queued work died with the child and
  no kernel kept running
