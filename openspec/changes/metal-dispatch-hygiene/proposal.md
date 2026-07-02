# Metal Dispatch Hygiene

Split out of `nanovdb-volume-rendering` so every proposal (volumes, SPPM soaks, neural training,
future long-kernel work) builds on the same guaranteed-cleanup foundation instead of re-inventing
it.

## Why

Abandoned Metal compute work (process killed mid-dispatch, exceptions between dispatches,
watchdog-length kernels) was not reliably cleaned up — the GPU stayed wedged until reboot
(observed repeatedly: "Metal slang-rhi WEDGED, needs reboot"). macOS offers no way to cancel
another process's GPU work, so the only robust fix is (a) never enqueue unbounded work and
(b) guarantee teardown on every exit path, verified by a kill harness.

## What Changes

- `MetalContext`: idempotent `destroy()` (safe on half-failed construction), context-manager
  protocol, one module-level lifecycle registry (`weakref.WeakSet`) serviced by a single `atexit`
  hook and chained SIGINT/SIGTERM handlers (main-thread-only install, restore-on-clean-destroy,
  never clobbers app handlers installed later).
- `app.py` main loop and `headless.py` cleanup route every exit path (exception, Ctrl-C, normal)
  through `destroy()`.
- Kill harness `tests/test_metal_cleanup.py` (13 hostless + 3 gpu-marked) + child driver
  `tests/metal_cleanup_child.py`: clean-exit probe, SIGKILL-mid-render → fresh-process probe
  within a time budget, atexit-teardown marker.
- CLAUDE.md: Metal dispatch hygiene is a standing requirement for all GPU work.

## Capabilities

### New Capabilities

- `metal-dispatch-hygiene`: bounded Metal compute dispatches + guaranteed GPU cleanup on every
  process-exit path, with a regression harness.

### Modified Capabilities

(none)

## Impact

- `src/skinny/metal_context.py`, `src/skinny/app.py`, `src/skinny/headless.py`
- `tests/test_metal_cleanup.py`, `tests/metal_cleanup_child.py` (new)
- `CLAUDE.md` (standing GPU-work requirement)
- The watchdog-bounded-command-buffer requirement's volume-march scenario is exercised by the
  in-flight `nanovdb-volume-rendering` change (which consumes this capability).
