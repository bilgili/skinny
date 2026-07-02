# Tasks — metal-dispatch-hygiene

Extracted from `nanovdb-volume-rendering` tasks 1.3–1.5 (implemented there, merged from here so
other proposals can build on it). The watchdog-bounded volume-march scenario stays with the
nanovdb change (task 6.5), which consumes this capability.

## 1. Teardown foundation

- [x] 1.1 `MetalContext`: idempotent `destroy()`, context-manager protocol, `atexit` +
      SIGINT/SIGTERM teardown hooks (weakref registry, single hook set, unregister/restore on
      clean close); wire `app.py` + `headless.py` exit paths through it
- [x] 1.2 Hostless unit tests: destroy idempotency, half-failed construction,
      context-manager exception path, atexit register/unregister, weakref-only registry,
      signal install/chain/restore/no-clobber, non-main-thread guard, SIGTERM subprocess
      end-to-end (13 tests)
- [x] 1.3 Kill harness (gpu-marked, guarded): clean-exit probe, SIGKILL-mid-render →
      fresh-subprocess probe within budget, atexit-marker on normal exit — all 3 GPU-verified
      green on Apple M5 Pro (Metal)
- [x] 1.4 CLAUDE.md: standing "Metal dispatch hygiene" requirement for all GPU work
