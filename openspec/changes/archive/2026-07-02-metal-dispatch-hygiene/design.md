# Design — Metal Dispatch Hygiene

Full rationale lives in `nanovdb-volume-rendering/design.md` decision **D5** (this change is its
extraction). Summary of the shipped mechanism:

## Decisions

- **Never cancel, never enqueue unbounded.** macOS cannot kill another process's GPU work; queued
  work dies with the process ONLY if no single command buffer overruns the watchdog. So the
  invariant is bounded command buffers (enforced per-kernel via `SKINNY_METAL` caps, e.g. the
  capped SSS march) + guaranteed drain/close on every exit path of the owning process.
- **One module-level lifecycle registry, not per-instance handler stacking.** A `weakref.WeakSet`
  of live contexts serviced by a single `atexit` hook + one chained SIGINT/SIGTERM handler pair:
  installed on empty→non-empty, removed on last clean destroy; weak refs so the hook never extends
  device lifetime; main-thread-only install (`signal.signal` raises elsewhere); chain semantics
  preserve `KeyboardInterrupt` behavior and later app-installed handlers.
- **Idempotent `destroy()` as the single teardown primitive** (`wait_for_idle` → surface
  unconfigure → `device.close()`, exactly once, tolerant of half-failed construction), reached via
  context manager, `try/finally` in `app.py`/`headless.py`, atexit, and signals.
- **Kill harness proves the invariant**, not the implementation: SIGKill a child mid-render, then
  a fresh probe process must construct a device and dispatch within a time budget.

## Risks

- [atexit/signal ordering vs GLFW/slangpy teardown] → hooks only drain+close if not already
  destroyed; idempotent; window state untouched.
- [Registry holds weakrefs ⇒ GC'd-without-destroy contexts are not drained at exit] → accepted:
  refcount-dead contexts have no owner; slang-rhi device destructor + process exit reclaim; the
  guarantee is scoped to contexts still alive at exit.
