# Design: renderer-command-interface

## Context

The queue already exists, already works, and is already the tested path.
`qt-render-threading` records its requirements. What is missing is the decision
that it is *the* interface rather than one of three.

The web session was built independently, with a lock and a background render
thread, and it is internally inconsistent: control actions take the lock,
parameter writes do not. That inconsistency is documented in the session's own
docstring (`handle_control` says it is "routed through the same lock the camera
/ render path uses"), which makes the omission in `set_param` look accidental
rather than deliberate.

## Goals / Non-Goals

**Goals**
- No front-end mutates a live renderer from a thread that does not own it.
- One command shape with a reply contract, usable interactively and headlessly.
- The shared control tree carries one thread-safety contract, not two.

**Non-Goals**
- Removing the web session's lock. The lock still protects the render/encode
  span; it stops being the *mutation* mechanism.
- Changing the queue's coalescing semantics.
- Merging `HeadlessRenderer`'s conveniences (`render_to_array`,
  `render_scene`, `render_animation`) — those stay; only what they call changes.

## Decisions

### D1 — The queue is the interface; the lock is an implementation detail

Front-ends post; the owning thread drains. The web session keeps its lock for
the render+encode span but no longer accepts mutations through it. This makes
the shared control tree's contract uniform: a setter always posts.

### D2 — `skinny-render` posts and drains synchronously

A headless caller has no second thread, so "post then drain immediately" is a
degenerate case of the same interface, not a special path. The value is
uniformity: the same setter code runs under all four front-ends, so a bug found
in one is found in all. Rejected: exempting headless because it is
single-threaded — that is exactly the reasoning that produced three paths.

### D3 — Panel setters post, and the proxy grows the verbs they need

`ui/panel/backend.py`'s six `node.setter(...)` call sites are the concrete
leak. The setters themselves are constructed by the shared UI tree, so binding
the tree to a proxy (as Qt does) fixes all six at once without touching
`backend.py` at all — which is the lazy and correct fix. Any verb the web needs
that the Qt proxy lacks is added to the proxy.

### D4 — Reply contract everywhere, reporting where it is useful

MCP already validates and reports edit outcomes; Qt uses replies for
GPU-producing docks. The web front-end currently reports nothing. Giving every
mutation a reply is free once D1 lands; whether the web *surfaces* the outcome
in the UI is a smaller follow-on and is not required by this change.

### D5 — The index-based button synthesis goes

`web_app.py:516-533` reaches into a Panel widget tree by index and increments
`clicks` to trigger a debug-camera action. It is a workaround for not having a
command path. Once the actions are posted commands, it is deleted rather than
re-plumbed.

## Risks / Trade-offs

- **Risk: web latency.** Posting adds a hop before the next frame picks up the
  change. The queue already coalesces last-write-wins, which is what a slider
  drag wants; the current direct write is not actually faster in any way the
  user perceives, since the value is only read at the next `update()`.
- **Risk: a posted command that expects to run before the next render.**
  Screenshot and resize currently hold the lock across the operation. They
  become post-with-reply and wait on the future — same ordering, explicit.
- **Risk: headless behaviour change.** Draining synchronously must not reorder
  anything relative to today's direct calls. Gate: render the same scenes
  before and after and require identical images.
- **Trade-off: one more indirection in the headless path.** Accepted for
  uniformity; headless renders are not latency-bound.

## Open Questions

- Should `HeadlessRenderer` expose the queue at all, or keep its current
  method surface and use the queue internally? Leaning: keep the surface,
  use the queue internally — the public API in `docs/PythonAPI.md` should not
  churn for an internal uniformity fix.
- Does the web session need per-session queues, or one per renderer? One per
  renderer — sessions already own their renderer.
