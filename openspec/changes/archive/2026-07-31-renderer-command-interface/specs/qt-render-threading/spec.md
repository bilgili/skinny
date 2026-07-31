# qt-render-threading (delta)

## MODIFIED Requirements

### Requirement: Single-threaded front-end owns and drains a command queue

**Every** front-end that owns a renderer SHALL own a render-thread command
queue and SHALL drain it at a point before the renderer is advanced for that
frame, so that another thread can mutate renderer state without touching the
renderer directly. This applies to the single-threaded interactive front-end,
the Qt front-end's render thread, the web session's background render thread,
and the headless driver — the latter draining synchronously, which is the
degenerate case of the same interface rather than a separate path.

Draining SHALL occur whether or not any in-process server is enabled, so the
ordering behavior does not depend on optional features being active.

#### Scenario: Commands posted from another thread apply

- **WHEN** another thread posts a mutation while the single-threaded front-end is
  running its main loop
- **THEN** the mutation is applied on the main loop's thread before the next frame
  is advanced, and accumulation resets for the changed state

#### Scenario: Empty queue costs nothing observable

- **WHEN** no commands are pending
- **THEN** the drain completes without blocking and the frame proceeds normally

#### Scenario: Web session mutations are posted, not applied in place

- **WHEN** a browser client changes a parameter, or a sidebar widget setter
  fires on the web server's worker thread
- **THEN** the mutation is posted to the session's queue and applied on the
  session's render thread, and no code path outside that thread mutates the
  live renderer

#### Scenario: Headless driving uses the same interface

- **WHEN** the headless driver applies scene, parameter or camera changes
  between renders
- **THEN** it posts and drains through the same queue, and the rendered images
  are identical to those the pre-change direct-call path produced
