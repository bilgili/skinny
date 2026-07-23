## MODIFIED Requirements

### Requirement: GUI interactions post render-thread commands

GUI actions that mutate renderer state SHALL be posted to a FIFO render-thread
command queue instead of synchronously mutating the live renderer from the GUI
thread. This includes camera/gizmo input, zoom/focus actions, render-target
resize, scene loading, scene picking, parameter edits, material edits, and
session restore/snapshot operations. High-rate commands such as resize and slider
updates MAY be coalesced, but visible ordering for distinct user actions SHALL be
preserved.

The command queue and the renderer proxy built on it SHALL be front-end-neutral:
they SHALL live in a module that imports no GUI toolkit, so that any front-end and
any in-process server thread can marshal renderer mutations through them. The Qt
module SHALL continue to expose them so existing imports remain valid.

The queue SHALL own the execution of pending commands — invoking each callback
against the renderer and delivering its result or its exception to the awaiting
caller — rather than leaving that loop to each caller. A caller that merely removes
pending commands without delivering replies would strand every awaited call until
its timeout, so command execution SHALL NOT be duplicated per front-end.

Any thread that is not the renderer's owning thread — including a GUI thread and
any in-process server thread — SHALL marshal both reads and writes of renderer
state through this queue. Off-thread reads are included because the scene graph
may be replaced by the streaming load thread.

#### Scenario: Rendering does not freeze common GUI interactions

- **WHEN** the renderer is actively accumulating frames
- **THEN** mouse/keyboard input, sidebar edits, menu actions, and dock controls
  enqueue commands and return to the Qt event loop without waiting for the
  current GPU frame to finish

#### Scenario: Commands apply between frames in order

- **WHEN** the GUI posts a camera drag followed by a parameter change
- **THEN** the render thread applies the drag before the parameter change, resets
  accumulation for the changed state, and subsequent frames reflect the ordered
  changes

#### Scenario: Queue is usable without a GUI toolkit

- **WHEN** the command queue is imported and exercised in a process with no GUI
  toolkit and no GPU context
- **THEN** it constructs, accepts posted commands, and executes them in order,
  without importing a GUI toolkit

#### Scenario: Awaited command receives its reply

- **WHEN** a caller posts a command expecting a reply and the owning thread executes
  pending commands
- **THEN** the caller receives the callback's return value, or its exception if it
  raised, without waiting for a timeout

#### Scenario: Non-Qt front-end drains the queue

- **WHEN** a non-Qt interactive front-end runs its main loop
- **THEN** it drains pending commands each iteration before advancing the renderer,
  so commands posted by another thread apply between frames in order

## ADDED Requirements

### Requirement: Single-threaded front-end owns and drains a command queue

The single-threaded interactive front-end SHALL own a render-thread command queue
and SHALL drain it once per main-loop iteration, at a point before the renderer is
advanced for that frame, so that another thread can mutate renderer state without
touching the renderer directly.

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
