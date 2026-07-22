# qt-render-threading Specification

## Purpose
Keep `skinny-gui` responsive while the renderer is accumulating by confining GPU
context and `Renderer` ownership to the Qt render worker thread. The GUI thread
interacts through queued commands and immutable snapshots, never through direct
calls into renderer operations that can upload resources, dispatch GPU work, wait
for backend fences, or tear down GPU state.
## Requirements
### Requirement: Qt renderer ownership is confined to the render thread

`skinny-gui` SHALL construct, use, and destroy the GPU context and `Renderer` on
the render thread, not on the Qt GUI thread. The GUI thread SHALL NOT directly
call renderer methods that may upload resources, rebuild scene state, dispatch
GPU work, wait for backend fences, or perform renderer cleanup.

#### Scenario: Startup does not block the GUI thread with renderer construction

- **WHEN** `skinny-gui` starts and creates its main window
- **THEN** GPU context creation, renderer construction, and initial scene loading
  run on the render thread, while the GUI event loop remains able to process Qt
  events

#### Scenario: Shutdown cleans up on the owning thread

- **WHEN** the Qt window closes
- **THEN** online training is disabled, the renderer is cleaned up, and the GPU
  context is destroyed on the render thread before the worker thread exits

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

### Requirement: GUI reads immutable snapshots instead of the live renderer

The render thread SHALL emit immutable frame/status/state snapshots for GUI
painting, status bars, session persistence, and inspector docks. GUI widgets MAY
cache and display the latest snapshot, but SHALL NOT use the snapshot as a mutable
backdoor into the live renderer.

#### Scenario: Status updates do not wait on the render lock

- **WHEN** the status bar displays accumulation, backend/GPU name, and
  online-training status
- **THEN** it uses the latest emitted snapshot and does not acquire a lock held by
  the render thread

#### Scenario: Session persistence snapshots renderer-owned state safely

- **WHEN** the application saves settings on close
- **THEN** the GUI requests a renderer snapshot from the render thread, receives
  immutable data, and serializes that data without reading renderer-owned objects
  directly

### Requirement: View-menu tool docks operate through the render-thread proxy
The `skinny-gui` View-menu tool docks SHALL open and function under render-thread ownership through the GUI-side renderer proxy, not the live renderer.
The five docks — Scene Graph, Material Graph, Python Material Editor, BXDF
Visualizer, and Camera Debug View — SHALL interact through mirrored reads,
immutable snapshots, and posted commands, the same mechanism the Controls sidebar
uses. No tool dock SHALL hold the live `Renderer` on the GUI
thread or call renderer methods that upload resources, dispatch GPU work, wait for
backend fences, or tear down GPU state directly from the GUI thread. Selecting a
tool dock from the View menu SHALL show a working dock, not a status-bar
placeholder message.

#### Scenario: Every View-menu dock opens

- **WHEN** the user selects Scene Graph, Material Graph, Python Material Editor,
  BXDF Visualizer, or Camera Debug View from the View menu
- **THEN** the corresponding dock is created (or shown/raised if already created)
  and displays live content, without emitting a "needs the snapshot-backed port"
  placeholder

#### Scenario: Tool dock edits enqueue commands without blocking

- **WHEN** a tool dock mutates renderer state (a Scene Graph node transform, a
  material or light override, a Material Graph topology edit, a Python material
  recompile) while the renderer is accumulating
- **THEN** the edit is posted to the render-thread command queue and the GUI
  returns to its event loop without waiting for the current GPU frame, and the
  applied change resets accumulation on the render thread

#### Scenario: Tool dock reads use snapshots, not the live renderer

- **WHEN** a tool dock displays renderer-derived state (the scene-graph tree, the
  material list, Python module names, camera parameters)
- **THEN** it reads from the latest emitted immutable snapshot or an explicit
  round-trip request, and does not read renderer-owned mutable objects directly
  from the GUI thread

### Requirement: GPU-producing tool docks deliver results to the GUI thread
GPU-producing tool docks SHALL run their GPU work on the render thread and deliver the produced result to the GUI thread asynchronously, never issuing the GPU call from the GUI thread.
This covers the Material Graph material preview, the BXDF/BSSRDF lobe evaluation,
and the Camera Debug embedded viewport — each produces a result (preview image,
evaluation grid, or debug frame) that SHALL be marshalled back to the dock on the
GUI thread.

#### Scenario: BXDF evaluation runs on the render thread

- **WHEN** the BXDF Visualizer requests a lobe evaluation for the selected material
- **THEN** the evaluation executes on the render thread and its result grid is
  marshalled to the GUI thread for display, with the GUI remaining responsive while
  the evaluation runs

#### Scenario: Material preview renders on the render thread

- **WHEN** the Material Graph requests a material preview
- **THEN** the preview is rendered on the render thread and its pixels are delivered
  to the dock on the GUI thread, debounced against rapid edits

#### Scenario: Camera Debug view renders as a second surface

- **WHEN** the Camera Debug View is open
- **THEN** the embedded debug viewport is owned and rendered on the render thread,
  its frames are emitted for GUI painting, and camera/display input from the dock is
  posted as render-thread commands rather than applied through a GUI-thread GPU call

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

