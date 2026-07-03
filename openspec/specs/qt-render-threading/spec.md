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
