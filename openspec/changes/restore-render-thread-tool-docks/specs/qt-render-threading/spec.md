## ADDED Requirements

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
