# qt-render-threading (delta)

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

**Marshalling SHALL cover writes that reach renderer state through an
intermediate object, not only writes to a top-level renderer attribute.** A
setter that mutates an object obtained from the renderer — a playback clock, a
camera, a film record — is a renderer mutation and MUST be posted. A proxy that
holds its own instance of such an object MUST route mutations of it to the
owning thread rather than absorbing them into the local copy.

**A front-end SHALL bind the shared control tree to a marshalling proxy, never
to the live renderer.** Binding the same tree to a proxy in one front-end and to
the live object in another gives one set of setters two contradictory
thread-safety contracts.

**A front-end that offers a control whose action is served by a host callback
SHALL supply that callback.** Where the shared tree falls back to calling the
renderer directly when a callback is absent, omitting it silently converts a
marshalled action into an unsynchronised one.

**A command that raises SHALL NOT be able to retire the owning thread.** The
loop that drains commands and advances the renderer SHALL survive an exception
raised by renderer state mutated concurrently, report it, and continue or
terminate the session visibly — never leave a session marked running with a
dead render thread. Terminating visibly means the clients attached to the
session are told, and every command still awaiting a reply is settled; stopping
the loop silently is indistinguishable from a slow frame.

**A render path SHALL leave no synchronisation primitive in a state that a
retry cannot recover from.** A guard that catches and retries is worthless if
the retry blocks: a frame fence reset before the exception-capable work that
precedes its submit stays unsignaled, so the next iteration waits on it forever.
Reset such a primitive immediately before the operation that signals it.

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

#### Scenario: A sub-object write is marshalled

- **WHEN** a control mutates state through an object reached from the renderer —
  setting playback state or scrubbing time on the playback clock
- **THEN** the mutation is applied on the owning thread and takes effect, rather
  than being absorbed by a proxy-local copy of that object and silently doing
  nothing

#### Scenario: A parameter edit cannot retire the render thread

- **WHEN** a parameter that inserts a key into a renderer-owned mapping is edited
  from a non-owning thread while the owning thread is computing accumulation
  state
- **THEN** the session keeps rendering — the mutation is marshalled so the
  concurrent-mutation error cannot arise, and if any command does raise, the loop
  reports it and the session does not remain marked running with no render thread

#### Scenario: Resolution changes are marshalled on every front-end

- **WHEN** a resolution control is used on any front-end that offers one
- **THEN** the render-target resize is applied on the owning thread, and no
  front-end reaches the renderer's resize path directly from its own thread —
  serialising it under a lock is not sufficient, since that leaves the
  destroy-and-recreate of the offscreen image, readback buffer, accumulation
  image and HUD overlay running on the caller's thread

#### Scenario: A retry after a failed frame is not blocked by the failed frame

- **WHEN** a render iteration raises after the frame fence has been waited on
  but before the submit that would signal it
- **THEN** the next iteration proceeds rather than blocking forever on that
  fence, so the failure counter can reach its limit and the session can give up
  visibly

#### Scenario: A terminal render failure reaches the client

- **WHEN** a session's render loop stops after repeated failures
- **THEN** each attached client is told the session failed, and any caller
  awaiting a command reply is settled rather than left to time out

#### Scenario: An unmarshalled renderer verb is refused, not passed through

- **WHEN** a front-end calls a renderer mutation verb on a marshalling proxy
  that has no marshalled implementation of it
- **THEN** the call is refused with an error naming the missing verb, rather
  than forwarded to the live renderer on the caller's thread
