# renderer-command-interface (delta)

## ADDED Requirements

### Requirement: One interface drives the renderer from every caller

Driving a renderer SHALL be done through one command interface — post,
post-with-reply, coalescing by key, drain on the thread that owns the renderer
— used by every front-end and every tool. No caller SHALL mutate a live
renderer from a thread that does not own it, and no front-end SHALL invent a
second mutation mechanism. The shared control tree SHALL carry one
thread-safety contract regardless of which front-end mounts it: the Qt
front-end binds it to a marshalling proxy while the web front-end currently
binds it to the live renderer, so the same setter is safe under one and racy
under the other.

#### Scenario: The shared control tree is safe under every front-end

- **WHEN** the shared control tree is built for any front-end and a setter
  fires
- **THEN** the setter posts a command, and the mutation is applied on the
  renderer's owning thread

#### Scenario: No unsynchronised write to a live renderer remains

- **WHEN** the source tree is searched for direct attribute writes into a
  renderer from a front-end thread that does not own it — including the web
  session's parameter path and the panel backend's widget setters
- **THEN** none remain

#### Scenario: Every mutation can report an outcome

- **WHEN** a caller needs to know whether a mutation succeeded
- **THEN** post-with-reply returns a settled result carrying success or the
  error, on every front-end, as the scene-control tool surface already relies
  on today

### Requirement: Command paths are covered without a device

The command paths of all front-ends SHALL be exercised by hostless tests
against a stub renderer, in the way the scene-control tool surface already is.
Coverage MUST NOT depend on constructing a real GPU context, since that is why
the web session's mutation path is untested today.

#### Scenario: Web command path is tested without a GPU

- **WHEN** the web session's parameter, camera, control and resize paths are
  driven against a stub renderer with no GPU device present
- **THEN** the posted commands and their ordering can be asserted, and the
  test neither skips nor requires a device

#### Scenario: Debug-camera actions are commands, not widget-tree pokes

- **WHEN** a browser client triggers a Camera Debug action
- **THEN** it is delivered as a posted command, and no code reaches into a
  widget tree by index to synthesise a click
