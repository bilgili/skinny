# renderer-output-fidelity (delta)

## ADDED Requirements

### Requirement: Composited overlays belong to the frame they are composited into

An overlay copied into a rendered frame SHALL have been prepared for that frame.
Where an overlay's staging buffer is filled on one render path and copied on
several, every path that copies it SHALL also fill it, so a frame produced by any
path composites current content rather than whatever a previous frame left in
staging. This covers the HUD text overlay, which is copied by both the windowed
and the offscreen paths but filled only by the windowed one.

#### Scenario: An offscreen frame composites current overlay content

- **WHEN** an offscreen frame is produced after the overlay content has changed —
  a screenshot taken from a session whose HUD text differs from the previous
  frame's
- **THEN** the output shows the current content, not the previous frame's

#### Scenario: Every copying path fills what it copies

- **WHEN** the render paths are compared for overlay staging
- **THEN** no path copies a staging buffer it does not fill, and a path added
  later that copies without filling fails a test

### Requirement: No per-frame descriptor write without a differing target

A descriptor write SHALL NOT be performed per frame when the binding already
points at the resource being written. Such a write is dead work whose presence
implies a target difference between render paths that does not exist, and whose
accompanying comments misdescribe the other path. The offscreen path's per-call
rewrite of the output-image binding is one instance: every write of that binding
targets the same offscreen image, and the windowed path blits rather than
rebinding.

#### Scenario: The offscreen path does not rebind what is already bound

- **WHEN** an offscreen frame is rendered
- **THEN** no descriptor write is issued for a binding whose current target is
  already the resource that would be written

#### Scenario: Comments describe what the code does

- **WHEN** a comment states which resource a render path binds a descriptor to
- **THEN** that path performs that binding, or the comment is corrected — the
  windowed path is not described as binding the output image to an acquired
  swapchain image when it blits to it instead
