## ADDED Requirements

### Requirement: A Metal denoiser obeys the dispatch-hygiene rules

A denoiser that submits Metal work SHALL be bound by the same hygiene rules as
every other piece of GPU work on this platform.

The denoiser SHALL be destroyed in the renderer's teardown sequence, before the
Metal device closes. Its teardown SHALL be idempotent, so a repeat from the
context manager, from the `atexit` hook, or from a signal handler is a no-op.

Denoiser work SHALL be bounded by construction. One denoise call SHALL submit
one bounded piece of work over one image pair, and SHALL NOT loop over a
per-pixel budget that grows with scene content.

A denoiser that submits on its own Metal command queue SHALL be ordered against
the render passes by the renderer's existing device-idle wait, and SHALL NOT
leave work in flight when teardown starts.

The Metal cleanup harness SHALL pass with a denoiser active before this change
merges, because the change adds GPU work and changes context lifecycle.

#### Scenario: Teardown releases the denoiser before the device closes

- **WHEN** a renderer with an active Metal denoiser is destroyed
- **THEN** the denoiser's Metal objects are released before `Device.close()`, and
  a second destroy is a no-op

#### Scenario: Teardown drains in-flight denoiser work

- **WHEN** teardown begins while a denoise submission is in flight
- **THEN** teardown waits for the device to go idle before releasing the
  denoiser's Metal objects

#### Scenario: The cleanup harness passes with a denoiser active

- **WHEN** the guarded Metal cleanup harness runs with a denoiser active
- **THEN** every probe passes and a fresh Metal context constructs afterwards
