# wavefront-kernel-ownership (delta)

## ADDED Requirements

### Requirement: Wavefront kernel names and shared pass constants have one owner

The wavefront kernel entry-point names SHALL be declared once in a
backend-neutral table imported by the driver and by both backend pass modules,
so that renaming a kernel is an import-time failure rather than a runtime one.
Pass constants that must be equal across the backends — bounce counts, stream
capacities, vertex and auxiliary strides, walk modes, reservoir stride, and the
ReSTIR default configuration — SHALL have one home. Constants that legitimately
differ per backend SHALL remain separate but MUST be pinned by a test that
states the reason for the difference; equality MUST NOT be forced on values
that are per-backend by design, such as the record-stack sizing formula and the
backend-specific rebuild-key elements. The dispatched kernel names and constant
values SHALL be byte-identical to before this change.

#### Scenario: A kernel rename cannot leave a backend behind

- **WHEN** a wavefront kernel entry point is renamed
- **THEN** every consumer resolves it from the shared table, and a stale name
  fails at import rather than producing a runtime dispatch failure on one
  backend

#### Scenario: Shared constants cannot drift

- **WHEN** a shared pass constant is changed
- **THEN** both backends observe the new value, and any constant that is
  deliberately per-backend is covered by a test naming why it differs

#### Scenario: No kernel-name literal remains outside the owner

- **WHEN** the driver and the two backend pass modules are searched for kernel
  entry-point string literals
- **THEN** none remain outside the owning table, and the golden test confirms
  each owned name equals its historical string
