# metal-backend — Delta (combine-graph-param-buffers)

## ADDED Requirements

### Requirement: Argument-table budget independent of graph count

The Metal wavefront buffer argument table SHALL stay within the 31-slot cap
independently of how many distinct MaterialX graph materials the scene carries:
graph params SHALL occupy exactly one buffer slot for any graph count. The
neural directional proposal with online training (record emission) active SHALL
build on the Metal backend for scenes with multiple graph materials, where it
previously overflowed the argument table. A build that still exceeds 31 buffers
SHALL fail with a clear error naming the kernel and slot count rather than a raw
`SLANG_FAIL`.

#### Scenario: Neural online training builds on a multi-graph scene

- **WHEN** the wavefront path pass is built on a Metal host for a scene with two
  or more MaterialX graph materials, with the neural proposal and online training
  active
- **THEN** every shade kernel's buffer count is at most 31, the pipelines create
  successfully, and the heaviest kernel's slot count is logged

#### Scenario: Over-budget build fails clearly

- **WHEN** a Metal wavefront build would exceed the 31-buffer argument-table cap
- **THEN** the renderer raises an error naming the kernel and its buffer count,
  instead of surfacing an unlabelled Metal pipeline-creation failure
