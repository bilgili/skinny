# renderer-module-structure (delta)

## ADDED Requirements

### Requirement: The per-frame path is scene sync, a pure frame plan, and execution

The renderer's per-frame path SHALL be split into three stages: scene sync
(the state-advancing work currently in `update`), a **pure** frame plan that
derives the frame's decisions as an inspectable value, and execution of that
plan against a target. The plan SHALL name the execution mode, the pass
sequence, the accumulation state and reset decision, any dispatch banding or
tiling, and which optional per-frame work is performed — and MUST hold no
device handles, so it can be derived and asserted with no GPU present. The
windowed and headless paths SHALL share one execution body and differ only in
their target; the barrier, execution-mode gate and dispatch block that are
currently duplicated between them MUST NOT remain duplicated. The plan SHALL
consume the accumulation reset decision from the parameter-registry owner
rather than re-deriving it. Dispatch sequence and rendered images MUST be
unchanged.

#### Scenario: The frame plan is derived without a device

- **WHEN** a frame plan is derived from renderer state in a process with no
  GPU device
- **THEN** it is produced and its pass sequence, execution mode and
  accumulation decision can be asserted

#### Scenario: Windowed and headless share one dispatch body

- **WHEN** the same frame plan is executed against a windowed target and an
  offscreen target
- **THEN** the recorded dispatch sequence is identical, and the two paths
  differ only in output destination, swapchain acquisition and presentation,
  and readback

#### Scenario: Ordering constraints are asserted, not implied

- **WHEN** the plan's step order is inspected
- **THEN** the constraints that are currently implicit in line order — notably
  that the pick-result drain precedes uniform packing — are expressed in the
  plan and asserted by test

#### Scenario: Images are unchanged by the split

- **WHEN** the parity matrix's pbrt-truth and self-consistency gates run
  before and after the split
- **THEN** the results are identical, not merely within tolerance
