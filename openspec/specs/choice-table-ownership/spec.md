# choice-table-ownership Specification

## Purpose
TBD - created by archiving change choice-table-owners. Update Purpose after archive.
## Requirements
### Requirement: Each enumerated axis has one owner for its values, labels and indices

Each enumerated render axis SHALL have one owning table declaring, per entry,
its CLI token, its renderer index and its display label — covering integrator,
tonemap, execution mode, reuse mode, detail-map mode, ReSTIR combination mode
and proposal preset. The CLI's `choices`, the headless lookup tables, the renderer's
display lists, and the GUI-thread proxy's placeholder names and defaults SHALL
be projections of that table. No consumer may restate an axis's membership,
ordering or labels. The table SHALL be dependency-free so any consumer can
import it, and it MUST NOT absorb validity rules, which the render envelope
owns.

#### Scenario: Adding an axis value touches one table

- **WHEN** a new integrator is added
- **THEN** its CLI token, renderer index and display label come from the single
  table, and no separate list needs editing to make it appear correctly in the
  CLI, the headless driver, the renderer's display list, or the GUI proxy

#### Scenario: No mirrored axis list remains

- **WHEN** the source tree is searched for literal lists or dicts of axis
  values, labels or indices outside the owning table
- **THEN** none remain — including the duplicated integrator-index dict in the
  headless driver, the second tonemap dict and its argparse choices, and the
  17 placeholder choice-name lists and 8 hardcoded defaults in the render
  session proxy

#### Scenario: Drifted labels are corrected and listed

- **WHEN** the pre-change label and value lists are compared against the table
- **THEN** every divergence is enumerated and individually resolved — notably
  the missing MLT integrator label, the single-entry tonemap placeholder
  against the real four-entry list, and the six divergent proxy placeholder
  lists — with each verified as drift rather than a deliberate stub before it
  is changed

