# session-settings (delta)

## ADDED Requirements

### Requirement: Persisted settings are merged, never replaced wholesale

Writing session settings SHALL merge the supplied keys into the settings
already on disk, preserving keys the writer does not own. A front-end MUST NOT
be able to delete another front-end's persisted state by exiting. Today
`save_settings` writes the supplied dict wholesale, and the two interactive
front-ends author overlapping-but-different key sets — 11 each, intersecting in
6 — so each erases five of the other's keys on exit.

#### Scenario: Exiting one front-end preserves the other's keys

- **WHEN** `skinny` writes its settings, then `skinny-gui` is opened and
  closed, and the settings file is inspected
- **THEN** `vulkan_window`, `neural_handoff`, `neural_trainer`,
  `train_precision` and `online_training` are still present with their prior
  values

#### Scenario: The reverse direction is equally safe

- **WHEN** `skinny-gui` writes its settings, then `skinny` is opened and
  closed
- **THEN** `open_docks`, `last_dirs`, `section_states`, `qt_geometry` and
  `qt_dock_state` are still present with their prior values

#### Scenario: A corrupt settings file does not propagate

- **WHEN** the settings file is unreadable or not valid JSON and a front-end
  writes settings
- **THEN** the write succeeds with the writer's keys and no exception escapes

### Requirement: One owner declares the session snapshot schema

The session snapshot SHALL be declared by one module that both interactive
front-ends consume, with a shared section (parameters, camera, gizmo mode,
backend, encoding, SPPM glossy roughness) and front-end-contributed sections
that the module preserves without interpreting. Front-ends MUST NOT re-author
the schema. Capture and restore SHALL be covered by hostless tests — there is
no test for either today.

#### Scenario: Capture and restore round-trip

- **WHEN** a session snapshot is captured from a stub renderer and restored
  onto an equivalent stub
- **THEN** every shared-section value round-trips exactly, and every
  contributed section is preserved verbatim

#### Scenario: Both front-ends' key sets are pinned

- **WHEN** the key set each interactive front-end contributes is compared
  against the declared schema
- **THEN** every contributed key is accounted for, and adding a key to one
  front-end without declaring it fails the test

### Requirement: One camera restore rule and one captured parameter set

Camera restore SHALL follow a single rule across front-ends, replacing the two
that disagree today — one raising the orbit distance cap to fit the restored
distance, the other clamping the distance to 50 and ignoring the cap. The
snapshot SHALL capture the full parameter set rather than the
visibility-filtered set, so that a scene with authored lighting does not
silently discard the fallback-light parameters at capture time; visibility
filtering applies to display and restore, not to what is stored.

#### Scenario: A wide orbit distance survives both front-ends

- **WHEN** a camera with an orbit distance beyond the default cap is persisted
  and restored in either interactive front-end
- **THEN** the restored distance is the persisted one, and the cap is adjusted
  to accommodate it

#### Scenario: Authored lighting does not discard parameters

- **WHEN** settings are captured while a scene with authored lighting is
  loaded, and later restored under a scene using fallback lighting
- **THEN** the fallback-light parameters are present with their persisted
  values

#### Scenario: Documented persistence holds on both front-ends

- **WHEN** `--neural-trainer`, `--train-precision` or `--online-training` is
  set and the front-end exits
- **THEN** the value is persisted and reapplied on the next start of either
  interactive front-end, matching what the CLI help states
