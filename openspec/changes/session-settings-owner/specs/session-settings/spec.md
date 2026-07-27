# session-settings (delta)

## ADDED Requirements

### Requirement: Persisted settings are merged entry-wise, never replaced wholesale

Writing session settings SHALL merge into the settings already on disk,
preserving state the writer does not own. A front-end MUST NOT be able to
delete another front-end's persisted state by exiting. Merging SHALL descend
into the two keys that hold sub-dictionaries assembled from partial sources —
the parameter snapshot and the last-used-directory map — because preserving
those keys without their entries does not preserve the state: the parameter
snapshot omits every dynamic parameter when no skin material is loaded, and the
directory map is written from a process-local cache seeded once at first use.
Every other key replaces. The load guard SHALL also cover a non-UTF-8 settings
file, which today raises an uncaught decode error — harmless while it only
occurs at startup, but fatal once the same read happens on the save path.

#### Scenario: Exiting one front-end preserves the other's keys

- **WHEN** `skinny` writes its settings, then `skinny-gui` is opened and
  closed, and the settings file is inspected
- **THEN** the window position and the neural-handoff, neural-trainer,
  train-precision and online-training keys are still present with their prior
  values

#### Scenario: The reverse direction is equally safe

- **WHEN** `skinny-gui` writes its settings, then `skinny` is opened and closed
- **THEN** the open-dock list, last-used directories, section states and the Qt
  geometry and dock-state blobs are still present with their prior values

#### Scenario: A partial snapshot does not erase what it omits

- **WHEN** a front-end's renderer snapshot request times out and its written
  dictionary therefore omits the parameter, camera and gizmo keys entirely
- **THEN** the previously persisted values of those keys survive

#### Scenario: Sub-dictionary entries survive a partial writer

- **WHEN** settings are written by a session whose parameter snapshot contains
  no dynamic parameters because no skin material is loaded, or whose
  last-used-directory map was seeded before another session recorded a
  directory
- **THEN** the previously persisted dynamic parameters and the other session's
  recorded directory are still present

#### Scenario: A corrupt settings file does not propagate

- **WHEN** the settings file is unreadable, not valid JSON, or not valid UTF-8,
  and a front-end writes settings
- **THEN** the write succeeds with the writer's keys and no exception escapes,
  on the save path as well as at startup

### Requirement: One camera restore rule and one captured parameter set

Camera restore SHALL go through the camera's own distance-setting method, which
already clamps to the minimum and raises the distance cap to fit — replacing
the two divergent hand-rolled rules that bypass it today, one of which
additionally clamps to a fixed maximum that guards nothing, since the front-end
that applies it re-reads the live cap on every widget interaction. The snapshot
SHALL capture the full parameter set rather than the visibility-filtered set,
so that a scene with authored lighting does not discard the fallback-light
parameters at capture time. No restore-time visibility filter SHALL be added:
restore runs before the asynchronous scene load, so the filter could never
fire, and fallback-light parameters are already inert under authored lighting
at the point of use.

#### Scenario: A wide orbit distance survives both front-ends

- **WHEN** a camera with an orbit distance beyond the default cap is persisted
  and restored in either interactive front-end, with no scene loaded
- **THEN** the restored distance is the persisted one and the cap is adjusted
  to accommodate it, in both front-ends

#### Scenario: Authored lighting does not discard parameters at capture

- **WHEN** settings are captured while a scene with authored lighting is loaded
- **THEN** the fallback-light parameters are present in the written snapshot
  with their current values, in both front-ends

### Requirement: The documented persistence contract is true

Persistence claimed by the command-line help SHALL hold for the front-ends the
help names, or the help SHALL be corrected. A key SHALL NOT be written by a
front-end that cannot restore it: writing without restoring would overwrite the
other front-end's persisted value with an argument default on every exit, which
merge-on-write cannot prevent for a key that is written. Where a front-end
reads renderer-owned state through a request that may time out, the written
value SHALL come from the settled request and never from a fallback that
carries a startup-resolved value in place of the live one. The persisted
online-training value SHALL be the user's request, not the result after a
prerequisite gate refused it.

#### Scenario: A key is persisted only if it is also restored

- **WHEN** a front-end writes a persisted key
- **THEN** that front-end also restores it, applying the same
  command-line-and-environment-over-persisted precedence the other interactive
  front-end applies

#### Scenario: Help text names only front-ends that persist

- **WHEN** the command-line help states that an option is persisted on the
  interactive front-ends
- **THEN** every front-end covered by that phrase persists it, and a front-end
  that persists nothing by design is excluded from the phrase

#### Scenario: A refused prerequisite does not erase the request

- **WHEN** online training is requested but refused at startup because its
  prerequisites are not met, and the session then exits
- **THEN** the persisted value still records the request, so a later session
  that does meet the prerequisites honours it
