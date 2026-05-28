## ADDED Requirements

### Requirement: Remembered directory per loader category
The system SHALL track the most recently used directory independently for each
file-loader category: `model`, `ibl`, and `lens`. A file-open dialog for a
category SHALL open at that category's remembered directory.

#### Scenario: Reopen at last-used directory
- **WHEN** the user has previously loaded a file of a category from directory `D`
  and reopens the dialog for that category
- **THEN** the dialog's initial directory is `D`

#### Scenario: Successful pick records the directory
- **WHEN** the user picks a file at path `D/file.ext` from a categorized dialog
- **THEN** the parent directory `D` is recorded as that category's last-used
  directory

#### Scenario: Cancelled pick does not change the remembered directory
- **WHEN** the user opens a categorized dialog and cancels without picking a file
- **THEN** the category's remembered directory is unchanged

### Requirement: Type-specific default fallback
The system SHALL open a category's dialog at that category's default directory
when the category has no remembered directory or the remembered directory no
longer exists on disk. Defaults are anchored at the repository root: `assets/`
for `model`, `hdrs/` for `ibl`, `lenses/` for `lens`.

#### Scenario: First-ever use of a category
- **WHEN** no directory has ever been recorded for a category
- **THEN** the dialog opens at that category's default directory

#### Scenario: Remembered directory was deleted
- **WHEN** a category's remembered directory no longer exists on disk
- **THEN** the dialog opens at that category's default directory instead

### Requirement: Shared model directory across model entry points
The sidebar "Load scene…" picker and the "File ▸ Open scene" menu SHALL share a
single `model` remembered directory.

#### Scenario: Menu load updates sidebar picker start dir
- **WHEN** the user loads a model via the "File ▸ Open scene" menu from directory `D`
- **THEN** the sidebar "Load scene…" picker subsequently opens at `D`

### Requirement: Persistence across restarts
Remembered directories SHALL persist in `~/.skinny/settings.json` under a
`last_dirs` key and be restored on the next launch. A directory SHALL be persisted
at the time it is recorded (write-through), independent of any shutdown hook.

#### Scenario: Survives restart
- **WHEN** the user records a directory for a category and later relaunches the app
- **THEN** the dialog for that category opens at the recorded directory

#### Scenario: Survives a front-end without a shutdown hook
- **WHEN** a directory is recorded in a front-end that performs no save on exit
- **THEN** the recorded directory is already written to `settings.json`

#### Scenario: Recording preserves other settings
- **WHEN** a directory is recorded
- **THEN** existing keys in `settings.json` (params, camera, geometry) are retained

### Requirement: Both front-ends honor the registry
Both the Qt and Panel/web front-ends SHALL resolve categorized file-dialog start
directories from the shared registry and record the chosen directory on a
successful pick.

#### Scenario: Qt front-end
- **WHEN** the Qt front-end opens a categorized dialog
- **THEN** the start directory comes from the registry and a successful pick is recorded

#### Scenario: Panel front-end
- **WHEN** the Panel front-end opens a categorized dialog
- **THEN** the start directory comes from the registry and a successful pick is recorded
