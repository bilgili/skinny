## MODIFIED Requirements

### Requirement: Selectable regimes and front-end selection

ReSTIR DI SHALL expose its reuse regime as configuration: spatial reuse on/off
and temporal reuse off/progressive/reprojected. This change SHALL implement the
spatial and progressive-temporal regimes; the reprojected regime SHALL be
reserved in the configuration enum but MAY be unimplemented (falling back) until
a follow-on change adds the motion-vector subsystem. The reuse mode SHALL be
selectable on every front-end (the `reuse_modes` list gains `"ReSTIR DI"`,
surfaced by the data-driven UI) and persisted in settings; changing the reuse
mode or any ReSTIR config value SHALL reset progressive accumulation.

The `Reuse` selector together with the ReSTIR config controls (regime, the
biased-combine toggle, the `M light` / `M bsdf` candidate counts, the spatial
neighbour count and radius, and the `M cap`) SHALL be presented in a dedicated
**ReSTIR** control group, separate from the general Render group, on every
interactive front-end — the windowed app, the Qt GUI, the web panel, and the
debug viewport. The group SHALL be defined once in the shared control tree so the
front-ends stay layout-identical, and SHALL remain present regardless of the
active reuse mode so the user can switch into ReSTIR from it.

#### Scenario: Reprojected mode is reserved but not yet active

- **WHEN** the reprojected temporal regime is selected before its follow-on
  change lands
- **THEN** ReSTIR falls back to a supported temporal regime rather than failing

#### Scenario: Changing ReSTIR config resets accumulation

- **WHEN** the reuse mode or a ReSTIR config value (candidate counts, neighbor
  count/radius, M-cap, biased toggle, regime) changes
- **THEN** progressive accumulation resets so the new configuration converges
  cleanly

#### Scenario: ReSTIR controls live in a dedicated group on every front-end

- **WHEN** any interactive front-end (the windowed app, the Qt GUI, the web
  panel, or the debug viewport) builds its control panel
- **THEN** the `Reuse` selector and the ReSTIR tuning controls appear together
  under a single dedicated **ReSTIR** group, separate from the Render group, and
  that group is present even when the active reuse mode is identity
