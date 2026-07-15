## ADDED Requirements

### Requirement: Corpus scene data is integrity-checked hostlessly

The hostless matrix tier SHALL verify, without a GPU, that the corpus scene
data the GPU gates depend on is self-consistent: for every manifest scene whose
`usd` asset exists on disk, every `texture:file` reference authored in that
asset SHALL resolve to an existing file (relative references resolve against
the asset's directory); and every manifest scene whose `.pbrt` source authors a
non-flat material or a named medium (`Material "subsurface"`,
`MakeNamedMedium`) SHALL declare a non-flat `material_class`, and every
manifest scene whose on-disk `.usda` asset authors a volume field
(`OpenVDBAsset`) SHALL declare `material_class: "volume"`. A scene whose
`usd` asset or `.pbrt` source is absent on the current checkout (e.g. a
worktree without the untracked asset tree, or a `usd:`-sourced scene whose
`file` name is informational) SHALL be skipped for that check, not failed.

#### Scenario: Deleted baked-env side-file is caught hostlessly

- **WHEN** a manifest scene's `.usda` asset references a baked
  `light_infinite_*_const.hdr` that has been deleted from `assets/`
- **THEN** the hostless integrity meta-test fails naming the scene and the
  dangling reference, before any GPU gate renders the scene against the wrong
  (fallback) environment

#### Scenario: Non-flat scene missing material_class is caught hostlessly

- **WHEN** a corpus `.pbrt` scene source authors `Material "subsurface"` but its
  manifest entry declares no `material_class`
- **THEN** the hostless integrity meta-test fails naming the scene, before the
  spectral envelope admits a spectral combo that the renderer refuses at
  scene-build time

### Requirement: Missing dome-light textures fail loudly at load

The USD loader SHALL emit a visible warning (stderr) when a `DomeLight` authors
a `texture:file` that does not resolve to an existing file, identifying the
prim and the dangling path, while retaining the existing fallback behavior
(no DomeLight upload; the built-in environment library remains active).

#### Scenario: Dangling DomeLight texture warns

- **WHEN** a USD scene whose DomeLight references a missing `.hdr` is loaded
- **THEN** a warning naming the DomeLight prim and the unresolved path is
  printed, and the scene still renders under the built-in environment

## MODIFIED Requirements

### Requirement: megakernel and wavefront produce the same image
For any integrator that runs in both execution modes, the harness SHALL assert
that the megakernel and wavefront renders of the same scene agree. The
exposure-aligned relMSE/FLIP between the two modes SHALL be within the tight
mode-equivalence tolerance for that scene — **except** where the two modes are
specified to run different algorithms: the subsurface interior walk (change
`pbrt-subsurface-3d-walk`) is a true 3D per-segment walk under the wavefront
mode and a watchdog-safe 1D slab under the megakernel, so a
subsurface-dominated scene MAY record a per-scene `self_consistency` `mode`
override, measured and tighten-only, with the measured delta noted in the
manifest entry. The override SHALL NOT be used to mask an unexplained
divergence on flat or volume scenes.

#### Scenario: Path megakernel matches Path wavefront
- **WHEN** the same scene is rendered with `(Path, megakernel)` and `(Path, wavefront)`
- **THEN** the exposure-aligned relMSE and FLIP between the two images are within
  the mode-equivalence tolerance

#### Scenario: BDPT megakernel matches BDPT wavefront
- **WHEN** the same scene is rendered with `(BDPT, megakernel)` and `(BDPT, wavefront)`
- **THEN** the exposure-aligned relMSE and FLIP between the two images are within
  the mode-equivalence tolerance

#### Scenario: Recorded subsurface mode divergence

- **WHEN** `subsurface_infinite` renders `path|megakernel` against the
  `path|wavefront` anchor at the manifest spp
- **THEN** the self-consistency gate asserts against the scene's recorded
  `mode` override (relMSE ≤ 0.05, FLIP ≤ 0.07; measured 0.0362/0.0554 at
  512 spp), and the gate fails if the divergence grows beyond it
