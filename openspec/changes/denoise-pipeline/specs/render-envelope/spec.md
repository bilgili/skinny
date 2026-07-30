## ADDED Requirements

### Requirement: Denoiser selection is an envelope rule

The render-envelope predicate SHALL own the denoiser rules, so the CLI guards
read them and do not restate them. The envelope query SHALL carry the requested
denoiser name and the denoise scale, and the predicate SHALL emit a distinct
reason code for each of:

- the resolved GPU backend cannot run the named denoiser;
- the named denoiser's optional dependency is not installed;
- a denoise scale other than 1.0 is requested with no denoiser.

Each new code SHALL be listed in the canonical code order and SHALL be owned by
exactly one consumer — a CLI guard, the renderer scene gate, or the recorded
unowned set — so the existing partition test keeps covering every code.

The parity matrix SHALL never set a denoiser in a query, so the swept set of
combinations and every recorded baseline SHALL be unchanged.

#### Scenario: Every denoiser code has an owner

- **WHEN** the code partition is checked
- **THEN** each new denoiser code appears in exactly one of the CLI guard sets,
  the renderer scene set, or the recorded unowned set

#### Scenario: The predicate reports all denoiser violations

- **WHEN** a query names a denoiser the backend cannot run and whose dependency
  is also missing
- **THEN** the verdict lists both violations, in canonical order, rather than
  only the first

#### Scenario: The matrix is unchanged

- **WHEN** the parity matrix is enumerated before and after this change
- **THEN** the set of valid combinations and their recorded skip reasons are
  identical
