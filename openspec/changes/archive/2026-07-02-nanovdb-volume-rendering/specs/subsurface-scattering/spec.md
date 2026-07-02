# subsurface-scattering (delta)

## MODIFIED Requirements

### Requirement: Volume transport is forward-compatible with heterogeneous free-standing media

The volume transport and medium representation SHALL be chosen so that
heterogeneous, free-standing participating media (e.g. the pbrt `disney-cloud`
NanoVDB model) plug in **without reworking the transport loop**. Specifically: the
random walk SHALL read the medium ONLY through a density seam — `densityAt(medium, p)`
(local density multiplier) and `mediumMajorant(medium, segment)` — dispatched by a
medium `kind` tag, so a volume source is a new `kind` plus two `case` bodies with no
change to the transport equations, the walk, NEE, RR, or integrator wiring. The medium
SHALL be a handle-referenced registry entry (not hardwired to a surface's
interior), the transport SHALL be majorant /
null-collision (Woodcock) tracking (so a constant `σ_t` is the degenerate case of a
spatially-varying density field), the boundary crossing SHALL be parameterized by
mode (dielectric refract vs index-matched pass-through), the per-collision
throughput SHALL be per-channel (`float3`), and the phase function SHALL be the
general Henyey-Greenstein already used. This contract is now exercised by two kinds:
`MEDIUM_HOMOGENEOUS` (the dielectric-bounded subsurface interior) and
`MEDIUM_NANOVDB` (grid-density free-standing media attached via `MediumInterface`
with index-matched boundaries — see the `heterogeneous-media` capability).
Homogeneous subsurface behavior SHALL be unchanged by the addition of the
heterogeneous kind. Spectral σ remains out of scope.

#### Scenario: homogeneous interior is the degenerate null-collision case

- **WHEN** a homogeneous subsurface interior is transported
- **THEN** it is handled by the same majorant/null-collision walk the heterogeneous
  grid kind uses (with `σ_max = σ_t` and a constant density), so no closed-form
  homogeneous-only transmittance path exists that the grid kind cannot reuse

#### Scenario: medium is reachable independently of the bounding surface material

- **WHEN** the renderer resolves the interior medium at a subsurface hit
- **THEN** it does so through a medium handle into a medium registry (not by reading
  the bounding material's BRDF params), which is how a free-standing
  `MediumInterface` registers and attaches a named medium to geometry through the
  same registry without changing the walk

#### Scenario: distinct media of different kinds coexist without conflict

- **WHEN** a scene contains two disjoint media of different kinds (a homogeneous
  subsurface object and a heterogeneous free-standing volume)
- **THEN** each is a separate registry entry resolved by its own handle/kind and
  transported by the same segment walk with its own `(kind, boundaryMode)`, with no
  shared mutable state — so they render correctly together; and the per-segment
  traversal is factored as a standalone function the free-standing path-loop reuses
  unchanged (overlapping / nested media, needing a medium priority stack, remain
  explicitly out of scope)

#### Scenario: adding the heterogeneous kind leaves homogeneous rendering unchanged

- **WHEN** the pre-existing subsurface parity scenes render after `MEDIUM_NANOVDB`
  lands
- **THEN** results are unchanged (same seeds → same image) and every subsurface gate
  stays green
