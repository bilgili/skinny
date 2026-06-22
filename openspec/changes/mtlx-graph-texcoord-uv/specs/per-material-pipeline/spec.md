## ADDED Requirements

### Requirement: Graph vertex inputs are rewritten or the graph falls back

The material code generator SHALL rewrite every MaterialXGenSlang vertex-data
input (`vd.*`) referenced by a graph fragment into the fragment function's
supplied parameters: object/world position → `P_in`, world normal → `N_in`,
world tangent → `T_in`, and the texture coordinate of the default UV set into
`UV_in`. The default UV set SHALL be recognised in both forms MaterialXGenSlang
emits — the explicit `<geompropvalue geomprop="UVMap">` form (`vd.i_geomprop_
UVMap`) and the default `<texcoord>` form (`vd.texcoord_0`) produced when an
`<image>` node has no explicit texture-coordinate input.

If, after all known rewrites, a graph body still references any `vd.*` vertex
input the generator does not pipe (for example a secondary UV set or a vertex
color set), the generator SHALL NOT emit that graph fragment. It SHALL instead
return no fragment so the material is routed through the flat / std_surface
parameter-buffer path, and SHALL surface the unhandled input. A graph fragment
SHALL never be emitted with an unresolved `vd.*` identifier that would fail to
compile.

#### Scenario: Default-UV image graph shades from mesh UVs

- **WHEN** a MaterialX graph drives a `standard_surface` input from an `<image>`
  node that has no explicit texture-coordinate input (the default UV set,
  emitted by MaterialXGenSlang as `vd.texcoord_0`)
- **THEN** the emitted graph fragment references `UV_in` for that lookup and
  contains no bare `vd.` identifier, and the material's shading pipeline compiles

#### Scenario: Unhandled vertex input falls back instead of failing to compile

- **WHEN** a MaterialX graph references a vertex input the generator does not
  pipe (a `vd.*` other than the supported position/normal/tangent/default-UV
  forms)
- **THEN** the generator emits no graph fragment for that material and the
  material is shaded through the flat / std_surface parameter-buffer path,
  rather than emitting a module that fails to compile
