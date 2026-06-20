## ADDED Requirements

### Requirement: gzip-compressed PLY input is read transparently

`read_ply` SHALL detect a gzip-compressed PLY by its magic bytes (`0x1f 0x8b`)
at the start of the file — independent of the filename — and decompress it in
memory before parsing. After decompression the existing ascii,
binary-little-endian, and binary-big-endian parse paths apply unchanged, and the
returned `PlyMesh` (points, indices, optional normals/UVs) SHALL be identical to
reading the equivalent uncompressed PLY. Uncompressed PLY input SHALL be parsed
exactly as before.

#### Scenario: gzipped binary PLY (`.ply.gz`)

- **WHEN** `read_ply` is given a path whose contents are a gzip stream wrapping a
  binary-little-endian PLY (e.g. pbrt's `dragon.ply.gz`)
- **THEN** the mesh is decompressed and parsed, returning the same points,
  indices, normals, and UVs as the uncompressed PLY

#### Scenario: gzipped ascii PLY

- **WHEN** `read_ply` is given a gzip stream wrapping an ascii PLY
- **THEN** the mesh is decompressed and parsed, equal to the uncompressed result

#### Scenario: uncompressed PLY unaffected

- **WHEN** `read_ply` is given an uncompressed PLY (ascii or binary)
- **THEN** the result is identical to the prior behavior (no decompression path
  taken)

#### Scenario: gzip wrapping non-PLY data still errors

- **WHEN** `read_ply` is given a gzip stream whose decompressed contents are not
  a PLY (first line is not `ply`)
- **THEN** a `ValueError` is raised, as for an uncompressed non-PLY file
