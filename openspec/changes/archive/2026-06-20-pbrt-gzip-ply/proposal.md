## Why

The pbrt v4 importer's PLY reader (`read_ply`) reads the file as raw bytes and
checks the first line for the `ply` magic. pbrt v4 scenes ship large meshes
**gzip-compressed** with a `.ply.gz` filename (e.g.
`sssdragon/geometry/dragon.ply.gz`, 69 MB packed / 137 MB raw), and pbrt-v4
itself transparently gunzips any `*.gz` PLY at load. Our reader does not, so the
gzip magic (`1f 8b`) fails the `ply` check and the shape is dropped:

```
[!  ] shape:plymesh — failed to read .../dragon.ply.gz:
      .../dragon.ply.gz: not a PLY file
```

The sssdragon imports without its subject mesh. This is a correctness gap: the
importer's purpose is pbrt→USD parity, and gzipped PLY is a first-class pbrt
input.

## What Changes

- `read_ply` SHALL transparently decompress a gzip-compressed PLY: detect the
  gzip magic bytes (`0x1f 0x8b`) at the head of the file (independent of the
  filename) and gunzip before parsing, then proceed through the existing
  ascii / binary-little / binary-big paths unchanged.
- Uncompressed PLY input is byte-for-byte unaffected (same parse path, same
  results).
- No renderer, USD-loader, or shader change; no new dependency (`gzip` is in the
  Python stdlib).

## Capabilities

### New Capabilities
- `pbrt-plymesh-gzip`: Transparent gzip decompression of `plymesh` PLY input in
  the pbrt importer, gated on the gzip magic bytes so both `.ply` and `.ply.gz`
  load identically.

## Impact

- **Code**: `src/skinny/pbrt/ply.py` (`read_ply` — magic-byte sniff + gunzip).
- **Tests**: `tests/` — unit test that a gzipped round-trip of a known mesh reads
  identically to the uncompressed mesh (ascii + binary), and that a corrupt /
  non-PLY gzip still raises `ValueError`.
- **Docs**: pbrt importer PLY-support note gains `.ply.gz`.
- **Behavior**: scenes referencing `.ply.gz` (sssdragon, etc.) now import their
  mesh; uncompressed PLY scenes unchanged.
