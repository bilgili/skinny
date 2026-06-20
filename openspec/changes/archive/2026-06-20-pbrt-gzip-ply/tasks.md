## 1. Reader

- [x] 1.1 In `read_ply`, sniff the gzip magic (`0x1f 0x8b`) on the raw bytes and
  `gzip.decompress` before the `ply` check; leave the rest of the parse unchanged.

## 2. Tests

- [x] 2.1 Unit test: gzip a known binary PLY and assert `read_ply` returns the
  same `PlyMesh` as the uncompressed source (points/indices/normals/uvs).
- [x] 2.2 Unit test: same for an ascii PLY.
- [x] 2.3 Unit test: gzip wrapping non-PLY data raises `ValueError`.

## 3. Verify

- [x] 3.1 `skinny-import-pbrt -o assets/dragon.usda
  ../pbrt-v4-scenes/sssdragon/dragon_10.pbrt` reports `shape:plymesh` as `ok`
  (no longer skipped).
- [x] 3.2 Headless Metal render of the imported dragon; show the image.

## 4. Docs

- [x] 4.1 Note `.ply.gz` support in the pbrt importer PLY-support documentation.
