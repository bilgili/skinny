"""On-disk BVH cache for baked meshes.

Stores zstd-compressed mesh buffers (vertex, index, BVH) keyed by a content
hash of the source geometry + bake parameters. An index.json file holds
metadata so cache hits can be validated without decompressing blobs.

Cache layout::

    ~/.skinny/mesh_cache/
        index.json                              # key → metadata mapping
        <geometry_hash>_<params_hash>.skmc      # zstd(vertex + index + bvh)
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import time
from pathlib import Path
from typing import TYPE_CHECKING

import zstandard as zstd

if TYPE_CHECKING:
    from skinny.mesh import Mesh, MeshSource

CACHE_DIR = Path.home() / ".skinny" / "mesh_cache"
INDEX_FILE = CACHE_DIR / "index.json"
_INDEX_VERSION = 1
_ZSTD_LEVEL = 3
_TAG = "[skinny:cache]"


def _fmt_size(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n / (1024 * 1024):.1f} MB"


def make_cache_key(
    source_hash: str,
    displacement_bytes: bytes | None,
    displacement_res: int,
    displacement_scale_world: float,
    normal_bytes: bytes | None,
    normal_res: int,
    normal_map_strength: float,
) -> str:
    """Build a two-tier structured cache key: geometry_params."""
    h = hashlib.sha256()
    h.update(struct.pack(
        "<i f i f",
        displacement_res,
        round(displacement_scale_world, 6),
        normal_res,
        round(normal_map_strength, 6),
    ))
    if displacement_bytes is not None:
        h.update(displacement_bytes)
    if normal_bytes is not None:
        h.update(normal_bytes)
    params_hash = h.hexdigest()[:16]
    return f"{source_hash[:16]}_{params_hash}"


def load_cache_index() -> dict:
    """Read index.json at startup. Returns empty dict on missing/corrupt."""
    if not INDEX_FILE.exists():
        print(f"{_TAG} no index found at {CACHE_DIR}")
        return {}
    try:
        data = json.loads(INDEX_FILE.read_text(encoding="utf-8"))
        if data.get("version") != _INDEX_VERSION:
            print(f"{_TAG} index version mismatch, starting fresh")
            return {}
        entries = data.get("entries", {})
        total_compressed = sum(e.get("compressed_size", 0) for e in entries.values())
        print(
            f"{_TAG} loaded index: {len(entries)} entries, "
            f"{_fmt_size(total_compressed)} on disk @ {CACHE_DIR}"
        )
        return entries
    except Exception as exc:  # noqa: BLE001
        print(f"{_TAG} corrupt index ({exc}), starting fresh")
        return {}


def _save_index(entries: dict) -> None:
    """Atomically write index.json."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        {"version": _INDEX_VERSION, "entries": entries},
        indent=2,
        sort_keys=True,
    )
    tmp = INDEX_FILE.with_suffix(".json.tmp")
    tmp.write_text(payload, encoding="utf-8")
    os.replace(tmp, INDEX_FILE)


def lookup_cached_mesh(
    index: dict,
    cache_key: str,
    source: "MeshSource",
) -> "Mesh | None":
    """Check index in memory; if hit, read + decompress blob and return Mesh."""
    entry = index.get(cache_key)
    if entry is None:
        return None

    blob_path = CACHE_DIR / f"{cache_key}.skmc"
    if not blob_path.exists():
        print(f"{_TAG} stale entry {cache_key} — blob missing, removing from index")
        index.pop(cache_key, None)
        _save_index(index)
        return None

    try:
        t0 = time.monotonic()
        compressed = blob_path.read_bytes()
        raw = zstd.decompress(compressed)
        dt_ms = (time.monotonic() - t0) * 1000
    except Exception as exc:  # noqa: BLE001
        print(f"{_TAG} corrupt blob {cache_key} ({exc}), removing")
        blob_path.unlink(missing_ok=True)
        index.pop(cache_key, None)
        _save_index(index)
        return None

    v_len = entry["vertex_bytes_len"]
    i_len = entry["index_bytes_len"]
    b_len = entry["bvh_bytes_len"]
    expected = v_len + i_len + b_len
    if len(raw) != expected:
        print(
            f"{_TAG} size mismatch {cache_key}: "
            f"expected {_fmt_size(expected)}, got {_fmt_size(len(raw))}, removing"
        )
        blob_path.unlink(missing_ok=True)
        index.pop(cache_key, None)
        _save_index(index)
        return None

    ratio = len(compressed) / max(len(raw), 1) * 100
    print(
        f"{_TAG} HIT '{source.name}' [{cache_key}] — "
        f"{_fmt_size(len(compressed))} on disk ({ratio:.0f}% of {_fmt_size(len(raw))}), "
        f"decompressed in {dt_ms:.0f} ms | {blob_path}"
    )

    from skinny.mesh import Mesh

    return Mesh(
        name=source.name,
        vertex_bytes=raw[:v_len],
        index_bytes=raw[v_len : v_len + i_len],
        bvh_bytes=raw[v_len + i_len :],
        num_vertices=entry["num_vertices"],
        num_triangles=entry["num_triangles"],
        num_nodes=entry["num_nodes"],
        normal_map=source.normal_map,
        roughness_map=source.roughness_map,
        displacement_map=source.displacement_map,
        normals_baked=entry.get("normals_baked", False),
    )


def save_cached_mesh(
    index: dict,
    cache_key: str,
    mesh: "Mesh",
) -> None:
    """Compress + write blob atomically, update index."""
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        raw = mesh.vertex_bytes + mesh.index_bytes + mesh.bvh_bytes
        t0 = time.monotonic()
        compressed = zstd.compress(raw, _ZSTD_LEVEL)
        dt_ms = (time.monotonic() - t0) * 1000

        blob_path = CACHE_DIR / f"{cache_key}.skmc"
        tmp = blob_path.with_suffix(".skmc.tmp")
        tmp.write_bytes(compressed)
        os.replace(tmp, blob_path)

        index[cache_key] = {
            "name": mesh.name,
            "num_vertices": mesh.num_vertices,
            "num_triangles": mesh.num_triangles,
            "num_nodes": mesh.num_nodes,
            "normals_baked": mesh.normals_baked,
            "vertex_bytes_len": len(mesh.vertex_bytes),
            "index_bytes_len": len(mesh.index_bytes),
            "bvh_bytes_len": len(mesh.bvh_bytes),
            "uncompressed_size": len(raw),
            "compressed_size": len(compressed),
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        _save_index(index)

        ratio = len(compressed) / max(len(raw), 1) * 100
        total_compressed = sum(e.get("compressed_size", 0) for e in index.values())
        print(
            f"{_TAG} SAVED '{mesh.name}' [{cache_key}] — "
            f"{_fmt_size(len(raw))} → {_fmt_size(len(compressed))} ({ratio:.0f}%), "
            f"compressed in {dt_ms:.0f} ms | {blob_path}"
        )
        print(
            f"{_TAG} index now {len(index)} entries, "
            f"{_fmt_size(total_compressed)} total on disk"
        )
    except OSError as exc:
        print(f"{_TAG} failed to save {cache_key}: {exc}")


def clear_mesh_cache() -> int:
    """Delete all cached blobs + index. Returns number of files removed."""
    if not CACHE_DIR.exists():
        return 0
    count = 0
    for p in CACHE_DIR.glob("*.skmc"):
        p.unlink(missing_ok=True)
        count += 1
    if INDEX_FILE.exists():
        INDEX_FILE.unlink()
        count += 1
    return count
