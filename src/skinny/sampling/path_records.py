"""Neural training-record (`.nrec`) format — renderer dump ↔ offline trainer.

The renderer's ``mainImageRecord`` megakernel entry
(``shaders/integrators/path_record.slang``) appends one fixed-size record per
guideable path vertex to a GPU buffer; ``Renderer.dump_path_records`` reads it
back and streams it to a ``.nrec`` file. The offline ``spline_flow`` trainer
(``render_records.py``) reads that file, rebuilds the canonical condition with
the header's scene AABB, and fits the proposal flow.

Both sides share THIS module's layout byte-for-byte. The record matches the
Slang ``PathRecord`` struct under ``-fvk-use-scalar-layout`` (float3 = 12 B
tight, no padding) → 64 B stride:

    float3 pos        world hit position (raw; normalised by the header AABB)
    float3 normal     world shading normal (== neuralCondition's N)
    float3 wo          world outgoing dir toward the previous vertex
    float3 wiLocal     sampled bounce dir in flow-local (x=T, y=N, z=B), y-up
    float3 contrib     (L_final - L_k)/beta_in_k = weight·Li  (RGB training weight)
    uint   depth       bounce index (0 = primary hit)

File = a 64 B header then a raw stream of records (count implied by file size):

    uint32  magic = 0x4E524543 ("NREC")
    uint32  version = 1
    uint32  record_stride = 64
    uint32  cond_dim = 9
    float32 bounds_min[3]      scene AABB min  (fc.sceneBoundsMin)
    float32 bounds_extent[3]   scene AABB size (fc.sceneBoundsExtent)
    uint32  reserved[6]
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

MAGIC = 0x4E524543  # 'NREC' little-endian
VERSION = 1
COND_DIM = 9

RECORD_DTYPE = np.dtype([
    ("pos", "<f4", (3,)),
    ("normal", "<f4", (3,)),
    ("wo", "<f4", (3,)),
    ("wi_local", "<f4", (3,)),
    ("contrib", "<f4", (3,)),
    ("depth", "<u4"),
])
RECORD_STRIDE = RECORD_DTYPE.itemsize  # 64
assert RECORD_STRIDE == 64, RECORD_STRIDE

HEADER_STRIDE = 64


def pack_header(bounds_min, bounds_extent) -> bytes:
    """64-byte ``.nrec`` header carrying the scene AABB the trainer must reuse."""
    bmin = np.asarray(bounds_min, dtype="<f4").reshape(3)
    bext = np.asarray(bounds_extent, dtype="<f4").reshape(3)
    head = struct.pack("<IIII", MAGIC, VERSION, RECORD_STRIDE, COND_DIM)
    head += bmin.tobytes() + bext.tobytes()
    head += struct.pack("<6I", 0, 0, 0, 0, 0, 0)
    assert len(head) == HEADER_STRIDE, len(head)
    return head


def records_from_buffer(raw: bytes, count: int) -> np.ndarray:
    """Parse the first ``count`` records out of a raw GPU drain buffer.

    The live online-training drain (``Renderer.drain_path_records_to_replay``)
    reads the append buffer (binding 36) and counter (binding 37) directly, so —
    unlike :func:`read_records` — there is no file header to skip. ``count`` is
    the GPU-reported counter, which can exceed the buffer capacity (the shader
    atomically bumps it even past the cap); this clamps to the bytes actually
    present so the reader never runs past the end. Returns a writable
    ``RECORD_DTYPE`` copy (detached from ``raw``) ready for ``ReplayBuffer.add``.
    """
    n = min(int(count), len(raw) // RECORD_STRIDE)
    if n <= 0:
        return np.empty(0, dtype=RECORD_DTYPE)
    return np.frombuffer(raw, dtype=RECORD_DTYPE, count=n).copy()


def read_records(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse a ``.nrec`` file → (records, bounds_min, bounds_extent).

    ``records`` is a structured array of ``RECORD_DTYPE``. Raises on bad magic /
    version / stride or a truncated header.
    """
    data = Path(path).read_bytes()
    if len(data) < HEADER_STRIDE:
        raise ValueError(f"{path}: truncated (< {HEADER_STRIDE} B header)")
    magic, ver, stride, cond = struct.unpack_from("<IIII", data, 0)
    if magic != MAGIC:
        raise ValueError(f"{path}: bad magic 0x{magic:08X} (want 0x{MAGIC:08X})")
    if ver != VERSION:
        raise ValueError(f"{path}: unsupported version {ver}")
    if stride != RECORD_STRIDE or cond != COND_DIM:
        raise ValueError(f"{path}: stride/cond {stride}/{cond} != {RECORD_STRIDE}/{COND_DIM}")
    bmin = np.frombuffer(data, dtype="<f4", count=3, offset=16).copy()
    bext = np.frombuffer(data, dtype="<f4", count=3, offset=28).copy()
    body = data[HEADER_STRIDE:]
    n = len(body) // RECORD_STRIDE
    records = np.frombuffer(body, dtype=RECORD_DTYPE, count=n).copy()
    return records, bmin, bext
