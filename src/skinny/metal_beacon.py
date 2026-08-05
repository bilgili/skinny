"""Kernel beacon — the mmap cell format, its writer/reader, and the kernel-id
table (change metal-kernel-beacon).

macOS cannot cancel a committed Metal command buffer, so an infinite kernel
wedges the GPU until reboot. slang-rhi hides the ``MTLCommandBuffer``, so no
native completion handler or ``MTLFunctionLog`` names the stuck function. The
beacon reports it anyway: the child process stamps the kernel id into a
memory-mapped cell right before each dispatch, and a parent process reads the
cell after a wall-clock timeout.

This module is device-free — it imports no GPU package. It owns:

* the 256-byte little-endian cell layout (frozen in ``design.md``);
* :class:`BeaconWriter` (child side) and :class:`BeaconReader` (parent side),
  paired by a seqlock so a torn read is observable, not reported as a kernel;
* :data:`KERNEL_NAMES`, the single append-only id <-> entry-point-name table.

The same 256-byte layout also describes the UMA GPU beacon buffer the traced
shader writes (``shaders/beacon.slang``); this module owns only the host mmap.
"""

from __future__ import annotations

import mmap
import os
import struct
from dataclasses import dataclass

# ── Cell layout (256 bytes, little-endian) ───────────────────────────
# | 0 magic u32 | 4 seq u32 | 8 kernel_id u32 | 12 phase u32 |
# | 16 trip u32 | 20 seq_check u32 | 24..256 reserved (zero) |

BEACON_BYTES: int = 256
BEACON_MAGIC: int = 0xB0AC0001
KERNEL_ID_NONE: int = 0
PHASE_ENTRY: int = 0

_OFF_MAGIC = 0
_OFF_SEQ = 4
_OFF_KERNEL_ID = 8
_OFF_PHASE = 12
_OFF_TRIP = 16
_OFF_SEQ_CHECK = 20


@dataclass(frozen=True)
class BeaconReport:
    """A validated snapshot of the beacon cell — what a reader hands back."""

    kernel_id: int
    kernel_name: str
    phase: int
    trip: int
    seq: int


# ── Kernel-identity table ────────────────────────────────────────────
# The single source of id <-> entry-point name. Id 0 is reserved for
# "<none>". Ids are APPEND-ONLY: a new kernel takes the next free id; an
# existing id is never renumbered and never reused. A pinned golden test
# enforces the mapping, so a renumber fails the build.
#
# Order below (and therefore the ids) mirrors source appearance: the
# megakernel first, then the 35 dispatched wavefront entries from the
# `_ENTRIES` lists in `vk_wavefront.py` (path, sppm, mlt, bdpt, neural,
# indirect), then the preview kernel, then the debug-raster kernels. The
# seven undispatched wf* shader entries (wfEmitRecords, wfFinishShade,
# wfPushRecord, wfResetRecords, wfTerminate, wfWriteDisplay,
# wfBdptConnectLane) have no Python dispatch site and are deliberately absent.
_KERNEL_ORDER: tuple[str, ...] = (
    # megakernel
    "mainImage",
    # wavefront path (WavefrontPathPass._ENTRIES)
    "wfPathGenerate",
    "wfPathIntersect",
    "wfBuildArgs",
    "wfScatter",
    "wfPathShadeFlat",
    "wfPathResolve",
    "wfPathShade",
    # wavefront SPPM (WavefrontSppmPass._ENTRIES)
    "wfSppmEye",
    "wfSppmGridCount",
    "wfSppmGridScanBlock",
    "wfSppmGridScanBlockSums",
    "wfSppmGridScanAdd",
    "wfSppmGridScatter",
    "wfSppmPhotonTrace",
    "wfSppmUpdate",
    # wavefront MLT (WavefrontMltPass._ENTRIES)
    "wfMltBootstrap",
    "wfMltInit",
    "wfMltMutate",
    "wfMltResolve",
    # wavefront BDPT (shared + staged_eye + walk variants)
    "wfBdptClassify",
    "wfBdptBuildArgs",
    "wfBdptScatter",
    "wfBdptConnectNee",
    "wfBdptConnectFull",
    "wfBdptResolve",
    "wfBdptGenEye",
    "wfBdptWalkClassify",
    "wfBdptBounceEye",
    "wfBdptWalk",
    "wfBdptLightTail",
    "wfBdptGenLight",
    "wfBdptBounceLight",
    "wfBdptSplat",
    # wavefront neural proposal + indirect paint
    "wfNeuralProposal",
    "wfIndirectPaint",
    # tool-dock preview kernel
    "previewMain",
    # tool-dock debug-raster compute kernels (debug_raster.slang)
    "clearImage",
    "clearDepth",
    "depthLines",
    "colorLines",
    "blendTris",
)

#: id -> entry-point name; 0 -> "<none>". Append-only (see _KERNEL_ORDER).
KERNEL_NAMES: dict[int, str] = {KERNEL_ID_NONE: "<none>"}
KERNEL_NAMES.update({i: name for i, name in enumerate(_KERNEL_ORDER, start=1)})

#: name -> id, derived once from KERNEL_NAMES.
_NAME_TO_ID: dict[str, int] = {name: kid for kid, name in KERNEL_NAMES.items()}


def kernel_id_for(entry: str) -> int:
    """Return the static id for an entry-point name. Raise ``KeyError`` for an
    entry the table does not know, so a call site cannot stamp a phantom."""
    return _NAME_TO_ID[entry]


def kernel_name_for(kernel_id: int) -> str:
    """Return the entry-point name for an id, or ``"<unknown:N>"`` for an id
    absent from the table. Never raises, so a report always resolves."""
    name = KERNEL_NAMES.get(kernel_id)
    return name if name is not None else f"<unknown:{kernel_id}>"


# ── mmap writer / reader ─────────────────────────────────────────────

class BeaconWriter:
    """Child-side owner of the mmap cell. Stamp the kernel id right BEFORE
    submitting the dispatch that may hang, so the id survives the wedge."""

    def __init__(self, path: str) -> None:
        self._path = path
        # Create/size the backing file to exactly one cell so it can be mmapped
        # and read from another process. A fresh cell has magic 0, which a
        # reader maps to None until the first stamp writes BEACON_MAGIC.
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            os.ftruncate(fd, BEACON_BYTES)
            self._mm = mmap.mmap(fd, BEACON_BYTES)
        finally:
            os.close(fd)
        self._seq = 0

    def stamp(self, kernel_id: int) -> None:
        """Bump ``seq``, write ``kernel_id`` (phase 0, trip 0), write
        ``seq_check = seq``, then ``msync``. Raise ``ValueError`` when
        ``kernel_id`` is not in :data:`KERNEL_NAMES`."""
        if kernel_id not in KERNEL_NAMES:
            raise ValueError(f"kernel_id {kernel_id} not in KERNEL_NAMES")
        self._seq += 1
        mm = self._mm
        # Seqlock order: magic + seq first, then the payload, then seq_check.
        # A reader accepts the snapshot only when seq == seq_check, so a read
        # that races the payload write is observably torn (rejected).
        struct.pack_into("<II", mm, _OFF_MAGIC, BEACON_MAGIC, self._seq)
        struct.pack_into("<III", mm, _OFF_KERNEL_ID, kernel_id, PHASE_ENTRY, 0)
        struct.pack_into("<I", mm, _OFF_SEQ_CHECK, self._seq)
        mm.flush()

    def close(self) -> None:
        mm = getattr(self, "_mm", None)
        if mm is not None:
            mm.close()
            self._mm = None


class BeaconReader:
    """Parent-side poller of the mmap cell. A stateless per-call read always
    sees the writer's latest flushed bytes and never holds the file open."""

    def __init__(self, path: str) -> None:
        self._path = path

    def read(self) -> BeaconReport | None:
        """Return the current cell, or ``None`` when the cell is smaller than
        :data:`BEACON_BYTES`, the magic is wrong, or ``seq != seq_check`` (a
        torn read). Never raises for a partial, missing, or torn cell."""
        try:
            with open(self._path, "rb") as f:
                data = f.read(BEACON_BYTES)
        except OSError:
            return None
        if len(data) < BEACON_BYTES:
            return None
        magic, seq = struct.unpack_from("<II", data, _OFF_MAGIC)
        kernel_id, phase, trip = struct.unpack_from("<III", data, _OFF_KERNEL_ID)
        (seq_check,) = struct.unpack_from("<I", data, _OFF_SEQ_CHECK)
        if magic != BEACON_MAGIC or seq != seq_check:
            return None
        return BeaconReport(
            kernel_id=kernel_id,
            kernel_name=kernel_name_for(kernel_id),
            phase=phase,
            trip=trip,
            seq=seq,
        )

    def close(self) -> None:
        pass
