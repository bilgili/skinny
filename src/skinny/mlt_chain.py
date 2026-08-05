"""MLT host chain-state orchestration (change renderer-module-carveout,
Stage A) — the device-free half of the Metropolis-light-transport host layer.

Companion to `mlt_bootstrap.py` (the pure numpy resample) and
`wavefront_driver.record_mlt_*` (the pure stage order): this module holds the
per-reset seed derivation, the mutation budget, the uniform-tail predicate, and
the bootstrap round-trip sequencing that both backends drive. Everything here
is importable and testable without a GPU device or a constructed `Renderer`;
`run_bootstrap` is device-free by taking the backend's submit/upload
primitives as callables.
"""

from __future__ import annotations

import struct
import zlib
from typing import Callable, Optional

from skinny import choice_tables

# Execution-mode index for wavefront, projected from the choice_tables owner
# (dependency-free, so this module still imports no GPU-touching code).
EXECUTION_WAVEFRONT = choice_tables.index_by_token(choice_tables.EXECUTION_MODE)["wavefront"]


def next_seed(frame_index: int) -> int:
    """Per-reset MLT replay seed (change mlt-integrator, design D3): stable
    across an accumulation run, decorrelated between consecutive resets, and
    REPRODUCIBLE ACROSS PROCESSES — the parity gate re-renders in a fresh
    interpreter and must get the same chains (design D6's deterministic budget
    mapping).

    Deliberately NOT derived from `Renderer._current_state_hash()`. That hash
    exists for change detection, where only equality *within* one process
    matters, and it hashes tuples containing str (`state_signature()` leads
    with "orbit"/"free") — so PYTHONHASHSEED randomizes it per process.
    Seeding MLT from it made every render irreproducible: the same scene scored
    self-consistency relMSE 0.17 / 0.25 / 1.10 across three runs, which is
    pass-or-fail by luck. `frame_index` alone already decorrelates resets (it
    advances between them) and is deterministic in a headless render, so it is
    both necessary and sufficient here.
    """
    # frame_index is a monotonic counter and mltSeed is a u32 shader field,
    # so mask to 32 bits — a signed "<i" pack raises struct.error past 2**31
    # (codex pre-merge review).
    return zlib.crc32(struct.pack("<I", int(frame_index) & 0xFFFFFFFF)) & 0xFFFFFFFF


def iterations_per_frame(width: int, height: int, num_chains: int) -> int:
    """Mutation iterations per accumulation frame: ~1 mutation/pixel/frame
    (`mpp_actual = iterations × nChains / pixels` is packed into the MLT
    uniform tail so the resolve divides by the ACTUAL budget, design D4)."""
    pixels = max(1, int(width) * int(height))
    return max(1, round(pixels / max(1, int(num_chains))))


def uniform_tail_active(integrator_index: int, reflected_uniform_layout: bool,
                        execution_mode_index: int, pass_built: bool) -> bool:
    """Whether `_pack_uniforms` must emit the ``#if defined(SKINNY_MLT)``
    FrameConstants tail — i.e. the dispatched shader's ``fc`` actually has
    those fields (codex pre-merge review).

    ``reflected_uniform_layout`` is
    :attr:`~skinny.gpu_backend.BackendCapabilities.has_reflected_record_layouts`
    — pass the capability, never a vendor flag.

    A backend without it (Vulkan) uses one oversized shared UBO, so appending
    the tail whenever MLT is the integrator is harmless — only the MLT ``.spv``
    reads the offsets. A backend with it (Metal) packs the blob per-dispatch and
    the drift guard asserts the blob length equals the reflected ``fc`` size, so
    the tail is packed ONLY when the MLT wavefront pass is the real consumer:
    integrator 3, wavefront mode, and the pass built. A megakernel-fallback MLT
    selection (execution mode != wavefront) or any non-MLT integrator gets the
    base layout, no tail — otherwise runtime integrator cycling crashes uniform
    packing."""
    if integrator_index != 3:
        return False
    if not reflected_uniform_layout:
        return True
    return execution_mode_index == EXECUTION_WAVEFRONT and pass_built


def run_bootstrap(mlt, *, seed: int, submit: Callable[[str], None],
                  upload_uniforms: Optional[Callable[[], None]] = None) -> None:
    """Synchronous MLT (re)seed at an accumulation reset (design D3), shared by
    both backends: bootstrap dispatch → weight readback → host CDF resample
    (b + chain seeds) → seed upload → chain-init dispatch → `b` publication.

    `submit(phase)` runs one awaited GPU phase, ``phase`` being ``"bootstrap"``
    or ``"init"`` — on Vulkan a one-shot command buffer recording the pass's
    `record_*`; on Metal the pass's own `dispatch_*` encoder (each ends in a
    `MetalFrameEncoder.submit`, which drains, so the readback between them sees
    finished GPU work without an explicit wait).

    `upload_uniforms` re-uploads the frame constants around the phases — the
    bootstrap/init kernels read `fc.mltSeed`, and the frame's resolve reads
    `fc.mltB`. Vulkan supplies it (one persistent UBO); Metal passes None
    because its blob is a per-dispatch argument packed inside `submit`.

    The caller sets its own `_mlt_seed` (and packs it into the uniforms) BEFORE
    calling — `seed` here is only the host resample's RNG seed.
    """
    from skinny.mlt_bootstrap import resample_chain_seeds

    mlt.b = 0.0
    mlt.seeded = False
    if upload_uniforms is not None:
        upload_uniforms()
    submit("bootstrap")
    weights = mlt.read_bootstrap_weights()
    b, seeds = resample_chain_seeds(weights, mlt.num_chains, seed)
    mlt.upload_chain_seeds(seeds)
    submit("init")
    mlt.b = b
    mlt.seeded = True
    # The frame's resolve reads fc.mltB — re-upload now that b is known (the
    # frame command buffer has not been submitted yet).
    if upload_uniforms is not None:
        upload_uniforms()
