"""The per-frame plan (change frame-plan-split).

`frame_derive.py` derives the frame's *constants*. This module derives the
frame's *decisions*: which execution mode runs, which integrator, what the
step order is, how the dispatch is banded, and which optional per-frame work
happens. The result is a value — `FramePlan` — that holds no buffers, no
command buffers and no pipelines, so it can be derived and asserted in a
process with no GPU device.

Three properties make the plan worth having:

- **The pass sequence is inspectable.** "Which passes will this frame run" used
  to be a fact about which branch of `render` you were reading. It is now
  `plan.steps`.
- **Ordering constraints are stated.** The pick-result drain must precede the
  uniform pack, or a satisfied pick disarms one frame late. That was a fact
  about two line numbers in two functions; it is now an invariant over
  `plan.steps` that `check_invariants` asserts on every derivation.
- **Banding is capability-driven.** The macOS GPU watchdog is why a dispatch is
  split into row bands. The caller passes `needs_watchdog_tiling`, not
  `is_metal`, so the reason travels with the decision.

The plan does NOT own the accumulation reset. `Renderer.update` decides it from
the `params.py` registry (change `param-registry-accumulation-reset`) and
publishes it as `accum_frame`; the plan consumes `accum_frame == 0` as
`first_frame`. Two owners for one decision is what that capability exists to
prevent.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# Execution-mode indices, mirrored from `renderer` so this module imports no
# GPU-touching code (renderer.py imports `vulkan` at module load).
EXECUTION_MEGAKERNEL = 0
EXECUTION_WAVEFRONT = 1

# Integrator index → the staged pass name `_ensure_wavefront_pass` takes.
INTEGRATOR_NAMES = {0: "path", 1: "bdpt", 2: "sppm", 3: "mlt"}

# Target pixels per Metal megakernel command buffer, per integrator, before the
# frame is split into more row bands (change metal-megakernel-watchdog-tiling).
# BDPT does the widest per-pixel work (eye × light subpaths + full s×t connection
# matrix, each connection a BSDF eval at both ends), so it needs a far smaller
# budget than the path tracer to stay under the macOS GPU watchdog on heavy
# (graph-material) scenes. Path/SPPM are cheap enough to keep the single
# full-frame dispatch on ordinary scenes.
MEGAKERNEL_BAND_PIXELS_DEFAULT = 8_000_000
MEGAKERNEL_BAND_PIXELS = {
    0: 8_000_000,   # Path — effectively one band until very large frames
    1: 200_000,     # BDPT — the wedging case; ~1280×720 → ~5 bands
    2: 8_000_000,   # SPPM eye pass — cheap per pixel
}

# Chains per MLT phase dispatch under watchdog tiling (design D7 of
# `mlt-integrator`), so a large `--chains` stays under the macOS GPU watchdog.
MLT_CHAIN_BATCH_DEFAULT = 16384

# ── Step vocabulary ──────────────────────────────────────────────────────
# The names in `FramePlan.steps`. One name per thing the frame does that an
# ordering constraint can refer to.

PICK_DRAIN = "pick_drain"
FENCE_WAIT = "fence_wait"
ACQUIRE = "acquire"
PACK_UNIFORMS = "pack_uniforms"
UPLOAD_MTLX = "upload_mtlx"
BEGIN_CMD = "begin_cmd"
ACCUM_BARRIER = "accum_barrier"
HUD = "hud"
MLT_BOOTSTRAP = "mlt_bootstrap"
DISPATCH = "dispatch"
OUTPUT_BARRIER = "output_barrier"
OUTPUT = "output"
RESTORE_BARRIER = "restore_barrier"
PRESENT_BARRIER = "present_barrier"
END_CMD = "end_cmd"
SUBMIT = "submit"
PRESENT = "present"
DRAIN = "drain"
READBACK = "readback"
ONLINE_SWAP = "online_swap"
ROTATE_FRAME = "rotate_frame"

TARGET_WINDOWED = "windowed"
TARGET_HEADLESS = "headless"

# ── Ordering invariants ──────────────────────────────────────────────────
# Each entry is (earlier, later, why). A pair is checked only when the plan
# contains both steps, so a target that omits one is not constrained by it.

ORDERING_INVARIANTS = (
    (PICK_DRAIN, PACK_UNIFORMS,
     "a satisfied pick must disarm in THIS frame's uniform buffer; drained "
     "later it disarms one frame late and the pick fires twice"),
    (PICK_DRAIN, DISPATCH,
     "the drain reads the tool buffer the dispatch overwrites"),
    (FENCE_WAIT, PICK_DRAIN,
     "the tool buffer is read only once the frame that wrote it has retired"),
    (PACK_UNIFORMS, DISPATCH,
     "the dispatch reads the frame constants the pack produces — on Metal the "
     "pack is also what sets the SPPM photon batch the dispatch consumes"),
    (MLT_BOOTSTRAP, DISPATCH,
     "the mutation frame resolves against the `b` the bootstrap measures"),
    (ACCUM_BARRIER, DISPATCH,
     "the previous frame's accumulation writes must be visible to this "
     "frame's reads"),
    (DISPATCH, OUTPUT_BARRIER,
     "the offscreen image transitions to TRANSFER_SRC only after it is written"),
    (OUTPUT_BARRIER, OUTPUT, "the blit/copy source must already be TRANSFER_SRC"),
    (OUTPUT, RESTORE_BARRIER,
     "the offscreen image returns to GENERAL only after it is read"),
    (SUBMIT, PRESENT, "present waits on the submit's signal semaphore"),
    (SUBMIT, DRAIN, "the drain waits the fence the submit signals"),
    (DRAIN, READBACK, "the readback reads host-visible memory the drain flushes"),
    (DISPATCH, ONLINE_SWAP,
     "weights stay frozen for the frame that reads them; the swap promotes "
     "pending weights for the NEXT frame only"),
)


class PlanOrderError(AssertionError):
    """A derived plan violates one of `ORDERING_INVARIANTS`."""


@dataclass(frozen=True)
class FramePlan:
    """One frame's decisions. Holds no device handles."""

    target: str
    """`TARGET_WINDOWED` or `TARGET_HEADLESS` — the only axis on which the
    windowed and headless paths may differ."""

    steps: tuple[str, ...]
    """Every step this frame performs, in execution order."""

    execution_mode: int
    """`EXECUTION_MEGAKERNEL` or `EXECUTION_WAVEFRONT`, already resolved."""

    integrator: str
    """`path` / `bdpt` / `sppm` / `mlt`. Meaningful under wavefront; under the
    megakernel the integrator is selected in-shader by `fc.integrator`."""

    accum_frame: int
    """The accumulation frame index `update()` published for this frame."""

    first_frame: bool
    """`accum_frame == 0` — the accumulation reset decision, consumed from its
    registry owner. Drives the SPPM first-frame flag and the MLT reseed."""

    megakernel_bands: int
    """Row bands the megakernel dispatch is split into. 1 = one full-frame
    command buffer."""

    mlt_iterations: int
    """Mutation iterations this frame. 0 when the integrator is not MLT."""

    mlt_chain_batch: int
    """Chains per MLT phase dispatch. 0 = no breadth tiling."""

    bound_heavy_eye: bool
    """Bound the wavefront eye submit per tile — set when the scene has a
    non-terminal non-flat material (change wavefront-nonflat-tiled-fallback)."""

    online_swap: bool
    """Perform the neural double-buffer swap at frame end."""

    def index(self, step: str) -> int:
        """Position of *step*, or -1 when this frame does not perform it."""
        return self.steps.index(step) if step in self.steps else -1

    def runs(self, step: str) -> bool:
        return step in self.steps


def megakernel_bands(needs_watchdog_tiling: bool, integrator_index: int,
                     width: int, height: int) -> int:
    """Row-band count for the megakernel dispatch, so no single command buffer
    exceeds the GPU watchdog budget.

    Banding is driven by the *capability* — a backend whose command buffers are
    watchdog-policed — not by which backend it happens to be. Without that
    capability the frame is one band, which is one full-frame dispatch and
    exactly what Vulkan has always done. `SKINNY_METAL_MEGAKERNEL_BANDS`
    overrides for tuning.
    """
    if not needs_watchdog_tiling:
        return 1
    override = os.environ.get("SKINNY_METAL_MEGAKERNEL_BANDS")
    if override:
        try:
            return max(1, int(override))
        except ValueError:
            pass
    budget = MEGAKERNEL_BAND_PIXELS.get(
        int(integrator_index), MEGAKERNEL_BAND_PIXELS_DEFAULT)
    pixels = int(width) * int(height)
    bands = (pixels + budget - 1) // budget
    return max(1, min(int(height), bands))


def mlt_chain_batch(needs_watchdog_tiling: bool) -> int:
    """Chains per MLT phase dispatch. 0 (no tiling) without the watchdog
    capability; `SKINNY_MLT_METAL_CHAIN_BATCH` overrides the default."""
    if not needs_watchdog_tiling:
        return 0
    return int(os.environ.get("SKINNY_MLT_METAL_CHAIN_BATCH",
                             str(MLT_CHAIN_BATCH_DEFAULT)))


def integrator_name(integrator_index: int) -> str:
    """Staged-pass name for an integrator index. Anything unrecognised is the
    path tracer, matching the renderer's own fallback."""
    return INTEGRATOR_NAMES.get(int(integrator_index), "path")


def _vulkan_steps(target: str, wavefront: bool, mlt: bool,
                  online_swap: bool) -> tuple[str, ...]:
    """The Vulkan step order, shared between the two targets.

    The windowed and headless paths differ only where marked. Note the pick
    drain sits AFTER the fence wait on both: the headless path used to drain
    first, which read the same bytes only because its previous call waited the
    same fence at its tail. Draining after the wait is identical there and is
    the only safe order windowed-side (see `baseline.md` §1.2).
    """
    steps = [FENCE_WAIT, PICK_DRAIN]
    if target == TARGET_WINDOWED:
        steps.append(ACQUIRE)
    steps += [PACK_UNIFORMS, UPLOAD_MTLX, BEGIN_CMD, ACCUM_BARRIER, HUD]
    if wavefront and mlt:
        steps.append(MLT_BOOTSTRAP)
    steps += [DISPATCH, OUTPUT_BARRIER, OUTPUT, RESTORE_BARRIER]
    if target == TARGET_WINDOWED:
        steps.append(PRESENT_BARRIER)
    steps += [END_CMD, SUBMIT, PRESENT if target == TARGET_WINDOWED else DRAIN]
    if online_swap:
        steps.append(ONLINE_SWAP)
    steps.append(ROTATE_FRAME)
    if target == TARGET_HEADLESS:
        # The readback reads host-visible memory the drain already flushed, so
        # it is the last thing the frame does — after the weight swap and the
        # frame-index rotation, exactly as the pre-split path ordered it.
        steps.append(READBACK)
    return tuple(steps)


def _metal_steps(target: str, wavefront: bool, mlt: bool,
                 online_swap: bool) -> tuple[str, ...]:
    """The Metal step order. No fences, no command buffers, no descriptor sets
    — resources bind at dispatch and each submit drains. The uniform blob is
    packed per dispatch rather than uploaded to a persistent buffer, so
    `PACK_UNIFORMS` still precedes `DISPATCH`."""
    steps = [PICK_DRAIN, HUD, UPLOAD_MTLX]
    if wavefront and mlt:
        steps.append(MLT_BOOTSTRAP)
    steps += [PACK_UNIFORMS, DISPATCH]
    if target == TARGET_WINDOWED:
        steps += [ACQUIRE, OUTPUT, PRESENT, DRAIN]
    else:
        steps += [DRAIN, READBACK]
    if online_swap:
        steps.append(ONLINE_SWAP)
    return tuple(steps)


def check_invariants(plan: FramePlan) -> None:
    """Raise `PlanOrderError` if the plan's step order breaks a stated
    constraint. Called on every derivation — the constraints are asserted, not
    documented."""
    for earlier, later, why in ORDERING_INVARIANTS:
        i, j = plan.index(earlier), plan.index(later)
        if i < 0 or j < 0:
            continue
        if i > j:
            raise PlanOrderError(
                f"{plan.target}: {earlier!r} must precede {later!r} — {why}")


def derive(*, target: str, execution_mode_index: int, integrator_index: int,
           accum_frame: int, width: int, height: int,
           needs_watchdog_tiling: bool, records_command_buffers: bool,
           mlt_num_chains: int, has_heavy_nonflat: bool,
           online_training: bool) -> FramePlan:
    """Derive this frame's plan from renderer state.

    Every argument is a scalar or a bool. Nothing here touches a device, so a
    plan can be derived — and its pass sequence, execution mode and
    accumulation decision asserted — with no GPU present.

    `records_command_buffers` distinguishes the two step orders: a backend that
    records into a command buffer and submits it against a fence (Vulkan) from
    one that binds at dispatch and drains per submit (Metal). Like
    `needs_watchdog_tiling`, it names the reason rather than the backend.
    """
    wavefront = int(execution_mode_index) == EXECUTION_WAVEFRONT
    integrator = integrator_name(integrator_index)
    mlt = wavefront and integrator == "mlt"
    build_steps = _vulkan_steps if records_command_buffers else _metal_steps
    plan = FramePlan(
        target=target,
        steps=build_steps(target, wavefront, mlt, bool(online_training)),
        execution_mode=int(execution_mode_index),
        integrator=integrator,
        accum_frame=int(accum_frame),
        first_frame=int(accum_frame) == 0,
        megakernel_bands=megakernel_bands(
            needs_watchdog_tiling, integrator_index, width, height),
        mlt_iterations=(
            _mlt_iterations(width, height, mlt_num_chains) if mlt else 0),
        mlt_chain_batch=mlt_chain_batch(needs_watchdog_tiling) if mlt else 0,
        bound_heavy_eye=bool(has_heavy_nonflat),
        online_swap=bool(online_training),
    )
    check_invariants(plan)
    return plan


def _mlt_iterations(width: int, height: int, num_chains: int) -> int:
    # Imported lazily: `mlt_chain` is device-free too, but keeping the import
    # local documents that the budget rule is owned there, not here.
    from skinny import mlt_chain

    return mlt_chain.iterations_per_frame(width, height, num_chains)
