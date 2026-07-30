"""The per-frame plan is a value, derived with no GPU present.

Change ``frame-plan-split``. Before the split, "which passes will this frame
run" was a fact about which branch of ``render`` you happened to be reading,
and the ordering constraints between the steps were facts about line numbers in
two near-identical functions. Both are now properties of a derived
``FramePlan``, and this file asserts them on a host with no device — which is
the whole point of making the plan pure.

Hostless: plain ``pytest`` runs it. Nothing here imports ``skinny.renderer``.
"""

from __future__ import annotations

import itertools

import pytest

from skinny import frame_plan, render_envelope

TARGETS = (frame_plan.TARGET_WINDOWED, frame_plan.TARGET_HEADLESS)
INTEGRATOR_INDICES = {"path": 0, "bdpt": 1, "sppm": 2, "mlt": 3}
EXECUTION_INDICES = {
    "megakernel": frame_plan.EXECUTION_MEGAKERNEL,
    "wavefront": frame_plan.EXECUTION_WAVEFRONT,
}


def _derive(integrator="path", execution_mode="wavefront",
            target=frame_plan.TARGET_WINDOWED, *, accum_frame=0,
            needs_watchdog_tiling=False, records_command_buffers=True,
            width=256, height=256, mlt_num_chains=16384,
            has_heavy_nonflat=False, online_training=False):
    return frame_plan.derive(
        target=target,
        execution_mode_index=EXECUTION_INDICES[execution_mode],
        integrator_index=INTEGRATOR_INDICES[integrator],
        accum_frame=accum_frame, width=width, height=height,
        needs_watchdog_tiling=needs_watchdog_tiling,
        records_command_buffers=records_command_buffers,
        mlt_num_chains=mlt_num_chains,
        has_heavy_nonflat=has_heavy_nonflat,
        online_training=online_training,
    )


def _in_envelope(integrator: str, execution_mode: str) -> bool:
    return render_envelope.evaluate(render_envelope.EnvelopeQuery(
        integrator=integrator, execution_mode=execution_mode)).ok


# ── The plan holds no device handles ─────────────────────────────────────


def test_frame_plan_module_imports_without_a_gpu_package():
    """Covered structurally by tests/test_pure_core_modules.py; asserted here
    too because a plan that needs a device is not a plan."""
    import skinny.frame_plan as module

    assert "vulkan" not in getattr(module, "__dict__", {})


def test_plan_fields_are_all_plain_values():
    plan = _derive(integrator="mlt", execution_mode="wavefront")
    for name, value in vars(plan).items():
        assert isinstance(value, (str, int, bool, tuple)), \
            f"{name} is {type(value).__name__} — the plan must hold no handles"


def test_plan_is_frozen():
    plan = _derive()
    with pytest.raises(Exception):
        plan.execution_mode = 99


# ── 3.3 Every integrator × execution mode × capability in the envelope ───

_MATRIX = [
    (integrator, mode, tiling, records, target)
    for integrator, mode, tiling, records, target in itertools.product(
        render_envelope.INTEGRATORS, render_envelope.EXECUTION_MODES,
        (False, True), (False, True), TARGETS)
    if _in_envelope(integrator, mode)
]


@pytest.mark.parametrize(
    "integrator,mode,tiling,records,target", _MATRIX,
    ids=[f"{i}-{m}-{'tiled' if t else 'untiled'}-"
         f"{'cmdbuf' if r else 'immediate'}-{g}"
         for i, m, t, r, g in _MATRIX])
def test_plan_derives_for_every_envelope_combination(
        integrator, mode, tiling, records, target):
    """A plan is produced, and its execution mode, integrator and accumulation
    decision are what was asked for — for every combination the render envelope
    admits, on either backend capability, for either target."""
    plan = _derive(integrator=integrator, execution_mode=mode, target=target,
                   needs_watchdog_tiling=tiling, records_command_buffers=records)
    assert plan.execution_mode == EXECUTION_INDICES[mode]
    assert plan.integrator == integrator
    assert plan.target == target
    assert plan.first_frame is True
    assert plan.steps, "a plan must name the steps its frame performs"
    assert frame_plan.DISPATCH in plan.steps
    # Every step name is one of the declared vocabulary — a typo'd step would
    # silently satisfy no invariant at all.
    assert set(plan.steps) <= {
        v for k, v in vars(frame_plan).items()
        if k.isupper() and isinstance(v, str) and not k.startswith("TARGET")}


@pytest.mark.parametrize("integrator", render_envelope.INTEGRATORS)
def test_only_wavefront_mlt_gets_a_mutation_budget(integrator):
    """The MLT budget and chain batch are MLT-under-wavefront decisions. Any
    other combination must report zero, not a stale non-zero the dispatch could
    read."""
    wave = _derive(integrator=integrator, execution_mode="wavefront",
                   needs_watchdog_tiling=True)
    if integrator == "mlt":
        assert wave.mlt_iterations > 0
        assert wave.mlt_chain_batch == frame_plan.MLT_CHAIN_BATCH_DEFAULT
        assert wave.runs(frame_plan.MLT_BOOTSTRAP)
    else:
        assert wave.mlt_iterations == 0
        assert wave.mlt_chain_batch == 0
        assert not wave.runs(frame_plan.MLT_BOOTSTRAP)


def test_megakernel_never_runs_the_mlt_bootstrap():
    """MLT has no megakernel variant. A megakernel-fixed session that cycles to
    MLT shows the path tracer (the safe SPPM wart) — it must not try to seed
    chains for a pass that will not run."""
    plan = _derive(integrator="mlt", execution_mode="megakernel")
    assert not plan.runs(frame_plan.MLT_BOOTSTRAP)
    assert plan.mlt_iterations == 0


# ── 3.2 Banding is capability-driven, not backend-driven ─────────────────


def test_without_the_watchdog_capability_every_frame_is_one_band():
    for integrator in render_envelope.INTEGRATORS:
        plan = _derive(integrator=integrator, needs_watchdog_tiling=False,
                       width=2560, height=1440)
        assert plan.megakernel_bands == 1


def test_bdpt_bands_more_finely_than_path_under_the_watchdog():
    path = _derive(integrator="path", needs_watchdog_tiling=True,
                   width=1280, height=720)
    bdpt = _derive(integrator="bdpt", needs_watchdog_tiling=True,
                   width=1280, height=720)
    assert bdpt.megakernel_bands > path.megakernel_bands >= 1


def test_band_count_never_exceeds_the_row_count():
    plan = _derive(integrator="bdpt", needs_watchdog_tiling=True,
                   width=8192, height=64)
    assert 1 <= plan.megakernel_bands <= 64


def test_band_override_is_honoured_and_a_bad_override_is_ignored(monkeypatch):
    monkeypatch.setenv("SKINNY_METAL_MEGAKERNEL_BANDS", "7")
    assert _derive(needs_watchdog_tiling=True).megakernel_bands == 7
    monkeypatch.setenv("SKINNY_METAL_MEGAKERNEL_BANDS", "not-a-number")
    assert _derive(needs_watchdog_tiling=True).megakernel_bands >= 1


# ── 3.4 Ordering constraints are asserted, not implied ───────────────────


@pytest.mark.parametrize(
    "integrator,mode,tiling,records,target", _MATRIX,
    ids=[f"{i}-{m}-{'tiled' if t else 'untiled'}-"
         f"{'cmdbuf' if r else 'immediate'}-{g}"
         for i, m, t, r, g in _MATRIX])
def test_pick_drain_precedes_uniform_pack(
        integrator, mode, tiling, records, target):
    """The constraint D5 named: a satisfied pick must disarm in THIS frame's
    uniform buffer. Drained after the pack it disarms one frame late and the
    pick fires a second time."""
    plan = _derive(integrator=integrator, execution_mode=mode, target=target,
                   needs_watchdog_tiling=tiling, records_command_buffers=records)
    assert plan.runs(frame_plan.PICK_DRAIN)
    assert plan.runs(frame_plan.PACK_UNIFORMS)
    assert plan.index(frame_plan.PICK_DRAIN) < plan.index(frame_plan.PACK_UNIFORMS)


def test_every_stated_invariant_holds_on_every_derived_plan():
    for integrator, mode, tiling, records, target in _MATRIX:
        plan = _derive(integrator=integrator, execution_mode=mode,
                       target=target, needs_watchdog_tiling=tiling,
                       records_command_buffers=records, online_training=True)
        frame_plan.check_invariants(plan)  # raises PlanOrderError on violation


def test_a_violated_invariant_is_detected():
    """Negative control: `check_invariants` is not vacuous."""
    plan = _derive()
    steps = list(plan.steps)
    i, j = steps.index(frame_plan.PICK_DRAIN), steps.index(frame_plan.PACK_UNIFORMS)
    steps[i], steps[j] = steps[j], steps[i]
    broken = frame_plan.FramePlan(**{**vars(plan), "steps": tuple(steps)})
    with pytest.raises(frame_plan.PlanOrderError, match="pick_drain"):
        frame_plan.check_invariants(broken)


def test_index_reports_missing_steps_rather_than_raising():
    plan = _derive(target=frame_plan.TARGET_HEADLESS)
    assert plan.index(frame_plan.PRESENT) == -1
    assert not plan.runs(frame_plan.PRESENT)


# ── D2: windowed and headless differ only in their target ────────────────


@pytest.mark.parametrize("records", (True, False))
def test_windowed_and_headless_share_every_step_that_is_not_target_work(records):
    """The two targets may differ only in output destination, swapchain
    acquisition and presentation, and readback. Any other divergence means a
    duplicated middle has grown back."""
    common = dict(integrator="path", execution_mode="wavefront",
                  records_command_buffers=records)
    windowed = _derive(target=frame_plan.TARGET_WINDOWED, **common)
    headless = _derive(target=frame_plan.TARGET_HEADLESS, **common)

    # Exactly the three things D2 allows a target to own: where the output
    # goes, whether a swapchain is acquired and presented, whether a readback
    # follows. (`OUTPUT` is the destination copy itself — a blit onto the
    # acquired image windowed-side; on Vulkan headless a copy into the readback
    # staging buffer, and on Metal headless the readback subsumes it.)
    target_work = {frame_plan.ACQUIRE, frame_plan.OUTPUT, frame_plan.PRESENT,
                   frame_plan.PRESENT_BARRIER, frame_plan.DRAIN,
                   frame_plan.READBACK}
    assert set(windowed.steps) - target_work == set(headless.steps) - target_work

    # And the shared steps stay in the same relative order on both.
    shared = [s for s in windowed.steps if s not in target_work]
    assert shared == [s for s in headless.steps if s not in target_work]


def test_headless_reads_back_only_after_the_drain():
    plan = _derive(target=frame_plan.TARGET_HEADLESS)
    assert plan.index(frame_plan.DRAIN) < plan.index(frame_plan.READBACK)


def test_the_neural_swap_runs_only_when_online_training_is_on():
    assert not _derive(online_training=False).runs(frame_plan.ONLINE_SWAP)
    on = _derive(online_training=True)
    assert on.runs(frame_plan.ONLINE_SWAP)
    # Weights stay frozen for the frame that read them.
    assert on.index(frame_plan.DISPATCH) < on.index(frame_plan.ONLINE_SWAP)


# ── D4: the plan consumes the accumulation decision, never re-derives it ──


@pytest.mark.parametrize("accum_frame,first", [(0, True), (1, False), (900, False)])
def test_first_frame_mirrors_the_published_accumulation_index(accum_frame, first):
    plan = _derive(accum_frame=accum_frame)
    assert plan.accum_frame == accum_frame
    assert plan.first_frame is first


def test_the_plan_module_does_not_import_the_parameter_registry():
    """`_current_state_hash` owns the reset decision (change
    param-registry-accumulation-reset). A plan that could re-derive it would be
    a second owner."""
    src = (frame_plan.__file__ and open(frame_plan.__file__, encoding="utf-8").read())
    assert "_current_state_hash" not in src
    assert "STATIC_PARAMS" not in src
