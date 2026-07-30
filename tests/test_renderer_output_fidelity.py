"""Overlay + descriptor fidelity of the render paths (renderer-output-fidelity).

Source-level and stub-level: the real paths need a GPU context, but the two
properties under test are structural — which paths fill the overlay they copy,
and whether a per-frame descriptor write targets something already bound.
"""

from __future__ import annotations

import inspect

from skinny.renderer import Renderer

# One Vulkan body for both targets since change frame-plan-split — `render` and
# `render_headless` are the two entry points into `_execute_vulkan_frame`, which
# is where the frame is recorded. Metal keeps its own `_render_scene_metal`.
RENDER_PATHS = ("_execute_vulkan_frame", "_render_scene_metal")


class _HudOverlay:
    def __init__(self) -> None:
        self.uploaded: list[bytes] = []
        self.copies = 0

    def upload(self, data: bytes) -> None:
        self.uploaded.append(bytes(data))

    def record_copy(self, cmd=None) -> None:
        self.copies += 1


class _HudRenderer:
    """Just enough renderer for `_sync_hud_overlay` -- no GPU."""

    _build_hud_bytes = Renderer._build_hud_bytes
    _sync_hud_overlay = Renderer._sync_hud_overlay

    def __init__(self, width: int = 240, height: int = 64) -> None:
        self.width = width
        self.height = height
        self.show_hud = True
        self.hud_text_lines: list[str] = []
        self._hud_font = Renderer._load_hud_font()
        self._hud_upload_key = None
        self.hud_overlay = _HudOverlay()


def test_offscreen_frame_composites_current_overlay_content() -> None:
    """The fill has to follow the text change, not lag a frame behind it."""
    r = _HudRenderer()
    r.hud_text_lines = ["frame one"]
    r._sync_hud_overlay()
    first = r.hud_overlay.uploaded[-1]

    r.hud_text_lines = ["frame two is longer"]
    r._sync_hud_overlay()
    second = r.hud_overlay.uploaded[-1]

    assert first != second, "the overlay still holds the previous frame's mask"
    assert r.hud_overlay.copies == 2


def test_unchanged_hud_is_not_re_rasterised_but_is_still_copied() -> None:
    r = _HudRenderer()
    r.hud_text_lines = ["steady"]
    r._sync_hud_overlay()
    r._sync_hud_overlay()

    assert len(r.hud_overlay.uploaded) == 1
    assert r.hud_overlay.copies == 2


def test_empty_hud_rasterises_to_the_zero_mask() -> None:
    """A headless render that sets no text is byte-unchanged: the mask it fills
    equals the zero fill written at init and on resize."""
    r = _HudRenderer()
    r.show_hud = False
    r._sync_hud_overlay()
    assert r.hud_overlay.uploaded == [bytes(r.width * r.height)]

    r2 = _HudRenderer()
    r2.hud_text_lines = []
    r2._sync_hud_overlay()
    assert r2.hud_overlay.uploaded == [bytes(r2.width * r2.height)]


def test_every_path_that_copies_the_overlay_also_fills_it() -> None:
    """No render path may touch `hud_overlay` directly -- fill and copy are one
    call, so a path added later cannot copy staging it never filled."""
    for name in RENDER_PATHS:
        body = inspect.getsource(getattr(Renderer, name))
        assert "self._sync_hud_overlay(" in body, f"{name} does not sync the HUD"
        assert "hud_overlay.record_copy" not in body, name
        assert "hud_overlay.upload" not in body, name


def test_only_the_sync_helper_touches_the_overlay_per_frame() -> None:
    src = inspect.getsource(Renderer)
    # Allocation sites (init + resize) zero-fill; nothing else uploads or copies.
    assert src.count("self.hud_overlay.upload(") == 3, (
        "hud_overlay.upload belongs to _sync_hud_overlay plus the two "
        "zero-fills at construction and resize"
    )
    assert src.count("self.hud_overlay.record_copy(") == 1


def test_offscreen_path_does_not_rebind_what_is_already_bound() -> None:
    """`render_headless` rewrote binding 1 every call to the image it already
    pointed at, with a comment claiming `render()` rebinds it to the acquired
    swapchain image. `render()` blits instead; the rewrite was dead work."""
    body = inspect.getsource(Renderer.render_headless)
    assert "dstBinding=1" not in body
    assert "vkUpdateDescriptorSets" not in body
    # ...and the docstring describes what `render()` actually does with it.
    assert "blits" in Renderer.render_headless.__doc__


def test_fence_is_reset_only_immediately_before_its_submit() -> None:
    """A fence reset early stays unsignaled across every exception-capable step
    that follows, so a caller that catches and retries blocks forever in the
    wait — the permanent freeze the web render-loop guard exists to prevent
    (codex pre-merge review, finding 1).
    """
    # Both targets go through the one body now (change frame-plan-split), so
    # there is one reset to check instead of two copies of the same rule.
    for name in ("_execute_vulkan_frame",):
        lines = [l.strip() for l in inspect.getsource(getattr(Renderer, name)).splitlines()]
        resets = [i for i, l in enumerate(lines) if l.startswith("vk.vkResetFences(")]
        assert resets, f"{name} has no fence reset"
        for i in resets:
            nxt = next(l for l in lines[i + 1:] if l and not l.startswith("#"))
            assert nxt.startswith("vk.vkQueueSubmit("), (
                f"{name}: fence reset is not immediately followed by its submit, "
                f"found {nxt!r}"
            )


# ── plan.steps vs the executor's real order (codex review, finding 3) ─────

# Each Vulkan step, and the source marker in `_execute_vulkan_frame` that
# performs it. `plan.steps` used to be a symbolic tuple nothing replayed, so the
# plan and the executor were two independent, unreconciled authorities on order.
# Until `gpu-backend-adapter`'s recording adapter can replay the plan for real
# (task 4.3), this pins them to each other by source order.
_VULKAN_STEP_MARKERS = {
    "fence_wait": "vk.vkWaitForFences(",
    "pick_drain": "self.poll_pick_result()",
    "pack_uniforms": "self.uniform_buffer.upload(",
    "upload_mtlx": "self.mtlx_skin_buffer.upload_sync(",
    "begin_cmd": "vk.vkBeginCommandBuffer(",
    "hud": "self._sync_hud_overlay(cmd)",
    "dispatch": "self._record_frame_dispatch(",
    "output": "target.record_output(cmd)",
    "end_cmd": "vk.vkEndCommandBuffer(",
    "fence_reset": "vk.vkResetFences(",
    "submit": "vk.vkQueueSubmit(",
    "rotate_frame": "self.current_frame = (f + 1)",
}


def test_plan_step_order_matches_the_executor_source_order() -> None:
    """Every step the plan names, that the shared body performs, appears in the
    body in the plan's order. A plan whose order drifted from the code it claims
    to describe would assert an invariant about nothing."""
    from skinny import frame_plan

    body = inspect.getsource(Renderer._execute_vulkan_frame)
    lines = body.splitlines()
    positions = {}
    for step, marker in _VULKAN_STEP_MARKERS.items():
        hits = [i for i, line in enumerate(lines) if marker in line]
        assert hits, f"no source marker for planned step {step!r}: {marker!r}"
        positions[step] = hits[0]

    for target in (frame_plan.TARGET_WINDOWED, frame_plan.TARGET_HEADLESS):
        plan = frame_plan.derive(
            target=target, execution_mode_index=frame_plan.EXECUTION_WAVEFRONT,
            integrator_index=0, accum_frame=0, width=64, height=64,
            needs_watchdog_tiling=False, records_command_buffers=True,
            mlt_num_chains=16384, has_heavy_nonflat=False)
        planned = [s for s in plan.steps if s in positions]
        actual = sorted(planned, key=positions.__getitem__)
        assert planned == actual, (
            f"{target}: plan order {planned} disagrees with the executor's "
            f"source order {actual}")


def test_the_plan_is_derived_after_the_pick_drain() -> None:
    """A pick callback mutates what the plan reads — `_on_autofocus_hit` sets
    `accum_frame = 0`. Deriving before the drain hands the dispatch a
    `first_frame` that disagrees with the packed `fc.accumFrame`
    (codex pre-merge review, finding 1)."""
    for name in ("_execute_vulkan_frame", "_render_windowed_metal", "render_headless"):
        lines = inspect.getsource(getattr(Renderer, name)).splitlines()
        drain = next(i for i, l in enumerate(lines)
                     if "self.poll_pick_result()" in l)
        derive = next(i for i, l in enumerate(lines)
                      if "self._derive_frame_plan(" in l)
        assert drain < derive, f"{name} derives the plan before draining picks"


def test_the_weight_swap_reads_online_training_live() -> None:
    """Arming online training is a frame-END decision; a start-of-frame snapshot
    defers an OFF->ON transition by a frame (codex pre-merge review, finding 4)."""
    for name in ("_execute_vulkan_frame", "render", "render_headless"):
        body = inspect.getsource(getattr(Renderer, name))
        if "_online_frame_end_swap" not in body:
            continue
        assert "if self._online_training:" in body, (
            f"{name} must gate the swap on live state, not a planned flag")
        assert "plan.online_swap" not in body
