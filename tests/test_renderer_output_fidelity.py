"""Overlay + descriptor fidelity of the render paths (renderer-output-fidelity).

Source-level and stub-level: the real paths need a GPU context, but the two
properties under test are structural — which paths fill the overlay they copy,
and whether a per-frame descriptor write targets something already bound.
"""

from __future__ import annotations

import inspect

from skinny.renderer import Renderer

RENDER_PATHS = ("render", "render_headless", "_render_scene_metal")


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
    for name in ("render", "render_headless"):
        lines = [l.strip() for l in inspect.getsource(getattr(Renderer, name)).splitlines()]
        resets = [i for i, l in enumerate(lines) if l.startswith("vk.vkResetFences(")]
        assert resets, f"{name} has no fence reset"
        for i in resets:
            nxt = next(l for l in lines[i + 1:] if l and not l.startswith("#"))
            assert nxt.startswith("vk.vkQueueSubmit("), (
                f"{name}: fence reset is not immediately followed by its submit, "
                f"found {nxt!r}"
            )
