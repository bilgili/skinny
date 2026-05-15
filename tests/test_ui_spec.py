"""Parity tests for the backend-agnostic UI spec.

These tests don't touch Vulkan — a stub renderer with the attributes
``build_app_ui`` reads is enough to assert tree structure.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from skinny.params import STATIC_PARAMS, build_all_params
from skinny.ui import spec
from skinny.ui.build_app_ui import (
    _DEDICATED_WIDGET_PATHS, build_main_ui,
)


# ── Stub renderer ──────────────────────────────────────────────────


@dataclass
class _Choice:
    name: str


def _named(*names: str) -> list[_Choice]:
    return [_Choice(n) for n in names]


class _StubRenderer:
    """Just enough surface area for ``build_main_ui`` to construct the
    tree. No Vulkan, no shaders.
    """

    def __init__(self) -> None:
        # Every choice_source referenced by params.STATIC_PARAMS.
        self.presets             = _named("Default")
        self.environments        = _named("studio.hdr")
        self.direct_light_modes  = _named("On", "Off")
        self.scatter_modes       = _named("BSSRDF", "Volume")
        self.integrator_modes    = _named("Path", "Direct")
        self.furnace_modes       = _named("Off", "On")
        self.models              = _named("(none)")
        self.detail_maps_modes   = _named("Off", "On")
        self.tattoos             = _named("(none)")

        # Indices.
        self.preset_index = 0
        self.env_index = 0
        self.direct_light_index = 0
        self.scatter_index = 0
        self.integrator_index = 0
        self.furnace_index = 0
        self.model_index = 0
        self.detail_maps_index = 0
        self.tattoo_index = 0

        # Continuous scalars (every non-mtlx continuous path).
        self.env_intensity        = 1.0
        self.mm_per_unit          = 5.0
        self.normal_map_strength  = 1.0
        self.displacement_scale_mm = 0.0
        self.tattoo_density       = 0.0
        self.light_elevation      = 0.0
        self.light_azimuth        = 0.0
        self.light_intensity      = 1.0
        self.light_color_r        = 1.0
        self.light_color_g        = 1.0
        self.light_color_b        = 1.0

        # MaterialX overrides bag (read by params._get_nested for mtlx.*).
        self.mtlx_overrides: dict = {}
        self._mtlx_skin_material = None  # no dynamic params

        # Resolution + scene state read by build_main_ui.
        self.width = 1280
        self.height = 720
        self._usd_scene = None
        self.scene_graph = None

    # Minimal action-API surface.
    def resize(self, w: int, h: int) -> None:
        self.width = int(w)
        self.height = int(h)

    def _update_light(self) -> None:
        pass


@pytest.fixture
def stub_renderer() -> _StubRenderer:
    return _StubRenderer()


# ── Tests ──────────────────────────────────────────────────────────


def _collect_bound_paths(node: spec.Node) -> list[str]:
    """Walk the tree and return the ParamSpec.path implied by every
    Slider/Combo getter via its closure cell. Both ``_add_param`` setters
    capture the path as a default arg, so we read ``setter.__defaults__``.
    """
    out: list[str] = []
    for n in spec.walk(node):
        if isinstance(n, (spec.Slider, spec.Combo)):
            defaults = getattr(n.setter, "__defaults__", None) or ()
            for d in defaults:
                if isinstance(d, str):
                    out.append(d)
                    break
    return out


def test_every_param_bound_exactly_once(stub_renderer):
    """Every entry from ``build_all_params`` either appears as a
    Slider/Combo bound to its path, or is on the dedicated-widget
    allowlist (light RGB + elev/az), exactly once.
    """
    tree = build_main_ui(stub_renderer)
    bound = _collect_bound_paths(tree)
    bound_set = set(bound)

    assert len(bound) == len(bound_set), (
        f"Duplicate path bindings in tree: {bound}"
    )

    expected = {p.path for p in build_all_params(stub_renderer)}
    # IBL + Direct Light params live in the scene-graph dock now, not in
    # the sidebar; their paths intentionally absent from the sidebar tree.
    sidebar_excluded = _DEDICATED_WIDGET_PATHS | {
        "env_index", "env_intensity",
        "direct_light_index", "light_intensity",
    }
    missing = expected - bound_set - sidebar_excluded
    assert not missing, f"Params missing from UI tree: {sorted(missing)}"

    extra = bound_set - expected
    assert not extra, f"Tree binds unknown paths: {sorted(extra)}"


def test_dedicated_widgets_not_double_bound(stub_renderer):
    """light_color_* and light_elev/az must NOT also appear as Slider rows
    — they're owned by the Color and DirectionPicker widgets.
    """
    tree = build_main_ui(stub_renderer)
    bound = set(_collect_bound_paths(tree))
    leaked = bound & _DEDICATED_WIDGET_PATHS
    assert not leaked, f"Dedicated widget paths leaked into sliders: {leaked}"


def test_top_level_section_order(stub_renderer):
    """Backends rely on stable section ordering. Lock it."""
    tree = build_main_ui(stub_renderer)
    titles = [c.title for c in tree.children
              if isinstance(c, (spec.Section, spec.DynamicSection))]
    assert titles == [
        "Resolution", "Capture", "Load Model",
        "Render", "Skin", "Detail",
        "Materials", "Scene Graph",
    ]


def test_dedicated_widgets_present(stub_renderer):
    """ResolutionPicker, ScreenshotPicker, FilePicker — sidebar widgets
    that don't map to a single ParamSpec.path.
    """
    tree = build_main_ui(stub_renderer)
    kinds = {type(n).__name__ for n in spec.walk(tree)}
    for required in ("ResolutionPicker", "ScreenshotPicker", "FilePicker"):
        assert required in kinds, f"Missing {required} in tree"


def test_dynamic_section_token_drives_rebuild(stub_renderer):
    """Dynamic section's ``rebuild_token()`` controls when the body
    rebuilds. Same scene id → no rebuild; new id → rebuild.
    """
    tree = build_main_ui(stub_renderer)
    dyn = next(
        n for n in tree.children if isinstance(n, spec.DynamicSection)
        and n.title == "Materials"
    )
    # Token is currently ``id(None)`` because _usd_scene is None.
    assert dyn.rebuild_token() == id(None)

    # Swap in a scene with materials.
    class _Mat:
        def __init__(self, name):
            self.name = name
            self.mtlx_target_name = None
            self.parameter_overrides = {}
            self.texture_paths = {}

    class _Scene:
        materials = [object(), _Mat("alpha")]

    stub_renderer._usd_scene = _Scene()
    stub_renderer.apply_material_override = lambda mid, k, v: None
    stub_renderer.toggle_material_furnace = lambda mid, v: None
    stub_renderer.iter_graph_uniforms = lambda mid: []

    assert dyn.rebuild_token() == id(stub_renderer._usd_scene)

    # Run the build closure and verify it produces widgets.
    sub = spec.UIBuilder()
    dyn.build(sub)
    # Material "alpha" → one Section with color + 7 sliders + furnace checkbox.
    sections = [n for n in sub.tree.children if isinstance(n, spec.Section)]
    assert len(sections) == 1
    assert sections[0].title == "alpha"


def test_window_openers_in_callbacks(stub_renderer):
    """Window-open callbacks are not rendered as sidebar buttons anymore;
    they live on AppCallbacks for the backend to expose (menu in Qt,
    button row in Panel). Confirm the dataclass still carries them.
    """
    from skinny.ui.build_app_ui import AppCallbacks

    fired: list[str] = []
    cb = AppCallbacks(
        open_scene_graph=lambda: fired.append("sg"),
        open_material_graph=lambda: fired.append("mg"),
        open_bxdf_visualizer=lambda: fired.append("bxdf"),
        open_debug_viewport=lambda: fired.append("dbg"),
    )
    cb.open_scene_graph()
    cb.open_material_graph()
    cb.open_bxdf_visualizer()
    cb.open_debug_viewport()
    assert fired == ["sg", "mg", "bxdf", "dbg"]

    # Sidebar tree no longer contains Button nodes for these openers.
    tree = build_main_ui(stub_renderer, callbacks=cb)
    button_labels = {b.label for b in spec.walk(tree) if isinstance(b, spec.Button)}
    for legacy in (
        "Scene Graph...", "Material Graph...", "BXDF Visualizer...",
        "Camera Debug View", "Top", "Left", "Back",
    ):
        assert legacy not in button_labels, (
            f"{legacy!r} should be host-rendered, not in the tree"
        )
