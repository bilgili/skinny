"""Single source of truth for Skinny's control UI tree.

`build_main_ui(renderer, callbacks)` returns a `Section` describing every
visible control. Both the Qt desktop backend and the Panel web backend
walk the same tree, so adding a control here adds it everywhere.

Renderer-only state (params, presets, materials, environments) is bound
via closures over `params._get_nested` / `_set_nested` and the renderer's
own action API. Backend-specific actions (open child windows, save a
screenshot to a path) come in through the `AppCallbacks` dataclass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

from skinny.params import (
    ParamSpec, RESOLUTION_PRESETS, _get_nested, _set_nested,
    build_all_params,
)
from skinny.presets import apply_preset
from skinny.ui.spec import Section, UIBuilder


# ── Constants ──────────────────────────────────────────────────────


# (label, encoder format string, file extension) — reused by both backends.
CAPTURE_FORMATS: list[tuple[str, str, str]] = [
    ("PNG",  "png",  "png"),
    ("JPEG", "jpeg", "jpg"),
    ("BMP",  "bmp",  "bmp"),
    ("EXR",  "exr",  "exr"),
    ("HDR",  "hdr",  "hdr"),
]


MODEL_FILE_FILTERS: list[tuple[str, str]] = [
    ("All supported", "*.usda *.usdc *.usdz *.obj"),
    ("USD files",     "*.usda *.usdc *.usdz"),
    ("OBJ files",     "*.obj"),
    ("All files",     "*.*"),
]


HDR_FILE_FILTERS: list[tuple[str, str]] = [
    ("HDR images",  "*.hdr *.exr *.pfm"),
    ("Radiance HDR", "*.hdr"),
    ("OpenEXR",      "*.exr"),
    ("PFM",          "*.pfm"),
    ("All files",    "*.*"),
]


# Light RGB and elev/az get dedicated widgets (color picker, direction
# picker), so we hide their individual sliders from the generic param
# loop. Same set the old Tk panel used (`control_panel._HIDDEN_PANEL_PATHS`).
_DEDICATED_WIDGET_PATHS: frozenset[str] = frozenset({
    "light_color_r", "light_color_g", "light_color_b",
    "light_elevation", "light_azimuth",
})


# ── Host-supplied callbacks ────────────────────────────────────────


@dataclass
class AppCallbacks:
    """Actions only the host (Qt app / Panel app) can implement.

    Window openers (``open_*``) are rendered as a menu in Qt and a row
    of buttons in Panel — they no longer appear in the shared widget
    tree because dock-vs-menu placement is backend-specific.

    ``capture_screenshot`` lets the web path acquire its session lock
    around ``save_screenshot``. ``load_model`` / ``load_hdr`` exist so
    the web backend can serialise file loads under the same lock.
    """
    open_scene_graph: Callable[[], None] = field(default=lambda: None)
    open_material_graph: Callable[[], None] = field(default=lambda: None)
    open_bxdf_visualizer: Callable[[], None] = field(default=lambda: None)
    open_debug_viewport: Callable[[], None] = field(default=lambda: None)
    capture_screenshot: Callable[[str], bytes] | None = None
    load_model: Callable[[Path], None] | None = None
    load_hdr: Callable[[Path], None] | None = None


# ── Param grouping ─────────────────────────────────────────────────


def _classify(p: ParamSpec) -> str:
    """Map a ParamSpec to its UI section key. Mirrors the rules in
    ``web_app._group_params`` so Qt and web stay layout-identical.
    """
    path = p.path
    if path == "preset_index" or path.startswith("mtlx."):
        return "Skin"
    if path in ("env_index", "env_intensity"):
        return "IBL"
    if path.startswith("light") or path == "direct_light_index":
        return "Direct Light"
    if path in ("normal_map_strength", "displacement_scale_mm",
                "detail_maps_index"):
        return "Detail"
    return "Render"


def _group_params(
    renderer,
) -> tuple[dict[str, list[ParamSpec]], list[ParamSpec]]:
    """Return ``(by_group, all_params)``. Empty groups are dropped."""
    all_params = build_all_params(renderer)
    groups: dict[str, list[ParamSpec]] = {
        "Render": [], "Skin": [], "Detail": [], "IBL": [], "Direct Light": [],
    }
    for p in all_params:
        groups[_classify(p)].append(p)
    return {k: v for k, v in groups.items() if v}, all_params


# ── Param → widget closures ────────────────────────────────────────


def _choice_labels(renderer, choice_source: str) -> list[str]:
    """Resolve a ParamSpec.choice_source into a list of display labels.
    Matches the ``getattr(c, 'name', str(c))`` rule the Tk + Panel paths
    used so labels stay identical.
    """
    raw = getattr(renderer, choice_source, None) or []
    return [getattr(c, "name", str(c)) for c in raw]


def _add_param(ui: UIBuilder, renderer, p: ParamSpec) -> None:
    """Append a Slider or Combo for one ParamSpec, with a setter that
    routes preset selections through ``apply_preset`` so the renderer's
    accumulation reset fires correctly.
    """
    if p.kind == "continuous":
        ui.slider(
            p.name,
            getter=lambda path=p.path: float(_get_nested(renderer, path)),
            setter=lambda v, path=p.path: _set_nested(renderer, path, float(v)),
            lo=p.lo, hi=p.hi, step=p.step,
        )
        return

    # Discrete — combo. preset_index has the side-effect of applying the
    # preset; everything else is a plain index write.
    def _get(path=p.path) -> int:
        return int(_get_nested(renderer, path))

    if p.path == "preset_index":
        def _set(idx: int, path: str = "preset_index") -> None:
            _set_nested(renderer, path, int(idx))
            presets = getattr(renderer, "presets", [])
            if 0 <= idx < len(presets):
                apply_preset(renderer, presets[idx])
    else:
        def _set(idx: int, path: str = p.path) -> None:
            _set_nested(renderer, path, int(idx))

    ui.combo(
        p.name, getter=_get, setter=_set,
        choices=lambda src=p.choice_source: _choice_labels(renderer, src),
    )


# ── Light helpers ──────────────────────────────────────────────────


def _add_light_color(ui: UIBuilder, renderer) -> None:
    def _get() -> tuple[float, float, float]:
        return (
            float(getattr(renderer, "light_color_r")),
            float(getattr(renderer, "light_color_g")),
            float(getattr(renderer, "light_color_b")),
        )

    def _set(rgb: tuple[float, float, float]) -> None:
        renderer.light_color_r = float(rgb[0])
        renderer.light_color_g = float(rgb[1])
        renderer.light_color_b = float(rgb[2])
        # Existing Tk + web both call _update_light after editing light
        # state so the cached spherical→cartesian direction stays in sync.
        if hasattr(renderer, "_update_light"):
            renderer._update_light()

    ui.color("Color", getter=_get, setter=_set)


def _add_light_direction(ui: UIBuilder, renderer) -> None:
    ui.direction_picker(
        "Direction",
        elev_getter=lambda: float(getattr(renderer, "light_elevation")),
        elev_setter=lambda v: _set_light_angle(renderer, "light_elevation", v),
        az_getter=lambda: float(getattr(renderer, "light_azimuth")),
        az_setter=lambda v: _set_light_angle(renderer, "light_azimuth", v),
    )


def _set_light_angle(renderer, attr: str, value: float) -> None:
    setattr(renderer, attr, float(value))
    if hasattr(renderer, "_update_light"):
        renderer._update_light()


# ── Resolution + capture ───────────────────────────────────────────


def _add_resolution(ui: UIBuilder, renderer) -> None:
    def _apply(w: int, h: int) -> tuple[int, int]:
        renderer.resize(int(w), int(h))
        return int(renderer.width), int(renderer.height)

    ui.resolution_picker(
        presets=RESOLUTION_PRESETS,
        width_getter=lambda: int(renderer.width),
        height_getter=lambda: int(renderer.height),
        on_apply=_apply,
    )


def _add_capture(ui: UIBuilder, renderer, capture_fn=None) -> None:
    """Screenshot picker. ``capture_fn(fmt) -> bytes`` lets the host wrap
    the call (e.g. acquire the per-session render lock in the web path).
    Defaults to a direct ``Renderer.save_screenshot`` into a ``BytesIO``.
    """
    if capture_fn is None:
        import io as _io
        def capture_fn(fmt: str) -> bytes:
            buf = _io.BytesIO()
            renderer.save_screenshot(buf, fmt)
            return buf.getvalue()

    ui.screenshot_picker(formats=CAPTURE_FORMATS, capture=capture_fn)


# ── Materials ──────────────────────────────────────────────────────


# Std-surface keys + their UI ranges. Same set both legacy paths used.
_MATERIAL_SLIDER_KEYS: list[tuple[str, float, float]] = [
    ("roughness",      0.04, 1.0),
    ("metallic",       0.0,  1.0),
    ("specular",       0.0,  1.0),
    ("opacity",        0.0,  1.0),
    ("ior",            1.0,  3.0),
    ("coat",           0.0,  1.0),
    ("coat_roughness", 0.0,  1.0),
]


def _coerce_color3(value, fallback=(0.72, 0.72, 0.72)) -> tuple[float, float, float]:
    if value is None:
        return fallback
    if hasattr(value, "asTuple"):
        seq = value.asTuple()
    elif hasattr(value, "__getitem__") and not isinstance(value, str):
        try:
            seq = (value[0], value[1], value[2])
        except (IndexError, TypeError):
            return fallback
    else:
        return fallback
    try:
        return float(seq[0]), float(seq[1]), float(seq[2])
    except (TypeError, ValueError):
        return fallback


def build_material_subtree(ui: UIBuilder, renderer) -> None:
    """Populate the Materials dynamic section. Called by both backends
    when the active scene changes (token = ``id(renderer._usd_scene)``).
    """
    scene = getattr(renderer, "_usd_scene", None)
    if scene is None or not getattr(scene, "materials", None):
        return

    # Skip the implicit skin material at index 0 — it has its own dedicated
    # "Skin" section above (mtlx.* sliders). Mirrors both legacy paths.
    for mat_id, mat in list(enumerate(scene.materials))[1:]:
        label = getattr(mat, "mtlx_target_name", None) or mat.name
        with ui.section(label, expanded=False):
            _add_material_block(ui, renderer, mat_id, mat)


def _add_material_block(ui: UIBuilder, renderer, mat_id: int, mat) -> None:
    # diffuseColor — hide swatch when texture-bound (texture overrides
    # the constant; the swatch would be dead state).
    if "diffuseColor" not in getattr(mat, "texture_paths", {}):
        def _diff_get(m=mat) -> tuple[float, float, float]:
            return _coerce_color3(m.parameter_overrides.get("diffuseColor"))

        def _diff_set(rgb, mid=mat_id) -> None:
            renderer.apply_material_override(mid, "diffuseColor", tuple(rgb))

        ui.color("diffuseColor", getter=_diff_get, setter=_diff_set)

    for key, lo, hi in _MATERIAL_SLIDER_KEYS:
        def _get(m=mat, k=key, default=0.5) -> float:
            cur = m.parameter_overrides.get(k)
            try:
                return float(cur) if cur is not None else default
            except (TypeError, ValueError):
                return default

        def _set(v, mid=mat_id, k=key) -> None:
            renderer.apply_material_override(mid, k, float(v))

        ui.slider(key, getter=_get, setter=_set, lo=lo, hi=hi)

    def _furnace_get(mid=mat_id) -> bool:
        # Renderer doesn't surface a per-mat read API; default off so the
        # UI mirrors the Tk + web "fresh checkbox per rebuild" behaviour.
        return False

    def _furnace_set(v: bool, mid=mat_id) -> None:
        renderer.toggle_material_furnace(mid, bool(v))

    ui.checkbox("Furnace", getter=_furnace_get, setter=_furnace_set)

    # Dynamic MaterialX graph uniforms (noise_octaves, color_mix_fg, …).
    # iter_graph_uniforms already filters out filename + string types.
    graph_uniforms = renderer.iter_graph_uniforms(mat_id)
    if graph_uniforms:
        with ui.section("MaterialX Graph Inputs", expanded=False):
            for u in graph_uniforms:
                _add_graph_uniform(ui, renderer, mat_id, mat, u)
            ui.button(
                "Reset to defaults",
                on_click=lambda mid=mat_id, ulist=graph_uniforms:
                    _reset_graph_uniforms(renderer, mid, ulist),
            )


def _add_graph_uniform(ui: UIBuilder, renderer, mat_id: int, mat, u) -> None:
    """One MaterialX graph UniformField → backend-agnostic widget.

    Type → widget mapping matches both legacy paths verbatim:
      - boolean → Checkbox
      - integer → IntSpin [0, 32]
      - float   → Slider [0, max(2 * default, 1)]
      - color3/4 → Color
      - vector2/3/4 → Vector [0, 4]
    """
    def _read():
        return mat.parameter_overrides.get(u.name, u.default)

    if u.type_name == "boolean":
        ui.checkbox(
            u.name,
            getter=lambda: bool(_read()) if _read() is not None else False,
            setter=lambda v, mid=mat_id, k=u.name:
                renderer.apply_material_override(mid, k, int(bool(v))),
        )
        return

    if u.type_name == "integer":
        def _ig() -> int:
            cur = _read()
            try:
                return int(cur) if cur is not None else 0
            except (TypeError, ValueError):
                return 0

        ui.int_spin(
            u.name, getter=_ig,
            setter=lambda v, mid=mat_id, k=u.name:
                renderer.apply_material_override(mid, k, int(v)),
            lo=0, hi=32,
        )
        return

    if u.type_name == "float":
        def _fg() -> float:
            cur = _read()
            try:
                return float(cur) if cur is not None else 0.0
            except (TypeError, ValueError):
                return 0.0

        # Heuristic upper bound (matches both legacy paths): 2× default with
        # a floor of 1.0 — covers typical [0,1] params and larger ones like
        # noise_diminish=0.5 or scale_pos_in2=6.
        hi = max(_fg() * 2.0, 1.0)
        ui.slider(
            u.name, getter=_fg,
            setter=lambda v, mid=mat_id, k=u.name:
                renderer.apply_material_override(mid, k, float(v)),
            lo=0.0, hi=hi, step=hi / 100.0,
        )
        return

    if u.type_name in ("color3", "color4"):
        def _cg() -> tuple[float, float, float]:
            return _coerce_color3(_read(), fallback=(0.8, 0.8, 0.8))

        ui.color(
            u.name, getter=_cg,
            setter=lambda rgb, mid=mat_id, k=u.name:
                renderer.apply_material_override(mid, k, tuple(rgb)),
        )
        return

    if u.type_name in ("vector2", "vector3", "vector4"):
        comps = {"vector2": 2, "vector3": 3, "vector4": 4}[u.type_name]

        def _vg() -> tuple[float, ...]:
            cur = _read()
            seq: list[float] = [0.0] * comps
            if cur is not None:
                src = cur.asTuple() if hasattr(cur, "asTuple") else cur
                for i in range(comps):
                    try:
                        seq[i] = float(src[i])
                    except (TypeError, IndexError, ValueError):
                        seq[i] = 0.0
            return tuple(seq)

        ui.vector(
            u.name, components=comps, getter=_vg,
            setter=lambda v, mid=mat_id, k=u.name:
                renderer.apply_material_override(mid, k, tuple(v)),
            lo=0.0, hi=4.0,
        )


def _reset_graph_uniforms(renderer, mat_id: int, uniforms) -> None:
    for u in uniforms:
        d = u.default
        if d is None:
            continue
        if hasattr(d, "asTuple"):
            d = d.asTuple()
        elif hasattr(d, "__getitem__") and not isinstance(d, (str, bytes)):
            d = tuple(d[i] for i in range(min(4, len(d))))
        renderer.apply_material_override(mat_id, u.name, d)


# ── Top-level builder ──────────────────────────────────────────────


def build_main_ui(renderer, callbacks: AppCallbacks | None = None) -> Section:
    """Single source of truth for both desktop + web UIs.

    Walking the returned tree visits every visible widget in the order
    they should appear; backends instantiate accordingly.
    """
    cb = callbacks or AppCallbacks()
    ui = UIBuilder()

    grouped, _all_params = _group_params(renderer)

    with ui.section("Resolution"):
        _add_resolution(ui, renderer)
    with ui.section("Capture"):
        _add_capture(ui, renderer, capture_fn=cb.capture_screenshot)

    # IBL + Direct Light intentionally omitted from the sidebar — they
    # live in the scene-graph dock now (the user edits DomeLight /
    # DistantLight / SphereLight prims directly there).
    section_order = ["Render", "Skin", "Detail"]
    for group in section_order:
        params_in_group = grouped.get(group)
        if not params_in_group:
            continue
        with ui.section(group, expanded=(group != "Skin")):
            for p in params_in_group:
                if p.path in _DEDICATED_WIDGET_PATHS:
                    continue
                _add_param(ui, renderer, p)

    # Materials + Scene Graph sections used to be sidebar dynamic-section
    # accordions. Removed — those live in dedicated docks (Material
    # Graph + Scene Graph) opened from the View menu.
    # Window-opener buttons are not in the tree — Qt renders them as a
    # menu and Panel as a row of buttons, both via the AppCallbacks
    # fields directly.

    return ui.tree
