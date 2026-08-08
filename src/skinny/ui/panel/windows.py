"""Browser-side child panes for the four legacy Tk windows.

Each builder returns a ``pn.Card`` with a Close button that removes the
card from its host ``pn.Column``. The host (in ``web_app``) keeps a dict
of open cards keyed by name so a second click on the sidebar button
focuses the existing pane rather than spawning a duplicate.
"""

from __future__ import annotations

import io
import logging
import math
from typing import Callable

import numpy as np
import panel as pn

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None  # type: ignore[assignment]

from skinny.bxdf_math import eval_grid, render_lobe_image
from skinny.mtlx_graph_view import (
    NodeGraphView, NodeView, build_view,
)
from skinny.ui import spec
from skinny.ui.panel.backend import PanelTreeBuilder
from skinny.ui.scene_edit_actions import apply_scene_property
from skinny.ui.scene_property_nodes import (
    graph_input_to_node,
    scene_property_to_node,
)

log = logging.getLogger(__name__)


# ── Shared helpers ────────────────────────────────────────────────


def _close_button(on_close: Callable[[], None]) -> pn.widgets.Button:
    btn = pn.widgets.Button(name="Close", button_type="default", width=80)
    btn.on_click(lambda _e: on_close())
    return btn


def _card(title: str, body: pn.viewable.Viewable, on_close) -> pn.Card:
    return pn.Card(
        _close_button(on_close), body,
        title=title, sizing_mode="stretch_width", collapsed=False,
    )


# ── Scene Graph ───────────────────────────────────────────────────


def build_scene_graph_pane(
    session, on_close: Callable[[], None],
) -> pn.Card:
    """Tree-view + property editor for the active USD scene graph."""
    from skinny.scene_graph import find_node_by_path, type_icon

    from pathlib import Path as _Path

    from skinny.settings import get_last_dir, record_last_dir
    from skinny.ui.scene_edit_actions import (
        SUPPORTED_LIGHT_TYPES,
        add_parent_for_node,
        has_editable_stage,
        is_deletable,
    )

    renderer = session.renderer
    selector = pn.widgets.Select(name="Node", options={}, size=16)
    props_col = pn.Column()
    state = {"node": None}  # currently-selected SceneGraphNode

    def _repopulate() -> None:
        graph = renderer.scene_graph
        opts: dict[str, str] = {}
        if graph is not None:
            def collect(node, depth):
                indent = "  " * depth
                icon = type_icon(node.type_name)
                label = f"{indent}{icon} {node.name}  ({node.type_name})"
                opts[label] = node.path
                for child in node.children:
                    collect(child, depth + 1)
            collect(graph, 0)
        selector.options = opts or {"(no scene loaded)": ""}

    _repopulate()

    # ── Editing controls (mirror the Qt dock; shared decision helpers) ──
    start_dir = str(get_last_dir("model") or "") or None
    file_sel = pn.widgets.FileSelector(start_dir, file_pattern="*", only_files=True)
    add_btn = pn.widgets.Button(name="Add model", button_type="primary")
    add_light_btn = pn.widgets.MenuButton(
        name="Add light",
        items=[(f"Add {light_type}", light_type) for light_type in SUPPORTED_LIGHT_TYPES],
        button_type="primary",
        disabled=not has_editable_stage(renderer),
    )
    del_btn = pn.widgets.Button(name="Delete node", button_type="danger", disabled=True)
    save_btn = pn.widgets.Button(name="Save edits", button_type="default")
    status = pn.pane.Alert("", alert_type="info", visible=False)

    def _set_status(msg: str, kind: str = "info") -> None:
        status.object = msg
        status.alert_type = kind
        status.visible = True

    def _on_add(_event) -> None:
        picked = file_sel.value
        if not picked:
            _set_status("No file selected", "warning")
            return
        path = _Path(picked[0])
        if path.suffix.lower() not in (".usda", ".usdc", ".usdz"):
            _set_status(f"Unsupported type {path.suffix}; USD only", "warning")
            return
        if getattr(renderer, "_usd_stage", None) is None:
            _set_status("Load a USD scene before adding a model.", "warning")
            return
        parent = add_parent_for_node(state["node"])
        try:
            new_path = session.run_on_render_thread(
                lambda r, p=str(path), parent=parent: r.add_model(
                    p, parent_prim_path=parent,
                )
            )
        except (ValueError, RuntimeError) as exc:
            _set_status(f"Add failed: {exc}", "danger")
            return
        record_last_dir("model", path.parent)
        _set_status(f"Added {new_path}", "success")

    def _on_delete(_event) -> None:
        node = state["node"]
        if not is_deletable(node):
            _set_status("This node cannot be deleted.", "warning")
            return
        try:
            session.run_on_render_thread(
                lambda r, p=node.path: r.remove_node(p)
            )
        except (ValueError, RuntimeError) as exc:
            _set_status(f"Delete failed: {exc}", "danger")
            return
        _set_status(f"Deleted {node.path}", "success")

    def _on_add_light(event) -> None:
        light_type = event.new
        if light_type not in SUPPORTED_LIGHT_TYPES:
            _set_status(f"Unsupported light type: {light_type}", "warning")
            return
        if not has_editable_stage(renderer):
            _set_status(
                "Load an editable USD scene before adding a light.", "warning",
            )
            return
        parent = add_parent_for_node(state["node"])
        try:
            new_path = session.run_on_render_thread(
                lambda r, t=light_type, parent=parent: r.add_light(
                    t, parent_prim_path=parent,
                )
            )
        except Exception as exc:  # noqa: BLE001 — non-fatal UI boundary
            _set_status(f"Add {light_type} failed: {exc}", "danger")
            return
        _set_status(f"Added {new_path}", "success")

    def _on_save(_event) -> None:
        if getattr(renderer, "_usd_edit_layer", None) is None:
            _set_status("No edits to save (no USD scene loaded).", "warning")
            return
        try:
            written = session.run_on_render_thread(lambda r: r.save_edits())
        except (ValueError, RuntimeError) as exc:
            _set_status(f"Save failed: {exc}", "danger")
            return
        _set_status(f"Saved edits to {written}", "success")

    add_btn.on_click(_on_add)
    add_light_btn.on_click(_on_add_light)
    del_btn.on_click(_on_delete)
    save_btn.on_click(_on_save)

    def on_select(event) -> None:
        path = event.new
        props_col.clear()
        state["node"] = None
        del_btn.disabled = True
        graph = renderer.scene_graph
        if not path or graph is None:
            return
        node = find_node_by_path(graph, path)
        if node is None:
            return
        state["node"] = node
        del_btn.disabled = not is_deletable(node)
        props_col.append(pn.pane.Markdown(
            f"**{node.name}** `{node.type_name}`\n\n`{node.path}`"
        ))
        if not node.properties:
            props_col.append(pn.pane.Markdown("*(no properties)*"))
            return
        for prop in node.properties:
            w = _build_scene_prop_widget(session, node, prop, report=_set_status)
            if w is not None:
                props_col.append(w)

    selector.param.watch(on_select, "value")

    # Repoll for scene swap / edit (id changes on rebuild; version is a backstop).
    _last_id = [-1]
    _last_ver = [-1]

    def poll() -> None:
        cur = id(renderer.scene_graph)
        ver = getattr(renderer, "_scene_graph_version", 0)
        if cur != _last_id[0] or ver != _last_ver[0]:
            _last_id[0] = cur
            _last_ver[0] = ver
            _repopulate()
            # Drop a stale selection whose node the edit removed.
            if state["node"] is not None and find_node_by_path(
                renderer.scene_graph, state["node"].path
            ) is None:
                state["node"] = None
                del_btn.disabled = True
                props_col.clear()
        save_btn.disabled = getattr(renderer, "_usd_edit_layer", None) is None
        add_light_btn.disabled = not has_editable_stage(renderer)

    pn.state.add_periodic_callback(poll, period=1000)

    controls = pn.Column(
        pn.Row(add_btn, add_light_btn, del_btn, save_btn),
        pn.Card(file_sel, title="Add model — pick a USD file", collapsed=True),
        status,
        sizing_mode="stretch_width",
    )
    return _card(
        "Scene Graph",
        pn.Column(
            controls,
            pn.Row(selector, props_col, sizing_mode="stretch_width"),
            sizing_mode="stretch_width",
        ),
        on_close,
    )


def _build_scene_prop_widget(
    session, node, prop, report=None,
) -> pn.viewable.Viewable | None:
    """Build one widget for a SceneGraphProperty edit.

    The per-prop-type switch that used to live here is gone: the mapping from a
    property to the control it needs is declared once in
    ``ui.scene_property_nodes`` (change ui-spec-scene-properties), and rendered
    by the Panel backend walker. This front-end supplies only the commit
    transport — every edit still routes through the shared
    ``apply_scene_property`` dispatcher on the render thread, and its returned
    reason surfaces in the status line via ``report``. The migration also gives
    Panel the ``lens_file`` / ``texture_file`` file pickers and the read-only
    colour / vector rows the local switch never had.
    """
    def commit(p, value) -> None:
        p.value = value
        reason = session.run_on_render_thread(
            lambda r, p=p, value=value: apply_scene_property(r, node, p, value)
        )
        if reason and report is not None:
            report(f"{p.display_name}: {reason}", "warning")

    spec_node = scene_property_to_node(prop, commit=commit)
    return PanelTreeBuilder(spec.Section(title="")).render_leaf(spec_node)


# ── BXDF Visualizer ───────────────────────────────────────────────


def build_bxdf_pane(
    session, on_close: Callable[[], None],
) -> pn.Card:
    """CPU-side analytic BSDF lobe viewer. Material is picked via combo
    (no scene-pick in the browser); shading frame is fixed to the
    tangent-space +Z normal so the user sees the analytic Lambert + GGX
    response of the material's ``parameter_overrides``.

    Mouse drag on the canvas orbits (yaw/pitch); wheel zooms. Material /
    angle changes re-run the eval; drag/zoom only re-rasterise the cached
    grid.
    """
    from bokeh.plotting import figure
    from bokeh.models import ColumnDataSource
    from bokeh.events import PanStart, Pan, MouseWheel

    renderer = session.renderer

    LOBE_SIZE = 360
    PITCH_LIMIT = math.pi * 0.49
    ORBIT_GAIN = 0.012
    ZOOM_STEP = 1.15
    ZOOM_RANGE = (0.1, 8.0)

    material_combo = pn.widgets.Select(name="Material", options={})
    theta = pn.widgets.FloatSlider(name="theta", start=0.0, end=89.0, value=30.0)
    phi = pn.widgets.FloatSlider(name="phi", start=0.0, end=359.0, value=0.0)
    lock = pn.widgets.RadioButtonGroup(
        name="Lock", options=["wi", "wo"], value="wi",
    )
    zoom_slider = pn.widgets.FloatSlider(
        name="Zoom", start=ZOOM_RANGE[0], end=ZOOM_RANGE[1], step=0.05, value=1.0,
    )
    status = pn.pane.Markdown("Select a material.")

    state: dict = {
        "yaw": math.radians(35.0),
        "pitch": math.radians(20.0),
        "zoom": 1.0,
        "dirs": None,
        "f": None,
        "drag_sx": None,
        "drag_sy": None,
    }

    # Bokeh figure hosting the lobe image. Disable axes/grid/toolbar so it
    # looks like a plain canvas; we just need its mouse events.
    img_source = ColumnDataSource(
        data=dict(image=[np.zeros((LOBE_SIZE, LOBE_SIZE), dtype=np.uint32)]),
    )
    fig = figure(
        width=LOBE_SIZE, height=LOBE_SIZE,
        x_range=(0, LOBE_SIZE), y_range=(0, LOBE_SIZE),
        tools="", toolbar_location=None, min_border=0,
        background_fill_color="#121218", outline_line_color=None,
    )
    fig.axis.visible = False
    fig.grid.visible = False
    fig.image_rgba("image", x=0, y=0, dw=LOBE_SIZE, dh=LOBE_SIZE, source=img_source)

    def _pil_to_rgba32(img) -> np.ndarray:
        arr = np.asarray(img.convert("RGBA"), dtype=np.uint8)
        rgba = arr.view(np.uint32).reshape(arr.shape[:2])
        # Bokeh's y axis runs bottom-up; flip so the image renders upright.
        return np.flipud(rgba).copy()

    def _scene_materials() -> list:
        scene = getattr(renderer, "_usd_scene", None)
        if scene is None:
            scene = getattr(renderer, "scene", None)
        if scene is None:
            return []
        return list(getattr(scene, "materials", []) or [])

    def _repopulate() -> None:
        mats = _scene_materials()
        opts: dict[str, int] = {}
        for i, mat in enumerate(mats):
            if i == 0:
                continue
            name = getattr(mat, "mtlx_target_name", None) or getattr(mat, "name", f"#{i}")
            opts[f"#{i}  {name}"] = i
        material_combo.options = opts or {"(no materials)": -1}

    _repopulate()
    _last_id = [-1]

    def poll() -> None:
        cur = getattr(renderer, "scene_version", 0)
        if cur != _last_id[0]:
            _last_id[0] = cur
            _repopulate()

    pn.state.add_periodic_callback(poll, period=1000)

    def _render_from_cache() -> None:
        dirs = state["dirs"]
        f = state["f"]
        if dirs is None or f is None:
            return
        img = render_lobe_image(
            dirs, f, state["yaw"], state["pitch"],
            size=LOBE_SIZE, log_scale=True, zoom=state["zoom"],
        )
        img_source.data = dict(image=[_pil_to_rgba32(img)])

    def _eval_and_render() -> None:
        mat_id = material_combo.value
        if mat_id is None or mat_id < 0:
            status.object = "Select a material."
            return
        mats = _scene_materials()
        if not (0 <= mat_id < len(mats)):
            return
        params = dict(getattr(mats[mat_id], "parameter_overrides", {}) or {})
        t = math.radians(float(theta.value))
        p = math.radians(float(phi.value))
        locked = np.array(
            [math.sin(t) * math.cos(p), math.sin(t) * math.sin(p), math.cos(t)],
            dtype=np.float64,
        )
        lock_mode = 0 if lock.value == "wi" else 1
        dirs, f = eval_grid(locked, lock_mode, 24, 48, params)
        state["dirs"] = dirs
        state["f"] = f
        _render_from_cache()
        name = getattr(mats[mat_id], "name", "?")
        max_f = float(f.max())
        status.object = f"#{mat_id} ({name}) — max f·cosθ = {max_f:.3f} [CPU analytic]"

    for w in (material_combo, theta, phi, lock):
        w.param.watch(lambda _e: _eval_and_render(), "value")

    def _on_zoom(event) -> None:
        state["zoom"] = float(event.new)
        _render_from_cache()
    zoom_slider.param.watch(_on_zoom, "value")

    # ── Mouse: drag = orbit, wheel = zoom ──────────────────────────
    def _on_pan_start(event) -> None:
        state["drag_sx"] = event.sx
        state["drag_sy"] = event.sy

    def _on_pan(event) -> None:
        if state["drag_sx"] is None:
            return
        dx = event.sx - state["drag_sx"]
        dy = event.sy - state["drag_sy"]
        state["drag_sx"] = event.sx
        state["drag_sy"] = event.sy
        state["yaw"] += dx * ORBIT_GAIN
        state["pitch"] = max(
            -PITCH_LIMIT, min(PITCH_LIMIT, state["pitch"] + dy * ORBIT_GAIN),
        )
        _render_from_cache()

    def _on_wheel(event) -> None:
        notches = 1 if event.delta > 0 else -1
        factor = ZOOM_STEP ** notches
        new_zoom = max(ZOOM_RANGE[0], min(ZOOM_RANGE[1], state["zoom"] * factor))
        state["zoom"] = new_zoom
        # Keep slider in sync without re-firing _on_zoom.
        zoom_slider.param.update(value=new_zoom)

    fig.on_event(PanStart, _on_pan_start)
    fig.on_event(Pan, _on_pan)
    fig.on_event(MouseWheel, _on_wheel)

    return _card(
        "BXDF Visualizer",
        pn.Column(
            material_combo,
            pn.Row(theta, phi),
            pn.Row(pn.pane.Markdown("**Lock:**"), lock),
            fig,
            zoom_slider,
            pn.pane.Markdown("*Drag to orbit · Wheel to zoom*"),
            status,
        ),
        on_close,
    )


# ── Material Graph ────────────────────────────────────────────────


def build_material_graph_pane(
    session, on_close: Callable[[], None],
) -> pn.Card:
    """Per-material node list + per-node input editor. No graphical node
    layout (that needs a JS plugin); table-style edit instead.
    """
    renderer = session.renderer

    material_combo = pn.widgets.Select(name="Material", options={})
    node_combo = pn.widgets.Select(name="Node", options={})
    inputs_col = pn.Column()
    status = pn.pane.Markdown("Pick a material to inspect.")

    state: dict = {"view": None, "by_mat": {}}

    def _scene_materials() -> list:
        scene = getattr(renderer, "_usd_scene", None)
        return list(getattr(scene, "materials", []) or []) if scene else []

    def _repopulate_materials() -> None:
        mats = _scene_materials()
        cm_map = getattr(renderer, "_mtlx_scene_materials", {}) or {}
        opts: dict[str, tuple[int, str, str]] = {}
        for i, mat in enumerate(mats):
            if i == 0:
                continue
            target = getattr(mat, "mtlx_target_name", None)
            if not target:
                cm = cm_map.get(i)
                target = getattr(cm, "target_name", None) if cm else None
            if target:
                opts[f"#{i}  {mat.name}  ({target})"] = (i, mat.name, target)
        material_combo.options = opts or {"(no MaterialX materials)": (-1, "", "")}

    def _load_view(mat_tuple) -> None:
        inputs_col.clear()
        node_combo.options = {}
        if not mat_tuple or mat_tuple[0] < 0:
            state["view"] = None
            return
        mid, name, target = mat_tuple
        lib = getattr(renderer, "_mtlx_library", None)
        if lib is None:
            status.object = "MaterialX library not loaded."
            return
        try:
            view = build_view(lib.document, mid, name, target)
        except Exception as exc:  # noqa: BLE001
            status.object = f"build_view error: {exc}"
            return
        if view is None:
            status.object = f"Could not resolve '{target}'."
            state["view"] = None
            return
        state["view"] = view
        suffix = "  [flat std_surface]" if view.flat else ""
        status.object = f"{len(view.nodes)} node(s){suffix}"
        # Output node first.
        nodes_sorted = sorted(view.nodes, key=lambda n: (not n.is_output, n.name))
        node_combo.options = {
            f"{'★' if n.is_output else ' '} {n.category} / {n.name}": n.name
            for n in nodes_sorted
        }

    def _refresh_inputs(node_name) -> None:
        inputs_col.clear()
        view = state["view"]
        if view is None or not node_name:
            return
        node = next((n for n in view.nodes if n.name == node_name), None)
        if node is None:
            return
        if not node.inputs:
            inputs_col.append(pn.pane.Markdown("*(no inputs)*"))
            return
        for inp in node.inputs:
            w = _build_graph_input_widget(session, view, node, inp)
            if w is not None:
                inputs_col.append(w)

    _repopulate_materials()
    _last_id = [-1]

    def poll() -> None:
        cur = getattr(renderer, "scene_version", 0)
        if cur != _last_id[0]:
            _last_id[0] = cur
            _repopulate_materials()

    pn.state.add_periodic_callback(poll, period=1000)

    material_combo.param.watch(lambda e: _load_view(e.new), "value")
    node_combo.param.watch(lambda e: _refresh_inputs(e.new), "value")

    return _card(
        "Material Graph",
        pn.Column(material_combo, node_combo, inputs_col, status),
        on_close,
    )


def _build_graph_input_widget(
    session, view: NodeGraphView, node: NodeView, port,
) -> pn.viewable.Viewable | None:
    """Build one widget for a material-graph input edit.

    The per-type switch moved to the shared ``graph_input_to_node`` mapper
    (change ui-spec-scene-properties); this front-end supplies only the commit
    transport (post to the render worker). The migration also gives Panel the
    ``vector2`` and ``filename`` inputs its local switch never had.
    """
    def commit(p, value) -> None:
        session.run_on_render_thread(
            lambda r, v=value: _apply_graph_edit(r, view, node, p, v)
        )

    spec_node = graph_input_to_node(port, commit=commit)
    return PanelTreeBuilder(spec.Section(title="")).render_leaf(spec_node)


def _apply_graph_edit(renderer, view, node, port, value) -> None:
    """Mirror of Qt MaterialGraphDock._apply_value_edit minus topology."""
    import MaterialX as mx

    lib = getattr(renderer, "_mtlx_library", None)
    if lib is None:
        return
    doc = lib.document

    def _find_node(name: str):
        target = doc.getChild(view.target_name)
        if target is not None:
            try:
                ss_input = target.getInput("surfaceshader")
                if ss_input is not None:
                    ss = ss_input.getConnectedNode()
                    if ss is not None and ss.getName() == name:
                        return ss
            except Exception:
                pass
        if view.nodegraph_name:
            ng = doc.getNodeGraph(view.nodegraph_name)
            if ng is not None:
                n = ng.getNode(name)
                if n is not None:
                    return n
        return None

    mx_node = _find_node(node.name)
    if mx_node is None:
        return
    inp = mx_node.getInput(port.name)
    if inp is None:
        try:
            inp = mx_node.addInput(port.name, port.type_name)
        except Exception:
            return

    t = port.type_name
    try:
        if t == "float":
            inp.setValue(float(value))
        elif t == "integer":
            inp.setValue(int(value))
        elif t == "boolean":
            inp.setValue(bool(value))
        elif t == "color3":
            r, g, b = (float(x) for x in value)
            inp.setValue(mx.Color3(r, g, b))
        elif t == "vector3":
            x, y, z = (float(v) for v in value)
            inp.setValue(mx.Vector3(x, y, z))
        else:
            return
    except Exception:
        return
    port.value = value

    if view.flat or node.is_output:
        renderer.apply_material_override(view.material_id, port.name, value)
        return
    try:
        renderer._gen_scene_materials()
        renderer._upload_graph_param_buffers()
        renderer._material_version += 1
    except Exception:
        pass


# ── Debug Viewport ────────────────────────────────────────────────


def build_debug_viewport_pane(
    session, on_close: Callable[[], None],
) -> pn.Card:
    """Embedded debug-camera view. Renders the legacy DebugViewport to an
    offscreen image and streams PNGs into a Panel image pane on a ~5 Hz
    timer. Pure server-side render; no browser interaction beyond the
    Top/Left/Back buttons.
    """
    from pathlib import Path as _Path

    from skinny.debug_viewport import DebugViewport

    ctx = session.ctx
    shader_dir = _Path(__file__).resolve().parents[2] / "shaders"

    image_pane = pn.pane.PNG(None, width=640)
    status = pn.pane.Markdown("Initialising debug viewport…")
    dv_holder: dict = {"dv": None}
    #: Latest frame the render thread produced, plus the closed flag. The render
    #: thread writes; the Bokeh timer reads. One slot, last-write-wins — a
    #: dropped debug frame costs nothing, and it is what lets the timer stay
    #: non-blocking.
    latest: dict = {"frame": None, "error": None, "closed": False}

    #: Everything below that takes a ``renderer`` argument runs on the render
    #: thread, posted through the session. The viewport owns GPU resources and
    #: writes ``renderer.debug_viewport``; the Bokeh thread only reads the
    #: latest slot and writes Panel widgets, which is the only place a Panel
    #: widget may be written from.

    def _ensure_dv(renderer) -> DebugViewport:
        if dv_holder["dv"] is not None:
            return dv_holder["dv"]
        try:
            dv = DebugViewport(
                vk_ctx=ctx, shader_dir=shader_dir,
                width=640, height=360, embedded=True,
            )
            dv.attach_renderer(renderer)
            renderer.debug_viewport = dv
            dv.open()
        except Exception as exc:
            raise RuntimeError(f"Debug viewport unavailable: {exc}") from exc
        dv_holder["dv"] = dv
        return dv

    def _render_frame(renderer) -> None:
        """Produce one debug frame into `latest`. Runs on the render thread.

        Returns nothing: this is a plain `post`, not an awaited request. The
        Bokeh timer used to wait on the reply, which blocked the single Tornado
        IOLoop for a whole render-loop iteration — up to 0.5 s once the frame
        converges and the loop starts sleeping — five times a second, starving
        every other session's video for as long as the pane stayed open.
        """
        if latest["closed"]:
            return
        try:
            dv = _ensure_dv(renderer)
            latest["frame"] = (
                dv.render_embedded(renderer) if dv.is_open else None,
                dv.camera_mode, dv._width, dv._height,
            )
            latest["error"] = None
        except Exception as exc:  # noqa: BLE001 — surfaced in the status line
            latest["frame"] = None
            latest["error"] = str(exc)

    def _apply_view(renderer, which: str) -> None:
        dv = _ensure_dv(renderer)
        if which == "top":
            dv.view_top()
        elif which == "left":
            dv.view_left()
        elif which == "back":
            dv.view_back()
        elif which == "reset":
            dv._reset_debug_camera()

    # Publish the action so the sidebar shortcut can post it without knowing
    # anything about this pane's widgets.
    session._debug_view_action = _apply_view

    def _tick() -> None:
        """Ask for the next debug frame and show the last one. Never blocks."""
        if latest["closed"]:
            return
        session.post(_render_frame, coalesce_key="debug_viewport_frame")
        if latest["error"] is not None:
            status.object = f"render failed: {latest['error']}"
            return
        frame = latest["frame"]
        if frame is None or Image is None:
            return
        pixels, mode, width, height = frame
        if pixels is None:
            return
        img = Image.frombytes("RGBA", (width, height), bytes(pixels))
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="PNG")
        image_pane.object = buf.getvalue()
        status.object = f"Mode: {mode}   Size: {width}×{height}"

    # Keep the handle: an un-stopped callback keeps firing after the pane is
    # closed, and `_render_frame` would rebuild the DebugViewport it had just
    # destroyed, forever, 5 times a second.
    tick_handle = pn.state.add_periodic_callback(_tick, period=200)

    def view(which: str):
        def _go(_e) -> None:
            try:
                session.debug_view(which)
            except Exception as exc:  # noqa: BLE001
                status.object = f"{which} view failed: {exc}"
        return _go

    btn_top = pn.widgets.Button(name="Top", width=60)
    btn_left = pn.widgets.Button(name="Left", width=60)
    btn_back = pn.widgets.Button(name="Back", width=60)
    btn_reset = pn.widgets.Button(name="Reset", width=60)
    btn_top.on_click(view("top"))
    btn_left.on_click(view("left"))
    btn_back.on_click(view("back"))
    btn_reset.on_click(view("reset"))

    def _real_close() -> None:
        # Order matters: stop producing before destroying. `closed` also stops
        # any `_render_frame` already queued from rebuilding the viewport.
        latest["closed"] = True
        if tick_handle is not None:
            try:
                tick_handle.stop()
            except Exception:  # noqa: BLE001 — already stopped
                pass
        session._debug_view_action = None
        if dv_holder["dv"] is not None:
            def teardown(renderer) -> None:
                dv = dv_holder["dv"]
                if dv is None:
                    return
                dv_holder["dv"] = None
                renderer.debug_viewport = None
                dv.destroy()

            try:
                session.run_on_render_thread(teardown)
            except Exception:
                # A dead or wedged render thread must not block closing the
                # pane, but a skipped GPU teardown is exactly what the Metal
                # dispatch-hygiene rules say must never be silent.
                log.exception(
                    "Debug viewport teardown did not run; its GPU resources "
                    "are leaked until the session is destroyed",
                )
        on_close()

    return _card(
        "Camera Debug View",
        pn.Column(
            pn.Row(btn_top, btn_left, btn_back, btn_reset),
            image_pane, status,
        ),
        _real_close,
    )
