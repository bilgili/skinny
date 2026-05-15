"""Browser-side child panes for the four legacy Tk windows.

Each builder returns a ``pn.Card`` with a Close button that removes the
card from its host ``pn.Column``. The host (in ``web_app``) keeps a dict
of open cards keyed by name so a second click on the sidebar button
focuses the existing pane rather than spawning a duplicate.
"""

from __future__ import annotations

import io
import math
import struct
from datetime import datetime
from typing import Callable, Optional

import numpy as np
import panel as pn

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None  # type: ignore[assignment]

from skinny.bxdf_math import eval_grid, render_lobe_image
from skinny.mtlx_graph_view import (
    _ADDABLE_CATEGORIES, NodeGraphView, NodeView, build_view,
)


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

    renderer = session.renderer
    selector = pn.widgets.Select(name="Node", options={}, size=16)
    props_col = pn.Column()

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

    def on_select(event) -> None:
        path = event.new
        props_col.clear()
        graph = renderer.scene_graph
        if not path or graph is None:
            return
        node = find_node_by_path(graph, path)
        if node is None:
            return
        props_col.append(pn.pane.Markdown(
            f"**{node.name}** `{node.type_name}`\n\n`{node.path}`"
        ))
        if not node.properties:
            props_col.append(pn.pane.Markdown("*(no properties)*"))
            return
        for prop in node.properties:
            w = _build_scene_prop_widget(session, node, prop)
            if w is not None:
                props_col.append(w)

    selector.param.watch(on_select, "value")

    # Repoll for scene swap.
    _last_id = [-1]

    def poll() -> None:
        cur = id(renderer.scene_graph)
        if cur != _last_id[0]:
            _last_id[0] = cur
            _repopulate()

    pn.state.add_periodic_callback(poll, period=1000)

    return _card(
        "Scene Graph",
        pn.Row(selector, props_col, sizing_mode="stretch_width"),
        on_close,
    )


def _build_scene_prop_widget(
    session, node, prop,
) -> pn.viewable.Viewable | None:
    """Build one widget for a SceneGraphProperty edit. Routes through
    ``Renderer.apply_*`` so accumulation resets fire.
    """
    renderer = session.renderer
    ref = node.renderer_ref or _find_material_ancestor(renderer, node)

    if prop.type_name == "bool" and prop.editable:
        w = pn.widgets.Checkbox(name=prop.display_name, value=bool(prop.value))

        def on_bool(event, p=prop, n=node):
            value = bool(event.new)
            p.value = value
            toggle = p.metadata.get("toggle", "node")
            with session._lock:
                if toggle == "subtree":
                    renderer.apply_subtree_enabled(n.path, value)
                else:
                    r = n.renderer_ref
                    if r is None:
                        return
                    if r.kind == "renderer_camera":
                        renderer.apply_camera_param(p.name, value)
                    else:
                        renderer.apply_node_enabled(r.kind, r.index, value)

        w.param.watch(on_bool, "value")
        return w

    if prop.type_name == "float" and prop.editable:
        lo = float(prop.metadata.get("min", 0.0))
        hi = float(prop.metadata.get("max", 1.0))
        w = pn.widgets.FloatSlider(
            name=prop.display_name, start=lo, end=hi,
            step=(hi - lo) / 100.0 if hi > lo else 0.01,
            value=float(prop.value),
        )

        def on_change(event, p=prop, r=ref):
            with session._lock:
                _apply_prop_value(renderer, r, p, float(event.new))

        w.param.watch(on_change, "value")
        return w

    if prop.type_name == "color3f" and prop.editable:
        c = prop.value
        r, g, b = float(c[0]), float(c[1]), float(c[2])
        hex_color = "#{:02x}{:02x}{:02x}".format(
            max(0, min(255, int(round(r * 255)))),
            max(0, min(255, int(round(g * 255)))),
            max(0, min(255, int(round(b * 255)))),
        )
        cw = pn.widgets.ColorPicker(name=prop.display_name, value=hex_color)

        def on_color(event, p=prop, r=ref):
            h = event.new.lstrip("#")
            rf = int(h[0:2], 16) / 255.0
            gf = int(h[2:4], 16) / 255.0
            bf = int(h[4:6], 16) / 255.0
            with session._lock:
                _apply_prop_value(renderer, r, p, (rf, gf, bf))

        cw.param.watch(on_color, "value")
        return cw

    if prop.type_name == "vec3f" and prop.editable:
        v = prop.value
        spins = [
            pn.widgets.FloatInput(
                name=f"{prop.display_name} {axis}",
                value=float(v[i]), step=0.05,
            )
            for i, axis in enumerate("XYZ")
        ]

        def on_vec3(_e, p=prop, r=ref, ws=spins):
            vals = tuple(float(w.value) for w in ws)
            with session._lock:
                _apply_vec3_value(renderer, r, node, p, vals)

        for w in spins:
            w.param.watch(on_vec3, "value")
        return pn.Row(*spins)

    if prop.type_name == "vec3f":
        v = prop.value
        return pn.pane.Markdown(
            f"**{prop.display_name}**: ({v[0]:.3f}, {v[1]:.3f}, {v[2]:.3f})"
        )

    if prop.type_name == "rel":
        return pn.pane.Markdown(f"**{prop.display_name}**: → `{prop.value}`")
    if prop.type_name == "asset":
        return pn.pane.Markdown(f"**{prop.display_name}**: `{prop.value}`")

    val_str = (
        f"{prop.value:.4f}" if isinstance(prop.value, float) else str(prop.value)
    )
    return pn.pane.Markdown(f"**{prop.display_name}**: {val_str}")


def _apply_prop_value(renderer, ref, prop, value) -> None:
    if ref is None:
        return
    if ref.kind == "material":
        renderer.apply_material_override(ref.index, prop.name, value)
    elif ref.kind in ("light_dir", "light_sphere"):
        light_type = "dir" if ref.kind == "light_dir" else "sphere"
        renderer.apply_light_override(light_type, ref.index, prop.name, value)
    elif ref.kind == "renderer_camera":
        renderer.apply_camera_param(prop.name, value)


def _apply_vec3_value(renderer, ref, node, prop, values) -> None:
    if ref is None:
        return
    if ref.kind == "renderer_camera":
        axis_kind = prop.metadata.get("camera_axis", "")
        if axis_kind == "target":
            keys = ("target_x", "target_y", "target_z")
        elif axis_kind == "position":
            keys = ("position_x", "position_y", "position_z")
        else:
            return
        for k, v in zip(keys, values):
            renderer.apply_camera_param(k, v)
        return
    if ref.kind != "instance":
        return
    translate = scale = (0.0, 0.0, 0.0)
    rotate = (0.0, 0.0, 0.0)
    for p in node.properties:
        if p.name == "translate":
            translate = values if p is prop else p.value
        elif p.name == "rotate":
            rotate = values if p is prop else p.value
        elif p.name == "scale":
            scale = values if p is prop else p.value
    renderer.apply_instance_transform(ref.index, translate, rotate, scale)


def _find_material_ancestor(renderer, node):
    from skinny.scene_graph import find_node_by_path
    graph = renderer.scene_graph
    if graph is None:
        return None
    parts = node.path.rstrip("/").split("/")
    for i in range(len(parts) - 1, 0, -1):
        parent_path = "/".join(parts[:i]) or "/"
        parent = find_node_by_path(graph, parent_path)
        if parent is not None and parent.renderer_ref is not None:
            if parent.renderer_ref.kind == "material":
                return parent.renderer_ref
    return None


# ── BXDF Visualizer ───────────────────────────────────────────────


def build_bxdf_pane(
    session, on_close: Callable[[], None],
) -> pn.Card:
    """CPU-side analytic BSDF lobe viewer. Material is picked via combo
    (no scene-pick in the browser); shading frame is fixed to the
    tangent-space +Z normal so the user sees the analytic Lambert + GGX
    response of the material's ``parameter_overrides``.
    """
    renderer = session.renderer

    material_combo = pn.widgets.Select(name="Material", options={})
    theta = pn.widgets.FloatSlider(name="theta", start=0.0, end=89.0, value=30.0)
    phi = pn.widgets.FloatSlider(name="phi", start=0.0, end=359.0, value=0.0)
    lock = pn.widgets.RadioButtonGroup(
        name="Lock", options=["wi", "wo"], value="wi",
    )
    image_pane = pn.pane.PNG(None, width=360)
    status = pn.pane.Markdown("Select a material.")

    state = {"yaw": math.radians(35.0), "pitch": math.radians(20.0)}

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
        cur = id(getattr(renderer, "_usd_scene", None))
        if cur != _last_id[0]:
            _last_id[0] = cur
            _repopulate()

    pn.state.add_periodic_callback(poll, period=1000)

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
        img = render_lobe_image(
            dirs, f, state["yaw"], state["pitch"], size=360, log_scale=True,
        )
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        image_pane.object = buf.getvalue()
        name = getattr(mats[mat_id], "name", "?")
        max_f = float(f.max())
        status.object = f"#{mat_id} ({name}) — max f·cosθ = {max_f:.3f} [CPU analytic]"

    for w in (material_combo, theta, phi, lock):
        w.param.watch(lambda _e: _eval_and_render(), "value")

    return _card(
        "BXDF Visualizer",
        pn.Column(
            material_combo,
            pn.Row(theta, phi),
            pn.Row(pn.pane.Markdown("**Lock:**"), lock),
            image_pane, status,
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
        cur = id(getattr(renderer, "_usd_scene", None))
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
    renderer = session.renderer
    label = pn.pane.Markdown(f"**{port.name}** ({port.type_name})")

    if port.connected_from:
        up, op = port.connected_from
        return pn.Row(
            label,
            pn.pane.Markdown(f"← `{up}.{op}`"),
        )

    t = port.type_name
    if t == "float":
        val = float(port.value) if port.value is not None else 0.0
        hi = max(val * 2.0, 1.0)
        w = pn.widgets.FloatSlider(
            name=port.name, start=0.0, end=hi, step=hi / 100.0, value=val,
        )

        def on_change(event):
            with session._lock:
                _apply_graph_edit(renderer, view, node, port, float(event.new))

        w.param.watch(on_change, "value")
        return pn.Row(label, w)

    if t in ("color3", "vector3"):
        v = port.value if isinstance(port.value, (list, tuple)) else (0.0, 0.0, 0.0)
        sliders = [
            pn.widgets.FloatSlider(
                name=ch, start=0.0, end=1.0, step=0.01,
                value=float(v[i]) if i < len(v) else 0.0,
            )
            for i, ch in enumerate("rgb" if t == "color3" else "xyz")
        ]

        def push(_e, ws=sliders):
            with session._lock:
                _apply_graph_edit(
                    renderer, view, node, port,
                    tuple(float(w.value) for w in ws),
                )

        for s in sliders:
            s.param.watch(push, "value")
        return pn.Row(label, *sliders)

    if t == "boolean":
        w = pn.widgets.Checkbox(name=port.name, value=bool(port.value))

        def on_change(event):
            with session._lock:
                _apply_graph_edit(renderer, view, node, port, bool(event.new))

        w.param.watch(on_change, "value")
        return pn.Row(label, w)

    if t == "integer":
        try:
            v = int(port.value or 0)
        except (TypeError, ValueError):
            v = 0
        w = pn.widgets.IntSlider(name=port.name, start=0, end=32, value=v)

        def on_change(event):
            with session._lock:
                _apply_graph_edit(renderer, view, node, port, int(event.new))

        w.param.watch(on_change, "value")
        return pn.Row(label, w)

    return pn.Row(label, pn.pane.Markdown(f"_(type {t} not editable)_"))


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

    renderer = session.renderer
    ctx = session.ctx
    shader_dir = _Path(__file__).resolve().parents[2] / "shaders"

    image_pane = pn.pane.PNG(None, width=640)
    status = pn.pane.Markdown("Initialising debug viewport…")
    dv_holder: dict = {"dv": None}

    def _ensure_dv() -> Optional[DebugViewport]:
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
            dv_holder["dv"] = dv
            return dv
        except Exception as exc:  # noqa: BLE001
            status.object = f"Debug viewport unavailable: {exc}"
            return None

    def _tick() -> None:
        dv = _ensure_dv()
        if dv is None or not dv.is_open:
            return
        with session._lock:
            try:
                pixels = dv.render_embedded(renderer)
            except Exception as exc:  # noqa: BLE001
                status.object = f"render failed: {exc}"
                return
        if pixels is None or Image is None:
            return
        img = Image.frombytes("RGBA", (dv._width, dv._height), bytes(pixels))
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="PNG")
        image_pane.object = buf.getvalue()
        status.object = f"Mode: {dv.camera_mode}   Size: {dv._width}×{dv._height}"

    pn.state.add_periodic_callback(_tick, period=200)

    def view(which: str):
        def _go(_e):
            dv = _ensure_dv()
            if dv is None:
                return
            with session._lock:
                if which == "top":
                    dv.view_top()
                elif which == "left":
                    dv.view_left()
                elif which == "back":
                    dv.view_back()
        return _go

    btn_top = pn.widgets.Button(name="Top", width=60)
    btn_left = pn.widgets.Button(name="Left", width=60)
    btn_back = pn.widgets.Button(name="Back", width=60)
    btn_top.on_click(view("top"))
    btn_left.on_click(view("left"))
    btn_back.on_click(view("back"))

    def reset(_e):
        dv = _ensure_dv()
        if dv is None:
            return
        with session._lock:
            dv._reset_debug_camera()

    btn_reset = pn.widgets.Button(name="Reset", width=60)
    btn_reset.on_click(reset)

    def _real_close() -> None:
        dv = dv_holder["dv"]
        if dv is not None:
            try:
                dv.destroy()
            except Exception:
                pass
            dv_holder["dv"] = None
            renderer.debug_viewport = None
        on_close()

    return _card(
        "Camera Debug View",
        pn.Column(
            pn.Row(btn_top, btn_left, btn_back, btn_reset),
            image_pane, status,
        ),
        _real_close,
    )
