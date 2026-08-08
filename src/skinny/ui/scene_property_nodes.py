"""Toolkit-free mapping from a scene property / material-graph input to a
``ui.spec`` node — the single owner of "what control does this value need".

The Qt and Panel scene docks each used to carry their own prop-type switch
(``scene_graph.py::_build_property_widget`` + eight helpers, and
``panel/windows.py::_build_scene_prop_widget``), and their own material-graph
input-row switch. Those two switches drifted (Panel silently dropped
``lens_file`` / ``texture_file`` / read-only colour / vec2, and the material
graph dropped ``vector2`` / ``filename``). This module is the one switch both
front-ends build their rows from, so a value renders the same in both or in
neither.

Each front-end injects only the parts that are genuinely per-toolkit:

* ``commit(prop_or_port, value)`` — how to *transport* the edit. The routing is
  already shared (``scene_edit_actions.apply_scene_property`` for scene props;
  the material-override path for graph inputs); Qt calls it directly, Panel
  wraps it in ``run_on_render_thread`` and surfaces the returned reason. This
  module never calls the renderer.
* ``get_live(prop)`` (scene props, optional) — how to read a *live* value for a
  control the user can drive without a graph rebuild (a camera scalar). ``None``
  → the snapshot ``prop.value``, which is the current Panel behaviour.

Design note (change ui-spec-scene-properties, decision 2.2): scene properties
and graph inputs are ONE node family — the existing ``ui.spec`` leaf types —
reached by two thin adapters, because they map onto the same widget vocabulary
and differ only in how a value is *sourced* (constraint metadata vs a bare
``PortView``), not in which control it needs.
"""

from __future__ import annotations

from typing import Any, Callable

from skinny.scene_graph import SceneGraphProperty
from skinny.ui import spec

# ── The renderable vocabulary these adapters emit ───────────────────
# Every node class either adapter can return. Both backends MUST dispatch each
# (asserted by tests/test_scene_property_nodes.py), which is what makes "a type
# handled in one front-end but not the other fails the build" structural.
RENDERABLE_NODE_TYPES: frozenset[type[spec.Node]] = frozenset({
    spec.Checkbox, spec.Slider, spec.Color, spec.Vector, spec.IntSpin,
    spec.FilePicker, spec.Label,
})

# Scene-property ``type_name``s the switch turns into an *editable* control;
# any other (or a non-editable prop) becomes a read-only ``Label``.
EDITABLE_SCENE_PROP_TYPES: frozenset[str] = frozenset({
    "bool", "float", "color3f", "vec3f", "vec2f", "int",
    "lens_file", "texture_file",
})
# Material-graph ``PortView.type_name``s the switch turns into an editable
# control (a bare port carries no constraint metadata, so ranges are per-type).
EDITABLE_GRAPH_INPUT_TYPES: frozenset[str] = frozenset({
    "float", "color3", "vector3", "vector2", "integer", "boolean", "filename",
})

# Shared constants (previously restated in each dock).
_GROWABLE_MAX = 1.0e9   # a "growable" float's upper bound (Panel parity; Qt's
#                          live rescale nicety is dropped for cross-front-end
#                          parity — recorded, not a silent loss).
_VEC_RANGE = 1.0e6      # transform-vector spinbox span (±)
_VEC_STEP = 0.05        # transform-vector single-step
_INT_RANGE = 1_000_000  # default int span when no min/max metadata
_LENS_FILTERS = [("USDA lens", "*.usda"), ("All files", "*.*")]
_TEXTURE_FILTERS = [("HDR images", "*.hdr *.exr *.pfm"), ("All files", "*.*")]

CommitFn = Callable[[Any, Any], None]


# ── Scene properties ────────────────────────────────────────────────


def scene_property_to_node(
    prop: SceneGraphProperty,
    *,
    commit: CommitFn,
    get_live: Callable[[SceneGraphProperty], Any] | None = None,
) -> spec.Node:
    """Map one ``SceneGraphProperty`` to a spec node.

    Never returns ``None``: a value the switch cannot make editable becomes a
    read-only ``Label``, so both front-ends show it identically instead of one
    silently dropping the row.
    """
    read = (lambda: get_live(prop)) if get_live is not None else (lambda: prop.value)
    t = prop.type_name
    name = prop.display_name

    if prop.editable and t == "bool":
        return spec.Checkbox(
            name, getter=lambda: bool(read()),
            setter=lambda v: commit(prop, bool(v)),
        )
    if prop.editable and t == "float":
        lo = float(prop.metadata.get("min", 0.0))
        hi = (_GROWABLE_MAX if prop.metadata.get("growable")
              else float(prop.metadata.get("max", 1.0)))
        return spec.Slider(
            name, getter=lambda: float(read()),
            setter=lambda v: commit(prop, float(v)), lo=lo, hi=hi,
        )
    if prop.editable and t == "color3f":
        return spec.Color(
            name, getter=lambda: _as_rgb(read()),
            setter=lambda v: commit(prop, tuple(float(c) for c in v)),
        )
    if prop.editable and t in ("vec3f", "vec2f"):
        n = 3 if t == "vec3f" else 2
        return spec.Vector(
            name, n, getter=lambda: tuple(float(c) for c in read()),
            setter=lambda v: commit(prop, tuple(float(c) for c in v)),
            lo=-_VEC_RANGE, hi=_VEC_RANGE, step=_VEC_STEP,
        )
    if prop.editable and t == "int":
        lo = int(prop.metadata.get("min", -_INT_RANGE))
        hi = int(prop.metadata.get("max", _INT_RANGE))
        return spec.IntSpin(
            name, getter=lambda: int(read()),
            setter=lambda v: commit(prop, int(v)), lo=lo, hi=hi,
        )
    if prop.editable and t == "lens_file":
        return spec.FilePicker(
            name, filters=_LENS_FILTERS,
            on_pick=lambda p: commit(prop, str(p)), category="lens",
        )
    if prop.editable and t == "texture_file":
        return spec.FilePicker(
            name, filters=_TEXTURE_FILTERS,
            # "ibl" keeps the last-used-directory key the Qt dome-texture picker
            # used before this seam existed (no silent settings-key change).
            on_pick=lambda p: commit(prop, str(p)), category="ibl",
        )
    return spec.Label(name, text=lambda: _format_readonly(prop, read()))


# ── Material-graph inputs ───────────────────────────────────────────


def graph_input_to_node(port, *, commit: CommitFn) -> spec.Node:
    """Map one material-graph ``PortView`` to a spec node.

    A ``PortView`` carries no constraint metadata, so ranges are per-type (the
    values Qt used, taken as canonical). A connected or unsupported port becomes
    a read-only ``Label``.
    """
    t = port.type_name
    name = port.name

    connected = getattr(port, "connected_from", None)
    if connected:
        up, op = connected
        return spec.Label(name, text=lambda: f"← {up}.{op}")

    if t == "float":
        return spec.Slider(
            name, getter=lambda: float(port.value or 0.0),
            setter=lambda v: commit(port, float(v)), lo=0.0, hi=1.0,
        )
    if t == "color3":
        return spec.Color(
            name, getter=lambda: _as_rgb(port.value),
            setter=lambda v: commit(port, tuple(float(c) for c in v)),
        )
    if t == "vector3":
        return spec.Vector(
            name, 3, getter=lambda: _as_vec(port.value, 3),
            setter=lambda v: commit(port, tuple(float(c) for c in v)),
            lo=0.0, hi=1.0, step=0.01,
        )
    if t == "vector2":
        return spec.Vector(
            name, 2, getter=lambda: _as_vec(port.value, 2),
            setter=lambda v: commit(port, tuple(float(c) for c in v)),
            lo=-_VEC_RANGE, hi=_VEC_RANGE, step=_VEC_STEP,
        )
    if t == "integer":
        return spec.IntSpin(
            name, getter=lambda: int(port.value or 0),
            setter=lambda v: commit(port, int(v)), lo=-1024, hi=1024,
        )
    if t == "boolean":
        return spec.Checkbox(
            name, getter=lambda: bool(port.value),
            setter=lambda v: commit(port, bool(v)),
        )
    if t == "filename":
        return spec.FilePicker(
            name, filters=[("All files", "*.*")],
            on_pick=lambda p: commit(port, str(p)), category="graph_input",
        )
    return spec.Label(name, text=lambda: f"(type {t} not editable)")


# ── Formatting helpers ──────────────────────────────────────────────


def _as_rgb(v) -> tuple[float, float, float]:
    if v is None:
        return (0.0, 0.0, 0.0)
    return (float(v[0]), float(v[1]), float(v[2]))


def _as_vec(v, n: int) -> tuple[float, ...]:
    if v is None:
        return tuple(0.0 for _ in range(n))
    return tuple(float(v[i]) for i in range(n))


def _format_readonly(prop: SceneGraphProperty, v) -> str:
    """Faithful to the Qt read-only formatting (change ui-spec-scene-properties
    folds Qt's per-type read-only labels here so Panel matches)."""
    t = prop.type_name
    if t == "vec3f":
        return f"({v[0]:.3f}, {v[1]:.3f}, {v[2]:.3f})"
    if t == "vec2f":
        return f"({v[0]:.3f}, {v[1]:.3f})"
    if t == "color3f":
        return f"rgb({v[0]:.2f}, {v[1]:.2f}, {v[2]:.2f})"
    if t == "rel":
        return f"→ {v}"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)
