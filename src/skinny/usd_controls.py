"""Apply a resolved USD control binding to a renderer.

The other half of the inversion in `scene_intake.resolve_control_binding`:
intake looks a target up and returns a `ControlBinding` description, and this
module turns that description into the `(getter, setter)` pair the UI and the
load-time defaults use. Nothing here reads a stage; nothing in intake writes to
a renderer.

`renderer` is duck-typed on purpose — the Qt and web front-ends pass a
marshalling proxy, not the live `Renderer`. That is why the writes go through
`params.set_param_value` and never `_set_nested`: `_set_nested` resolves
*through* a proxy to the live object behind it, so an `mtlx.*` write would
insert into the renderer's own mapping from the GUI thread (the web freeze) or
into a proxy mirror that posts nothing (the Qt drop).

This module holds no GPU dependency, so the UI and its tests import it without
pulling in `skinny.renderer`.
"""

from __future__ import annotations

from typing import Callable

from skinny.scene_intake import ControlBinding, resolve_control_binding


def accessors_for(
    renderer, binding: ControlBinding
) -> tuple[Callable[[], object], Callable[[object], None]]:
    """Turn a resolved binding into `(getter, setter)` against `renderer`.

    An inert binding yields no-op closures, so a malformed declaration leaves
    the widget present-but-dead rather than breaking the panel.
    """
    from skinny.params import _get_nested, set_param_value

    kind = binding.kind

    if kind in ("renderer", "mtlx"):
        path = binding.param_path
        return (lambda: _get_nested(renderer, path),
                lambda v: set_param_value(renderer, path, v))

    if kind == "material":
        mid, key = binding.material_id, binding.input_name

        def _get():
            return renderer._usd_scene.materials[mid].parameter_overrides.get(key)

        def _set(v):
            renderer.apply_material_override(mid, key, v)

        return (_get, _set)

    if kind == "usd":
        attr = binding.attribute

        def _get():
            return attr.Get()

        def _set(v):
            attr.Set(v)
            renderer._usd_live_dirty = True

        return (_get, _set)

    return (lambda: None, lambda _v: None)


def control_accessors(
    renderer, spec
) -> tuple[Callable[[], object], Callable[[object], None]]:
    """Resolve `spec` against the renderer's scene and stage, then bind it."""
    return accessors_for(renderer, resolve_control_binding(
        spec,
        scene=getattr(renderer, "_usd_scene", None),
        stage=getattr(renderer, "_usd_stage", None),
    ))
