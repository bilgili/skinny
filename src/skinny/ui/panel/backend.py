"""Walks a ``Section`` tree and produces a Panel sidebar.

Phase 5 scope: basic widget set + DynamicSection rebuild on token change.
DirectionPicker remains a placeholder until Phase 6.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import panel as pn

from skinny.ui import spec


# ── Dynamic-section bookkeeping ────────────────────────────────────


@dataclass
class _DynSection:
    """One DynamicSection's runtime state. Owns its own pulls list so
    old widgets die with the section on rebuild.
    """
    node: spec.DynamicSection
    body: pn.Column
    last_token: Any
    pulls: list[Callable[[], None]] = field(default_factory=list)


# ── Builder ────────────────────────────────────────────────────────


class PanelTreeBuilder:
    """Walks the tree once and returns the assembled ``pn.Accordion``
    layout. Periodic ``_tick`` rebuilds dynamic sections and refreshes
    pulled state.
    """

    PULL_INTERVAL_MS = 200

    def __init__(self, root: spec.Section) -> None:
        self.root = root
        self._pulls: list[Callable[[], None]] = []
        self._dyn: list[_DynSection] = []

        sections: list[tuple[str, pn.viewable.Viewable]] = []
        active: list[int] = []
        for child in root.children:
            entry = self._build_top_level(child)
            if entry is None:
                continue
            sections.append(entry[0])
            if entry[1]:
                active.append(len(sections) - 1)

        self.layout = pn.Accordion(*sections, active=active)
        # Drive the first dynamic rebuild now so the initial render
        # already shows materials / scene-graph contents.
        for dyn in self._dyn:
            self._rebuild_dynamic(dyn)

    # ── Pull loop ─────────────────────────────────────────────────

    def register_periodic(self) -> None:
        """Call after the layout has been served (inside
        ``create_panel_app``) so periodic refresh ticks fire on the
        Bokeh document.
        """
        pn.state.add_periodic_callback(self._tick, period=self.PULL_INTERVAL_MS)

    def _tick(self) -> None:
        for pull in self._pulls:
            _safe_call(pull)
        for dyn in self._dyn:
            self._maybe_rebuild_dynamic(dyn)
            for pull in dyn.pulls:
                _safe_call(pull)

    # ── Top-level walker ──────────────────────────────────────────

    def _build_top_level(
        self, node: spec.Node,
    ) -> tuple[tuple[str, pn.viewable.Viewable], bool] | None:
        if isinstance(node, spec.Section):
            body = pn.Column()
            for child in node.children:
                w = self._build(child, self._pulls)
                if w is not None:
                    body.append(w)
            return (node.title, body), node.expanded
        if isinstance(node, spec.DynamicSection):
            body = pn.Column()
            dyn = _DynSection(node=node, body=body, last_token=object())
            self._dyn.append(dyn)
            return (node.title, body), node.expanded
        # Bare leaf at top level — wrap in a single-row Accordion entry.
        w = self._build(node, self._pulls)
        if w is None:
            return None
        return ("(item)", pn.Column(w)), True

    # ── Generic walker ────────────────────────────────────────────

    def _build(
        self, node: spec.Node, pulls: list[Callable[[], None]],
    ) -> pn.viewable.Viewable | None:
        if isinstance(node, spec.Section):
            return self._build_section(node, pulls)
        if isinstance(node, spec.DynamicSection):
            # Nested dynamic section — uncommon but legal. Embed a pn.Card
            # whose body is rebuilt on token change.
            body = pn.Column()
            dyn = _DynSection(node=node, body=body, last_token=object())
            self._dyn.append(dyn)
            return pn.Card(body, title=node.title, collapsed=not node.expanded)
        if isinstance(node, spec.Slider):
            return self._build_slider(node, pulls)
        if isinstance(node, spec.Combo):
            return self._build_combo(node, pulls)
        if isinstance(node, spec.Color):
            return self._build_color(node, pulls)
        if isinstance(node, spec.Checkbox):
            return self._build_checkbox(node, pulls)
        if isinstance(node, spec.Vector):
            return self._build_vector(node, pulls)
        if isinstance(node, spec.IntSpin):
            return self._build_int_spin(node, pulls)
        if isinstance(node, spec.Button):
            return self._build_button(node)
        if isinstance(node, spec.FilePicker):
            return self._build_file_picker(node)
        if isinstance(node, spec.ResolutionPicker):
            return self._build_resolution_picker(node)
        if isinstance(node, spec.ScreenshotPicker):
            return self._build_screenshot_picker(node)
        if isinstance(node, spec.DirectionPicker):
            return self._build_direction_picker(node, pulls)
        return pn.pane.Markdown(f"*[unknown: {type(node).__name__}]*")

    # ── Layout primitives ─────────────────────────────────────────

    def _build_section(
        self, node: spec.Section, pulls: list[Callable[[], None]],
    ) -> pn.viewable.Viewable:
        items = []
        for child in node.children:
            w = self._build(child, pulls)
            if w is not None:
                items.append(w)
        return pn.Card(*items, title=node.title, collapsed=not node.expanded)

    # ── Dynamic rebuild ───────────────────────────────────────────

    def _maybe_rebuild_dynamic(self, dyn: _DynSection) -> None:
        try:
            token = dyn.node.rebuild_token()
        except Exception:  # noqa: BLE001
            return
        if token == dyn.last_token:
            return
        self._rebuild_dynamic(dyn)

    def _rebuild_dynamic(self, dyn: _DynSection) -> None:
        try:
            dyn.last_token = dyn.node.rebuild_token()
        except Exception:  # noqa: BLE001
            return
        dyn.pulls.clear()
        dyn.body.clear()
        sub_ub = spec.UIBuilder()
        try:
            dyn.node.build(sub_ub)
        except Exception as exc:  # noqa: BLE001
            dyn.body.append(pn.pane.Alert(f"Build failed: {exc}", alert_type="danger"))
            return
        for child in sub_ub.tree.children:
            w = self._build(child, dyn.pulls)
            if w is not None:
                dyn.body.append(w)

    # ── Leaf widgets ──────────────────────────────────────────────

    def _build_slider(
        self, node: spec.Slider, pulls: list[Callable[[], None]],
    ) -> pn.viewable.Viewable:
        cur = float(node.getter())
        step = node.step if node.step > 0 else (node.hi - node.lo) / 100.0
        w = pn.widgets.FloatSlider(
            name=node.name, start=node.lo, end=node.hi,
            step=step, value=cur,
        )

        def on_change(event) -> None:
            node.setter(float(event.new))

        w.param.watch(on_change, "value")

        def pull() -> None:
            v = float(node.getter())
            if abs(w.value - v) > 1e-5:
                with _Suppress(w, on_change):
                    w.value = v
        pulls.append(pull)
        return w

    def _build_combo(
        self, node: spec.Combo, pulls: list[Callable[[], None]],
    ) -> pn.viewable.Viewable:
        labels = list(node.choices())
        opts = {label: i for i, label in enumerate(labels)} if labels else {"(none)": -1}
        cur = int(node.getter())
        value = cur if 0 <= cur < len(labels) else (0 if labels else -1)
        w = pn.widgets.Select(name=node.name, options=opts, value=value)

        def on_change(event) -> None:
            if event.new is None or event.new == -1:
                return
            node.setter(int(event.new))

        w.param.watch(on_change, "value")

        def pull() -> None:
            new_labels = list(node.choices())
            new_opts = {l: i for i, l in enumerate(new_labels)} if new_labels else {"(none)": -1}
            cur_idx = int(node.getter())
            with _Suppress(w, on_change):
                if w.options != new_opts:
                    w.options = new_opts
                if 0 <= cur_idx < len(new_labels) and w.value != cur_idx:
                    w.value = cur_idx
        pulls.append(pull)
        return w

    def _build_color(
        self, node: spec.Color, pulls: list[Callable[[], None]],
    ) -> pn.viewable.Viewable:
        r, g, b = node.getter()
        w = pn.widgets.ColorPicker(name=node.name, value=_rgb_to_hex(r, g, b))

        def on_change(event) -> None:
            node.setter(_hex_to_rgb(event.new))

        w.param.watch(on_change, "value")

        def pull() -> None:
            cur = _rgb_to_hex(*node.getter())
            if w.value != cur:
                with _Suppress(w, on_change):
                    w.value = cur
        pulls.append(pull)
        return w

    def _build_checkbox(
        self, node: spec.Checkbox, pulls: list[Callable[[], None]],
    ) -> pn.viewable.Viewable:
        w = pn.widgets.Checkbox(name=node.name, value=bool(node.getter()))

        def on_change(event) -> None:
            node.setter(bool(event.new))

        w.param.watch(on_change, "value")

        def pull() -> None:
            v = bool(node.getter())
            if w.value != v:
                with _Suppress(w, on_change):
                    w.value = v
        pulls.append(pull)
        return w

    def _build_vector(
        self, node: spec.Vector, pulls: list[Callable[[], None]],
    ) -> pn.viewable.Viewable:
        cur = node.getter()
        sliders: list[pn.widgets.FloatSlider] = []
        labels = "xyzw"

        def push() -> None:
            node.setter(tuple(s.value for s in sliders))

        widgets: list[pn.viewable.Viewable] = [pn.pane.Markdown(f"**{node.name}**")]
        for i in range(node.components):
            v = float(cur[i]) if i < len(cur) else 0.0
            sw = pn.widgets.FloatSlider(
                name=f"{node.name}.{labels[i]}", start=node.lo, end=node.hi,
                step=(node.hi - node.lo) / 100.0, value=v,
            )
            sw.param.watch(lambda _e, _push=push: _push(), "value")
            sliders.append(sw)
            widgets.append(sw)

        def pull() -> None:
            cur_now = node.getter()
            for i, s in enumerate(sliders):
                if i >= len(cur_now):
                    continue
                v = float(cur_now[i])
                if abs(s.value - v) > 1e-5:
                    s.value = v
        pulls.append(pull)
        return pn.Column(*widgets)

    def _build_int_spin(
        self, node: spec.IntSpin, pulls: list[Callable[[], None]],
    ) -> pn.viewable.Viewable:
        w = pn.widgets.IntSlider(
            name=node.name, start=node.lo, end=node.hi, value=int(node.getter()),
        )

        def on_change(event) -> None:
            node.setter(int(event.new))

        w.param.watch(on_change, "value")

        def pull() -> None:
            v = int(node.getter())
            if w.value != v:
                with _Suppress(w, on_change):
                    w.value = v
        pulls.append(pull)
        return w

    def _build_button(self, node: spec.Button) -> pn.viewable.Viewable:
        w = pn.widgets.Button(name=node.label)
        w.on_click(lambda _e: node.on_click())
        return w

    def _build_direction_picker(
        self, node: spec.DirectionPicker, pulls: list[Callable[[], None]],
    ) -> pn.viewable.Viewable:
        # Web has no native arcball; expose elev + az as numeric sliders.
        # Same renderer state the Qt arcball ultimately writes — picking
        # via numbers is less ergonomic but functionally identical.
        elev = pn.widgets.FloatSlider(
            name=f"{node.name} elev", start=-90.0, end=90.0,
            step=1.0, value=float(node.elev_getter()),
        )
        az = pn.widgets.FloatSlider(
            name=f"{node.name} az", start=-180.0, end=180.0,
            step=1.0, value=float(node.az_getter()),
        )

        def on_elev(event) -> None:
            node.elev_setter(float(event.new))

        def on_az(event) -> None:
            node.az_setter(float(event.new))

        elev.param.watch(on_elev, "value")
        az.param.watch(on_az, "value")

        def pull() -> None:
            ev = float(node.elev_getter())
            av = float(node.az_getter())
            if abs(elev.value - ev) > 1e-3:
                with _Suppress(elev, on_elev):
                    elev.value = ev
            if abs(az.value - av) > 1e-3:
                with _Suppress(az, on_az):
                    az.value = av

        pulls.append(pull)
        return pn.Column(elev, az)

    def _build_file_picker(self, node: spec.FilePicker) -> pn.viewable.Viewable:
        glob = node.filters[0][1] if node.filters else "*"
        start = str(node.start_dir) if node.start_dir else None
        sel = pn.widgets.FileSelector(start, file_pattern=glob, only_files=True)
        btn = pn.widgets.Button(name=node.label, button_type="primary")
        status = pn.pane.Alert("", alert_type="info", visible=False)

        def on_click(_event) -> None:
            picked = sel.value
            if not picked:
                status.object = "No file selected"
                status.alert_type = "warning"
                status.visible = True
                return
            try:
                node.on_pick(Path(picked[0]))
                status.object = f"Loaded {Path(picked[0]).name}"
                status.alert_type = "success"
            except Exception as exc:  # noqa: BLE001
                status.object = f"Error: {exc}"
                status.alert_type = "danger"
            status.visible = True

        btn.on_click(on_click)
        return pn.Column(sel, btn, status)

    def _build_resolution_picker(
        self, node: spec.ResolutionPicker,
    ) -> pn.viewable.Viewable:
        names = [name for name, _w, _h in node.presets]

        def find_idx(w: int, h: int) -> int:
            for i, (_n, pw, ph) in enumerate(node.presets):
                if pw == w and ph == h:
                    return i
            return 0

        cur_w = int(node.width_getter())
        cur_h = int(node.height_getter())
        preset = pn.widgets.Select(
            name="Preset", options=names, value=names[find_idx(cur_w, cur_h)],
        )
        w_in = pn.widgets.IntInput(
            name="Width", value=cur_w, start=64, end=8192, step=8,
        )
        h_in = pn.widgets.IntInput(
            name="Height", value=cur_h, start=64, end=8192, step=8,
        )
        apply_btn = pn.widgets.Button(name="Apply Resolution", button_type="primary")
        guard = {"suppress": False}

        def do_apply(w: int, h: int) -> None:
            actual_w, actual_h = node.on_apply(int(w), int(h))
            guard["suppress"] = True
            try:
                w_in.value = actual_w
                h_in.value = actual_h
                preset.value = names[find_idx(actual_w, actual_h)]
            finally:
                guard["suppress"] = False

        def on_preset(event) -> None:
            if guard["suppress"]:
                return
            for n, pw, ph in node.presets:
                if n == event.new:
                    if pw == 0 or ph == 0:
                        return
                    do_apply(pw, ph)
                    return

        def on_apply(_event) -> None:
            do_apply(int(w_in.value), int(h_in.value))

        preset.param.watch(on_preset, "value")
        apply_btn.on_click(on_apply)
        return pn.Column(preset, w_in, h_in, apply_btn)

    def _build_screenshot_picker(
        self, node: spec.ScreenshotPicker,
    ) -> pn.viewable.Viewable:
        labels = [label for label, _f, _e in node.formats]
        fmt_select = pn.widgets.Select(name="Format", options=labels, value=labels[0])

        def callback() -> io.BytesIO:
            label = fmt_select.value
            for lab, fmt_str, _ext in node.formats:
                if lab == label:
                    return io.BytesIO(node.capture(fmt_str))
            return io.BytesIO(b"")

        download = pn.widgets.FileDownload(
            callback=callback, filename="skinny.png",
            button_type="primary", label="Screenshot", embed=False,
        )

        def update_filename(*_args) -> None:
            for lab, _f, ext in node.formats:
                if lab == fmt_select.value:
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    download.filename = f"skinny_{ts}.{ext}"
                    return

        fmt_select.param.watch(update_filename, "value")
        update_filename()
        return pn.Column(fmt_select, download)


# ── Helpers ────────────────────────────────────────────────────────


def _safe_call(pull: Callable[[], None]) -> None:
    try:
        pull()
    except Exception:
        # Widget destroyed during a dynamic-section rebuild. Caller's
        # pulls list will be replaced on next tick.
        pass


class _Suppress:
    """Detach a watcher for the duration of an assignment. Panel re-emits
    ``param.watch`` on every value write, so the pull callbacks would
    otherwise re-fire user setters and loop.
    """

    def __init__(self, widget, callback) -> None:
        self.widget = widget
        self.callback = callback
        self._watcher = None

    def __enter__(self):
        watchers = list(self.widget.param.watchers.get("value", []))
        for w in watchers:
            if w.fn is self.callback:
                self.widget.param.unwatch(w)
                self._watcher = w
                break
        return self

    def __exit__(self, *_exc):
        if self._watcher is not None:
            self.widget.param.watch(self.callback, "value")


def _rgb_to_hex(r: float, g: float, b: float) -> str:
    return "#{:02x}{:02x}{:02x}".format(_b(r), _b(g), _b(b))


def _hex_to_rgb(hex_str: str) -> tuple[float, float, float]:
    h = hex_str.lstrip("#")
    return (
        int(h[0:2], 16) / 255.0,
        int(h[2:4], 16) / 255.0,
        int(h[4:6], 16) / 255.0,
    )


def _b(c: float) -> int:
    return max(0, min(255, int(round(c * 255.0))))
