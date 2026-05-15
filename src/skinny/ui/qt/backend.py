"""Walks a ``Section`` tree and produces a Qt sidebar.

Phase 5 scope: basic widget set + DynamicSection rebuild. DirectionPicker
remains a placeholder until Phase 6.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from PySide6.QtCore import Qt, QSignalBlocker, QTimer
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox, QColorDialog, QComboBox, QDoubleSpinBox, QFileDialog,
    QGroupBox, QHBoxLayout, QLabel, QLayout, QLayoutItem, QPushButton,
    QSlider, QSpinBox, QVBoxLayout, QWidget,
)

from skinny.ui import spec
from skinny.ui.qt.direction_picker import build_direction_widget


# ── Dynamic-section bookkeeping ────────────────────────────────────


@dataclass
class _DynSection:
    """One DynamicSection's runtime state. Owns its own pull list so old
    widgets die with the section on rebuild rather than leaking into the
    main pull loop.
    """
    node: spec.DynamicSection
    body_widget: QWidget
    body_layout: QVBoxLayout
    last_token: Any
    pulls: list[Callable[[], None]] = field(default_factory=list)


# ── Builder ────────────────────────────────────────────────────────


class QtTreeBuilder:
    """Walks a tree once, instantiates widgets, and registers per-frame
    "pull" callbacks so external state changes show up in the widgets.
    """

    PULL_INTERVAL_MS = 100

    def __init__(self, root: spec.Section, parent: QWidget) -> None:
        self.root = root
        self.parent = parent
        self._pulls: list[Callable[[], None]] = []
        self._dyn: list[_DynSection] = []

        self._layout = QVBoxLayout(parent)
        self._layout.setContentsMargins(4, 4, 4, 4)
        self._layout.setSpacing(4)
        for child in root.children:
            self._build_node(self._layout, child, self._pulls)
        self._layout.addStretch(1)

        self._timer = QTimer(parent)
        self._timer.setInterval(self.PULL_INTERVAL_MS)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

    # ── Walker ────────────────────────────────────────────────────

    def _build_node(
        self, layout: QLayout, node: spec.Node,
        pulls: list[Callable[[], None]],
    ) -> None:
        if isinstance(node, spec.Section):
            self._build_section(layout, node, pulls)
        elif isinstance(node, spec.DynamicSection):
            self._build_dynamic_section(layout, node)
        elif isinstance(node, spec.Slider):
            self._build_slider(layout, node, pulls)
        elif isinstance(node, spec.Combo):
            self._build_combo(layout, node, pulls)
        elif isinstance(node, spec.Color):
            self._build_color(layout, node, pulls)
        elif isinstance(node, spec.Checkbox):
            self._build_checkbox(layout, node, pulls)
        elif isinstance(node, spec.Vector):
            self._build_vector(layout, node, pulls)
        elif isinstance(node, spec.IntSpin):
            self._build_int_spin(layout, node, pulls)
        elif isinstance(node, spec.Button):
            self._build_button(layout, node)
        elif isinstance(node, spec.FilePicker):
            self._build_file_picker(layout, node)
        elif isinstance(node, spec.ResolutionPicker):
            self._build_resolution_picker(layout, node)
        elif isinstance(node, spec.ScreenshotPicker):
            self._build_screenshot_picker(layout, node)
        elif isinstance(node, spec.DirectionPicker):
            self._build_direction_picker(layout, node, pulls)
        else:
            self._build_placeholder(layout, f"[unknown: {type(node).__name__}]")

    def _tick(self) -> None:
        for pull in self._pulls:
            _safe_call(pull)
        for dyn in self._dyn:
            self._maybe_rebuild_dynamic(dyn)
            for pull in dyn.pulls:
                _safe_call(pull)

    # ── Layout primitives ─────────────────────────────────────────

    def _build_section(
        self, layout: QLayout, node: spec.Section,
        pulls: list[Callable[[], None]],
    ) -> None:
        box = QGroupBox(node.title)
        box.setCheckable(True)
        box.setChecked(node.expanded)
        outer = QVBoxLayout(box)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(2)
        body = QWidget(box)
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(2)
        outer.addWidget(body)
        for child in node.children:
            self._build_node(body_layout, child, pulls)
        body.setVisible(node.expanded)
        box.toggled.connect(body.setVisible)
        layout.addWidget(box)

    def _build_dynamic_section(
        self, layout: QLayout, node: spec.DynamicSection,
    ) -> None:
        box = QGroupBox(node.title)
        box.setCheckable(True)
        box.setChecked(node.expanded)
        outer = QVBoxLayout(box)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(2)
        body = QWidget(box)
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(2)
        outer.addWidget(body)
        body.setVisible(node.expanded)
        box.toggled.connect(body.setVisible)
        layout.addWidget(box)

        dyn = _DynSection(
            node=node, body_widget=body, body_layout=body_layout,
            last_token=object(),  # never matches first token() result
        )
        self._dyn.append(dyn)
        # Initial fill.
        self._maybe_rebuild_dynamic(dyn)

    def _maybe_rebuild_dynamic(self, dyn: _DynSection) -> None:
        try:
            token = dyn.node.rebuild_token()
        except Exception:  # noqa: BLE001
            return
        if token == dyn.last_token:
            return
        dyn.last_token = token

        # Tear down old body widgets + pulls.
        _clear_layout(dyn.body_layout)
        dyn.pulls.clear()

        # Run the build closure into a fresh sub-builder. UIBuilder's tree
        # root is itself a Section; walk its children into our body layout.
        sub_ub = spec.UIBuilder()
        try:
            dyn.node.build(sub_ub)
        except Exception as exc:  # noqa: BLE001
            self._build_placeholder(dyn.body_layout, f"[build failed: {exc}]")
            return
        for child in sub_ub.tree.children:
            self._build_node(dyn.body_layout, child, dyn.pulls)

    def _build_placeholder(self, layout: QLayout, text: str) -> None:
        lbl = QLabel(text)
        lbl.setStyleSheet("color: #888; font-style: italic;")
        layout.addWidget(lbl)

    # ── Leaf widgets ──────────────────────────────────────────────

    def _build_slider(
        self, layout: QLayout, node: spec.Slider,
        pulls: list[Callable[[], None]],
    ) -> None:
        row = _row()
        row.addWidget(_label(node.name))
        slider = QSlider(Qt.Horizontal)
        slider.setRange(0, 1000)
        spin = QDoubleSpinBox()
        spin.setRange(node.lo, node.hi)
        spin.setDecimals(3)
        step = node.step if node.step > 0 else (node.hi - node.lo) / 100.0
        spin.setSingleStep(step)

        span = max(node.hi - node.lo, 1e-9)
        def to_int(v: float) -> int:
            return int(round((v - node.lo) / span * 1000.0))
        def from_int(i: int) -> float:
            return node.lo + (i / 1000.0) * span

        cur = float(node.getter())
        with QSignalBlocker(slider), QSignalBlocker(spin):
            slider.setValue(to_int(cur))
            spin.setValue(cur)

        def on_slider(i: int) -> None:
            v = from_int(i)
            with QSignalBlocker(spin):
                spin.setValue(v)
            node.setter(v)

        def on_spin(v: float) -> None:
            with QSignalBlocker(slider):
                slider.setValue(to_int(v))
            node.setter(v)

        slider.valueChanged.connect(on_slider)
        spin.valueChanged.connect(on_spin)

        row.addWidget(slider, stretch=1)
        row.addWidget(spin)
        layout.addLayout(row)

        def pull() -> None:
            v = float(node.getter())
            if abs(spin.value() - v) > 1e-5:
                with QSignalBlocker(spin), QSignalBlocker(slider):
                    spin.setValue(v)
                    slider.setValue(to_int(v))
        pulls.append(pull)

    def _build_combo(
        self, layout: QLayout, node: spec.Combo,
        pulls: list[Callable[[], None]],
    ) -> None:
        row = _row()
        row.addWidget(_label(node.name))
        combo = QComboBox()
        options = list(node.choices())
        combo.addItems(options)
        cur = int(node.getter())
        if 0 <= cur < combo.count():
            with QSignalBlocker(combo):
                combo.setCurrentIndex(cur)

        def on_change(idx: int) -> None:
            if idx < 0:
                return
            node.setter(idx)

        combo.currentIndexChanged.connect(on_change)
        row.addWidget(combo, stretch=1)
        layout.addLayout(row)

        def pull() -> None:
            new_options = list(node.choices())
            cur_idx = int(node.getter())
            if [combo.itemText(i) for i in range(combo.count())] != new_options:
                with QSignalBlocker(combo):
                    combo.clear()
                    combo.addItems(new_options)
            if 0 <= cur_idx < combo.count() and combo.currentIndex() != cur_idx:
                with QSignalBlocker(combo):
                    combo.setCurrentIndex(cur_idx)
        pulls.append(pull)

    def _build_color(
        self, layout: QLayout, node: spec.Color,
        pulls: list[Callable[[], None]],
    ) -> None:
        row = _row()
        row.addWidget(_label(node.name))
        swatch = QPushButton()
        swatch.setFixedWidth(48)
        swatch.setFixedHeight(20)

        def apply_swatch(rgb: tuple[float, float, float]) -> None:
            r, g, b = rgb
            swatch.setStyleSheet(
                f"background-color: rgb({_b(r)}, {_b(g)}, {_b(b)}); "
                f"border: 1px solid #555;"
            )

        apply_swatch(node.getter())

        def on_click() -> None:
            r, g, b = node.getter()
            initial = QColor(_b(r), _b(g), _b(b))
            chosen = QColorDialog.getColor(initial, self.parent, node.name)
            if not chosen.isValid():
                return
            rgb = (chosen.redF(), chosen.greenF(), chosen.blueF())
            node.setter(rgb)
            apply_swatch(rgb)

        swatch.clicked.connect(on_click)
        row.addWidget(swatch)
        row.addStretch(1)
        layout.addLayout(row)

        def pull() -> None:
            apply_swatch(node.getter())
        pulls.append(pull)

    def _build_checkbox(
        self, layout: QLayout, node: spec.Checkbox,
        pulls: list[Callable[[], None]],
    ) -> None:
        cb = QCheckBox(node.name)
        cb.setChecked(bool(node.getter()))
        cb.toggled.connect(lambda v: node.setter(bool(v)))
        layout.addWidget(cb)

        def pull() -> None:
            v = bool(node.getter())
            if cb.isChecked() != v:
                with QSignalBlocker(cb):
                    cb.setChecked(v)
        pulls.append(pull)

    def _build_vector(
        self, layout: QLayout, node: spec.Vector,
        pulls: list[Callable[[], None]],
    ) -> None:
        outer = QVBoxLayout()
        outer.addWidget(_label(node.name))
        spins: list[QDoubleSpinBox] = []
        labels = "xyzw"
        cur = node.getter()

        def push() -> None:
            node.setter(tuple(s.value() for s in spins))

        for i in range(node.components):
            row = _row()
            row.addWidget(_label(labels[i], width=12))
            spin = QDoubleSpinBox()
            spin.setRange(node.lo, node.hi)
            spin.setDecimals(3)
            spin.setSingleStep((node.hi - node.lo) / 100.0)
            with QSignalBlocker(spin):
                spin.setValue(float(cur[i]) if i < len(cur) else 0.0)
            spin.valueChanged.connect(lambda _v: push())
            spins.append(spin)
            row.addWidget(spin, stretch=1)
            outer.addLayout(row)
        layout.addLayout(outer)

        def pull() -> None:
            cur_now = node.getter()
            for i, s in enumerate(spins):
                if i >= len(cur_now):
                    continue
                v = float(cur_now[i])
                if abs(s.value() - v) > 1e-5:
                    with QSignalBlocker(s):
                        s.setValue(v)
        pulls.append(pull)

    def _build_int_spin(
        self, layout: QLayout, node: spec.IntSpin,
        pulls: list[Callable[[], None]],
    ) -> None:
        row = _row()
        row.addWidget(_label(node.name))
        spin = QSpinBox()
        spin.setRange(node.lo, node.hi)
        spin.setValue(int(node.getter()))
        spin.valueChanged.connect(lambda v: node.setter(int(v)))
        row.addWidget(spin)
        row.addStretch(1)
        layout.addLayout(row)

        def pull() -> None:
            v = int(node.getter())
            if spin.value() != v:
                with QSignalBlocker(spin):
                    spin.setValue(v)
        pulls.append(pull)

    def _build_button(self, layout: QLayout, node: spec.Button) -> None:
        btn = QPushButton(node.label)
        btn.clicked.connect(node.on_click)
        layout.addWidget(btn)

    def _build_direction_picker(
        self, layout: QLayout, node: spec.DirectionPicker,
        pulls: list[Callable[[], None]],
    ) -> None:
        container = build_direction_widget(node)
        layout.addWidget(container)
        widget = container._direction_widget  # type: ignore[attr-defined]
        pulls.append(widget.refresh_from_state)

    def _build_file_picker(self, layout: QLayout, node: spec.FilePicker) -> None:
        btn = QPushButton(node.label)

        def on_click() -> None:
            filt = ";;".join(f"{label} ({glob})" for label, glob in node.filters)
            start = str(node.start_dir) if node.start_dir else ""
            path, _ = QFileDialog.getOpenFileName(self.parent, node.label, start, filt)
            if path:
                node.on_pick(Path(path))

        btn.clicked.connect(on_click)
        layout.addWidget(btn)

    def _build_resolution_picker(
        self, layout: QLayout, node: spec.ResolutionPicker,
    ) -> None:
        outer = QVBoxLayout()
        preset = QComboBox()
        preset.addItems([name for name, _w, _h in node.presets])
        outer.addWidget(_label("Preset"))
        outer.addWidget(preset)

        wh_row = _row()
        w_spin = QSpinBox(); w_spin.setRange(64, 8192); w_spin.setSingleStep(8)
        h_spin = QSpinBox(); h_spin.setRange(64, 8192); h_spin.setSingleStep(8)
        w_spin.setValue(node.width_getter())
        h_spin.setValue(node.height_getter())
        wh_row.addWidget(w_spin); wh_row.addWidget(QLabel("×")); wh_row.addWidget(h_spin)
        outer.addLayout(wh_row)

        apply_btn = QPushButton("Apply")
        outer.addWidget(apply_btn)

        def find_preset_index(w: int, h: int) -> int:
            for i, (_n, pw, ph) in enumerate(node.presets):
                if pw == w and ph == h:
                    return i
            return 0  # "Custom"

        def on_preset(idx: int) -> None:
            if idx < 0:
                return
            _name, pw, ph = node.presets[idx]
            if pw == 0 or ph == 0:
                return
            with QSignalBlocker(w_spin), QSignalBlocker(h_spin):
                w_spin.setValue(pw); h_spin.setValue(ph)
            do_apply()

        def do_apply() -> None:
            actual_w, actual_h = node.on_apply(int(w_spin.value()), int(h_spin.value()))
            with QSignalBlocker(w_spin), QSignalBlocker(h_spin), QSignalBlocker(preset):
                w_spin.setValue(actual_w); h_spin.setValue(actual_h)
                preset.setCurrentIndex(find_preset_index(actual_w, actual_h))

        preset.currentIndexChanged.connect(on_preset)
        apply_btn.clicked.connect(do_apply)
        layout.addLayout(outer)

    def _build_screenshot_picker(
        self, layout: QLayout, node: spec.ScreenshotPicker,
    ) -> None:
        outer = QVBoxLayout()
        fmt = QComboBox()
        fmt.addItems([label for label, _f, _e in node.formats])
        outer.addWidget(_label("Format"))
        outer.addWidget(fmt)
        btn = QPushButton("Screenshot")
        outer.addWidget(btn)

        def on_click() -> None:
            label = fmt.currentText()
            for lab, fmt_str, ext in node.formats:
                if lab == label:
                    path, _ = QFileDialog.getSaveFileName(
                        self.parent, "Save screenshot",
                        f"skinny.{ext}", f"{label} (*.{ext})",
                    )
                    if not path:
                        return
                    data = node.capture(fmt_str)
                    Path(path).write_bytes(data)
                    return

        btn.clicked.connect(on_click)
        layout.addLayout(outer)


# ── Helpers ────────────────────────────────────────────────────────


def _safe_call(pull: Callable[[], None]) -> None:
    try:
        pull()
    except RuntimeError:
        # Widget destroyed during a dynamic-section rebuild. Caller's
        # pulls list will be discarded on the next tick.
        pass


def _clear_layout(layout: QLayout) -> None:
    while layout.count():
        item: QLayoutItem = layout.takeAt(0)
        w = item.widget()
        if w is not None:
            w.setParent(None)
            w.deleteLater()
        else:
            sub = item.layout()
            if sub is not None:
                _clear_layout(sub)


def _row() -> QHBoxLayout:
    h = QHBoxLayout()
    h.setContentsMargins(0, 0, 0, 0)
    h.setSpacing(4)
    return h


def _label(text: str, *, width: int = 120) -> QLabel:
    lbl = QLabel(text)
    lbl.setMinimumWidth(width)
    return lbl


def _b(c: float) -> int:
    return max(0, min(255, int(round(c * 255.0))))
