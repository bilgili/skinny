"""Programmatic widget-tree builder + dataclass nodes.

The tree is a pure data description — no Qt, no Panel, no Tk imports here.
Each backend walks the same tree and instantiates its own widgets.

Bindings are expressed as plain ``getter() -> value`` and ``setter(value)``
callables so the spec doesn't need to know about ``params._get_nested``,
material overrides, or any other renderer internals — the call sites
(``build_app_ui``) inject the right closures.

Layout primitives (``section``, ``dynamic_section``) use a context manager
so the builder reads top-to-bottom like ``with`` blocks. Inside a section,
calls like ``ui.slider(...)`` append a child node to the current parent.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence


Number = float | int


# ── Node dataclasses ────────────────────────────────────────────────


@dataclass
class Node:
    """Common base — gives backends a single isinstance() target."""


@dataclass
class Section(Node):
    """Collapsible group with a title and ordered children.

    Qt → ``QGroupBox`` (or a collapsible body). Panel → ``pn.Card`` /
    ``pn.Accordion`` entry.
    """
    title: str
    expanded: bool = True
    children: list[Node] = field(default_factory=list)


@dataclass
class DynamicSection(Node):
    """A section whose contents are rebuilt when ``rebuild_token()`` changes.

    Use for material lists, scene-graph trees, etc., where the children
    depend on renderer state that can change at runtime (model load,
    material refresh).

    ``build`` receives a fresh ``UIBuilder`` rooted at this section's body
    and is expected to populate it. Backends keep the previous build's
    widget handles and dispose them before re-running ``build``.
    """
    title: str
    rebuild_token: Callable[[], Any]
    build: Callable[["UIBuilder"], None]
    expanded: bool = True


@dataclass
class Slider(Node):
    name: str
    getter: Callable[[], float]
    setter: Callable[[float], None]
    lo: float
    hi: float
    step: float = 0.0
    """0.0 means "backend chooses" (typically ``(hi - lo) / 100``)."""


@dataclass
class Combo(Node):
    """Discrete picker. ``choices`` is re-evaluated on rebuild so dynamic
    lists (presets, environments, models) refresh correctly.
    """
    name: str
    getter: Callable[[], int]
    setter: Callable[[int], None]
    choices: Callable[[], Sequence[str]]


@dataclass
class Color(Node):
    """RGB picker. Getter/setter exchange ``(r, g, b)`` floats in [0, 1]."""
    name: str
    getter: Callable[[], tuple[float, float, float]]
    setter: Callable[[tuple[float, float, float]], None]


@dataclass
class Checkbox(Node):
    name: str
    getter: Callable[[], bool]
    setter: Callable[[bool], None]


@dataclass
class Vector(Node):
    """N-component float vector (2/3/4). Getter/setter exchange a tuple."""
    name: str
    components: int
    getter: Callable[[], tuple[float, ...]]
    setter: Callable[[tuple[float, ...]], None]
    lo: float = 0.0
    hi: float = 1.0


@dataclass
class IntSpin(Node):
    name: str
    getter: Callable[[], int]
    setter: Callable[[int], None]
    lo: int = 0
    hi: int = 32


@dataclass
class Button(Node):
    label: str
    on_click: Callable[[], None]


@dataclass
class FilePicker(Node):
    """Opens an OS file dialog. ``filters`` is a list of ``(label, glob)``
    pairs (e.g. ``[("USD", "*.usda *.usdc *.usdz"), ("OBJ", "*.obj")]``).
    """
    label: str
    filters: list[tuple[str, str]]
    on_pick: Callable[[Path], None]
    start_dir: Path | None = None


@dataclass
class ResolutionPicker(Node):
    """Bundles preset combo + W/H entries + Apply button.

    ``presets`` is the same shape as ``params.RESOLUTION_PRESETS``:
    a list of ``(name, width, height)``. The ``(0, 0)`` "Custom" sentinel
    is honoured.
    """
    presets: list[tuple[str, int, int]]
    width_getter: Callable[[], int]
    height_getter: Callable[[], int]
    on_apply: Callable[[int, int], tuple[int, int]]
    """Returns the actual ``(W, H)`` the renderer settled on after clamp."""


@dataclass
class ScreenshotPicker(Node):
    """Format combo + capture button. ``formats`` is ``[(label, fmt, ext)]``.

    ``capture(fmt) -> bytes`` is host-agnostic. The Qt backend pops a Save
    dialog and writes the bytes to the chosen path; the Panel backend
    streams them to the browser via ``pn.widgets.FileDownload``.
    """
    formats: list[tuple[str, str, str]]
    capture: Callable[[str], bytes]


@dataclass
class DirectionPicker(Node):
    """Arcball-style 3D direction picker for light direction.

    Getters/setters work in degrees (matching renderer attributes
    ``light_elevation`` / ``light_azimuth`` and the existing presets).
    """
    name: str
    elev_getter: Callable[[], float]
    elev_setter: Callable[[float], None]
    az_getter: Callable[[], float]
    az_setter: Callable[[float], None]


# ── Builder ─────────────────────────────────────────────────────────


class UIBuilder:
    """Stack-based builder. ``with ui.section(...)`` pushes/pops the
    parent node; convenience methods append into the current parent.
    """

    def __init__(self, root: Section | None = None) -> None:
        self._root = root if root is not None else Section(title="", expanded=True)
        self._stack: list[Section | DynamicSection] = [self._root]

    # ── Tree access ────────────────────────────────────────────────

    @property
    def tree(self) -> Section:
        return self._root

    def _current(self) -> Section | DynamicSection:
        return self._stack[-1]

    def _append(self, node: Node) -> Node:
        parent = self._current()
        # DynamicSection has no .children — its body is built by the
        # rebuild closure on demand. Misuse here is a programming error.
        if isinstance(parent, DynamicSection):
            raise TypeError(
                "Cannot append directly to a DynamicSection; populate it "
                "from inside its build callable."
            )
        parent.children.append(node)
        return node

    # ── Layout primitives ──────────────────────────────────────────

    @contextmanager
    def section(self, title: str, *, expanded: bool = True) -> Iterator[Section]:
        s = Section(title=title, expanded=expanded)
        self._append(s)
        self._stack.append(s)
        try:
            yield s
        finally:
            self._stack.pop()

    def dynamic_section(
        self,
        title: str,
        *,
        rebuild_token: Callable[[], Any],
        build: Callable[["UIBuilder"], None],
        expanded: bool = True,
    ) -> DynamicSection:
        d = DynamicSection(
            title=title, rebuild_token=rebuild_token, build=build,
            expanded=expanded,
        )
        self._append(d)
        return d

    # ── Leaf widgets ───────────────────────────────────────────────

    def slider(
        self, name: str, getter: Callable[[], float],
        setter: Callable[[float], None], lo: float, hi: float,
        step: float = 0.0,
    ) -> Slider:
        return self._append(Slider(name, getter, setter, lo, hi, step))  # type: ignore[return-value]

    def combo(
        self, name: str, getter: Callable[[], int],
        setter: Callable[[int], None], choices: Callable[[], Sequence[str]],
    ) -> Combo:
        return self._append(Combo(name, getter, setter, choices))  # type: ignore[return-value]

    def color(
        self, name: str,
        getter: Callable[[], tuple[float, float, float]],
        setter: Callable[[tuple[float, float, float]], None],
    ) -> Color:
        return self._append(Color(name, getter, setter))  # type: ignore[return-value]

    def checkbox(
        self, name: str, getter: Callable[[], bool],
        setter: Callable[[bool], None],
    ) -> Checkbox:
        return self._append(Checkbox(name, getter, setter))  # type: ignore[return-value]

    def vector(
        self, name: str, components: int,
        getter: Callable[[], tuple[float, ...]],
        setter: Callable[[tuple[float, ...]], None],
        lo: float = 0.0, hi: float = 1.0,
    ) -> Vector:
        if components not in (2, 3, 4):
            raise ValueError("Vector.components must be 2, 3, or 4")
        return self._append(Vector(name, components, getter, setter, lo, hi))  # type: ignore[return-value]

    def int_spin(
        self, name: str, getter: Callable[[], int],
        setter: Callable[[int], None], lo: int = 0, hi: int = 32,
    ) -> IntSpin:
        return self._append(IntSpin(name, getter, setter, lo, hi))  # type: ignore[return-value]

    def button(self, label: str, on_click: Callable[[], None]) -> Button:
        return self._append(Button(label, on_click))  # type: ignore[return-value]

    def file_picker(
        self, label: str, filters: list[tuple[str, str]],
        on_pick: Callable[[Path], None], start_dir: Path | None = None,
    ) -> FilePicker:
        return self._append(FilePicker(label, filters, on_pick, start_dir))  # type: ignore[return-value]

    def resolution_picker(
        self, presets: list[tuple[str, int, int]],
        width_getter: Callable[[], int], height_getter: Callable[[], int],
        on_apply: Callable[[int, int], tuple[int, int]],
    ) -> ResolutionPicker:
        return self._append(  # type: ignore[return-value]
            ResolutionPicker(presets, width_getter, height_getter, on_apply)
        )

    def screenshot_picker(
        self, formats: list[tuple[str, str, str]],
        capture: Callable[[str], bytes],
    ) -> ScreenshotPicker:
        return self._append(ScreenshotPicker(formats, capture))  # type: ignore[return-value]

    def direction_picker(
        self, name: str,
        elev_getter: Callable[[], float], elev_setter: Callable[[float], None],
        az_getter: Callable[[], float], az_setter: Callable[[float], None],
    ) -> DirectionPicker:
        return self._append(DirectionPicker(  # type: ignore[return-value]
            name, elev_getter, elev_setter, az_getter, az_setter,
        ))


# ── Tree walking helpers (used by both backends + tests) ─────────────


def walk(node: Node) -> Iterator[Node]:
    """Yield ``node`` then every descendant (DFS). DynamicSection bodies
    are NOT walked — their contents are dynamic by definition.
    """
    yield node
    children: list[Node] = []
    if isinstance(node, Section):
        children = node.children
    for c in children:
        yield from walk(c)
