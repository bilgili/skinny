"""Python Material Editor — dockable code editor with hot-reload.

Tracks the Python-authored slangpile material currently bound by the
active scene (see ``Renderer.active_python_module``). The user edits the
source in a `QPlainTextEdit`, hits **Compile & Reload**, and the renderer
re-runs the slangpile codegen → slangc → pipeline-rebuild chain via
`MaterialReloader`. Errors land in an inline read-only output panel.

Layout (top→bottom):

    QDockWidget("Python Material Editor")
    └── QSplitter (vertical)
        ├── QWidget
        │   ├── header row    (module path label + dirty marker)
        │   ├── CodeEditor    (QPlainTextEdit + Python syntax highlight)
        │   └── button row    (Compile & Reload, Revert)
        └── output panel      (QPlainTextEdit read-only)
"""

from __future__ import annotations

import re
from pathlib import Path

from PySide6.QtCore import (
    QRect, QRegularExpression, QSignalBlocker, QSize, Qt, QTimer, Signal,
)
from PySide6.QtGui import (
    QColor, QFont, QKeySequence, QPainter, QShortcut, QSyntaxHighlighter,
    QTextCharFormat, QTextCursor, QTextFormat,
)
from PySide6.QtWidgets import (
    QComboBox, QDockWidget, QHBoxLayout, QLabel, QPlainTextEdit, QPushButton,
    QSplitter, QTextEdit, QVBoxLayout, QWidget,
)

from skinny.material_reloader import (
    MaterialReloader, ReloadResult, resolve_source_path,
)


# ─── Syntax highlighter ─────────────────────────────────────────────


_KEYWORDS = (
    "False", "None", "True", "and", "as", "assert", "async", "await",
    "break", "class", "continue", "def", "del", "elif", "else", "except",
    "finally", "for", "from", "global", "if", "import", "in", "is",
    "lambda", "nonlocal", "not", "or", "pass", "raise", "return", "try",
    "while", "with", "yield",
)

_BUILTINS = (
    "abs", "all", "any", "bool", "dict", "enumerate", "float", "int",
    "isinstance", "len", "list", "max", "min", "print", "range", "round",
    "set", "str", "sum", "tuple", "type", "zip", "max", "min",
)


class PythonSyntaxHighlighter(QSyntaxHighlighter):
    """Lightweight regex-based Python highlighter.

    Covers keywords, builtins, ``sp.*`` slangpile calls, decorators,
    numbers, strings (single/double + triple-quoted across lines), and
    line comments. No semantic analysis — `QSyntaxHighlighter` is plenty
    for the editor's scope.
    """

    def __init__(self, document) -> None:
        super().__init__(document)

        def fmt(color: str, *, italic: bool = False, bold: bool = False) -> QTextCharFormat:
            f = QTextCharFormat()
            f.setForeground(QColor(color))
            if italic:
                f.setFontItalic(True)
            if bold:
                f.setFontWeight(QFont.Bold)
            return f

        self._fmt_keyword = fmt("#c586c0", bold=True)
        self._fmt_builtin = fmt("#4ec9b0")
        self._fmt_sp = fmt("#dcdcaa")
        self._fmt_decorator = fmt("#dcdcaa", italic=True)
        self._fmt_number = fmt("#b5cea8")
        self._fmt_string = fmt("#ce9178")
        self._fmt_comment = fmt("#6a9955", italic=True)
        self._fmt_self = fmt("#569cd6", italic=True)

        kw_pattern = r"\b(?:" + "|".join(_KEYWORDS) + r")\b"
        bi_pattern = r"\b(?:" + "|".join(_BUILTINS) + r")\b"

        self._rules: list[tuple[QRegularExpression, QTextCharFormat]] = [
            (QRegularExpression(kw_pattern), self._fmt_keyword),
            (QRegularExpression(bi_pattern), self._fmt_builtin),
            (QRegularExpression(r"\bself\b"), self._fmt_self),
            (QRegularExpression(r"\bsp\.[A-Za-z_][A-Za-z0-9_]*"), self._fmt_sp),
            (QRegularExpression(r"@[A-Za-z_][A-Za-z0-9_.]*"), self._fmt_decorator),
            (QRegularExpression(r"\b\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\b"), self._fmt_number),
            (QRegularExpression(r"'[^'\\\n]*(?:\\.[^'\\\n]*)*'"), self._fmt_string),
            (QRegularExpression(r'"[^"\\\n]*(?:\\.[^"\\\n]*)*"'), self._fmt_string),
            (QRegularExpression(r"#[^\n]*"), self._fmt_comment),
        ]

        # Triple-quoted strings span lines, so they need block-state tracking.
        self._tri_double = QRegularExpression(r'"""')
        self._tri_single = QRegularExpression(r"'''")

    def highlightBlock(self, text: str) -> None:
        for rx, fmt_ in self._rules:
            it = rx.globalMatch(text)
            while it.hasNext():
                m = it.next()
                self.setFormat(m.capturedStart(), m.capturedLength(), fmt_)

        self._highlight_triple(text, self._tri_double, 1, self._fmt_string)
        self._highlight_triple(text, self._tri_single, 2, self._fmt_string)

    def _highlight_triple(
        self, text: str, delim: QRegularExpression, state_id: int,
        fmt_: QTextCharFormat,
    ) -> None:
        # 0 = normal, 1 = inside """, 2 = inside '''.
        prev_state = self.previousBlockState()
        start = 0
        if prev_state != state_id:
            m = delim.match(text)
            if not m.hasMatch():
                return
            start = m.capturedStart()
        # Walk delimiter pairs through the block.
        while True:
            m = delim.match(text, start + (3 if prev_state == state_id else 0))
            if not m.hasMatch():
                self.setFormat(start, len(text) - start, fmt_)
                self.setCurrentBlockState(state_id)
                return
            end = m.capturedStart() + 3
            self.setFormat(start, end - start, fmt_)
            start = end
            prev_state = -1
            # Look for another opener on the same line.
            m2 = delim.match(text, start)
            if not m2.hasMatch():
                return
            start = m2.capturedStart()
            prev_state = 0  # force inside-block branch on next iter


# ─── Editor widget with line numbers ────────────────────────────────


class _LineNumberArea(QWidget):
    def __init__(self, editor: "CodeEditor") -> None:
        super().__init__(editor)
        self._editor = editor

    def sizeHint(self) -> QSize:
        return QSize(self._editor.line_number_area_width(), 0)

    def paintEvent(self, event) -> None:
        self._editor.paint_line_numbers(event)


class CodeEditor(QPlainTextEdit):
    """`QPlainTextEdit` with line-number gutter, monospace font, and
    tab-as-4-spaces input.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        font = QFont("Menlo")
        font.setStyleHint(QFont.Monospace)
        font.setPointSize(11)
        self.setFont(font)
        self.setTabStopDistance(4 * self.fontMetrics().horizontalAdvance(" "))
        self.setLineWrapMode(QPlainTextEdit.NoWrap)

        self._line_area = _LineNumberArea(self)
        self.blockCountChanged.connect(self._update_line_area_width)
        self.updateRequest.connect(self._update_line_area)
        self.cursorPositionChanged.connect(self._highlight_current_line)
        self._update_line_area_width(0)
        self._highlight_current_line()

    def line_number_area_width(self) -> int:
        digits = max(2, len(str(max(1, self.blockCount()))))
        return 10 + self.fontMetrics().horizontalAdvance("9") * digits

    def _update_line_area_width(self, _new_block_count: int) -> None:
        self.setViewportMargins(self.line_number_area_width(), 0, 0, 0)

    def _update_line_area(self, rect: QRect, dy: int) -> None:
        if dy:
            self._line_area.scroll(0, dy)
        else:
            self._line_area.update(0, rect.y(), self._line_area.width(), rect.height())
        if rect.contains(self.viewport().rect()):
            self._update_line_area_width(0)

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        super().resizeEvent(event)
        cr = self.contentsRect()
        self._line_area.setGeometry(QRect(cr.left(), cr.top(),
                                          self.line_number_area_width(),
                                          cr.height()))

    def _highlight_current_line(self) -> None:
        extra: list[QTextEdit.ExtraSelection] = []
        if not self.isReadOnly():
            sel = QTextEdit.ExtraSelection()
            color = QColor("#2a2d2e")
            sel.format.setBackground(color)
            sel.format.setProperty(QTextFormat.FullWidthSelection, True)
            sel.cursor = self.textCursor()
            sel.cursor.clearSelection()
            extra.append(sel)
        self.setExtraSelections(extra)

    def paint_line_numbers(self, event) -> None:
        painter = QPainter(self._line_area)
        painter.fillRect(event.rect(), QColor("#1e1e1e"))
        block = self.firstVisibleBlock()
        block_number = block.blockNumber()
        top = int(self.blockBoundingGeometry(block).translated(self.contentOffset()).top())
        bottom = top + int(self.blockBoundingRect(block).height())
        pen = QColor("#858585")
        painter.setPen(pen)
        painter.setFont(self.font())
        right_pad = 4
        width = self._line_area.width() - right_pad
        while block.isValid() and top <= event.rect().bottom():
            if block.isVisible() and bottom >= event.rect().top():
                number = str(block_number + 1)
                painter.drawText(0, top, width,
                                 self.fontMetrics().height(),
                                 Qt.AlignRight, number)
            block = block.next()
            top = bottom
            bottom = top + int(self.blockBoundingRect(block).height())
            block_number += 1

    def keyPressEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        # Tab → 4 spaces; Shift+Tab → outdent up to 4 spaces.
        if event.key() == Qt.Key_Tab and not (event.modifiers() & Qt.ShiftModifier):
            self.insertPlainText("    ")
            return
        if event.key() == Qt.Key_Backtab:
            cursor = self.textCursor()
            cursor.beginEditBlock()
            cursor.movePosition(QTextCursor.StartOfLine)
            line = cursor.block().text()
            strip = 0
            for ch in line[:4]:
                if ch == " ":
                    strip += 1
                else:
                    break
            for _ in range(strip):
                cursor.deleteChar()
            cursor.endEditBlock()
            return
        super().keyPressEvent(event)


# ─── Dock ─────────────────────────────────────────────────────────────


class PythonMaterialEditorDock(QDockWidget):
    """Edits one Python-authored slangpile material at a time, identified
    by a module name like ``python_materials.preview_surface_material``.

    The dock owns no GPU state and never touches the live renderer on the GUI
    thread. Under render-thread ownership `renderer` is the `QtRendererProxy`:
    the module-list read and the `MaterialReloader` reload both run on the render
    worker via `renderer.request(...)`, and their results are marshalled back to
    the GUI thread through the `_modules_ready` / `_reload_ready` signals.
    """

    # Cross-thread result carriers: the render worker's future-callbacks emit
    # these; Qt delivers them queued onto the GUI thread.
    _modules_ready = Signal(object)
    _reload_ready = Signal(object)

    def __init__(
        self, renderer, parent: QWidget | None = None,
    ) -> None:
        super().__init__("Python Material Editor", parent)
        self.renderer = renderer
        self.setAllowedAreas(Qt.AllDockWidgetAreas)
        self._modules_inflight = False
        self._reload_inflight = False
        self._modules_ready.connect(self._apply_modules)
        self._reload_ready.connect(self._apply_reload_result)
        self._module_name: str | None = None
        self._source_path: Path | None = None
        self._dirty: bool = False
        self._loading: bool = False

        splitter = QSplitter(Qt.Vertical)
        self.setWidget(splitter)

        # ── Top: header + editor + buttons ──
        top = QWidget()
        top_layout = QVBoxLayout(top)
        top_layout.setContentsMargins(6, 6, 6, 6)
        top_layout.setSpacing(4)

        # Material selector — dropdown of every Python material declared
        # by the current scene. Empty when the scene has no Python materials.
        selector_row = QHBoxLayout()
        selector_row.addWidget(QLabel("Material:"))
        self._material_combo = QComboBox()
        self._material_combo.setMinimumWidth(280)
        self._material_combo.setSizeAdjustPolicy(
            QComboBox.AdjustToContentsOnFirstShow,
        )
        self._material_combo.currentIndexChanged.connect(
            self._on_combo_changed,
        )
        selector_row.addWidget(self._material_combo, stretch=1)
        top_layout.addLayout(selector_row)

        self._header = QLabel("No Python material in current scene.")
        hf = self._header.font()
        hf.setBold(True)
        self._header.setFont(hf)
        top_layout.addWidget(self._header)

        self.editor = CodeEditor()
        self._highlighter = PythonSyntaxHighlighter(self.editor.document())
        self.editor.textChanged.connect(self._on_text_changed)
        # Dark palette to match VSCode-ish defaults.
        self.editor.setStyleSheet(
            "QPlainTextEdit { background-color: #1e1e1e; color: #d4d4d4; "
            "border: 1px solid #3c3c3c; }"
        )
        top_layout.addWidget(self.editor, stretch=1)

        button_row = QHBoxLayout()
        self._undo_btn = QPushButton("Undo")
        self._undo_btn.setShortcut("Ctrl+Z")
        self._undo_btn.clicked.connect(self.editor.undo)
        button_row.addWidget(self._undo_btn)
        self._redo_btn = QPushButton("Redo")
        # Both Ctrl+Y and Ctrl+Shift+Z are common bindings; wire both.
        self._redo_btn.setShortcut("Ctrl+Y")
        self._redo_secondary = QShortcut(QKeySequence("Ctrl+Shift+Z"), self)
        # Window-scope so the shortcut works while the viewport has focus.
        self._redo_secondary.setContext(Qt.WindowShortcut)
        self._redo_secondary.activated.connect(self.editor.redo)
        self._redo_btn.clicked.connect(self.editor.redo)
        button_row.addWidget(self._redo_btn)
        # Mirror editor's undo-availability into the buttons so Ctrl+Z is
        # a no-op (instead of falling through to viewport-camera Z) when
        # there's nothing to undo.
        self.editor.undoAvailable.connect(self._undo_btn.setEnabled)
        self.editor.redoAvailable.connect(self._redo_btn.setEnabled)
        self._undo_btn.setEnabled(False)
        self._redo_btn.setEnabled(False)

        self._compile_btn = QPushButton("Compile && Reload")
        self._compile_btn.setShortcut("Ctrl+B")
        self._compile_btn.clicked.connect(self._on_compile_clicked)
        button_row.addWidget(self._compile_btn)
        self._revert_btn = QPushButton("Revert")
        self._revert_btn.clicked.connect(self._on_revert_clicked)
        button_row.addWidget(self._revert_btn)
        self._status = QLabel("")
        button_row.addWidget(self._status, stretch=1)
        top_layout.addLayout(button_row)

        splitter.addWidget(top)

        # ── Bottom: read-only output panel ──
        self.output = QPlainTextEdit()
        self.output.setReadOnly(True)
        self.output.setMaximumBlockCount(2000)
        self.output.setStyleSheet(
            "QPlainTextEdit { background-color: #1e1e1e; color: #d4d4d4; "
            "font-family: Menlo, monospace; border: 1px solid #3c3c3c; }"
        )
        self.output.setPlaceholderText("Compiler output appears here.")
        splitter.addWidget(self.output)

        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)

        self._set_buttons_enabled(False)

        # USD scene loads asynchronously on a background thread; the
        # dropdown may be empty when the dock first opens. Poll until the
        # renderer's USD scene reports at least one Python material, then
        # stop. Cheap (one list comprehension per tick).
        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(400)
        self._poll_timer.timeout.connect(self._poll_for_modules)
        self._poll_timer.start()

    # ── Public API ────────────────────────────────────────────────

    def refresh_from_renderer(self) -> None:
        """Refresh the dropdown + buffer from the renderer's current scene.

        Called after a scene load. The module list is fetched from the render
        worker and applied on the GUI thread (`_apply_modules`), keeping the
        current selection when it still lives in the scene; `set_active_module`
        preserves unsaved edits and warns.
        """
        self._request_modules()

    def _request_modules(self) -> None:
        """Ask the render worker for the scene's Python module list."""
        if self._modules_inflight:
            return
        self._modules_inflight = True
        fut = self.renderer.request(lambda r: r.scene_python_modules())
        fut.add_done_callback(self._on_modules_future)

    def _on_modules_future(self, fut) -> None:
        # Runs on the render worker thread; marshal to the GUI via a signal.
        try:
            modules = list(fut.result())
        except Exception:  # noqa: BLE001
            modules = []
        self._modules_ready.emit(modules)

    def _apply_modules(self, modules: "list[str]") -> None:
        self._modules_inflight = False
        if not modules:
            return
        self._poll_timer.stop()
        target = (
            self._module_name
            if self._module_name in modules
            else modules[0]
        )
        self._set_combo_items(modules, selected=target)
        self.set_active_module(target)

    def set_active_module(self, module_name: str | None) -> None:
        if module_name == self._module_name and not self._dirty:
            self._sync_combo_selection(module_name)
            return
        if self._dirty:
            self._append_output(
                f"Buffer has unsaved edits for {self._module_name!r}; "
                f"not switching to {module_name!r}. Revert or compile first.\n",
            )
            # Snap the combo back to whatever's currently loaded so the
            # dropdown doesn't lie about the editor state.
            self._sync_combo_selection(self._module_name)
            return

        if module_name is None:
            self._module_name = None
            self._source_path = None
            self._loading = True
            self.editor.setPlainText("")
            self.editor.setReadOnly(True)
            self._loading = False
            self._header.setText("No Python material in current scene.")
            self._set_buttons_enabled(False)
            self._dirty = False
            return

        path = resolve_source_path(module_name)
        if path is None:
            self._module_name = None
            self._source_path = None
            self._header.setText(
                f"Material {module_name!r} not found on disk."
            )
            self._loading = True
            self.editor.setPlainText("")
            self.editor.setReadOnly(True)
            self._loading = False
            self._set_buttons_enabled(False)
            return

        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            self._append_output(f"Failed to read {path}: {exc}\n")
            return

        self._module_name = module_name
        self._source_path = path
        self._loading = True
        self.editor.setReadOnly(False)
        self.editor.setPlainText(text)
        # Reset undo history so Ctrl+Z can't roll back past the freshly
        # loaded buffer into the previous module's edits.
        self.editor.document().clearUndoRedoStacks()
        self._loading = False
        self._dirty = False
        self._update_header()
        self._set_buttons_enabled(True)
        self._sync_combo_selection(module_name)

    # ── Internal ──────────────────────────────────────────────────

    def _set_buttons_enabled(self, on: bool) -> None:
        self._compile_btn.setEnabled(on)
        self._revert_btn.setEnabled(on and self._dirty)

    def _set_combo_items(
        self, modules: "list[str]", *, selected: "str | None",
    ) -> None:
        with QSignalBlocker(self._material_combo):
            self._material_combo.clear()
            for mod in modules:
                # Display the trailing module name (stripped of the
                # `python_materials.` prefix) and keep the full dotted
                # name as item data for round-tripping.
                label = mod.split(".", 1)[1] if mod.startswith(
                    "python_materials.",
                ) else mod
                self._material_combo.addItem(label, mod)
            if selected is not None:
                idx = self._material_combo.findData(selected)
                if idx >= 0:
                    self._material_combo.setCurrentIndex(idx)
        self._material_combo.setEnabled(bool(modules))

    def _sync_combo_selection(self, module_name: "str | None") -> None:
        if module_name is None:
            with QSignalBlocker(self._material_combo):
                self._material_combo.setCurrentIndex(-1)
            return
        idx = self._material_combo.findData(module_name)
        if idx < 0 or idx == self._material_combo.currentIndex():
            return
        with QSignalBlocker(self._material_combo):
            self._material_combo.setCurrentIndex(idx)

    def _on_combo_changed(self, idx: int) -> None:
        if idx < 0:
            return
        module_name = self._material_combo.itemData(idx)
        if not isinstance(module_name, str) or module_name == self._module_name:
            return
        self.set_active_module(module_name)

    def _poll_for_modules(self) -> None:
        """Tick handler: request the scene's Python modules from the render
        worker while the dropdown stays empty. Stops polling once at least one
        module appears (USD load completed, applied in `_apply_modules`) or
        when the user has already loaded a buffer.
        """
        if self._dirty or self._module_name is not None:
            self._poll_timer.stop()
            return
        self._request_modules()

    def _on_text_changed(self) -> None:
        if self._loading:
            return
        if not self._dirty:
            self._dirty = True
            self._update_header()
        self._revert_btn.setEnabled(self._module_name is not None)

    def _on_revert_clicked(self) -> None:
        if self._source_path is None:
            return
        try:
            text = self._source_path.read_text(encoding="utf-8")
        except OSError as exc:
            self._append_output(f"Failed to read {self._source_path}: {exc}\n")
            return
        self._loading = True
        self.editor.setPlainText(text)
        self._loading = False
        self._dirty = False
        self._update_header()
        self._revert_btn.setEnabled(False)
        self._set_status("Reverted from disk.", ok=True)

    def _on_compile_clicked(self) -> None:
        if self._module_name is None or self._reload_inflight:
            return
        self._set_status("Compiling…", ok=None)
        # Disable buttons while the reload runs on the render worker so users
        # don't queue two slangc compiles on top of each other.
        self._compile_btn.setEnabled(False)
        self._revert_btn.setEnabled(False)
        self._reload_inflight = True
        module = self._module_name
        source = self.editor.toPlainText()
        # The reload (codegen + pipeline rebuild) is GPU work — run it on the
        # render worker, which owns the live renderer, and collect the result.
        fut = self.renderer.request(
            lambda r, m=module, s=source: MaterialReloader(r).reload(m, s),
        )
        fut.add_done_callback(self._on_reload_future)

    def _on_reload_future(self, fut) -> None:
        # Runs on the render worker thread; marshal the result to the GUI.
        try:
            result = fut.result()
        except Exception as exc:  # noqa: BLE001
            result = ReloadResult(
                ok=False, stage="pipeline", message=repr(exc),
            )
        self._reload_ready.emit(result)

    def _apply_reload_result(self, result: "ReloadResult") -> None:
        self._reload_inflight = False
        self._compile_btn.setEnabled(True)
        self._revert_btn.setEnabled(self._dirty)
        if result.ok:
            self._dirty = False
            self._update_header()
            self._revert_btn.setEnabled(False)
            self._set_status(
                f"Compiled in {result.duration_ms:.0f} ms.", ok=True,
            )
            self._append_output(
                f"[ok] {self._module_name} reloaded "
                f"({result.duration_ms:.0f} ms).\n",
            )
        else:
            self._set_status(
                f"Failed at {result.stage} ({result.duration_ms:.0f} ms).",
                ok=False,
            )
            self._append_output(
                f"[fail:{result.stage}] {self._module_name}:\n"
                f"{result.message}\n",
            )

    def _update_header(self) -> None:
        if self._module_name is None:
            return
        marker = "*" if self._dirty else ""
        path = str(self._source_path) if self._source_path else ""
        self._header.setText(f"{marker}{self._module_name}    —    {path}")

    def _set_status(self, text: str, *, ok: bool | None) -> None:
        if ok is True:
            color = "#73c991"
        elif ok is False:
            color = "#f48771"
        else:
            color = "#cccccc"
        self._status.setStyleSheet(f"color: {color};")
        self._status.setText(text)

    def _append_output(self, text: str) -> None:
        # `_strip_ansi` keeps slangc colour codes from leaking through.
        self.output.appendPlainText(_strip_ansi(text).rstrip("\n"))


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)
