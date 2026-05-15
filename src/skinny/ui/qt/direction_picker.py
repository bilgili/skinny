"""Arcball-style light direction widget.

A small sphere with a draggable dot. Drag rotates the current direction
through ``ui.direction_math.rotate_by_delta``; the new ``(elev, az)`` is
written back via the spec's getters/setters.
"""

from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QPainter, QPen
from PySide6.QtWidgets import QHBoxLayout, QLabel, QSizePolicy, QWidget

from skinny.ui import direction_math
from skinny.ui.spec import DirectionPicker


class DirectionPickerWidget(QWidget):
    SIZE = 96
    R = 40.0

    def __init__(self, node: DirectionPicker, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.node = node
        self.setMinimumSize(self.SIZE, self.SIZE)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.setMouseTracking(False)

        self._a: np.ndarray | None = None
        self._D: np.ndarray = direction_math.eulers_to_direction(
            float(node.elev_getter()), float(node.az_getter()),
        )

    def refresh_from_state(self) -> None:
        """Re-read elev/az from the renderer and repaint. Called by the
        backend's per-tick pull so external edits (presets, keyboard) show
        up here.
        """
        D_new = direction_math.eulers_to_direction(
            float(self.node.elev_getter()), float(self.node.az_getter()),
        )
        if not np.array_equal(D_new, self._D):
            self._D = D_new
            self.update()

    # ── Mouse ──────────────────────────────────────────────────────

    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.LeftButton:
            return
        cx, cy = self.width() / 2.0, self.height() / 2.0
        self._a = direction_math.arcball_vec(
            event.position().x(), event.position().y(), cx, cy, self.R,
        )

    def mouseMoveEvent(self, event) -> None:
        if self._a is None:
            return
        cx, cy = self.width() / 2.0, self.height() / 2.0
        b = direction_math.arcball_vec(
            event.position().x(), event.position().y(), cx, cy, self.R,
        )
        self._D = direction_math.rotate_by_delta(self._D, self._a, b)
        self._a = b

        el, az = direction_math.direction_to_eulers(self._D)
        self.node.elev_setter(el)
        self.node.az_setter(az)
        self.update()

    def mouseReleaseEvent(self, _event) -> None:
        self._a = None

    # ── Paint ──────────────────────────────────────────────────────

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)

        cx = self.width() / 2.0
        cy = self.height() / 2.0
        r = self.R

        # Sphere outline + meridians as orientation guides.
        painter.setPen(QPen(QColor(140, 140, 140), 1))
        painter.setBrush(QBrush(QColor(40, 40, 40)))
        painter.drawEllipse(QPointF(cx, cy), r, r)
        painter.setPen(QPen(QColor(85, 85, 85), 1))
        painter.drawLine(QPointF(cx - r, cy), QPointF(cx + r, cy))
        painter.drawLine(QPointF(cx, cy - r), QPointF(cx, cy + r))
        # Half-width ellipse hints at the front/back great circle.
        painter.drawEllipse(QRectF(cx - r * 0.5, cy - r, r, 2 * r))

        # Direction dot. Filled when in front (+Z), outlined when behind.
        px = cx + r * float(self._D[0])
        py = cy - r * float(self._D[1])
        front = float(self._D[2]) >= 0.0
        size = 4.5
        if front:
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(QColor(255, 224, 102)))
        else:
            painter.setPen(QPen(QColor(255, 224, 102), 1))
            painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(QPointF(px, py), size, size)


def build_direction_widget(node: DirectionPicker) -> QWidget:
    """Returns a labelled container with the arcball widget."""
    container = QWidget()
    layout = QHBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(6)
    label = QLabel(node.name)
    label.setMinimumWidth(120)
    layout.addWidget(label)
    widget = DirectionPickerWidget(node, container)
    layout.addWidget(widget)
    layout.addStretch(1)
    container._direction_widget = widget  # type: ignore[attr-defined]
    return container
