"""Qt port of ``bxdf_visualizer.BXDFVisualizer``.

Renders the per-material BXDF lobe to a 2D image (PIL → QPixmap).
Reuses the pure-numpy + Pillow evaluator + rasteriser from
``skinny.bxdf_visualizer`` unchanged.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QMouseEvent, QPixmap, QWheelEvent
from PySide6.QtWidgets import (
    QButtonGroup, QCheckBox, QDockWidget, QGroupBox, QHBoxLayout, QLabel,
    QPushButton, QRadioButton, QSlider, QVBoxLayout, QWidget,
)

from skinny.bxdf_math import eval_grid, render_lobe_image
from skinny.renderer import _hashable_value


class _LobeCanvas(QLabel):
    """``QLabel`` showing the rendered lobe pixmap. Forwards mouse drag
    and wheel events to its parent dock via callbacks so the lobe orbits
    + zooms.
    """

    def __init__(
        self, size: int,
        on_drag: callable, on_wheel: callable, parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setFixedSize(size, size)
        self.setStyleSheet("background-color: black; border: 1px solid #333;")
        self._on_drag = on_drag
        self._on_wheel = on_wheel
        self._last: tuple[float, float] | None = None

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            self._last = (event.position().x(), event.position().y())

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._last is None:
            return
        x, y = event.position().x(), event.position().y()
        dx = x - self._last[0]
        dy = y - self._last[1]
        self._last = (x, y)
        self._on_drag(dx, dy)

    def mouseReleaseEvent(self, _event: QMouseEvent) -> None:
        self._last = None

    def wheelEvent(self, event: QWheelEvent) -> None:
        notches = event.angleDelta().y() / 120.0
        if notches != 0:
            self._on_wheel(int(notches))


class BXDFDock(QDockWidget):
    """Non-modal dock hosting the lobe canvas + material/pick/direction
    controls.
    """

    LOBE_SIZE = 480
    EVAL_DEBOUNCE_MS = 120

    def __init__(self, renderer, viewport, parent: QWidget | None = None) -> None:
        super().__init__("BXDF Visualizer", parent)
        self.renderer = renderer
        self.viewport = viewport  # ``RenderViewport``, owns arm_scene_pick
        self.setAllowedAreas(Qt.AllDockWidgetAreas)

        self._material_id: int = -1
        self._pick_state: Optional[dict] = None
        self._entrance_state: Optional[dict] = None
        self._mode: str = "bxdf"
        self._lock_mode: int = 0
        self._log_scale: bool = True
        self._theta_deg: float = 30.0
        self._phi_deg: float = 0.0
        self._yaw: float = math.radians(35.0)
        self._pitch: float = math.radians(20.0)
        self._zoom: float = 1.0

        self._cached_dirs: Optional[np.ndarray] = None
        self._cached_f: Optional[np.ndarray] = None
        self._cached_gpu: bool = False
        self._pending_dirs: Optional[np.ndarray] = None
        self._last_material_hash: int = self._compute_material_hash()

        self._build_widgets()

        # Watch for renderer-side material edits + scene-graph swaps so
        # the lobe re-evaluates when the user moves a slider in the main
        # sidebar.
        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(200)
        self._poll_timer.timeout.connect(self._poll_material_changes)
        self._poll_timer.start()

    # ── Layout ────────────────────────────────────────────────────

    def _build_widgets(self) -> None:
        host = QWidget()
        outer = QVBoxLayout(host)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(4)

        self._material_label = QLabel("Material: (no pick)")
        outer.addWidget(self._material_label)

        # Mode.
        mode_box = QGroupBox("Mode")
        mb = QHBoxLayout(mode_box)
        self._mode_bxdf = QRadioButton("BXDF (surface lobe)")
        self._mode_bssrdf = QRadioButton("BSSRDF (skin diffusion)")
        self._mode_bxdf.setChecked(True)
        self._mode_bxdf.toggled.connect(self._on_mode_changed)
        mb.addWidget(self._mode_bxdf)
        mb.addWidget(self._mode_bssrdf)
        mb.addStretch(1)
        outer.addWidget(mode_box)

        # Pick controls.
        pick_box = QGroupBox("Scene point")
        pb = QHBoxLayout(pick_box)
        self._pick_btn = QPushButton("Pick exit point")
        self._pick_btn.clicked.connect(self._on_pick_click)
        pb.addWidget(self._pick_btn)
        self._pick_readout = QLabel("No pick yet.")
        pb.addWidget(self._pick_readout, stretch=1)
        self._entrance_btn = QPushButton("Pick entrance")
        self._entrance_btn.setEnabled(False)
        self._entrance_btn.clicked.connect(self._on_entrance_pick_click)
        pb.addWidget(self._entrance_btn)
        self._entrance_readout = QLabel("(BSSRDF only)")
        pb.addWidget(self._entrance_readout)
        outer.addWidget(pick_box)

        # Directions.
        dir_box = QGroupBox("Directions")
        dl = QVBoxLayout(dir_box)
        lock_row = QHBoxLayout()
        self._lock_wi = QRadioButton("Lock wi (sweep wo)")
        self._lock_wi.setChecked(True)
        self._lock_wo = QRadioButton("Lock wo (sweep wi)")
        lock_group = QButtonGroup(dir_box)
        lock_group.addButton(self._lock_wi, 0)
        lock_group.addButton(self._lock_wo, 1)
        lock_group.idToggled.connect(self._on_lock_changed)
        lock_row.addWidget(self._lock_wi)
        lock_row.addWidget(self._lock_wo)
        self._log_cb = QCheckBox("Log scale")
        self._log_cb.setChecked(True)
        self._log_cb.toggled.connect(self._on_log_toggled)
        lock_row.addWidget(self._log_cb)
        lock_row.addStretch(1)
        dl.addLayout(lock_row)

        # theta + phi sliders.
        for label_text, attr, lo, hi, init in (
            ("theta", "_theta_slider", 0, 89, 30),
            ("phi",   "_phi_slider",   0, 359, 0),
        ):
            row = QHBoxLayout()
            row.addWidget(QLabel(label_text + ":"))
            sl = QSlider(Qt.Horizontal)
            sl.setRange(lo, hi); sl.setValue(init)
            sl.valueChanged.connect(self._schedule_eval)
            row.addWidget(sl, stretch=1)
            val_lbl = QLabel(f"{init}")
            sl.valueChanged.connect(lambda v, l=val_lbl: l.setText(str(v)))
            row.addWidget(val_lbl)
            dl.addLayout(row)
            setattr(self, attr, sl)
        outer.addWidget(dir_box)

        # Lobe canvas.
        canvas_box = QGroupBox("Lobe")
        cl = QVBoxLayout(canvas_box)
        self._canvas = _LobeCanvas(
            self.LOBE_SIZE, on_drag=self._on_canvas_drag,
            on_wheel=self._on_canvas_wheel, parent=canvas_box,
        )
        cl.addWidget(self._canvas, alignment=Qt.AlignCenter)
        outer.addWidget(canvas_box, stretch=1)

        self._status = QLabel("Pick a material and a scene point.")
        outer.addWidget(self._status)

        self.setWidget(host)

    # ── Mode + scale ──────────────────────────────────────────────

    def _on_mode_changed(self, _checked: bool) -> None:
        self._mode = "bxdf" if self._mode_bxdf.isChecked() else "bssrdf"
        bssrdf = self._mode == "bssrdf"
        self._entrance_btn.setEnabled(bssrdf)
        if not bssrdf:
            self._entrance_readout.setText("(BSSRDF only)")
        elif self._entrance_state is None:
            self._entrance_readout.setText("Entrance not picked.")
        self._schedule_eval()

    def _on_lock_changed(self, btn_id: int, checked: bool) -> None:
        if checked:
            self._lock_mode = int(btn_id)
            self._schedule_eval()

    def _on_log_toggled(self, checked: bool) -> None:
        self._log_scale = bool(checked)
        # Log scale only affects PIL rasterisation; no GPU re-eval needed.
        self._do_render()

    # ── Pick flow ─────────────────────────────────────────────────

    def _on_pick_click(self) -> None:
        if self.viewport is None:
            self._status.setText("Pick unavailable: no viewport bound.")
            return
        self.viewport.arm_scene_pick(self._on_pick_result)
        self._status.setText("Click the main viewport to pick exit point.")

    def _on_entrance_pick_click(self) -> None:
        if self.viewport is None:
            self._status.setText("Pick unavailable: no viewport bound.")
            return
        self.viewport.arm_scene_pick(self._on_entrance_pick_result)
        self._status.setText("Click the main viewport to pick BSSRDF entrance.")

    def _on_pick_result(self, result: Optional[dict]) -> None:
        if result is None:
            self._pick_readout.setText("Pick missed: no scene hit.")
            self._status.setText("Pick missed. Try again.")
            return
        self._pick_state = result
        p = result["position"]; n = result["normal"]
        mat_id = int(result["material_id"])
        self._pick_readout.setText(
            f"matId={mat_id}  P=({p[0]:+.2f}, {p[1]:+.2f}, {p[2]:+.2f})"
            f"  N=({n[0]:+.2f}, {n[1]:+.2f}, {n[2]:+.2f})"
        )
        self._set_active_material(mat_id)
        self._status.setText("Scene point captured. Evaluating BXDF…")
        self._schedule_eval()

    def _on_entrance_pick_result(self, result: Optional[dict]) -> None:
        if result is None:
            self._entrance_readout.setText("Entrance pick missed.")
            return
        self._entrance_state = result
        p = result["position"]
        self._entrance_readout.setText(
            f"xi=({p[0]:+.2f}, {p[1]:+.2f}, {p[2]:+.2f})"
        )
        self._schedule_eval()

    def _set_active_material(self, mat_id: int) -> None:
        self._material_id = mat_id
        mats = self._scene_materials()
        if 0 <= mat_id < len(mats):
            name = getattr(mats[mat_id], "name", "?")
            self._material_label.setText(f"Material: #{mat_id} — {name}")
        else:
            self._material_label.setText(f"Material: #{mat_id}")

    def _scene_materials(self) -> list:
        scene = getattr(self.renderer, "scene", None)
        if scene is None:
            return []
        return list(getattr(scene, "materials", []) or [])

    # ── Eval scheduling ───────────────────────────────────────────

    def _schedule_eval(self) -> None:
        """Coalesce rapid slider drags into one GPU eval after ``EVAL_DEBOUNCE_MS``."""
        # Read the latest slider values into our cached scalars before
        # firing — Qt's valueChanged delivered them in order, but the
        # eval reads from these attrs.
        self._theta_deg = float(self._theta_slider.value())
        self._phi_deg = float(self._phi_slider.value())
        QTimer.singleShot(self.EVAL_DEBOUNCE_MS, self._do_eval)

    def _locked_dir(self) -> np.ndarray:
        theta = math.radians(self._theta_deg)
        phi = math.radians(self._phi_deg)
        return np.array(
            [math.sin(theta) * math.cos(phi),
             math.sin(theta) * math.sin(phi),
             math.cos(theta)],
            dtype=np.float64,
        )

    def _make_dirs_grid(self, n_theta: int, n_phi: int) -> np.ndarray:
        thetas = (np.arange(n_theta) + 0.5) / n_theta * (math.pi * 0.5)
        phis = np.arange(n_phi) / n_phi * (2.0 * math.pi)
        sin_t = np.sin(thetas)
        cos_t = np.cos(thetas)
        cos_p = np.cos(phis)
        sin_p = np.sin(phis)
        dirs = np.empty((n_theta, n_phi, 3), dtype=np.float64)
        dirs[..., 0] = sin_t[:, None] * cos_p[None, :]
        dirs[..., 1] = sin_t[:, None] * sin_p[None, :]
        dirs[..., 2] = cos_t[:, None]
        return dirs

    def _do_eval(self) -> None:
        n_theta, n_phi = 24, 48
        if self._pick_state is None:
            self._status.setText("Pick a scene point to evaluate a material.")
            return
        if self._material_id < 0:
            self._status.setText("Pick captured no valid material id.")
            return
        if self._mode == "bssrdf" and self._entrance_state is None:
            self._status.setText("BSSRDF mode: pick an entrance point first.")
            return

        req = {
            "material_id": self._material_id,
            "position": self._pick_state["position"],
            "normal": self._pick_state["normal"],
            "tangent": self._pick_state["tangent"],
            "uv": self._pick_state["uv"],
            "locked_dir": self._locked_dir(),
            "lock_mode": self._lock_mode,
            "n_theta": n_theta,
            "n_phi": n_phi,
        }
        self._pending_dirs = self._make_dirs_grid(n_theta, n_phi)
        try:
            if self._mode == "bssrdf":
                req["entrance_position"] = self._entrance_state["position"]
                self.renderer.request_bssrdf_eval(req, self._on_gpu_eval_result)
            else:
                self.renderer.request_bxdf_eval(req, self._on_gpu_eval_result)
            return
        except Exception as exc:  # noqa: BLE001
            print(f"[skinny] GPU eval failed: {exc}")
            self._status.setText(f"GPU eval failed: {exc}")

        # CPU fallback (analytic Lambert+GGX, no graph procedurals).
        mats = self._scene_materials()
        if not (0 <= self._material_id < len(mats)):
            return
        params = dict(getattr(mats[self._material_id], "parameter_overrides", {}) or {})
        dirs, f = eval_grid(self._locked_dir(), self._lock_mode, n_theta, n_phi, params)
        self._cache_and_render(dirs, f, gpu=False)

    def _on_gpu_eval_result(self, grid: np.ndarray) -> None:
        if self._pending_dirs is None:
            return
        dirs = self._pending_dirs
        f = grid.astype(np.float64)
        self._cache_and_render(dirs, f, gpu=True)

    def _cache_and_render(
        self, dirs: np.ndarray, f: np.ndarray, gpu: bool,
    ) -> None:
        self._cached_dirs = dirs
        self._cached_f = f
        self._cached_gpu = gpu
        self._do_render()

    def _do_render(self) -> None:
        if self._cached_dirs is None or self._cached_f is None:
            return
        img = render_lobe_image(
            self._cached_dirs, self._cached_f,
            self._yaw, self._pitch, size=self.LOBE_SIZE,
            log_scale=self._log_scale, zoom=self._zoom,
        )
        # PIL → QImage (RGB888) → QPixmap. Convert to RGBA8888 in case
        # render_lobe_image grows an alpha channel later.
        rgb = img.convert("RGB")
        data = rgb.tobytes("raw", "RGB")
        qimg = QImage(data, rgb.width, rgb.height, 3 * rgb.width, QImage.Format_RGB888).copy()
        self._canvas.setPixmap(QPixmap.fromImage(qimg))

        max_f = float(self._cached_f.max())
        mats = self._scene_materials()
        name = mats[self._material_id].name if 0 <= self._material_id < len(mats) else "?"
        src = "GPU" if self._cached_gpu else "CPU"
        scale = "log" if self._log_scale else "lin"
        self._status.setText(
            f"Material #{self._material_id} ({name})  "
            f"max f·cosθ = {max_f:.3f}  [{src} | {scale}]"
        )

    # ── Canvas mouse + wheel ──────────────────────────────────────

    def _on_canvas_drag(self, dx: float, dy: float) -> None:
        self._yaw += dx * 0.012
        self._pitch = max(
            -math.pi * 0.49, min(math.pi * 0.49, self._pitch + dy * 0.012),
        )
        # Orbit is a local re-rasterisation; cached grid is reused.
        self._do_render()

    def _on_canvas_wheel(self, notches: int) -> None:
        # 1.15× per notch, clamped so the hemisphere stays visible.
        factor = 1.15 ** notches
        self._zoom = max(0.1, min(8.0, self._zoom * factor))
        self._do_render()

    # ── Live material state ───────────────────────────────────────

    def _compute_material_hash(self) -> int:
        r = self.renderer
        version = int(getattr(r, "_material_version", 0))
        mtlx = getattr(r, "mtlx_overrides", {}) or {}
        try:
            payload = tuple(sorted(
                (k, _hashable_value(v)) for k, v in mtlx.items()
            ))
        except Exception:
            payload = ()
        return hash((version, payload))

    def _poll_material_changes(self) -> None:
        h = self._compute_material_hash()
        if h == self._last_material_hash:
            return
        self._last_material_hash = h
        if self._pick_state is not None:
            self._schedule_eval()
