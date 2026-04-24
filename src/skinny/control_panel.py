"""Tkinter control panel — sliders + combos for every entry in ALL_PARAMS.

This runs in the same process/thread as the GLFW/Vulkan render loop. Each
frame, `app.main()` calls `panel.tick()`, which (a) pumps any pending Tk
events via `root.update()` and (b) pushes the renderer's current values back
into the widgets so keyboard edits and preset applications show up on screen.

Deliberately generated from `ALL_PARAMS`: adding a new ParamSpec over in
`app.py` automatically produces a new slider here. All writes go through
`_set_nested`, the same path the keyboard UI uses, so the accumulation reset
in `Renderer._current_state_hash` fires consistently.
"""

from __future__ import annotations

import math
import tkinter as tk
from tkinter import colorchooser, messagebox, simpledialog, ttk
from typing import Any

import numpy as np

from skinny.presets import apply_preset
from skinny.settings import delete_user_preset, save_user_preset


# ── Trackball helpers (module-private) ──────────────────────────────

def _arcball_vec(px: float, py: float, cx: float, cy: float, r: float) -> np.ndarray:
    """Project canvas pixel (px, py) onto the unit hemisphere centered at (cx, cy)."""
    x = (px - cx) / r
    y = -(py - cy) / r   # screen-y grows downward; flip to world y-up
    d2 = x * x + y * y
    if d2 <= 1.0:
        return np.array([x, y, math.sqrt(1.0 - d2)])
    inv = 1.0 / math.sqrt(d2)
    return np.array([x * inv, y * inv, 0.0])


def _rotate_by_delta(D: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rodrigues rotation of D by the rotation that takes unit vector a to unit vector b."""
    axis = np.cross(a, b)
    axis_len = float(np.linalg.norm(axis))
    if axis_len < 1e-8:
        return D
    axis = axis / axis_len
    cos_t = float(np.clip(np.dot(a, b), -1.0, 1.0))
    sin_t = axis_len   # |a||b|sinθ = |a×b| for unit a, b
    return (
        D * cos_t
        + np.cross(axis, D) * sin_t
        + axis * np.dot(axis, D) * (1.0 - cos_t)
    )


def _direction_to_eulers(D: np.ndarray) -> tuple[float, float]:
    """Inverse of renderer._update_light's spherical→Cartesian mapping."""
    el = math.degrees(math.asin(max(-1.0, min(1.0, float(D[1])))))
    az = math.degrees(math.atan2(float(D[0]), float(D[2])))
    return el, az


def _eulers_to_direction(el_deg: float, az_deg: float) -> np.ndarray:
    el = math.radians(el_deg)
    az = math.radians(az_deg)
    return np.array([
        math.cos(el) * math.sin(az),
        math.sin(el),
        math.cos(el) * math.cos(az),
    ])


def _rgb_floats_to_hex(r: float, g: float, b: float) -> str:
    rr = max(0, min(255, int(round(r * 255.0))))
    gg = max(0, min(255, int(round(g * 255.0))))
    bb = max(0, min(255, int(round(b * 255.0))))
    return f"#{rr:02x}{gg:02x}{bb:02x}"


_HIDDEN_PANEL_PATHS = {
    "light_color_r", "light_color_g", "light_color_b",
    "light_elevation", "light_azimuth",
}


class ControlPanel:
    def __init__(self, renderer, all_params: list) -> None:
        self.renderer = renderer
        self.params = all_params
        # Guards against our sync-from-renderer writes re-entering the
        # Scale/Combobox callbacks (which would clobber the value back).
        self._suppress_cb = False

        self.root: tk.Tk | None = tk.Tk()
        self.root.title("Skinny Controls")
        self.root.geometry("380x780")
        # Widgets indexed by dotted path: (variable, primary widget, value label).
        self._widgets: dict[str, tuple[tk.Variable, tk.Widget, ttk.Label | None]] = {}
        # Custom-row widget refs (filled in by _build_widgets).
        self._color_swatch: tk.Canvas | None = None
        self._color_swatch_rect: int | None = None
        self._direction_preview: tk.Canvas | None = None
        self._direction_preview_dot: int | None = None
        self._direction_popup: _DirectionPickerPopup | None = None
        self._build_widgets()

    # ── Widget construction ─────────────────────────────────────────

    def _build_widgets(self) -> None:
        from skinny.app import _get_nested  # late import — app imports this module

        container = ttk.Frame(self.root)
        container.pack(fill="both", expand=True, padx=4, pady=4)

        color_row_built = False
        direction_row_built = False

        for p in self.params:
            if p.path in _HIDDEN_PANEL_PATHS:
                # Collapse the three RGB sliders into a swatch + Pick... row,
                # and the elevation/azimuth sliders into an arrow-preview + Pick...
                # row. Built once at the position of the first-seen path so the
                # custom rows slot in where the original sliders used to be.
                if p.path.startswith("light_color_"):
                    if not color_row_built:
                        self._build_color_row(container)
                        color_row_built = True
                else:
                    if not direction_row_built:
                        self._build_direction_row(container)
                        direction_row_built = True
                continue

            row = ttk.Frame(container)
            row.pack(fill="x", pady=1)
            ttk.Label(row, text=p.name, width=18, anchor="w").pack(side="left")

            if p.kind == "continuous":
                var = tk.DoubleVar(value=float(_get_nested(self.renderer, p.path)))
                scale = ttk.Scale(
                    row,
                    from_=p.lo,
                    to=p.hi,
                    variable=var,
                    orient="horizontal",
                    command=lambda v, path=p.path: self._on_continuous(path, float(v)),
                )
                scale.pack(side="left", fill="x", expand=True, padx=(0, 4))
                val_lbl = ttk.Label(row, width=7, anchor="e",
                                    text=f"{float(var.get()):.3f}")
                val_lbl.pack(side="left")
                self._widgets[p.path] = (var, scale, val_lbl)
            else:
                choices = getattr(self.renderer, p.choice_source)
                names = [self._choice_label(c) for c in choices]
                current = int(_get_nested(self.renderer, p.path))
                var = tk.StringVar(value=names[current] if names else "")
                combo = ttk.Combobox(
                    row,
                    textvariable=var,
                    values=names,
                    state="readonly",
                )
                combo.pack(side="left", fill="x", expand=True)
                combo.bind(
                    "<<ComboboxSelected>>",
                    lambda _e, path=p.path, w=combo: self._on_discrete(path, w.current()),
                )
                self._widgets[p.path] = (var, combo, None)

                # Save / Delete buttons live directly under the Preset combo
                # so the user can write the current slider values back to
                # ~/.skinny/presets/<name>.json, or remove a user entry.
                if p.path == "preset_index":
                    btn_row = ttk.Frame(container)
                    btn_row.pack(fill="x", padx=4, pady=(0, 4))
                    ttk.Button(
                        btn_row, text="Save as user preset...",
                        command=self._on_save_preset,
                    ).pack(side="left", padx=(0, 4))
                    self._delete_btn = ttk.Button(
                        btn_row, text="Delete",
                        command=self._on_delete_preset,
                    )
                    self._delete_btn.pack(side="left")
                    self._update_delete_btn_state()

    def _build_color_row(self, container: ttk.Frame) -> None:
        row = ttk.Frame(container)
        row.pack(fill="x", pady=1)
        ttk.Label(row, text="Light color", width=18, anchor="w").pack(side="left")

        canvas = tk.Canvas(row, width=36, height=18, bd=1, relief="sunken",
                           highlightthickness=0)
        canvas.pack(side="left", padx=(0, 4))
        fill = _rgb_floats_to_hex(
            float(getattr(self.renderer, "light_color_r")),
            float(getattr(self.renderer, "light_color_g")),
            float(getattr(self.renderer, "light_color_b")),
        )
        rect_id = canvas.create_rectangle(0, 0, 36, 18, fill=fill, outline="")
        self._color_swatch = canvas
        self._color_swatch_rect = rect_id

        ttk.Button(row, text="Pick...", width=8,
                   command=self._on_pick_color).pack(side="left")

    def _build_direction_row(self, container: ttk.Frame) -> None:
        row = ttk.Frame(container)
        row.pack(fill="x", pady=1)
        ttk.Label(row, text="Light direction", width=18, anchor="w").pack(side="left")

        canvas = tk.Canvas(row, width=36, height=36, bg="grey25",
                           highlightthickness=0)
        canvas.pack(side="left", padx=(0, 4))
        canvas.create_oval(3, 3, 33, 33, outline="grey60", fill="grey30")
        D = _eulers_to_direction(
            float(getattr(self.renderer, "light_elevation")),
            float(getattr(self.renderer, "light_azimuth")),
        )
        dot_id = canvas.create_oval(0, 0, 0, 0, fill="", outline="")
        self._direction_preview = canvas
        self._direction_preview_dot = dot_id
        self._refresh_direction_preview(D)

        ttk.Button(row, text="Pick...", width=8,
                   command=self._on_pick_direction).pack(side="left")

    def _refresh_color_swatch(self) -> None:
        canvas = self._color_swatch
        rect_id = self._color_swatch_rect
        if canvas is None or rect_id is None:
            return
        fill = _rgb_floats_to_hex(
            float(getattr(self.renderer, "light_color_r")),
            float(getattr(self.renderer, "light_color_g")),
            float(getattr(self.renderer, "light_color_b")),
        )
        try:
            canvas.itemconfig(rect_id, fill=fill)
        except tk.TclError:
            pass

    def _refresh_direction_preview(self, D: np.ndarray | None = None) -> None:
        canvas = self._direction_preview
        dot_id = self._direction_preview_dot
        if canvas is None or dot_id is None:
            return
        if D is None:
            D = _eulers_to_direction(
                float(getattr(self.renderer, "light_elevation")),
                float(getattr(self.renderer, "light_azimuth")),
            )
        cx, cy, r = 18.0, 18.0, 13.0
        px = cx + r * float(D[0])
        py = cy - r * float(D[1])
        front = float(D[2]) >= 0.0
        try:
            canvas.coords(dot_id, px - 3, py - 3, px + 3, py + 3)
            canvas.itemconfig(
                dot_id,
                fill=("#ffe066" if front else ""),
                outline=("" if front else "#ffe066"),
                width=(0 if front else 1),
            )
        except tk.TclError:
            pass

    @staticmethod
    def _choice_label(choice: Any) -> str:
        return getattr(choice, "name", str(choice))

    # ── Callbacks ───────────────────────────────────────────────────

    def _on_continuous(self, path: str, value: float) -> None:
        if self._suppress_cb:
            return
        from skinny.app import _set_nested

        _set_nested(self.renderer, path, value)
        if path.startswith("light"):
            self.renderer._update_light()
        # Update value label immediately (otherwise it only refreshes on tick).
        entry = self._widgets.get(path)
        if entry is not None and entry[2] is not None:
            entry[2].configure(text=f"{value:.3f}")

    def _on_discrete(self, path: str, index: int) -> None:
        if self._suppress_cb:
            return
        from skinny.app import _set_nested

        if index < 0:
            return
        _set_nested(self.renderer, path, index)
        if path == "preset_index":
            apply_preset(self.renderer, self.renderer.presets[index])
            self._update_delete_btn_state()

    # ── Light color / direction pickers ─────────────────────────────

    def _on_pick_color(self) -> None:
        r = float(getattr(self.renderer, "light_color_r"))
        g = float(getattr(self.renderer, "light_color_g"))
        b = float(getattr(self.renderer, "light_color_b"))
        init = (
            max(0, min(255, int(round(r * 255.0)))),
            max(0, min(255, int(round(g * 255.0)))),
            max(0, min(255, int(round(b * 255.0)))),
        )
        result = colorchooser.askcolor(initialcolor=init, parent=self.root,
                                       title="Light color")
        if result is None or result[0] is None:
            return
        rf, gf, bf = (float(c) / 255.0 for c in result[0])
        self._on_continuous("light_color_r", rf)
        self._on_continuous("light_color_g", gf)
        self._on_continuous("light_color_b", bf)
        self._refresh_color_swatch()

    def _on_pick_direction(self) -> None:
        popup = self._direction_popup
        if popup is not None and popup.is_open():
            popup.focus()
            return
        self._direction_popup = _DirectionPickerPopup(self)

    def _on_direction_popup_closed(self) -> None:
        self._direction_popup = None

    # ── Preset save / delete / refresh ──────────────────────────────

    def _on_save_preset(self) -> None:
        from dataclasses import fields

        from skinny.renderer import SkinParameters

        name = simpledialog.askstring(
            "Save preset", "Preset name:", parent=self.root
        )
        if not name or not name.strip():
            return
        name = name.strip()

        values: dict[str, float] = {}
        for f in fields(SkinParameters):
            if f.name == "scattering_coefficient":
                continue  # vec3 — presets intentionally skip it
            values[f"skin.{f.name}"] = float(getattr(self.renderer.skin, f.name))

        try:
            save_user_preset(name, values)
        except OSError as exc:
            messagebox.showerror("Save preset", f"Failed to save preset:\n{exc}")
            return

        self.renderer.refresh_user_presets()
        self._refresh_preset_combo()

        # Select the newly-saved preset so the Delete button lights up.
        for i, preset in enumerate(self.renderer.presets):
            if preset.name == name:
                self.renderer.preset_index = i
                break
        self._update_delete_btn_state()

    def _on_delete_preset(self) -> None:
        idx = int(self.renderer.preset_index)
        if not (0 <= idx < len(self.renderer.presets)):
            return
        preset = self.renderer.presets[idx]
        if preset.is_builtin:
            return
        if not messagebox.askyesno(
            "Delete preset", f"Delete '{preset.name}'?", parent=self.root
        ):
            return

        delete_user_preset(preset.name)
        self.renderer.refresh_user_presets()
        self.renderer.preset_index = 0
        if self.renderer.presets:
            apply_preset(self.renderer, self.renderer.presets[0])
        self._refresh_preset_combo()
        self._update_delete_btn_state()

    def _refresh_preset_combo(self) -> None:
        entry = self._widgets.get("preset_index")
        if entry is None:
            return
        _var, combo, _ = entry
        names = [self._choice_label(p) for p in self.renderer.presets]
        combo.configure(values=names)

    def _update_delete_btn_state(self) -> None:
        btn = getattr(self, "_delete_btn", None)
        if btn is None:
            return
        idx = int(self.renderer.preset_index)
        presets = self.renderer.presets
        deletable = 0 <= idx < len(presets) and not presets[idx].is_builtin
        try:
            btn.configure(state=("normal" if deletable else "disabled"))
        except tk.TclError:
            pass

    # ── Window geometry ─────────────────────────────────────────────

    def get_geometry(self) -> str | None:
        if self.root is None:
            return None
        try:
            return self.root.winfo_geometry()
        except tk.TclError:
            return None

    def apply_geometry(self, geom: str) -> None:
        if self.root is None or not geom:
            return
        try:
            self.root.geometry(geom)
        except tk.TclError:
            pass

    # ── Per-frame update ────────────────────────────────────────────

    def tick(self) -> None:
        if self.root is None:
            return
        from skinny.app import _get_nested

        self._suppress_cb = True
        try:
            for p in self.params:
                entry = self._widgets.get(p.path)
                if entry is None:
                    continue  # hidden paths (RGB, elev/az) — no slider row
                var, _widget, val_lbl = entry
                cur = _get_nested(self.renderer, p.path)
                if p.kind == "continuous":
                    cur_f = float(cur)
                    if abs(float(var.get()) - cur_f) > 1e-5:
                        var.set(cur_f)
                    if val_lbl is not None:
                        val_lbl.configure(text=f"{cur_f:.3f}")
                else:
                    choices = getattr(self.renderer, p.choice_source)
                    idx = int(cur)
                    if 0 <= idx < len(choices):
                        name = self._choice_label(choices[idx])
                        if var.get() != name:
                            var.set(name)
        finally:
            self._suppress_cb = False

        self._refresh_color_swatch()
        self._refresh_direction_preview()
        self._update_delete_btn_state()

        try:
            self.root.update()
        except tk.TclError:
            # User clicked the Tk window's close button. Null the handle so
            # subsequent ticks are a cheap no-op; the Vulkan window keeps
            # running with keyboard control intact.
            self.root = None

    def destroy(self) -> None:
        if self.root is None:
            return
        try:
            self.root.destroy()
        except tk.TclError:
            pass
        self.root = None


# ── Light direction popup ───────────────────────────────────────────


class _DirectionPickerPopup:
    """Non-modal Toplevel with a trackball-style sphere for aiming the scene light.

    Click-drag on the canvas rotates the current light direction by the
    arcball delta between successive cursor positions. The light dir lives
    on the renderer (as elev/az); we write it back through the panel's
    _on_continuous so the accumulation-reset path fires exactly like a slider.
    """

    CANVAS = 300
    R = 120.0

    def __init__(self, panel: ControlPanel) -> None:
        self.panel = panel
        self._alive = True

        top = tk.Toplevel(panel.root)
        top.title("Light direction")
        top.protocol("WM_DELETE_WINDOW", self.close)
        self.top = top

        canvas = tk.Canvas(top, width=self.CANVAS, height=self.CANVAS,
                           bg="grey10", highlightthickness=0)
        canvas.pack(padx=8, pady=8)
        self.canvas = canvas

        cx = cy = self.CANVAS / 2.0
        r = self.R
        canvas.create_oval(cx - r, cy - r, cx + r, cy + r,
                           outline="grey60", fill="grey25")
        # Equator + two meridians as static orientation guides.
        canvas.create_line(cx - r, cy, cx + r, cy, fill="grey45")
        canvas.create_line(cx, cy - r, cx, cy + r, fill="grey45")
        canvas.create_oval(cx - r * 0.5, cy - r, cx + r * 0.5, cy + r,
                           outline="grey45")

        self._dot = canvas.create_oval(0, 0, 0, 0, fill="", outline="")

        self.readout = ttk.Label(top, text="", anchor="center")
        self.readout.pack(fill="x", padx=8)
        ttk.Button(top, text="Close", command=self.close).pack(pady=(4, 8))

        self._D = _eulers_to_direction(
            float(getattr(panel.renderer, "light_elevation")),
            float(getattr(panel.renderer, "light_azimuth")),
        )
        self._a: np.ndarray | None = None
        self._redraw()

        canvas.bind("<ButtonPress-1>", self._on_press)
        canvas.bind("<B1-Motion>", self._on_drag)

    # ── Mouse handling ──────────────────────────────────────────────

    def _on_press(self, event) -> None:
        cx = cy = self.CANVAS / 2.0
        self._a = _arcball_vec(event.x, event.y, cx, cy, self.R)

    def _on_drag(self, event) -> None:
        if self._a is None:
            return
        cx = cy = self.CANVAS / 2.0
        b = _arcball_vec(event.x, event.y, cx, cy, self.R)
        self._D = _rotate_by_delta(self._D, self._a, b)
        self._a = b

        el, az = _direction_to_eulers(self._D)
        self.panel._on_continuous("light_elevation", el)
        self.panel._on_continuous("light_azimuth", az)
        self._redraw()

    # ── Drawing ─────────────────────────────────────────────────────

    def _redraw(self) -> None:
        cx = cy = self.CANVAS / 2.0
        r = self.R
        px = cx + r * float(self._D[0])
        py = cy - r * float(self._D[1])
        front = float(self._D[2]) >= 0.0
        size = 7
        try:
            self.canvas.coords(self._dot, px - size, py - size, px + size, py + size)
            self.canvas.itemconfig(
                self._dot,
                fill=("#ffe066" if front else ""),
                outline=("#ffe066" if not front else "#8a6a00"),
                width=(1 if not front else 1),
            )
        except tk.TclError:
            return

        el, az = _direction_to_eulers(self._D)
        hemi = "front" if front else "back"
        self.readout.configure(
            text=f"elev: {el:+6.1f}°   az: {az:+7.1f}°   ({hemi})"
        )

    # ── Lifecycle ───────────────────────────────────────────────────

    def is_open(self) -> bool:
        if not self._alive:
            return False
        try:
            return bool(self.top.winfo_exists())
        except tk.TclError:
            return False

    def focus(self) -> None:
        try:
            self.top.lift()
            self.top.focus_set()
        except tk.TclError:
            pass

    def close(self) -> None:
        self._alive = False
        try:
            self.top.destroy()
        except tk.TclError:
            pass
        self.panel._on_direction_popup_closed()
