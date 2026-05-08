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
from datetime import datetime
from pathlib import Path
from tkinter import colorchooser, filedialog, messagebox, simpledialog, ttk
from typing import Any

import numpy as np

from skinny.params import RESOLUTION_PRESETS, find_resolution_preset_index
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
        self.root.geometry("720x580")
        # Widgets indexed by dotted path: (variable, primary widget, value label).
        self._widgets: dict[str, tuple[tk.Variable, tk.Widget, ttk.Label | None]] = {}
        # Custom-row widget refs (filled in by _build_widgets).
        self._color_swatch: tk.Canvas | None = None
        self._color_swatch_rect: int | None = None
        self._direction_preview: tk.Canvas | None = None
        self._direction_preview_dot: int | None = None
        self._direction_popup: _DirectionPickerPopup | None = None
        self._material_container: ttk.Frame | None = None
        self._material_outer: ttk.LabelFrame | None = None
        self._last_scene_id: int = -1
        self._scene_graph_window = None
        self._build_widgets()

    # ── Widget construction ─────────────────────────────────────────

    _LABEL_WIDTH = 16

    def _build_widgets(self) -> None:
        columns = ttk.Frame(self.root)
        columns.pack(fill="both", expand=True, padx=4, pady=4)

        left_col = ttk.Frame(columns)
        left_col.pack(side="left", fill="both", expand=True, padx=(0, 2))
        right_col = ttk.Frame(columns)
        right_col.pack(side="left", fill="both", expand=True, padx=(2, 0))

        # ── Left: Resolution ──
        self._build_resolution_section(left_col)

        # ── Left: Capture ──
        self._build_capture_section(left_col)

        # ── Left: Scene Graph button ──
        sg_btn = ttk.Button(
            left_col, text="Scene Graph...",
            command=self._on_open_scene_graph,
        )
        sg_btn.pack(fill="x", padx=2, pady=2)

        # ── Left: Render settings ──
        render_frame = ttk.LabelFrame(left_col, text="Render", padding=4)
        render_frame.pack(fill="x", padx=2, pady=2)

        for p in self.params:
            if p.path.startswith(("skin.", "mtlx.")) or p.path.startswith("light_"):
                continue
            if p.path in _HIDDEN_PANEL_PATHS:
                continue
            self._build_param_row(render_frame, p)
            if p.path == "preset_index":
                btn_row = ttk.Frame(render_frame)
                btn_row.pack(fill="x", pady=(0, 4))
                ttk.Button(
                    btn_row, text="Save preset...",
                    command=self._on_save_preset,
                ).pack(side="left", padx=(0, 4))
                self._delete_btn = ttk.Button(
                    btn_row, text="Delete",
                    command=self._on_delete_preset,
                )
                self._delete_btn.pack(side="left")
                self._update_delete_btn_state()

        # ── Left: Light ──
        light_frame = ttk.LabelFrame(left_col, text="Light", padding=4)
        light_frame.pack(fill="x", padx=2, pady=2)

        self._build_color_row(light_frame)
        self._build_direction_row(light_frame)
        for p in self.params:
            if p.path.startswith("light_") and p.path not in _HIDDEN_PANEL_PATHS:
                self._build_param_row(light_frame, p)

        # ── Right: Materials (skin + scene) ──
        self._build_material_widgets(right_col)

    def _build_resolution_section(self, container: ttk.Frame) -> None:
        frame = ttk.LabelFrame(container, text="Resolution", padding=4)
        frame.pack(fill="x", padx=2, pady=2)

        cur_w = int(getattr(self.renderer, "width", 1280))
        cur_h = int(getattr(self.renderer, "height", 720))
        preset_idx = find_resolution_preset_index(cur_w, cur_h)

        preset_row = ttk.Frame(frame)
        preset_row.pack(fill="x", pady=1)
        ttk.Label(preset_row, text="Preset", width=self._LABEL_WIDTH,
                  anchor="w").pack(side="left")
        self._resolution_preset_var = tk.StringVar(
            value=RESOLUTION_PRESETS[preset_idx][0]
        )
        preset_combo = ttk.Combobox(
            preset_row,
            textvariable=self._resolution_preset_var,
            values=[name for name, _w, _h in RESOLUTION_PRESETS],
            state="readonly",
        )
        preset_combo.pack(side="left", fill="x", expand=True)
        preset_combo.bind(
            "<<ComboboxSelected>>",
            lambda _e: self._on_resolution_preset_selected(),
        )
        self._resolution_preset_combo = preset_combo

        wh_row = ttk.Frame(frame)
        wh_row.pack(fill="x", pady=1)
        ttk.Label(wh_row, text="Width × Height", width=self._LABEL_WIDTH,
                  anchor="w").pack(side="left")
        self._resolution_width_var = tk.StringVar(value=str(cur_w))
        self._resolution_height_var = tk.StringVar(value=str(cur_h))
        w_entry = ttk.Entry(
            wh_row, textvariable=self._resolution_width_var, width=6,
        )
        w_entry.pack(side="left", padx=(0, 2))
        ttk.Label(wh_row, text="×").pack(side="left", padx=2)
        h_entry = ttk.Entry(
            wh_row, textvariable=self._resolution_height_var, width=6,
        )
        h_entry.pack(side="left", padx=(2, 6))
        # Commit on Return or focus-out.
        for entry in (w_entry, h_entry):
            entry.bind("<Return>", lambda _e: self._on_resolution_apply())
            entry.bind("<FocusOut>", lambda _e: self._on_resolution_apply())
        ttk.Button(
            wh_row, text="Apply", width=8,
            command=self._on_resolution_apply,
        ).pack(side="left")

    def _on_resolution_preset_selected(self) -> None:
        name = self._resolution_preset_var.get()
        for entry_name, w, h in RESOLUTION_PRESETS:
            if entry_name == name:
                if w == 0 or h == 0:  # "Custom" — leave entries alone
                    return
                self._resolution_width_var.set(str(w))
                self._resolution_height_var.set(str(h))
                self._apply_resolution(w, h)
                return

    def _on_resolution_apply(self) -> None:
        try:
            w = int(self._resolution_width_var.get())
            h = int(self._resolution_height_var.get())
        except (TypeError, ValueError):
            return
        self._apply_resolution(w, h)

    def _apply_resolution(self, width: int, height: int) -> None:
        try:
            self.renderer.resize(width, height)
        except Exception as exc:
            messagebox.showerror(
                "Resize failed", f"Could not resize to {width}x{height}:\n{exc}",
                parent=self.root,
            )
            return
        # Re-read what the renderer actually settled on (may be clamped /
        # rounded to a workgroup multiple) and update the entries.
        actual_w = int(self.renderer.width)
        actual_h = int(self.renderer.height)
        self._resolution_width_var.set(str(actual_w))
        self._resolution_height_var.set(str(actual_h))
        idx = find_resolution_preset_index(actual_w, actual_h)
        self._resolution_preset_var.set(RESOLUTION_PRESETS[idx][0])

    _CAPTURE_FORMAT_EXTENSIONS = {
        "PNG":  ("png",  "PNG image",         ".png"),
        "JPEG": ("jpeg", "JPEG image",        ".jpg"),
        "BMP":  ("bmp",  "Bitmap",            ".bmp"),
        "EXR":  ("exr",  "OpenEXR (linear)",  ".exr"),
        "HDR":  ("hdr",  "Radiance HDR",      ".hdr"),
    }

    def _build_capture_section(self, container: ttk.Frame) -> None:
        frame = ttk.LabelFrame(container, text="Capture", padding=4)
        frame.pack(fill="x", padx=2, pady=2)

        row = ttk.Frame(frame)
        row.pack(fill="x", pady=1)
        ttk.Label(row, text="Format", width=self._LABEL_WIDTH,
                  anchor="w").pack(side="left")
        self._capture_format_var = tk.StringVar(value="PNG")
        ttk.Combobox(
            row,
            textvariable=self._capture_format_var,
            values=list(self._CAPTURE_FORMAT_EXTENSIONS.keys()),
            state="readonly",
        ).pack(side="left", fill="x", expand=True, padx=(0, 6))
        ttk.Button(
            row, text="Screenshot", width=12,
            command=self._on_screenshot,
        ).pack(side="left")

    def _on_screenshot(self) -> None:
        label = self._capture_format_var.get()
        spec = self._CAPTURE_FORMAT_EXTENSIONS.get(label)
        if spec is None:
            return
        fmt, desc, ext = spec
        default_name = f"skinny_{datetime.now():%Y%m%d_%H%M%S}{ext}"
        path = filedialog.asksaveasfilename(
            parent=self.root,
            title="Save screenshot",
            defaultextension=ext,
            initialfile=default_name,
            filetypes=[(desc, f"*{ext}"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            self.renderer.save_screenshot(path, fmt)
        except Exception as exc:
            messagebox.showerror(
                "Screenshot failed",
                f"Could not save {label} screenshot:\n{exc}",
                parent=self.root,
            )

    def _on_load_model(self) -> None:
        path = filedialog.askopenfilename(
            parent=self.root,
            title="Open model",
            filetypes=[
                ("All supported", "*.usda *.usdc *.usdz *.obj"),
                ("USD files", "*.usda *.usdc *.usdz"),
                ("OBJ files", "*.obj"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        try:
            self.renderer.load_model_from_path(Path(path))
        except Exception as exc:
            messagebox.showerror(
                "Load failed",
                f"Could not load model:\n{exc}",
                parent=self.root,
            )

    def _build_param_row(self, container: ttk.Frame, p) -> None:
        from skinny.params import _get_nested

        row = ttk.Frame(container)
        row.pack(fill="x", pady=1)
        ttk.Label(row, text=p.name, width=self._LABEL_WIDTH, anchor="w").pack(
            side="left",
        )

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
            var = tk.StringVar(value=names[current] if 0 <= current < len(names) else "")
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

            if p.path == "model_index":
                ttk.Button(
                    row, text="Load...", width=6,
                    command=self._on_load_model,
                ).pack(side="left", padx=(4, 0))

    def _build_material_widgets(self, container: ttk.Frame) -> None:
        """Per-material editor section. Rebuilt when the loaded model changes
        so the widget list always reflects the current scene's materials.
        """
        self._material_container = container
        if self._material_outer is not None:
            for p in self.params:
                if p.path.startswith("mtlx."):
                    self._widgets.pop(p.path, None)
            self._material_outer.destroy()
            self._material_outer = None

        scene = getattr(self.renderer, "_usd_scene", None)
        self._last_scene_id = id(scene)

        outer = ttk.LabelFrame(container, text="Materials", padding=4)
        outer.pack(fill="x", padx=2, pady=2)
        self._material_outer = outer

        has_skin = scene is not None and any(
            m.mtlx_target_name == "M_skinny_skin_default"
            for m in scene.materials
        )

        mtlx_params = [p for p in self.params if p.path.startswith("mtlx.")]
        if mtlx_params and has_skin:
            skin_section = _CollapsibleSection(
                outer, title="Skin Material", expanded=True,
            )
            skin_section.pack(fill="x", padx=2, pady=2)
            for p in mtlx_params:
                self._build_param_row(skin_section.body, p)
            furnace_var = tk.BooleanVar(value=False)
            ttk.Checkbutton(
                skin_section.body, text="Furnace", variable=furnace_var,
                command=lambda v=furnace_var: self.renderer.toggle_material_furnace(0, v.get()),
            ).pack(fill="x", pady=1)

        editable = (
            list(enumerate(scene.materials))[1:]
            if scene is not None and scene.materials else []
        )
        for mat_id, mat in editable:
            section = _CollapsibleSection(outer, title=mat.name, expanded=False)
            section.pack(fill="x", padx=2, pady=2)
            body = section.body

            self._build_mat_color_row(body, mat_id, mat)
            for key, lo, hi in (
                ("roughness",      0.04, 1.0),
                ("metallic",       0.0,  1.0),
                ("specular",       0.0,  1.0),
                ("opacity",        0.0,  1.0),
                ("ior",            1.0,  3.0),
                ("coat",           0.0,  1.0),
                ("coat_roughness", 0.0,  1.0),
            ):
                self._build_mat_slider_row(body, mat_id, mat, key, lo, hi)
            furnace_var = tk.BooleanVar(value=False)
            ttk.Checkbutton(
                body, text="Furnace", variable=furnace_var,
                command=lambda v=furnace_var, mid=mat_id: self.renderer.toggle_material_furnace(mid, v.get()),
            ).pack(fill="x", pady=1)

    def _build_mat_color_row(self, parent: ttk.Frame, mat_id: int, mat) -> None:
        # Hide the swatch when diffuseColor is texture-bound — the texture
        # drives shading there, the constant override is dead state.
        if "diffuseColor" in mat.texture_paths:
            return
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=1)
        ttk.Label(row, text="diffuseColor", width=14, anchor="w").pack(side="left")
        diff = mat.parameter_overrides.get("diffuseColor")
        r, g, b = self._color3_to_floats(diff)
        canvas = tk.Canvas(row, width=36, height=18, bd=1, relief="sunken",
                           highlightthickness=0)
        canvas.pack(side="left", padx=(0, 4))
        fill = _rgb_floats_to_hex(r, g, b)
        rect_id = canvas.create_rectangle(0, 0, 36, 18, fill=fill, outline="")
        ttk.Button(
            row, text="Pick...", width=8,
            command=lambda c=canvas, rid=rect_id, mid=mat_id, m=mat:
                self._on_pick_material_color(c, rid, mid, m),
        ).pack(side="left")

    def _build_mat_slider_row(
        self, parent: ttk.Frame, mat_id: int, mat,
        key: str, lo: float, hi: float,
    ) -> None:
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=1)
        ttk.Label(row, text=key, width=14, anchor="w").pack(side="left")
        current = mat.parameter_overrides.get(key)
        try:
            val = float(current) if current is not None else 0.5
        except (TypeError, ValueError):
            val = 0.5
        var = tk.DoubleVar(value=val)
        scale = ttk.Scale(
            row, from_=lo, to=hi, variable=var, orient="horizontal",
            command=lambda v, mid=mat_id, k=key, lbl_ref=[None]:
                self._on_material_slider(mid, k, float(v), lbl_ref[0]),
        )
        scale.pack(side="left", fill="x", expand=True, padx=(0, 4))
        val_lbl = ttk.Label(row, width=7, anchor="e", text=f"{val:.3f}")
        val_lbl.pack(side="left")
        # Re-bind so the lambda's closure captures the real label after
        # creation (lambda default-arg trick).
        scale.configure(
            command=lambda v, mid=mat_id, k=key, lbl=val_lbl:
                self._on_material_slider(mid, k, float(v), lbl),
        )

    def _on_material_slider(
        self, mat_id: int, key: str, value: float, lbl: ttk.Label | None
    ) -> None:
        if self._suppress_cb:
            return
        self.renderer.apply_material_override(mat_id, key, float(value))
        if lbl is not None:
            try:
                lbl.configure(text=f"{value:.3f}")
            except tk.TclError:
                pass

    def _on_pick_material_color(
        self, canvas: tk.Canvas, rect_id: int, mat_id: int, mat,
    ) -> None:
        diff = mat.parameter_overrides.get("diffuseColor")
        r, g, b = self._color3_to_floats(diff)
        init = (
            max(0, min(255, int(round(r * 255)))),
            max(0, min(255, int(round(g * 255)))),
            max(0, min(255, int(round(b * 255)))),
        )
        result = colorchooser.askcolor(
            color="#%02x%02x%02x" % init, title=f"{mat.name} diffuseColor"
        )
        if result is None or result[0] is None:
            return
        rr, gg, bb = result[0]
        rf, gf, bf = rr / 255.0, gg / 255.0, bb / 255.0
        # Store as a 3-tuple — pack_flat_material's _override_to_color3
        # path accepts tuple/list/numpy/Color3 alike.
        self.renderer.apply_material_override(
            mat_id, "diffuseColor", (rf, gf, bf)
        )
        try:
            canvas.itemconfig(rect_id, fill=_rgb_floats_to_hex(rf, gf, bf))
        except tk.TclError:
            pass

    @staticmethod
    def _color3_to_floats(value) -> tuple[float, float, float]:
        if value is None:
            return 0.72, 0.72, 0.72
        if hasattr(value, "asTuple"):
            seq = value.asTuple()
        elif hasattr(value, "__getitem__") and not isinstance(value, str):
            try:
                seq = (value[0], value[1], value[2])
            except (IndexError, TypeError):
                return 0.72, 0.72, 0.72
        else:
            return 0.72, 0.72, 0.72
        return float(seq[0]), float(seq[1]), float(seq[2])

    def _build_color_row(self, container: ttk.Frame) -> None:
        row = ttk.Frame(container)
        row.pack(fill="x", pady=1)
        ttk.Label(row, text="Color", width=self._LABEL_WIDTH, anchor="w").pack(side="left")

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
        ttk.Label(row, text="Direction", width=self._LABEL_WIDTH, anchor="w").pack(side="left")

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
        from skinny.params import _set_nested

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
        from skinny.params import _set_nested

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

    def _on_open_scene_graph(self) -> None:
        if self._scene_graph_window is not None and self._scene_graph_window.is_open():
            self._scene_graph_window.focus()
            return
        from skinny.scene_graph_window import SceneGraphWindow
        if self.renderer.scene_graph is None:
            messagebox.showinfo(
                "Scene Graph",
                "No scene graph available.\nLoad a USD scene first.",
            )
            return
        self._scene_graph_window = SceneGraphWindow(self)

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
        name = simpledialog.askstring(
            "Save preset", "Preset name:", parent=self.root
        )
        if not name or not name.strip():
            return
        name = name.strip()

        values: dict[str, float] = {}
        for k, v in getattr(self.renderer, "mtlx_overrides", {}).items():
            if isinstance(v, (int, float)):
                values[f"mtlx.{k}"] = float(v)
            elif isinstance(v, (list, tuple)):
                for i, comp in enumerate(v):
                    if isinstance(comp, (int, float)):
                        values[f"mtlx.{k}.{i}"] = float(comp)

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
        from skinny.params import _get_nested

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
                    names = [self._choice_label(c) for c in choices]
                    combo = _widget
                    if list(combo["values"]) != names:
                        combo.configure(values=names)
                    idx = int(cur)
                    if 0 <= idx < len(choices):
                        name = self._choice_label(choices[idx])
                        if var.get() != name:
                            var.set(name)
        finally:
            self._suppress_cb = False

        cur_scene = getattr(self.renderer, "_usd_scene", None)
        if id(cur_scene) != self._last_scene_id:
            self._build_material_widgets(self._material_container)

        self._refresh_color_swatch()
        self._refresh_direction_preview()
        self._update_delete_btn_state()

        if self._scene_graph_window is not None:
            if self._scene_graph_window.is_open():
                self._scene_graph_window.tick()
            else:
                self._scene_graph_window = None

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


# ── Collapsible section ─────────────────────────────────────────────


class _CollapsibleSection(ttk.Frame):
    """Header bar with a disclosure triangle that hides/shows a body frame."""

    def __init__(self, parent, title: str, expanded: bool = False) -> None:
        super().__init__(parent)
        self._expanded = expanded
        self._title = title

        self._header = ttk.Label(
            self, text=self._header_text(), anchor="w", cursor="hand2",
        )
        self._header.pack(fill="x")
        self._header.bind("<Button-1>", lambda _e: self.toggle())

        self.body = ttk.Frame(self, padding=(8, 2, 2, 4))
        if expanded:
            self.body.pack(fill="x")

    def _header_text(self) -> str:
        marker = "▼" if self._expanded else "▶"
        return f"{marker} {self._title}"

    def toggle(self) -> None:
        self._expanded = not self._expanded
        self._header.configure(text=self._header_text())
        if self._expanded:
            self.body.pack(fill="x")
        else:
            self.body.pack_forget()
