"""BXDF visualizer — separate Tk window for inspecting per-material lobes.

Workflow:

1. User clicks "Pick" → next click in the main viewport captures the
   shading frame (P, N, T, UV, materialId) via the GPU's main_pass
   piggyback path (renderer.request_scene_pick).
2. User picks a material via the per-material button row. The active
   material's `parameter_overrides` drive the BRDF evaluation.
3. The 3D lobe is generated CPU-side using a Lambert + GGX-Smith
   standard_surface approximation and rendered to a PIL image displayed
   in a Tk Canvas. The user drags on the canvas to orbit the camera.

First cut deliberately CPU-only (uses constant `parameter_overrides`,
not MaterialX graph procedurals). Phase 2 will hook a dedicated GPU
compute path so marble / wood / etc. evaluate identically to the main
render. See `~/.claude/plans/i-would-like-to-lexical-kettle.md`.
"""

from __future__ import annotations

import math
import tkinter as tk
from tkinter import ttk
from typing import Optional

import numpy as np

from skinny.renderer import _hashable_value

try:
    from PIL import Image, ImageTk
except ImportError as _exc:  # pragma: no cover — pillow is a hard dep
    raise RuntimeError(
        "BXDF visualizer requires Pillow; install via `pip install Pillow`."
    ) from _exc


# ── BRDF evaluation (CPU, standard_surface diffuse + GGX-Smith) ────


def _ggx_smith_g1(n_dot_v: float, alpha: float) -> float:
    k = alpha * alpha / 2.0
    return n_dot_v / max(n_dot_v * (1.0 - k) + k, 1e-6)


def _ggx_d(n_dot_h: float, alpha: float) -> float:
    a2 = alpha * alpha
    denom = (n_dot_h * n_dot_h) * (a2 - 1.0) + 1.0
    return a2 / max(math.pi * denom * denom, 1e-8)


def _fresnel_schlick(cos_theta: float, F0: np.ndarray) -> np.ndarray:
    return F0 + (1.0 - F0) * (1.0 - cos_theta) ** 5


def eval_std_surface(
    wi: np.ndarray, wo: np.ndarray, params: dict[str, object],
) -> np.ndarray:
    """Lambert + GGX-Smith reflectance for one (wi, wo) pair in tangent space.

    Inputs are tangent-space directions where +Z is the shading normal. Returns
    RGB reflectance f(wi, wo) (no cosine factor). Matches the diffuse +
    specular terms of `mtlx_std_surface.slang::evalStdSurfaceBSDF` for the
    common opaque case (no coat, no sheen, no transmission).
    """
    n = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    wi = wi.astype(np.float64)
    wo = wo.astype(np.float64)
    n_dot_wi = float(np.dot(n, wi))
    n_dot_wo = float(np.dot(n, wo))
    if n_dot_wi <= 0.0 or n_dot_wo <= 0.0:
        return np.zeros(3, dtype=np.float64)

    base_color = np.array(params.get("base_color", (0.8, 0.8, 0.8)), dtype=np.float64)
    metalness = float(params.get("metalness", params.get("metallic", 0.0)))
    specular = float(params.get("specular", 1.0))
    roughness = float(params.get("specular_roughness", params.get("roughness", 0.5)))
    ior = float(params.get("specular_IOR", params.get("ior", 1.5)))
    base = float(params.get("base", 1.0))

    alpha = max(roughness * roughness, 1e-3)

    # Diffuse Lambertian lobe. The (1 - metallic) factor matches
    # standard_surface's metalness coupling.
    f_d = (base * base_color / math.pi) * (1.0 - metalness)

    # GGX-Smith specular lobe. F0 blends from dielectric (ior-derived) to
    # tinted-metal (base_color) by `metalness`.
    f0_diel_scalar = ((ior - 1.0) / (ior + 1.0)) ** 2
    f0_diel = specular * np.array([f0_diel_scalar] * 3, dtype=np.float64)
    F0 = f0_diel * (1.0 - metalness) + base_color * metalness

    h = wi + wo
    h_len = float(np.linalg.norm(h))
    if h_len < 1e-6:
        return f_d * n_dot_wi
    h = h / h_len
    n_dot_h = max(float(np.dot(n, h)), 0.0)
    v_dot_h = max(float(np.dot(wo, h)), 0.0)

    D = _ggx_d(n_dot_h, alpha)
    G = _ggx_smith_g1(n_dot_wi, alpha) * _ggx_smith_g1(n_dot_wo, alpha)
    F = _fresnel_schlick(v_dot_h, F0)
    f_s = (D * G * F) / max(4.0 * n_dot_wi * n_dot_wo, 1e-6)

    # MaterialX convention is response = f * cos(theta_i). Matches what the
    # path tracer's `BSDFEval.response` returns to NEE.
    return (f_d + f_s) * n_dot_wi


def eval_grid(
    locked_dir: np.ndarray,
    lock_mode: int,
    n_theta: int,
    n_phi: int,
    params: dict[str, object],
) -> tuple[np.ndarray, np.ndarray]:
    """Sample the lobe over a (theta, phi) grid on the upper hemisphere.

    Returns:
        dirs:  (n_theta, n_phi, 3) tangent-space directions of the swept
               axis. The locked axis sits at `locked_dir`.
        f:     (n_theta, n_phi, 3) RGB reflectance per direction. Hemisphere
               only — lower hemisphere is implicit zero in the Lambert /
               GGX standard_surface lobe used here.
    """
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

    f = np.zeros((n_theta, n_phi, 3), dtype=np.float64)
    for i in range(n_theta):
        for j in range(n_phi):
            d = dirs[i, j]
            if lock_mode == 0:  # lock wi, sweep wo
                f[i, j] = eval_std_surface(locked_dir, d, params)
            else:  # lock wo, sweep wi
                f[i, j] = eval_std_surface(d, locked_dir, params)
    return dirs, f


# ── Lobe geometry → 2D projection → PIL image ──────────────────────


def _euler_to_rot(yaw: float, pitch: float) -> np.ndarray:
    """Camera orbit rotation. Yaw around world +Z (tangent normal), pitch
    tilts the view up/down toward the equator."""
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]], dtype=np.float64)
    return Rx @ Rz


def render_lobe_image(
    dirs: np.ndarray,
    f: np.ndarray,
    yaw: float,
    pitch: float,
    size: int = 480,
    log_scale: bool = True,
    zoom: float = 1.0,
) -> Image.Image:
    """Rasterise the lobe to a `size × size` PIL image.

    Vertex radius = baseline + magnitude. Even a zero-magnitude BSDF
    still renders a visible hemisphere outline so the user can tell the
    eval ran. Per-quad fill colour encodes magnitude (heatmap). Quad
    strip from the grid is z-sorted (painter's algorithm) and drawn as
    filled polygons. Pure numpy + Pillow.
    """
    # Sanitise: NaN / Inf collapses everything; replace with zero.
    f = np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)
    f = np.maximum(f, 0.0)
    nt, npx = dirs.shape[:2]
    lum = 0.2126 * f[..., 0] + 0.7152 * f[..., 1] + 0.0722 * f[..., 2]
    lum_max = max(float(lum.max()), 1e-6)
    # `lum_norm` ∈ [0, 1] drives both the radius bulge and the heatmap
    # colour intensity. Log scale compresses the high-end so glancing-
    # angle lobes and dim BSSRDF responses stay visible alongside a
    # bright GGX peak.
    if log_scale:
        # log(1 + k*x) / log(1 + k*max) with k=20: gentle compression
        # that maps 0→0, max→1, with the lower decade boosted.
        K = 20.0
        lum_norm = np.log1p(K * lum) / float(np.log1p(K * lum_max))
        color_lum = np.log1p(K * f) / float(np.log1p(K * max(f.max(), 1e-6)))
    else:
        lum_norm = lum / lum_max
        color_lum = f / max(float(f.max()), 1e-6)
    lum_norm = np.clip(lum_norm, 0.0, 1.0)
    color_lum = np.clip(color_lum, 0.0, 1.0)

    # Radius = baseline hemisphere + magnitude bulge. Baseline keeps the
    # lobe visible even for zero / tiny BRDFs; magnitude lets the user
    # see the lobe shape.
    BASE_R = 0.20
    BULGE_R = 0.80
    radius = BASE_R + BULGE_R * lum_norm
    verts = dirs * radius[..., None]

    # Append a north-pole row at theta=0 so the cap closes — otherwise
    # the grid leaves a visible disk hole at the top of the hemisphere.
    pole_r = float(radius[0].max())
    pole = np.tile([0.0, 0.0, pole_r], (npx, 1))
    verts_ext = np.concatenate(
        [pole[None, :, :], verts], axis=0,
    )  # (nt+1, np, 3)

    R = _euler_to_rot(yaw, pitch)
    cam = verts_ext @ R.T  # (rows, cols, 3)

    # Orthographic projection: pixel = center + scale * (cam.x, -cam.z).
    # Y depth (cam.y) is the painter's-algorithm sort key. `zoom` scales
    # the projection — mouse wheel on the canvas drives it.
    half = size * 0.5
    scale = half * 0.78 * max(zoom, 0.05)
    px = half + scale * cam[..., 0]
    py = half - scale * cam[..., 2]

    img = Image.new("RGB", (size, size), (18, 18, 26))
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    # Unit-sphere guide circle + axes.
    cx = cy = half
    guide_r = scale * 1.0
    draw.ellipse([cx - guide_r, cy - guide_r, cx + guide_r, cy + guide_r],
                 outline=(60, 60, 80))
    draw.line([cx - guide_r, cy, cx + guide_r, cy], fill=(55, 55, 75))
    draw.line([cx, cy - guide_r, cx, cy + guide_r], fill=(55, 55, 75))

    rows = nt + 1
    cols = npx
    # Quad list with depth keys. Average y over four corners.
    quads: list[tuple[float, tuple, tuple]] = []
    for i in range(rows - 1):
        for j in range(cols):
            j2 = (j + 1) % cols
            v00 = cam[i, j]
            v01 = cam[i, j2]
            v10 = cam[i + 1, j]
            v11 = cam[i + 1, j2]
            depth = 0.25 * (v00[1] + v01[1] + v10[1] + v11[1])
            poly = (
                (px[i, j], py[i, j]),
                (px[i, j2], py[i, j2]),
                (px[i + 1, j2], py[i + 1, j2]),
                (px[i + 1, j], py[i + 1, j]),
            )
            i_lobe = max(i - 1, 0)
            # Heatmap: dim base for low magnitude, full colour at peak.
            # log_scale path uses lum_norm / color_lum which already
            # carry the log compression.
            m = float(lum_norm[i_lobe, j])
            color = color_lum[i_lobe, j]
            rgb = (
                int(round(40 + 215 * color[0] * max(m, 0.25))),
                int(round(40 + 215 * color[1] * max(m, 0.25))),
                int(round(40 + 215 * color[2] * max(m, 0.25))),
            )
            quads.append((depth, poly, rgb))

    quads.sort(key=lambda q: q[0])
    for _, poly, rgb in quads:
        draw.polygon(poly, fill=rgb, outline=(110, 110, 130))

    return img


# ── Visualizer Tk Toplevel ─────────────────────────────────────────


class BXDFVisualizer:
    """Non-modal Tk Toplevel hosting the lobe canvas + material picker."""

    LOBE_SIZE = 480

    def __init__(self, panel) -> None:
        self.panel = panel
        self.renderer = panel.renderer
        self._alive = True

        top = tk.Toplevel(panel.root)
        top.title("BXDF Visualizer")
        top.geometry("840x680")
        top.protocol("WM_DELETE_WINDOW", self.close)
        self.top = top

        # Active material is whatever the most recent pick captured.
        # No explicit picker — the user picks a point in the viewport
        # and the visualizer evaluates that point's material.
        self._material_id: int = -1
        self._material_label = ttk.Label(top, text="Material: (no pick)")
        self._material_label.pack(fill="x", padx=8, pady=(8, 0))

        # ── Mode toggle (BXDF vs BSSRDF) ─────────────────────
        mode_frame = ttk.LabelFrame(top, text="Mode")
        mode_frame.pack(fill="x", padx=8, pady=(0, 4))
        self._mode_var = tk.StringVar(value="bxdf")
        ttk.Radiobutton(
            mode_frame, text="BXDF (surface lobe)",
            variable=self._mode_var, value="bxdf",
            command=self._on_mode_changed,
        ).pack(side="left", padx=4, pady=2)
        ttk.Radiobutton(
            mode_frame, text="BSSRDF (skin diffusion)",
            variable=self._mode_var, value="bssrdf",
            command=self._on_mode_changed,
        ).pack(side="left", padx=4, pady=2)

        # ── Pick controls ─────────────────────────────────────
        pick_frame = ttk.LabelFrame(top, text="Scene point")
        pick_frame.pack(fill="x", padx=8, pady=4)
        self._pick_btn = ttk.Button(
            pick_frame, text="Pick exit point", command=self._on_pick_click,
        )
        self._pick_btn.pack(side="left", padx=4, pady=4)
        self._pick_readout = ttk.Label(pick_frame, text="No pick yet.")
        self._pick_readout.pack(side="left", padx=8)
        self._pick_state: Optional[dict] = None

        # Entrance-point pick row (only relevant in BSSRDF mode).
        ent_frame = ttk.Frame(pick_frame)
        ent_frame.pack(side="right")
        self._entrance_btn = ttk.Button(
            ent_frame, text="Pick entrance",
            command=self._on_entrance_pick_click,
            state="disabled",
        )
        self._entrance_btn.pack(side="left", padx=4)
        self._entrance_readout = ttk.Label(ent_frame, text="(BSSRDF only)")
        self._entrance_readout.pack(side="left", padx=4)
        self._entrance_state: Optional[dict] = None

        # ── Direction + lock controls ─────────────────────────
        dir_frame = ttk.LabelFrame(top, text="Directions")
        dir_frame.pack(fill="x", padx=8, pady=4)
        self._lock_var = tk.IntVar(value=0)
        ttk.Radiobutton(
            dir_frame, text="Lock wi (sweep wo)",
            variable=self._lock_var, value=0,
            command=self._redraw,
        ).grid(row=0, column=0, sticky="w", padx=4)
        ttk.Radiobutton(
            dir_frame, text="Lock wo (sweep wi)",
            variable=self._lock_var, value=1,
            command=self._redraw,
        ).grid(row=0, column=1, sticky="w", padx=4)
        # Log-scale toggle. Compresses the BSDF magnitude with
        # log(1 + k·f) so a bright GGX peak doesn't dwarf dim grazing-
        # angle and BSSRDF responses. On by default — linear is rarely
        # readable when both a specular peak and a diffuse base exist.
        self._log_var = tk.IntVar(value=1)
        ttk.Checkbutton(
            dir_frame, text="Log scale",
            variable=self._log_var,
            command=self._on_log_toggled,
        ).grid(row=0, column=2, sticky="w", padx=4)
        ttk.Label(dir_frame, text="theta:").grid(row=1, column=0, sticky="e")
        self._theta_var = tk.DoubleVar(value=30.0)
        ttk.Scale(
            dir_frame, from_=0.0, to=89.0, orient="horizontal",
            variable=self._theta_var, command=lambda _v: self._redraw(),
            length=240,
        ).grid(row=1, column=1, columnspan=2, sticky="we", padx=4)
        ttk.Label(dir_frame, text="phi:").grid(row=2, column=0, sticky="e")
        self._phi_var = tk.DoubleVar(value=0.0)
        ttk.Scale(
            dir_frame, from_=0.0, to=359.0, orient="horizontal",
            variable=self._phi_var, command=lambda _v: self._redraw(),
            length=240,
        ).grid(row=2, column=1, columnspan=2, sticky="we", padx=4)

        # ── Lobe canvas ───────────────────────────────────────
        lobe_frame = ttk.LabelFrame(top, text="Lobe")
        lobe_frame.pack(fill="both", expand=True, padx=8, pady=4)
        self._canvas = tk.Canvas(
            lobe_frame, width=self.LOBE_SIZE, height=self.LOBE_SIZE,
            bg="black", highlightthickness=0,
        )
        self._canvas.pack(padx=4, pady=4)
        self._photo: Optional[ImageTk.PhotoImage] = None
        self._image_item = self._canvas.create_image(
            self.LOBE_SIZE // 2, self.LOBE_SIZE // 2,
        )
        self._yaw = math.radians(35.0)
        self._pitch = math.radians(20.0)
        self._zoom = 1.0
        self._last_drag: Optional[tuple[float, float]] = None
        self._canvas.bind("<ButtonPress-1>", self._on_drag_press)
        self._canvas.bind("<B1-Motion>", self._on_drag_move)
        self._canvas.bind("<ButtonRelease-1>", lambda _e: self._on_drag_release())
        # Mouse wheel: Windows / macOS use <MouseWheel> with event.delta
        # (multiples of 120); Linux X11 uses <Button-4> / <Button-5>.
        # Bind both so the same handler runs on every platform.
        self._canvas.bind("<MouseWheel>", self._on_wheel)
        self._canvas.bind("<Button-4>", lambda _e: self._zoom_step(+1))
        self._canvas.bind("<Button-5>", lambda _e: self._zoom_step(-1))

        # ── Bottom status bar ─────────────────────────────────
        self._status = ttk.Label(top, text="Pick a material and a scene point.")
        self._status.pack(fill="x", padx=8, pady=(0, 8))

        # Cached last grid + dirs so mouse orbit only re-rasters the
        # PIL image, not a full GPU eval. `_needs_eval` triggers a new
        # GPU dispatch on the next tick; `_needs_render` triggers only a
        # rotate-and-paint cycle from the cached grid.
        self._cached_dirs: Optional[np.ndarray] = None
        self._cached_f: Optional[np.ndarray] = None
        self._cached_gpu: bool = False
        self._needs_eval = False
        self._needs_render = False
        # Coalesce rapid slider updates into a single GPU eval after a
        # short idle window.
        self._eval_after_id: Optional[str] = None
        # Track renderer-side material state hash so control-panel
        # slider drags re-eval the lobe. Covers both per-material
        # `apply_material_override` (bumps `_material_version`) and
        # skin / MaterialX overrides (mutate `mtlx_overrides`).
        self._last_material_hash: int = self._compute_material_hash()

    # ── Scene material helpers ───────────────────────────────

    def _scene_materials(self) -> list:
        scene = getattr(self.renderer, "scene", None)
        if scene is None:
            return []
        return list(getattr(scene, "materials", []) or [])

    def _set_active_material(self, mat_id: int) -> None:
        self._material_id = mat_id
        mats = self._scene_materials()
        if 0 <= mat_id < len(mats):
            name = getattr(mats[mat_id], "name", "?")
            self._material_label.configure(text=f"Material: #{mat_id} — {name}")
        else:
            # Scene.materials may be empty (no USD scene) or sparsely
            # populated relative to the GPU-side material table. The GPU
            # eval indexes the materialTypes SSBO directly so it doesn't
            # care; just show the raw id.
            self._material_label.configure(text=f"Material: #{mat_id}")

    # ── Pick flow ────────────────────────────────────────────

    def _compute_material_hash(self) -> int:
        """Coarse hash of renderer material state.

        Combines the per-material version counter (bumped by
        `apply_material_override`) with the global MaterialX overrides
        dict (skin sliders mutate this). Camera / light / accumulation
        are intentionally excluded — those don't change the BSDF.
        """
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

    def _on_log_toggled(self) -> None:
        # Log scale only affects PIL rasterisation; no GPU re-eval needed.
        self._needs_render = True

    def _on_mode_changed(self) -> None:
        bssrdf = self._mode_var.get() == "bssrdf"
        self._entrance_btn.configure(state="normal" if bssrdf else "disabled")
        if not bssrdf:
            self._entrance_readout.configure(text="(BSSRDF only)")
        else:
            if self._entrance_state is None:
                self._entrance_readout.configure(text="Entrance not picked.")
        self._needs_eval = True

    def _on_pick_click(self) -> None:
        input_handler = getattr(self.panel, "_input_handler", None)
        if input_handler is None:
            self._status.configure(
                text="Pick unavailable: app input handler not bound."
            )
            return
        input_handler.arm_bxdf_pick(self._on_pick_result)
        self._status.configure(
            text="Click the main viewport to pick exit point."
        )

    def _on_entrance_pick_click(self) -> None:
        input_handler = getattr(self.panel, "_input_handler", None)
        if input_handler is None:
            self._status.configure(
                text="Pick unavailable: app input handler not bound."
            )
            return
        input_handler.arm_bxdf_pick(self._on_entrance_pick_result)
        self._status.configure(
            text="Click the main viewport to pick BSSRDF entrance."
        )

    def _on_entrance_pick_result(self, result: Optional[dict]) -> None:
        if not self._alive:
            return
        if result is None:
            self._entrance_readout.configure(text="Entrance pick missed.")
            return
        self._entrance_state = result
        p = result["position"]
        self._entrance_readout.configure(
            text=f"xi=({p[0]:+.2f}, {p[1]:+.2f}, {p[2]:+.2f})"
        )
        self._needs_eval = True

    def _on_pick_result(self, result: Optional[dict]) -> None:
        if not self._alive:
            return
        if result is None:
            self._pick_readout.configure(text="Pick missed: no scene hit.")
            self._status.configure(text="Pick missed. Try again.")
            return
        self._pick_state = result
        p = result["position"]
        n = result["normal"]
        mat_id = result["material_id"]
        self._pick_readout.configure(
            text=(
                f"matId={mat_id}  P=({p[0]:+.2f}, {p[1]:+.2f}, {p[2]:+.2f})"
                f"  N=({n[0]:+.2f}, {n[1]:+.2f}, {n[2]:+.2f})"
            )
        )
        self._set_active_material(mat_id)
        self._status.configure(text="Scene point captured. Evaluating BXDF…")
        self._needs_eval = True

    # ── Lobe drawing ─────────────────────────────────────────

    def _locked_dir(self) -> np.ndarray:
        theta = math.radians(float(self._theta_var.get()))
        phi = math.radians(float(self._phi_var.get()))
        return np.array(
            [math.sin(theta) * math.cos(phi),
             math.sin(theta) * math.sin(phi),
             math.cos(theta)],
            dtype=np.float64,
        )

    def _active_params(self) -> Optional[dict]:
        idx = self._material_id
        mats = self._scene_materials()
        if not (0 <= idx < len(mats)):
            return None
        mat = mats[idx]
        return dict(getattr(mat, "parameter_overrides", {}) or {})

    def _redraw(self) -> None:
        # Slider / mode / radio change → schedule a GPU eval. Debounced
        # so rapid drags coalesce into one dispatch.
        self._schedule_eval()

    def _schedule_eval(self) -> None:
        try:
            if self._eval_after_id is not None:
                self.top.after_cancel(self._eval_after_id)
        except tk.TclError:
            pass
        try:
            self._eval_after_id = self.top.after(120, self._fire_eval)
        except tk.TclError:
            self._eval_after_id = None

    def _fire_eval(self) -> None:
        self._eval_after_id = None
        self._needs_eval = True

    def _do_eval(self) -> None:
        n_theta, n_phi = 24, 48
        mode = self._mode_var.get()

        if self._pick_state is None:
            self._status.configure(text="Pick a scene point to evaluate a material.")
            return

        idx = self._material_id
        if idx < 0:
            self._status.configure(text="Pick captured no valid material id.")
            return

        if mode == "bssrdf" and self._entrance_state is None:
            self._status.configure(
                text="BSSRDF mode: pick an entrance point first."
            )
            return

        # GPU path. Doesn't depend on `Scene.materials` — the shader
        # reads the materialTypes / stdSurfaceParams / graph SSBOs
        # directly by id so an empty or sparse CPU material list is
        # fine.
        req = {
            "material_id": idx,
            "position": self._pick_state["position"],
            "normal": self._pick_state["normal"],
            "tangent": self._pick_state["tangent"],
            "uv": self._pick_state["uv"],
            "locked_dir": self._locked_dir(),
            "lock_mode": int(self._lock_var.get()),
            "n_theta": n_theta,
            "n_phi": n_phi,
        }
        self._pending_dirs = self._make_dirs_grid(n_theta, n_phi)
        self._pending_n_theta = n_theta
        self._pending_n_phi = n_phi
        try:
            if mode == "bssrdf":
                req["entrance_position"] = self._entrance_state["position"]
                self.renderer.request_bssrdf_eval(req, self._on_gpu_eval_result)
            else:
                self.renderer.request_bxdf_eval(req, self._on_gpu_eval_result)
            return
        except Exception as exc:
            print(f"[skinny] GPU eval failed: {exc}")
            self._status.configure(text=f"GPU eval failed: {exc}")

        # CPU fallback only when the scene.materials list has the picked
        # id AND the GPU dispatch raised. Analytic Lambert+GGX (no
        # MaterialX graphs).
        params = self._active_params()
        if params is None:
            return
        dirs, f = eval_grid(
            self._locked_dir(), int(self._lock_var.get()),
            n_theta, n_phi, params,
        )
        self._cache_and_render(dirs, f, gpu=False)

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

    def _on_gpu_eval_result(self, grid: np.ndarray) -> None:
        if not self._alive or not hasattr(self, "_pending_dirs"):
            return
        dirs = self._pending_dirs
        f = grid.astype(np.float64)
        # Diagnostic so the user can see whether the GPU dispatch
        # produced any response. A flat-zero grid points to a degenerate
        # shading frame, wrong materialId, or a graph that returned
        # zero base_color.
        print(
            f"[bxdf-vis] GPU eval result: shape={f.shape} "
            f"min={float(f.min()):.4f} max={float(f.max()):.4f} "
            f"mean={float(f.mean()):.4f}",
            flush=True,
        )
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
        log_scale = bool(self._log_var.get())
        img = render_lobe_image(
            self._cached_dirs, self._cached_f,
            self._yaw, self._pitch, size=self.LOBE_SIZE,
            log_scale=log_scale, zoom=self._zoom,
        )
        self._photo = ImageTk.PhotoImage(img)
        try:
            self._canvas.itemconfig(self._image_item, image=self._photo)
        except tk.TclError:
            return
        max_f = float(self._cached_f.max())
        idx = self._material_id
        mats = self._scene_materials()
        name = mats[idx].name if 0 <= idx < len(mats) else "?"
        src = "GPU" if self._cached_gpu else "CPU"
        scale = "log" if bool(self._log_var.get()) else "lin"
        self._status.configure(
            text=(
                f"Material #{idx} ({name})  max f·cosθ = {max_f:.3f}  "
                f"[{src} | {scale}]"
            )
        )

    # ── Mouse orbit ──────────────────────────────────────────

    def _on_drag_press(self, event) -> None:
        self._last_drag = (event.x, event.y)

    def _on_drag_move(self, event) -> None:
        if self._last_drag is None:
            return
        dx = event.x - self._last_drag[0]
        dy = event.y - self._last_drag[1]
        self._last_drag = (event.x, event.y)
        self._yaw += dx * 0.012
        self._pitch = max(
            -math.pi * 0.49,
            min(math.pi * 0.49, self._pitch + dy * 0.012),
        )
        # Orbit is local: just re-rasterise the cached grid, no GPU eval.
        self._needs_render = True

    def _on_drag_release(self) -> None:
        self._last_drag = None

    def _on_wheel(self, event) -> None:
        # event.delta is +/- 120 per notch on Windows / macOS.
        steps = int(event.delta / 120) if event.delta else 0
        if steps == 0 and event.delta:
            steps = 1 if event.delta > 0 else -1
        if steps:
            self._zoom_step(steps)

    def _zoom_step(self, steps: int) -> None:
        # Geometric zoom: each notch = 1.15× in or out. Clamp so the
        # lobe stays inside the canvas and a fully-zoomed-out hemisphere
        # is still visible.
        factor = 1.15 ** steps
        self._zoom = max(0.1, min(8.0, self._zoom * factor))
        self._needs_render = True

    # ── Lifecycle ────────────────────────────────────────────

    def tick(self) -> None:
        if not self._alive:
            return
        # Watch renderer material state; control-panel sliders bump
        # `_material_version` or mutate `mtlx_overrides`. Debounce so a
        # drag fires one eval after the slider settles, not per step.
        h = self._compute_material_hash()
        if h != self._last_material_hash:
            self._last_material_hash = h
            if self._pick_state is not None:
                self._schedule_eval()
        if self._needs_eval:
            self._needs_eval = False
            self._do_eval()
        elif self._needs_render:
            self._needs_render = False
            self._do_render()
        try:
            self.top.update_idletasks()
        except tk.TclError:
            self._alive = False

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
