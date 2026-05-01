"""Application entry point — creates a GLFW window with Vulkan surface and runs the render loop."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import glfw

from skinny.vk_context import VulkanContext
from skinny.renderer import Renderer
from skinny.settings import ensure_dirs, load_settings, save_settings

WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720


@dataclass
class ParamSpec:
    name: str
    path: str
    kind: str  # "continuous" or "discrete"
    step: float = 0.0
    lo: float = 0.0
    hi: float = 0.0
    choice_source: str | None = None  # for discrete: attribute on renderer


def _cont(name: str, path: str, step: float, lo: float, hi: float) -> ParamSpec:
    return ParamSpec(name, path, "continuous", step, lo, hi)


def _disc(name: str, path: str, choice_source: str) -> ParamSpec:
    return ParamSpec(name, path, "discrete", choice_source=choice_source)


def build_dynamic_params(renderer) -> list[ParamSpec]:
    """Return one ParamSpec per editable MaterialX uniform on the active
    skin material, excluding fields already covered by STATIC_PARAMS.
    Empty when the runtime hasn't loaded a material.
    """
    cm = getattr(renderer, "_mtlx_skin_material", None)
    if cm is None or not getattr(cm, "uniform_block", None):
        return []
    static_paths = {p.path for p in STATIC_PARAMS}
    ganged_targets = set()
    for targets in _GANGED_MTLX_FIELDS.values():
        for t in targets:
            ganged_targets.add(t)
    from skinny.materialx_runtime import ui_specs_from_uniform_block
    out: list[ParamSpec] = []
    for spec in ui_specs_from_uniform_block(cm.uniform_block):
        if spec["path"] in static_paths:
            continue
        field_name = spec["path"].split(".")[1] if "." in spec["path"] else ""
        if field_name in ganged_targets:
            continue
        out.append(ParamSpec(
            name=spec["name"], path=spec["path"], kind=spec["kind"],
            step=spec["step"], lo=spec["lo"], hi=spec["hi"],
        ))
    return out


def build_all_params(renderer) -> list[ParamSpec]:
    """Concatenate the static base + material-driven dynamic params."""
    return STATIC_PARAMS + build_dynamic_params(renderer)


# Discrete first so Preset / Environment show up at the top.
STATIC_PARAMS: list[ParamSpec] = [
    _disc("Preset",            "preset_index",                "presets"),
    _disc("Environment",       "env_index",                   "environments"),
    _cont("IBL intensity",     "env_intensity",               0.05, 0.0,  3.0),
    _cont("mm per unit",       "mm_per_unit",                 5.0,  1.0,  500.0),
    _disc("Direct light",      "direct_light_index",          "direct_light_modes"),
    _disc("Scattering",        "scatter_index",               "scatter_modes"),
    _disc("Sampling",          "integrator_index",            "integrator_modes"),
    _disc("Furnace mode",      "furnace_index",               "furnace_modes"),
    _disc("Head model",        "head_index",                  "head_models"),
    _disc("Detail maps",       "detail_maps_index",           "detail_maps_modes"),
    _cont("Normal map strength", "normal_map_strength",       0.05, 0.0,  2.0),
    _disc("Subdivision",       "subdivision_index",           "subdivision_modes"),
    _cont("Displacement (mm)", "displacement_scale_mm",       0.05, 0.0,  2.0),
    _disc("Tattoo",            "tattoo_index",                "tattoos"),
    _cont("Tattoo density",    "tattoo_density",              0.05, 0.0,  1.0),

    _cont("Melanin",            "mtlx.layer_top_melanin",            0.01, 0.0,  1.0),
    _cont("Hemoglobin",         "mtlx.layer_middle_hemoglobin",      0.01, 0.0,  1.0),
    _cont("Blood oxygenation",  "mtlx.layer_middle_blood_oxygenation", 0.05, 0.0, 1.0),
    _cont("Epidermis thickness", "mtlx.layer_top_thickness",         0.02, 0.01, 1.0),
    _cont("Dermis thickness",   "mtlx.layer_middle_thickness",       0.1,  0.1,  5.0),
    _cont("Subcut thickness",   "mtlx.layer_bottom_thickness",       0.2,  0.5,  10.0),
    _cont("Anisotropy (g)",     "mtlx.layer_top_anisotropy",         0.02, 0.0,  0.99),
    _cont("Roughness",          "mtlx.skin_bsdf_roughness",          0.02, 0.01, 1.0),
    _cont("IOR",                "mtlx.skin_bsdf_ior",                0.02, 1.0,  2.0),

    _cont("Pore density",       "mtlx.skin_bsdf_pore_density",       0.05, 0.0,  1.0),
    _cont("Pore depth",         "mtlx.skin_bsdf_pore_depth",         0.05, 0.0,  1.0),
    _cont("Vellus hair density", "mtlx.skin_bsdf_hair_density",      0.05, 0.0,  1.0),
    _cont("Vellus hair tilt",   "mtlx.skin_bsdf_hair_tilt",          0.05, 0.0,  1.0),

    _cont("Light elevation",    "light_elevation",             5.0, -90.0, 90.0),
    _cont("Light azimuth",      "light_azimuth",               5.0, -180.0, 180.0),
    _cont("Light intensity",    "light_intensity",             0.2,  0.0,  20.0),
    _cont("Light color R",      "light_color_r",               0.05, 0.0,  1.0),
    _cont("Light color G",      "light_color_g",               0.05, 0.0,  1.0),
    _cont("Light color B",      "light_color_b",               0.05, 0.0,  1.0),
]

# Backward-compat alias. New code should call build_all_params(renderer)
# to get the live list including dynamic material params; older code that
# imports ALL_PARAMS directly still gets the static base.
ALL_PARAMS = STATIC_PARAMS


_GANGED_MTLX_FIELDS: dict[str, list[str]] = {
    "layer_top_anisotropy": ["layer_middle_anisotropy", "layer_bottom_anisotropy"],
    "skin_bsdf_ior": ["layer_top_ior", "layer_middle_ior", "layer_bottom_ior"],
    "layer_top_scattering_coeff": ["layer_middle_scattering_coeff"],
}

_SKIN_TO_MTLX: dict[str, str] = {
    "skin.melanin_fraction":       "mtlx.layer_top_melanin",
    "skin.hemoglobin_fraction":    "mtlx.layer_middle_hemoglobin",
    "skin.blood_oxygenation":      "mtlx.layer_middle_blood_oxygenation",
    "skin.epidermis_thickness_mm": "mtlx.layer_top_thickness",
    "skin.dermis_thickness_mm":    "mtlx.layer_middle_thickness",
    "skin.subcut_thickness_mm":    "mtlx.layer_bottom_thickness",
    "skin.anisotropy_g":           "mtlx.layer_top_anisotropy",
    "skin.roughness":              "mtlx.skin_bsdf_roughness",
    "skin.ior":                    "mtlx.skin_bsdf_ior",
    "skin.pore_density":           "mtlx.skin_bsdf_pore_density",
    "skin.pore_depth":             "mtlx.skin_bsdf_pore_depth",
    "skin.hair_density":           "mtlx.skin_bsdf_hair_density",
    "skin.hair_tilt":              "mtlx.skin_bsdf_hair_tilt",
}


def _get_nested(obj, path):
    """Resolve `path` on `obj`. Routes:
    - "mtlx.<field>"          → obj.mtlx_overrides[field]   (scalar)
    - "mtlx.<field>.<idx>"    → obj.mtlx_overrides[field][idx]  (vector comp)
    - "<a>.<b>"               → getattr chain (legacy)
    Falls back to the active material's uniform_block default when an
    `mtlx.*` path hasn't been explicitly overridden yet, so the slider
    starts at the authored MaterialX value.
    """
    parts = path.split(".")
    if parts[0] == "mtlx" and len(parts) >= 2:
        field_name = parts[1]
        overrides = getattr(obj, "mtlx_overrides", {})
        cm = getattr(obj, "_mtlx_skin_material", None)
        if field_name in overrides:
            value = overrides[field_name]
        else:
            # Fall back to the live SkinParameters → MaterialX mapping so
            # the slider tracks the biological-parameter sliders until the
            # user explicitly touches it. After that, the override wins.
            try:
                base = obj._mtlx_skin_overrides()
            except (AttributeError, TypeError):
                base = {}
            if field_name in base:
                value = base[field_name]
            elif cm is not None:
                value = next(
                    (uf.default for uf in cm.uniform_block if uf.name == field_name),
                    0.0,
                )
            else:
                value = 0.0
        if len(parts) == 3:
            idx = int(parts[2])
            if isinstance(value, (tuple, list)):
                return float(value[idx]) if idx < len(value) else 0.0
            return 0.0
        return float(value) if isinstance(value, (int, float)) else value
    for p in parts:
        obj = getattr(obj, p)
    return obj


def _set_nested(obj, path, value):
    parts = path.split(".")
    if parts[0] == "mtlx" and len(parts) >= 2:
        field_name = parts[1]
        overrides = obj.mtlx_overrides
        if len(parts) == 3:
            idx = int(parts[2])
            cur = overrides.get(field_name)
            if cur is None:
                cm = getattr(obj, "_mtlx_skin_material", None)
                cur = next(
                    (uf.default for uf in cm.uniform_block if uf.name == field_name),
                    (0.0, 0.0, 0.0),
                ) if cm is not None else (0.0, 0.0, 0.0)
            cur_list = list(cur) if isinstance(cur, (tuple, list)) else [0.0, 0.0, 0.0]
            while len(cur_list) <= idx:
                cur_list.append(0.0)
            cur_list[idx] = float(value)
            val_tuple = tuple(cur_list)
            overrides[field_name] = val_tuple
            for linked in _GANGED_MTLX_FIELDS.get(field_name, ()):
                overrides[linked] = val_tuple
        else:
            fval = float(value)
            overrides[field_name] = fval
            for linked in _GANGED_MTLX_FIELDS.get(field_name, ()):
                overrides[linked] = fval
        return
    for p in parts[:-1]:
        obj = getattr(obj, p)
    setattr(obj, parts[-1], value)


# preset_index intentionally stays out of the saved snapshot: the user's
# custom preset list can change between sessions, so a stored index loses
# meaning. The underlying param values restore themselves directly.
_NON_PERSISTED_PARAMS = {"preset_index"}


def _snapshot_params(renderer, params: list[ParamSpec] | None = None) -> dict[str, float | int]:
    out: dict[str, float | int] = {}
    for p in (params if params is not None else build_all_params(renderer)):
        if p.path in _NON_PERSISTED_PARAMS:
            continue
        val = _get_nested(renderer, p.path)
        if p.kind == "continuous":
            out[p.path] = float(val)
        else:
            out[p.path] = int(val)
    return out


def _apply_saved_params(renderer, saved_params, params: list[ParamSpec] | None = None) -> None:
    if not isinstance(saved_params, dict):
        return
    for old_key, new_key in _SKIN_TO_MTLX.items():
        if old_key in saved_params and new_key not in saved_params:
            saved_params[new_key] = saved_params[old_key]
    for p in (params if params is not None else build_all_params(renderer)):
        if p.path in _NON_PERSISTED_PARAMS or p.path not in saved_params:
            continue
        raw = saved_params[p.path]
        try:
            if p.kind == "continuous":
                val = float(np.clip(float(raw), p.lo, p.hi))
            else:
                choices = getattr(renderer, p.choice_source, None) or []
                if not choices:
                    continue
                idx = int(raw)
                if not (0 <= idx < len(choices)):
                    continue
                val = idx
        except (TypeError, ValueError):
            continue
        _set_nested(renderer, p.path, val)


def _snapshot_camera(renderer) -> dict:
    orbit = renderer.orbit_camera
    free = renderer.free_camera
    return {
        "mode": renderer.camera_mode,
        "orbit": {
            "yaw": float(orbit.yaw),
            "pitch": float(orbit.pitch),
            "distance": float(orbit.distance),
            "fov": float(orbit.fov),
            "target": [float(orbit.target[0]), float(orbit.target[1]), float(orbit.target[2])],
        },
        "free": {
            "position": [float(free.position[0]), float(free.position[1]), float(free.position[2])],
            "yaw": float(free.yaw),
            "pitch": float(free.pitch),
            "fov": float(free.fov),
            "move_speed": float(free.move_speed),
        },
    }


def _apply_saved_camera(renderer, saved_cam) -> None:
    if not isinstance(saved_cam, dict):
        return

    def _vec3(raw, fallback):
        if isinstance(raw, (list, tuple)) and len(raw) == 3:
            try:
                return np.array([float(raw[0]), float(raw[1]), float(raw[2])], dtype=np.float32)
            except (TypeError, ValueError):
                pass
        return fallback

    def _flt(raw, fallback):
        try:
            return float(raw)
        except (TypeError, ValueError):
            return fallback

    orbit_raw = saved_cam.get("orbit")
    if isinstance(orbit_raw, dict):
        o = renderer.orbit_camera
        o.yaw = _flt(orbit_raw.get("yaw"), o.yaw)
        o.pitch = float(np.clip(
            _flt(orbit_raw.get("pitch"), o.pitch), -np.pi / 2 + 0.01, np.pi / 2 - 0.01
        ))
        o.distance = float(np.clip(_flt(orbit_raw.get("distance"), o.distance), 0.5, 50.0))
        o.fov = float(np.clip(_flt(orbit_raw.get("fov"), o.fov), 1.0, 170.0))
        o.target = _vec3(orbit_raw.get("target"), o.target)

    free_raw = saved_cam.get("free")
    if isinstance(free_raw, dict):
        f = renderer.free_camera
        f.position = _vec3(free_raw.get("position"), f.position)
        f.yaw = _flt(free_raw.get("yaw"), f.yaw)
        f.pitch = float(np.clip(
            _flt(free_raw.get("pitch"), f.pitch), -np.pi / 2 + 0.01, np.pi / 2 - 0.01
        ))
        f.fov = float(np.clip(_flt(free_raw.get("fov"), f.fov), 1.0, 170.0))
        f.move_speed = float(np.clip(_flt(free_raw.get("move_speed"), f.move_speed), 0.05, 50.0))

    mode = saved_cam.get("mode")
    if mode in ("orbit", "free"):
        renderer.camera_mode = mode


class InputHandler:
    """Manages GLFW callbacks and maps input to renderer state."""

    def __init__(self, window, renderer: Renderer) -> None:
        self.window = window
        self.renderer = renderer
        self.selected_param = 0
        # Live param list — static base + dynamic MaterialX inputs from
        # the active skin material. Built once after the renderer's MtlX
        # runtime has loaded so uniform_block reflection is available.
        self.params: list[ParamSpec] = build_all_params(renderer)

        # Mouse state
        self._last_mx = 0.0
        self._last_my = 0.0
        self._left_down = False
        self._right_down = False
        self._middle_down = False

        glfw.set_cursor_pos_callback(window, self._on_mouse_move)
        glfw.set_mouse_button_callback(window, self._on_mouse_button)
        glfw.set_scroll_callback(window, self._on_scroll)
        glfw.set_key_callback(window, self._on_key)

        self._print_help()
        self._print_param()

    def _on_mouse_button(self, _win, button, action, _mods):
        if button == glfw.MOUSE_BUTTON_LEFT:
            self._left_down = action == glfw.PRESS
        elif button == glfw.MOUSE_BUTTON_RIGHT:
            self._right_down = action == glfw.PRESS
        elif button == glfw.MOUSE_BUTTON_MIDDLE:
            self._middle_down = action == glfw.PRESS
        if action == glfw.PRESS:
            self._last_mx, self._last_my = glfw.get_cursor_pos(self.window)

    def _on_mouse_move(self, _win, mx, my):
        dx = mx - self._last_mx
        dy = my - self._last_my
        self._last_mx = mx
        self._last_my = my

        cam = self.renderer.camera
        if self.renderer.camera_mode == "orbit":
            if self._left_down:
                cam.orbit(dx, dy)
            elif self._right_down:
                cam.pan(dx, dy)
        else:  # free
            if self._left_down:
                cam.look(dx, dy)

    def _on_scroll(self, _win, _xoff, yoff):
        self.renderer.camera.zoom(yoff)

    def _on_key(self, win, key, _scancode, action, _mods):
        if action not in (glfw.PRESS, glfw.REPEAT):
            return

        if key == glfw.KEY_TAB:
            self.selected_param = (self.selected_param + 1) % len(self.params)
            self._print_param()
        elif key == glfw.KEY_LEFT_SHIFT and action == glfw.PRESS:
            self.selected_param = (self.selected_param - 1) % len(self.params)
            self._print_param()
        elif key in (glfw.KEY_UP, glfw.KEY_RIGHT):
            self._adjust_param(1)
        elif key in (glfw.KEY_DOWN, glfw.KEY_LEFT):
            self._adjust_param(-1)
        elif key == glfw.KEY_R:
            self._reset_params()
        elif key == glfw.KEY_H:
            self._print_help()
        elif key == glfw.KEY_P:
            self._print_all_params()
        elif key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(win, True)
        elif key == glfw.KEY_F1 or key == glfw.KEY_SPACE:
            self.renderer.show_hud = not self.renderer.show_hud
        elif key == glfw.KEY_C:
            self.renderer.toggle_camera_mode()
            print(f"[Camera mode: {self.renderer.camera_mode}]")
        elif key == glfw.KEY_F:
            self.renderer.reset_camera()
            print("[Camera recentred on head]")
        elif glfw.KEY_1 <= key <= glfw.KEY_9:
            idx = key - glfw.KEY_1
            if idx < len(self.params):
                self.selected_param = idx
                self._print_param()

    def update(self, dt: float) -> None:
        """Poll continuous inputs (free-cam WASDQE). Called once per frame."""
        if self.renderer.camera_mode != "free":
            return
        w = self.window
        f = (glfw.get_key(w, glfw.KEY_W) == glfw.PRESS) - (glfw.get_key(w, glfw.KEY_S) == glfw.PRESS)
        r = (glfw.get_key(w, glfw.KEY_D) == glfw.PRESS) - (glfw.get_key(w, glfw.KEY_A) == glfw.PRESS)
        u = (glfw.get_key(w, glfw.KEY_E) == glfw.PRESS) - (glfw.get_key(w, glfw.KEY_Q) == glfw.PRESS)
        if f or r or u:
            self.renderer.camera.move(float(f), float(r), float(u), dt)

    def _adjust_param(self, direction: int) -> None:
        p = self.params[self.selected_param]
        if p.kind == "continuous":
            val = _get_nested(self.renderer, p.path)
            val = float(np.clip(val + direction * p.step, p.lo, p.hi))
            _set_nested(self.renderer, p.path, val)
            if p.path.startswith("light"):
                self.renderer._update_light()
        else:
            choices = getattr(self.renderer, p.choice_source)
            if not choices:
                return
            current = int(_get_nested(self.renderer, p.path))
            new = (current + direction) % len(choices)
            _set_nested(self.renderer, p.path, new)
            if p.choice_source == "presets":
                from skinny.presets import apply_preset
                apply_preset(self.renderer, self.renderer.presets[new])
        self._print_param()

    def _reset_params(self) -> None:
        from skinny.renderer import SkinParameters
        self.renderer.skin = SkinParameters()
        self.renderer.mtlx_overrides.clear()
        self.renderer.mtlx_overrides.update(self.renderer._mtlx_skin_overrides())
        self.renderer.light_azimuth = 45.0
        self.renderer.light_elevation = 35.0
        self.renderer.light_intensity = 5.0
        self.renderer.light_color_r = 0.624
        self.renderer.light_color_g = 0.583
        self.renderer.light_color_b = 0.520
        self.renderer._update_light()
        print("[Reset all parameters to defaults]")
        self._print_param()

    def _param_value_str(self, p: ParamSpec) -> str:
        """Human-readable current value for a param (value + position bar)."""
        if p.kind == "continuous":
            val = float(_get_nested(self.renderer, p.path))
            bar_len = 16
            frac = (val - p.lo) / (p.hi - p.lo) if p.hi > p.lo else 0.0
            filled = int(np.clip(frac, 0, 1) * bar_len)
            bar = "#" * filled + "-" * (bar_len - filled)
            return f"{val:.3f}  [{bar}]"
        choices = getattr(self.renderer, p.choice_source, [])
        idx = int(_get_nested(self.renderer, p.path))
        if 0 <= idx < len(choices):
            label = getattr(choices[idx], "name", str(choices[idx]))
        else:
            label = "(none)"
        return f"{label}  [{idx + 1}/{len(choices)}]"

    def _print_param(self) -> None:
        p = self.params[self.selected_param]
        print(f"  [{self.selected_param + 1}/{len(self.params)}] {p.name}: {self._param_value_str(p)}")

    def _print_all_params(self) -> None:
        print("\n--- Current Parameters ---")
        for i, p in enumerate(self.params):
            marker = " >> " if i == self.selected_param else "    "
            print(f"{marker}{p.name}: {self._param_value_str(p)}")
        cam = self.renderer.camera
        if self.renderer.camera_mode == "orbit":
            print(
                f"    Camera[orbit]: yaw={np.degrees(cam.yaw):.1f} "
                f"pitch={np.degrees(cam.pitch):.1f} dist={cam.distance:.2f}"
            )
        else:
            p = cam.position
            print(
                f"    Camera[free]: pos=({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f}) "
                f"yaw={np.degrees(cam.yaw):.1f} pitch={np.degrees(cam.pitch):.1f} "
                f"speed={cam.move_speed:.2f}"
            )
        print("---\n")

    def build_hud_lines(self) -> list[str]:
        """Text lines that the renderer rasterises into the on-screen HUD each frame."""
        r = self.renderer
        p = self.params[self.selected_param]

        if r.camera_mode == "orbit":
            cam_line = "Camera: Orbit    L drag orbit  R drag pan  scroll zoom"
        else:
            cam_line = (
                f"Camera: Free     L drag look  WASD move  Q/E down/up  "
                f"scroll speed ({r.camera.move_speed:.2f})"
            )

        lines = [
            f"SKINNY  {self._fps_str()}   samples: {r.accum_frame + 1}",
            f"Environment: {r.env_name}",
            cam_line,
            "",
            f"[{self.selected_param + 1}/{len(self.params)}] {p.name}: {self._param_value_str(p)}",
            "",
            "Tab / Shift    : next / prev param",
            "Arrows         : adjust parameter",
            "1-9            : jump to parameter",
            "C  camera   F  recentre   R  reset   P  print   H  help   Space  HUD",
            "Esc            : quit",
        ]
        return lines

    def _fps_str(self) -> str:
        fps = getattr(self.renderer, "_fps_smooth", 0.0)
        return f"{fps:5.1f} fps" if fps > 0 else "  --- fps"

    @staticmethod
    def _print_help() -> None:
        print(
            "\n=== Skinny Controls ===\n"
            "  C                 : toggle Orbit / Free camera (default: Orbit)\n"
            "  [Orbit]  L drag   : orbit around head\n"
            "           R drag   : pan target\n"
            "           scroll   : zoom\n"
            "  [Free]   L drag   : mouse look\n"
            "           W A S D  : move forward / left / back / right\n"
            "           Q / E    : move down / up\n"
            "           scroll   : adjust move speed\n"
            "  Tab / Shift       : next / previous parameter\n"
            "  Up/Down arrows    : increase / decrease parameter\n"
            "  1-9               : jump directly to parameter\n"
            "  F                 : recentre camera on head\n"
            "  R                 : reset all to defaults\n"
            "  P                 : print all parameters\n"
            "  H                 : show this help\n"
            "  Space / F1        : toggle on-screen HUD\n"
            "  Esc               : quit\n"
            "=======================\n"
        )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(prog="skinny")
    parser.add_argument(
        "scene", nargs="?", type=Path, default=None,
        help="Path to a USD stage (.usda / .usdc / .usdz).",
    )
    parser.add_argument(
        "--usd", type=Path, default=None,
        help="(deprecated, use positional arg) Path to a USD stage.",
    )
    parser.add_argument(
        "--usdMtlx", action="store_true", default=False,
        help="Rely on USD's built-in usdMtlx plugin for .mtlx file "
             "resolution instead of the MaterialX API fallback.",
    )
    args = parser.parse_args()

    scene_path: Path | None = args.scene or args.usd
    usd_scene = None
    if scene_path is not None:
        from skinny.usd_loader import load_scene_from_usd
        usd_scene = load_scene_from_usd(
            scene_path, use_usd_mtlx_plugin=args.usdMtlx,
        )
        n_inst = len(usd_scene.instances)
        n_mat = len(usd_scene.materials)
        print(f"[USD] {scene_path.name}: {n_inst} instances, {n_mat} materials")

    ensure_dirs()
    saved = load_settings()

    if not glfw.init():
        raise RuntimeError("Failed to initialise GLFW")

    if not glfw.vulkan_supported():
        glfw.terminate()
        raise RuntimeError("GLFW reports Vulkan is not supported on this system")

    glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)
    glfw.window_hint(glfw.RESIZABLE, glfw.FALSE)
    window = glfw.create_window(WINDOW_WIDTH, WINDOW_HEIGHT, "Skinny", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("Failed to create GLFW window")

    # Restore last Vulkan-window position (size is fixed — RESIZABLE=FALSE).
    vw = saved.get("vulkan_window")
    if isinstance(vw, dict):
        x, y = vw.get("x"), vw.get("y")
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            glfw.set_window_pos(window, int(x), int(y))

    vk_ctx = VulkanContext(window, WINDOW_WIDTH, WINDOW_HEIGHT)

    repo_root = Path(__file__).resolve().parents[2]
    renderer = Renderer(
        vk_ctx=vk_ctx,
        shader_dir=Path(__file__).parent / "shaders",
        hdr_dir=repo_root / "hdrs",
        head_dir=repo_root / "heads",
        tattoo_dir=repo_root / "tattoos",
        usd_scene=usd_scene,
    )

    _apply_saved_params(renderer, saved.get("params", {}))
    _apply_saved_camera(renderer, saved.get("camera"))
    renderer._update_light()

    input_handler = InputHandler(window, renderer)

    from skinny.control_panel import ControlPanel
    panel = ControlPanel(renderer, input_handler.params)

    tk_geom = saved.get("tk_window")
    if isinstance(tk_geom, str):
        panel.apply_geometry(tk_geom)

    prev_time = time.perf_counter()
    while not glfw.window_should_close(window):
        glfw.poll_events()

        now = time.perf_counter()
        dt = now - prev_time
        prev_time = now

        input_handler.update(dt)
        panel.tick()
        renderer.update(dt)
        renderer.hud_text_lines = input_handler.build_hud_lines()
        renderer.render()

    # Snapshot state before tearing things down. Write failures are swallowed
    # so a read-only home dir can't break shutdown.
    try:
        out: dict = {
            "vulkan_window": _window_pos_dict(window),
            "params": _snapshot_params(renderer, input_handler.params),
            "camera": _snapshot_camera(renderer),
        }
        geom = panel.get_geometry()
        if geom:
            out["tk_window"] = geom
        save_settings(out)
    except OSError:
        pass

    panel.destroy()
    renderer.cleanup()
    vk_ctx.destroy()
    glfw.terminate()


def _window_pos_dict(window) -> dict[str, int]:
    pos = glfw.get_window_pos(window)
    return {"x": int(pos[0]), "y": int(pos[1])}


if __name__ == "__main__":
    main()
