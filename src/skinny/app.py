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


# Discrete first so Preset / Environment show up at the top.
ALL_PARAMS: list[ParamSpec] = [
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

    _cont("Melanin",            "skin.melanin_fraction",       0.01, 0.0,  1.0),
    _cont("Hemoglobin",         "skin.hemoglobin_fraction",    0.01, 0.0,  1.0),
    _cont("Blood oxygenation",  "skin.blood_oxygenation",      0.05, 0.0,  1.0),
    _cont("Epidermis thickness", "skin.epidermis_thickness_mm", 0.02, 0.01, 1.0),
    _cont("Dermis thickness",   "skin.dermis_thickness_mm",    0.1,  0.1,  5.0),
    _cont("Subcut thickness",   "skin.subcut_thickness_mm",    0.2,  0.5,  10.0),
    _cont("Anisotropy (g)",     "skin.anisotropy_g",           0.02, 0.0,  0.99),
    _cont("Roughness",          "skin.roughness",              0.02, 0.01, 1.0),
    _cont("IOR",                "skin.ior",                    0.02, 1.0,  2.0),

    _cont("Pore density",       "skin.pore_density",           0.05, 0.0,  1.0),
    _cont("Pore depth",         "skin.pore_depth",             0.05, 0.0,  1.0),
    _cont("Vellus hair density", "skin.hair_density",          0.05, 0.0,  1.0),
    _cont("Vellus hair tilt",   "skin.hair_tilt",              0.05, 0.0,  1.0),

    _cont("Light elevation",    "light_elevation",             5.0, -90.0, 90.0),
    _cont("Light azimuth",      "light_azimuth",               5.0, -180.0, 180.0),
    _cont("Light intensity",    "light_intensity",             0.2,  0.0,  20.0),
    _cont("Light color R",      "light_color_r",               0.05, 0.0,  1.0),
    _cont("Light color G",      "light_color_g",               0.05, 0.0,  1.0),
    _cont("Light color B",      "light_color_b",               0.05, 0.0,  1.0),
]


def _get_nested(obj, path):
    parts = path.split(".")
    for p in parts:
        obj = getattr(obj, p)
    return obj


def _set_nested(obj, path, value):
    parts = path.split(".")
    for p in parts[:-1]:
        obj = getattr(obj, p)
    setattr(obj, parts[-1], value)


# preset_index intentionally stays out of the saved snapshot: the user's
# custom preset list can change between sessions, so a stored index loses
# meaning. The underlying param values restore themselves directly.
_NON_PERSISTED_PARAMS = {"preset_index"}


def _snapshot_params(renderer) -> dict[str, float | int]:
    out: dict[str, float | int] = {}
    for p in ALL_PARAMS:
        if p.path in _NON_PERSISTED_PARAMS:
            continue
        val = _get_nested(renderer, p.path)
        if p.kind == "continuous":
            out[p.path] = float(val)
        else:
            out[p.path] = int(val)
    return out


def _apply_saved_params(renderer, saved_params) -> None:
    if not isinstance(saved_params, dict):
        return
    for p in ALL_PARAMS:
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
            self.selected_param = (self.selected_param + 1) % len(ALL_PARAMS)
            self._print_param()
        elif key == glfw.KEY_LEFT_SHIFT and action == glfw.PRESS:
            self.selected_param = (self.selected_param - 1) % len(ALL_PARAMS)
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
            if idx < len(ALL_PARAMS):
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
        p = ALL_PARAMS[self.selected_param]
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
        p = ALL_PARAMS[self.selected_param]
        print(f"  [{self.selected_param + 1}/{len(ALL_PARAMS)}] {p.name}: {self._param_value_str(p)}")

    def _print_all_params(self) -> None:
        print("\n--- Current Parameters ---")
        for i, p in enumerate(ALL_PARAMS):
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
        p = ALL_PARAMS[self.selected_param]

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
            f"[{self.selected_param + 1}/{len(ALL_PARAMS)}] {p.name}: {self._param_value_str(p)}",
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
    )

    _apply_saved_params(renderer, saved.get("params", {}))
    _apply_saved_camera(renderer, saved.get("camera"))
    renderer._update_light()

    input_handler = InputHandler(window, renderer)

    from skinny.control_panel import ControlPanel
    panel = ControlPanel(renderer, ALL_PARAMS)

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
            "params": _snapshot_params(renderer),
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
