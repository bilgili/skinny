"""Small command queue used by the Qt render worker.

The queue is deliberately free of Qt imports so it can be unit-tested without a
GUI or GPU. GUI code posts renderer mutations; the render worker drains and
executes them on the thread that owns frame rendering.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any

from skinny.params import STATIC_PARAMS, _set_nested
from skinny.playback import PlaybackClock
from skinny.scene_graph import copy_scene_graph


@dataclass(frozen=True)
class QtRendererConfig:
    scene_path: Path | None
    gpu_pref: str
    use_usd_mtlx: bool
    execution_mode: str
    bdpt_walk: str
    initial_integrator: str | None
    neural_handoff: str
    neural_trainer: str
    train_precision: str
    online_training: bool
    reuse: str | None
    lobe_samplers: str | None
    backend: str
    encoding: str
    sppm_glossy_roughness: float | None
    width: int
    height: int
    requested_backend: str = "auto"
    spectral: bool = False


@dataclass(frozen=True)
class FrameSnapshot:
    pixels: bytes
    width: int
    height: int
    accum_frame: int
    gpu_name: str
    online_training: dict[str, Any]


@dataclass(frozen=True)
class DebugFrame:
    """One embedded Camera-Debug-viewport frame, produced on the render worker."""
    pixels: bytes
    width: int
    height: int


@dataclass(frozen=True)
class RendererStateSnapshot:
    width: int
    height: int
    gpu_name: str
    params: dict[str, float | int] = field(default_factory=dict)
    camera: dict[str, Any] = field(default_factory=dict)
    gizmo_mode: int | None = None
    encoding: str = "E0"
    sppm_glossy_roughness: float | None = None
    online_training: dict[str, Any] = field(default_factory=dict)
    choices: dict[str, list[str]] = field(default_factory=dict)
    uses_default_lights: bool = True


@dataclass(frozen=True)
class _LensProj:
    enabled: bool


@dataclass(frozen=True)
class _CameraProj:
    """Read-only mirror of the camera params the Scene Graph dock displays."""
    fov: float = 0.0
    near: float = 0.0
    far: float = 0.0
    fstop: float = 0.0
    focus_distance: float = 0.0
    yaw: float = 0.0
    pitch: float = 0.0
    distance: float = 0.0
    max_distance: float = 0.0
    lens: _LensProj | None = None


@dataclass(frozen=True)
class _MaterialProj:
    python_module: str | None = None
    name: str | None = None
    mtlx_target_name: str | None = None
    mtlx_scene_target: str | None = None
    parameter_overrides: dict = field(default_factory=dict)


@dataclass(frozen=True)
class _UsdSceneProj:
    materials: tuple[_MaterialProj, ...] = ()


@dataclass(frozen=True)
class SceneStateSnapshot:
    """Immutable projection of renderer-owned scene state the Scene Graph and
    BXDF docks read on the GUI thread — built on the render worker, never a live
    handle."""
    scene_graph: Any = None
    scene_graph_version: int = 0
    # `id()` of the LIVE renderer-owned tree (captured before copying), stable
    # across polls while that object is unchanged. The dock keys structural-change
    # detection off this + the version, not `id(scene_graph)` — the latter is a
    # fresh copy every refresh (see `copy_scene_graph`) and would trip every tick.
    scene_graph_id: int = 0
    usd_scene: _UsdSceneProj | None = None
    scene: _UsdSceneProj | None = None
    usd_scene_id: int = 0
    has_usd_stage: bool = False
    has_usd_edit_layer: bool = False
    camera: _CameraProj | None = None
    material_version: int = 0
    mtlx_overrides: dict = field(default_factory=dict)


def _proj_scene(scene, cm_map=None) -> "_UsdSceneProj | None":
    if scene is None:
        return None
    cm_map = cm_map or {}
    mats = getattr(scene, "materials", None) or []
    out = []
    for i, m in enumerate(mats):
        cm = cm_map.get(i)
        out.append(_MaterialProj(
            python_module=getattr(m, "python_module", None),
            name=getattr(m, "name", None),
            mtlx_target_name=getattr(m, "mtlx_target_name", None),
            mtlx_scene_target=getattr(cm, "target_name", None) if cm else None,
            parameter_overrides=dict(getattr(m, "parameter_overrides", {}) or {}),
        ))
    return _UsdSceneProj(materials=tuple(out))


def build_scene_state(renderer) -> SceneStateSnapshot:
    """Project the renderer's scene state into an immutable snapshot. Runs on the
    render worker (via `proxy.request`), so it may touch the live renderer."""
    cam = getattr(renderer, "camera", None)
    camera = None
    if cam is not None:
        lens = getattr(cam, "lens", None)
        camera = _CameraProj(
            fov=float(getattr(cam, "fov", 0.0)),
            near=float(getattr(cam, "near", 0.0)),
            far=float(getattr(cam, "far", 0.0)),
            fstop=float(getattr(cam, "fstop", 0.0)),
            focus_distance=float(getattr(cam, "focus_distance", 0.0)),
            yaw=float(getattr(cam, "yaw", 0.0)),
            pitch=float(getattr(cam, "pitch", 0.0)),
            distance=float(getattr(cam, "distance", 0.0)),
            max_distance=float(getattr(cam, "max_distance", 0.0)),
            lens=_LensProj(bool(lens.enabled)) if lens is not None else None,
        )
    usd_scene = getattr(renderer, "_usd_scene", None)
    cm_map = getattr(renderer, "_mtlx_scene_materials", {}) or {}
    live_graph = getattr(renderer, "scene_graph", None)
    return SceneStateSnapshot(
        # Copy the tree, don't leak the live renderer-owned one: the worker keeps
        # mutating/reassigning `renderer.scene_graph` on its thread while the GUI
        # reads the snapshot (data race). The snapshot must be a detached copy;
        # `scene_graph_id` carries the LIVE tree's identity for change detection.
        scene_graph=copy_scene_graph(live_graph),
        scene_graph_version=int(getattr(renderer, "_scene_graph_version", 0)),
        scene_graph_id=id(live_graph) if live_graph is not None else 0,
        usd_scene=_proj_scene(usd_scene, cm_map),
        scene=_proj_scene(getattr(renderer, "scene", None)),
        usd_scene_id=id(usd_scene) if usd_scene is not None else 0,
        has_usd_stage=getattr(renderer, "_usd_stage", None) is not None,
        has_usd_edit_layer=getattr(renderer, "_usd_edit_layer", None) is not None,
        camera=camera,
        material_version=int(getattr(renderer, "_material_version", 0)),
        mtlx_overrides=dict(getattr(renderer, "mtlx_overrides", {}) or {}),
    )


# Sentinel stored on the proxy for `_usd_stage` / `_usd_edit_layer`: the dock only
# tests presence (`is None` / `is not None`), never the object itself.
_SCENE_STATE_PRESENT = object()


@dataclass
class _Choice:
    name: str


@dataclass
class _AttrBag:
    pass


def _default_choice_names() -> dict[str, list[str]]:
    return {
        "presets": ["Default"],
        "environments": ["studio.hdr"],
        "direct_light_modes": ["On", "Off"],
        "scatter_modes": ["BSSRDF", "Volume"],
        "integrator_modes": ["Path", "BDPT", "SPPM"],
        "proposal_preset_modes": ["bsdf"],
        "reuse_modes": ["Off"],
        "coat_sampler_modes": ["Default"],
        "spec_sampler_modes": ["Default"],
        "diff_sampler_modes": ["Default"],
        "restir_regime_modes": ["Initial"],
        "restir_combination_modes": ["Unbiased", "Biased"],
        "tonemap_modes": ["Filmic"],
        "furnace_modes": ["Off", "On"],
        "models": ["(none)"],
        "detail_maps_modes": ["Off"],
        "tattoos": ["(none)"],
    }


def choice_names_from_renderer(renderer) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for param in STATIC_PARAMS:
        if param.kind != "discrete" or param.choice_source is None:
            continue
        raw = getattr(renderer, param.choice_source, None) or []
        out[param.choice_source] = [getattr(item, "name", str(item)) for item in raw]
    return out


def renderer_online_status(renderer) -> dict[str, Any]:
    try:
        return dict(renderer.online_training_status())
    except Exception:  # noqa: BLE001
        return {}


class QtRendererProxy:
    """GUI-side mirror of renderer state.

    The proxy lets the existing shared Qt UI tree read and write plausible
    renderer state without holding the live GPU renderer on the GUI thread.
    Setters update local state immediately, then enqueue the real mutation for
    the render thread.
    """

    def __init__(
        self,
        command_queue: "RenderCommandQueue",
        *,
        width: int,
        height: int,
        backend: str,
        encoding: str,
        sppm_glossy_roughness: float | None,
    ) -> None:
        object.__setattr__(self, "_commands", command_queue)
        object.__setattr__(self, "_lock", Lock())
        object.__setattr__(self, "_values", self._default_values(width, height))
        object.__setattr__(self, "_choices", {
            key: [_Choice(name) for name in names]
            for key, names in _default_choice_names().items()
        })
        object.__setattr__(self, "_gpu_name", "starting")
        object.__setattr__(self, "_backend_name", backend)
        object.__setattr__(self, "_encoding", encoding)
        object.__setattr__(self, "_sppm_glossy_roughness", sppm_glossy_roughness)
        object.__setattr__(self, "_suppress_posts", False)
        object.__setattr__(self, "mtlx_overrides", {})
        object.__setattr__(self, "_mtlx_skin_material", None)
        object.__setattr__(self, "_usd_scene", None)
        object.__setattr__(self, "_usd_controls", [])
        object.__setattr__(self, "scene_graph", None)
        object.__setattr__(self, "_scene_graph_version", 0)
        object.__setattr__(self, "_scene_graph_id", 0)
        object.__setattr__(self, "clock", PlaybackClock(has_animation=False))
        object.__setattr__(self, "film", _AttrBag())
        self.film.iso = 100.0
        self.film.exposure_time = 1.0
        object.__setattr__(self, "has_usd_camera", False)
        # Scene Graph dock reads (refreshed from a worker `SceneStateSnapshot`).
        object.__setattr__(self, "scene", None)
        object.__setattr__(self, "camera", None)
        object.__setattr__(self, "_usd_stage", None)
        object.__setattr__(self, "_usd_edit_layer", None)
        object.__setattr__(self, "_material_version", 0)
        object.__setattr__(self, "_usd_scene_id", 0)
        self._values["uses_default_lights"] = True

    def _default_values(self, width: int, height: int) -> dict[str, Any]:
        values: dict[str, Any] = {
            "width": int(width),
            "height": int(height),
            "camera_mode": "orbit",
            "light_color_r": 1.0,
            "light_color_g": 1.0,
            "light_color_b": 1.0,
            "light_elevation": 0.0,
            "light_azimuth": 0.0,
        }
        for param in STATIC_PARAMS:
            if param.path in values:
                continue
            values[param.path] = float(param.lo) if param.kind == "continuous" else 0
        values.update({
            "env_intensity": 1.0,
            "mm_per_unit": 5.0,
            "normal_map_strength": 1.0,
            "light_intensity": 1.0,
        })
        return values

    def apply_snapshot(self, snapshot: RendererStateSnapshot) -> None:
        with self._lock:
            self._values.update(snapshot.params)
            if "film.iso" in snapshot.params:
                self.film.iso = snapshot.params["film.iso"]
            if "film.exposure_time" in snapshot.params:
                self.film.exposure_time = snapshot.params["film.exposure_time"]
            self._values["width"] = snapshot.width
            self._values["height"] = snapshot.height
            for key, names in snapshot.choices.items():
                self._choices[key] = [_Choice(name) for name in names]
            self._gpu_name = snapshot.gpu_name
            self._encoding = snapshot.encoding
            self._sppm_glossy_roughness = snapshot.sppm_glossy_roughness
            self._values["uses_default_lights"] = snapshot.uses_default_lights

    def post(self, callback: Callable[[Any], Any], *, coalesce_key: str | None = None) -> None:
        self._commands.post(callback, coalesce_key=coalesce_key)

    def request(self, callback: Callable[[Any], Any]) -> Future[Any]:
        return self._commands.post_with_reply(callback)

    def resize(self, w: int, h: int) -> None:
        self.width = int(w)
        self.height = int(h)
        self.post(
            lambda renderer, w=int(w), h=int(h): renderer.resize(w, h),
            coalesce_key="resize",
        )

    def load_model_from_path(self, path: Path) -> None:
        self.post(lambda renderer, path=Path(path): renderer.load_model_from_path(path))

    # ── Scene Graph dock surface ──────────────────────────────────────────
    # Reads come from a worker-built `SceneStateSnapshot`; mutations post to the
    # render worker. The four edits whose result the GUI uses (add/save/texture/
    # lens) return a `Future` the dock resolves asynchronously.

    def refresh_scene_state(self) -> "Future[SceneStateSnapshot]":
        return self.request(build_scene_state)

    def apply_scene_state(self, state: SceneStateSnapshot) -> None:
        self.scene_graph = state.scene_graph
        self._scene_graph_version = state.scene_graph_version
        self._scene_graph_id = state.scene_graph_id
        self._usd_scene = state.usd_scene
        self.scene = state.scene if state.scene is not None else state.usd_scene
        self._usd_stage = _SCENE_STATE_PRESENT if state.has_usd_stage else None
        self._usd_edit_layer = (
            _SCENE_STATE_PRESENT if state.has_usd_edit_layer else None
        )
        self.camera = state.camera
        self._material_version = state.material_version
        self._usd_scene_id = state.usd_scene_id
        self.mtlx_overrides = dict(state.mtlx_overrides)

    def set_gizmo_target(self, index: int) -> None:
        self.post(
            lambda r, i=int(index): r.set_gizmo_target(i),
            coalesce_key="gizmo_target",
        )

    def apply_subtree_enabled(self, path: str, value: bool) -> None:
        self.post(lambda r, p=path, v=value: r.apply_subtree_enabled(p, v))

    def apply_node_enabled(self, path: str, value: bool) -> None:
        self.post(lambda r, p=path, v=value: r.apply_node_enabled(p, v))

    def apply_camera_param(self, name: str, value: Any) -> None:
        self.post(
            lambda r, n=name, v=value: r.apply_camera_param(n, v),
            coalesce_key=f"camera_param:{name}",
        )

    def apply_material_override(self, index: int, name: str, value: Any) -> None:
        self.post(
            lambda r, i=index, n=name, v=value: r.apply_material_override(i, n, v),
            coalesce_key=f"mat_override:{index}:{name}",
        )

    def apply_light_override(
        self, light_type: str, index: int, name: str, value: Any,
    ) -> None:
        self.post(
            lambda r, t=light_type, i=index, n=name, v=value:
                r.apply_light_override(t, i, n, v),
            coalesce_key=f"light_override:{light_type}:{index}:{name}",
        )

    def apply_instance_transform(self, path, translate, rotate, scale) -> None:
        self.post(
            lambda r, p=path, t=translate, ro=rotate, s=scale:
                r.apply_instance_transform(p, t, ro, s),
            coalesce_key=f"instance_xform:{path}",
        )

    def set_transform(self, path, matrix) -> None:
        self.post(
            lambda r, p=path, m=matrix: r.set_transform(p, m),
            coalesce_key=f"set_transform:{path}",
        )

    def remove_node(self, path: str) -> "Future[Any]":
        return self.request(lambda r, p=path: r.remove_node(p))

    def add_model(self, path, parent_prim_path=None) -> "Future[Any]":
        return self.request(
            lambda r, p=path, pp=parent_prim_path: r.add_model(
                p, parent_prim_path=pp,
            ),
        )

    def add_light(self, light_type, parent_prim_path=None) -> "Future[Any]":
        return self.request(
            lambda r, lt=light_type, pp=parent_prim_path: r.add_light(
                lt, parent_prim_path=pp,
            ),
        )

    def save_edits(self) -> "Future[Any]":
        return self.request(lambda r: r.save_edits())

    def apply_dome_light_texture(self, index: int, path) -> "Future[Any]":
        return self.request(
            lambda r, i=index, p=path: r.apply_dome_light_texture(i, p),
        )

    def apply_camera_lens_file(self, path) -> "Future[Any]":
        return self.request(lambda r, p=path: r.apply_camera_lens_file(p))

    # ── BXDF dock GPU eval ────────────────────────────────────────────────
    # The renderer computes the lobe/BSSRDF grid and invokes `callback(grid)`;
    # under render-thread ownership that runs on the worker, so the dock passes a
    # callback that marshals the grid onto the GUI thread.

    def request_bxdf_eval(self, req, callback, on_error=None) -> None:
        self.post(self._eval_runner("request_bxdf_eval", req, callback, on_error))

    def request_bssrdf_eval(self, req, callback, on_error=None) -> None:
        self.post(self._eval_runner("request_bssrdf_eval", req, callback, on_error))

    @staticmethod
    def _eval_runner(method: str, req, callback, on_error):
        def run(r) -> None:
            try:
                getattr(r, method)(req, callback)
            except Exception as exc:  # noqa: BLE001
                if on_error is None:
                    raise
                on_error(exc)  # worker-thread CPU fallback
        return run

    # ── Material Graph dock ───────────────────────────────────────────────

    def render_material_preview(self, material_id, prim, size) -> "Future[Any]":
        return self.request(
            lambda r, m=material_id, p=prim, s=size:
                r.render_material_preview(m, p, size=s),
        )

    def save_screenshot(self, buf, fmt: str) -> None:
        def run(r, fmt=fmt) -> bytes:
            import io as _io
            inner = _io.BytesIO()
            r.save_screenshot(inner, fmt)
            return inner.getvalue()
        buf.write(self.request(run).result(timeout=30.0))

    def ensure_env_uploaded(self) -> None:
        def run(r) -> None:
            r._ensure_env_uploaded()
            r._material_version += 1
        self.post(run, coalesce_key="ensure_env_uploaded")

    def online_training_status(self) -> dict[str, Any]:
        return {}

    def _update_light(self) -> None:
        self.post(lambda renderer: renderer._update_light(), coalesce_key="light")

    def set_path(self, path: str, value: Any) -> None:
        object.__setattr__(self, "_suppress_posts", True)
        try:
            _set_nested(self, path, value)
        finally:
            object.__setattr__(self, "_suppress_posts", False)

        def set_nested(renderer, path=path, value=value) -> None:
            _set_nested(renderer, path, value)

        self.post(set_nested, coalesce_key=f"param:{path}")

    @property
    def gpu_name(self) -> str:
        with self._lock:
            return str(self._gpu_name)

    def __getattr__(self, name: str) -> Any:
        with self._lock:
            if name in self._choices:
                return self._choices[name]
            if name in self._values:
                return self._values[name]
        raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_") or name in {
            "mtlx_overrides", "_mtlx_skin_material", "_usd_scene",
            "_usd_controls", "scene_graph", "_scene_graph_version", "clock",
            "has_usd_camera", "film", "camera", "scene",
        }:
            object.__setattr__(self, name, value)
            return
        with self._lock:
            self._values[name] = value
        if getattr(self, "_suppress_posts", False):
            return

        def set_attr(renderer, name=name, value=value) -> None:
            setattr(renderer, name, value)

        self.post(set_attr, coalesce_key=f"attr:{name}")


@dataclass(frozen=True)
class RenderCommand:
    """One renderer-thread operation."""

    callback: Callable[[Any], Any]
    coalesce_key: str | None = None
    reply: Future[Any] | None = None


class RenderCommandQueue:
    """Thread-safe FIFO queue with optional last-write-wins coalescing.

    A coalesced command keeps the position of the first pending command with the
    same key, but replaces its callback. That preserves ordering against
    distinct user actions while preventing resize drags or slider streams from
    growing the queue without bound.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._pending: list[RenderCommand] = []

    def post(
        self,
        callback: Callable[[Any], Any],
        *,
        coalesce_key: str | None = None,
    ) -> None:
        command = RenderCommand(callback, coalesce_key)
        with self._lock:
            if coalesce_key is not None:
                for idx, existing in enumerate(self._pending):
                    if existing.coalesce_key == coalesce_key:
                        self._pending[idx] = command
                        return
            self._pending.append(command)

    def post_with_reply(self, callback: Callable[[Any], Any]) -> Future[Any]:
        """Post a command and return a future completed by the render worker."""
        future: Future[Any] = Future()
        command = RenderCommand(callback, reply=future)
        with self._lock:
            self._pending.append(command)
        return future

    def drain(self) -> list[RenderCommand]:
        with self._lock:
            commands = self._pending
            self._pending = []
        return commands

    def __len__(self) -> int:
        with self._lock:
            return len(self._pending)
