"""Scene intake — the one interface from a USD stage to a `SceneUpdate` value.

Intake reads a stage (in full, as a streamed batch, or at a time code) and
returns a value. It holds no reference to the renderer, imports nothing from
it, and never mutates it: `resolve_control_binding` returns a *description* of
a binding, and the renderer performs the write.

The renderer consumes a `SceneUpdate` through one application path
(`Renderer.apply_scene_update`). Everything that used to differ between the
three adoption paths — initial load, streamed metadata, post-edit resync — is
a field of the update, not a separate code path. `SceneUpdate` has three
constructors, one per trigger; see `openspec/specs/scene-intake/spec.md`.

`SceneUpdate` is a transfer object, not a pure value: it carries the live
`Usd.Stage` the renderer takes ownership of, and (for the same reason) the
`SkeletalScene` handle, which retains that stage plus a `UsdSkel.Cache`.
Splitting the skeletal handle back out would force every call site to thread
two returns for no gained purity, since the stage is already aboard.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional

import numpy as np
from pxr import Usd, UsdGeom, UsdLux

from skinny.mesh import MeshSource
from skinny.playback import PlaybackClock
from skinny.scene import CameraOverride, LensSystem, LightDir, Scene
from skinny.usd_loader import (
    AnimationIndex,
    ControlSpec,
    SkeletalScene,
    SkinnedMeshBinding,
    _extract_camera,
    _extract_distant_light,
    _extract_lens_system,
    _extract_sphere_light,
    _light_color_radiance,
    _read_open_stage,
    _read_usd_stage,
    _smooth_normals,
    _up_axis_rt,
    _world_transform,
    bake_usd_prim,
    build_animation_index,
    build_playback_clock,
    compute_joint_matrices,
    extract_skeletal_bindings,
    extract_ui_controls,
    lbs_points,
)
from skinny.mesh_cache import load_cache_index

# Re-exported so consumers get the whole per-frame surface from one module.
__all__ = [
    "ControlBinding",
    "SceneUpdate",
    "TimeSample",
    "adopt_scene",
    "compute_joint_matrices",
    "deform_skinned_mesh",
    "dome_light_intensity",
    "read_at_time",
    "read_lens_file",
    "read_open_stage",
    "read_stage",
    "resolve_control_binding",
]

# `Scene.mm_per_unit`'s sentinel default. A stage that reports it is telling us
# nothing, so adopting it would clobber the renderer's own skin scale.
_MM_PER_UNIT_SENTINEL = 120.0

_LUMINANCE = np.array([0.2126, 0.7152, 0.0722], np.float32)


# ─── Values ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ControlBinding:
    """What a `ControlSpec.target` resolves to, with no write performed.

    Intake resolves the target against the scene and stage it was given; the
    renderer reads `kind` and applies the value to its own state. `kind` is
    one of `renderer`, `mtlx`, `material`, `usd`, or `inert` — the last
    carrying `reason` for a target that could not be resolved, so a bad
    declaration leaves the widget present-but-dead instead of raising.
    """

    kind: str
    param_path: str = ""          # renderer / mtlx: dotted parameter path
    material_id: int = -1         # material: index into `Scene.materials`
    input_name: str = ""          # material: override key
    attribute: object = None      # usd: the resolved live `Usd.Attribute`
    reason: str = ""              # inert: why it did not resolve


@dataclass
class TimeSample:
    """Animated stage state re-read at one time code.

    Every array is already in the renderer's stored form: transforms are
    post-multiplied by the up-axis correction and light/camera vectors are
    rotated by it, so the caller applies the sample verbatim.

    `read_lights` / `read_camera` say whether that part of the stage was read
    at all. They are not the same as "the lists came back empty": a caller that
    skipped lights must leave the scene's lights alone, not clear them.
    """

    # A frame number, or the `Usd.TimeCode` the caller passed.
    time_code: object
    instance_transforms: dict[str, np.ndarray] = field(default_factory=dict)
    read_lights: bool = False
    lights_dir: list = field(default_factory=list)
    lights_sphere: list = field(default_factory=list)
    read_camera: bool = False
    camera_override: Optional[CameraOverride] = None


@dataclass
class SceneUpdate:
    """One scene change, ready for `Renderer.apply_scene_update`.

    A full load replaces everything; a streamed batch replaces the metadata
    and leaves `pending_prims` for the caller to bake; a post-edit resync
    replaces geometry while preserving renderer-side runtime state. One type,
    three constructors — `read_stage`, `read_open_stage` and `adopt_scene`
    build them; nothing else should fill the per-trigger flags by hand.
    """

    scene: Scene

    # Stage-derived state the renderer takes ownership of. `None` means
    # "this update carries no opinion" — the renderer keeps what it has.
    stage: Optional["Usd.Stage"] = None
    scene_graph: object = None
    anim_index: Optional[AnimationIndex] = None
    clock: Optional[PlaybackClock] = None
    up_axis_rt: Optional[np.ndarray] = None
    skeletal: Optional[SkeletalScene] = None
    controls: Optional[list[ControlSpec]] = None

    # Mesh sources still to bake, as `(source, transform, material_id)`. The
    # streaming path bakes these off-thread and appends the results; a
    # synchronous read leaves this empty because it baked inline.
    pending_prims: list = field(default_factory=list)

    # Whether the six stage-derived fields above are an opinion about the WHOLE
    # stage, so `None` means "this stage has none" rather than "not read".
    # True for the triggers that replace the loaded stage. It is the difference
    # between a failed index build clearing the animation state and it leaving
    # the *previous* stage's index in place, pointed at a stage that is gone.
    replaces_stage_state: bool = False

    # ── Per-trigger steps (baseline.md verdict table) ──
    # Label to append to `Renderer.models` when this update should enter the
    # USD-active state; `None` when the renderer is already in it.
    activate_label: Optional[str] = None
    # Adopt `scene.mm_per_unit`. Off for the re-callable synchronous path,
    # whose headless callers set the scale themselves.
    adopt_mm_per_unit: bool = False
    # Preserve instance-enabled / light-enabled flags and live material
    # overrides across the replacement. On for a resync only: on a full load
    # the previous scene's authored overrides would win over the newly
    # authored ones, since `parameter_overrides` mixes both.
    carry_runtime_state: bool = False
    # "always" | "if_first_or_authored" | "never"
    frame_camera: str = "never"
    # Apply authored `skinny:ui:default` values. Load-time only; re-applying
    # on a resync would clobber the user's later edits.
    apply_control_defaults: bool = False

    @classmethod
    def streamed(
        cls,
        scene: Scene,
        *,
        stage,
        scene_graph=None,
        anim_index=None,
        clock=None,
        up_axis_rt=None,
        skeletal=None,
        controls=None,
        pending_prims=(),
    ) -> "SceneUpdate":
        """Initial interactive load: the stage is new and owns everything."""
        return cls(
            scene=scene, stage=stage, scene_graph=scene_graph,
            anim_index=anim_index, clock=clock, up_axis_rt=up_axis_rt,
            skeletal=skeletal, controls=list(controls or []),
            pending_prims=list(pending_prims),
            replaces_stage_state=True,
            adopt_mm_per_unit=True,
            frame_camera="always",
            apply_control_defaults=True,
        )

    @classmethod
    def adopted(cls, scene: Scene, *, stage=None) -> "SceneUpdate":
        """Synchronous re-callable adoption (headless / parity harness).

        Re-callable per frame with a caller-mutated stage, so it must not
        adopt the scale, must not re-apply control defaults, and frames the
        camera only on the first call or when the scene authors one.
        """
        return cls(
            scene=scene, stage=stage,
            activate_label="USD: (headless)",
            frame_camera="if_first_or_authored",
        )

    @classmethod
    def resynced(cls, scene: Scene, *, stage, scene_graph=None) -> "SceneUpdate":
        """Post-edit re-read: replace geometry, keep the user's runtime state
        and the camera exactly where they left it."""
        return cls(
            scene=scene, stage=stage, scene_graph=scene_graph,
            carry_runtime_state=True,
            frame_camera="never",
        )

    @classmethod
    def replacing(
        cls, scene: Scene, *, stage, scene_graph=None, label: str,
        anim_index=None, clock=None, up_axis_rt=None, skeletal=None,
        controls=None,
    ) -> "SceneUpdate":
        """Force-replace: a brand-new stage supersedes whatever was loaded.

        Unlike a resync there is no outgoing scene worth preserving, and the
        new stage's physical scale must win — otherwise a replaced scene would
        leave later content rendering at the previous stage's scale.
        """
        return cls(
            scene=scene, stage=stage, scene_graph=scene_graph,
            anim_index=anim_index, clock=clock, up_axis_rt=up_axis_rt,
            skeletal=skeletal, controls=list(controls or []),
            replaces_stage_state=True,
            activate_label=label,
            adopt_mm_per_unit=True,
            frame_camera="never",
        )

    @property
    def adopts_mm_per_unit(self) -> bool:
        """Whether `scene.mm_per_unit` is a real opinion worth adopting."""
        return (
            self.adopt_mm_per_unit
            and float(self.scene.mm_per_unit) != _MM_PER_UNIT_SENTINEL
        )


# ─── Reading a stage ──────────────────────────────────────────────────


def _stage_state(stage) -> tuple:
    """Animation index, clock, up-axis rotation, skeletal handle, controls.

    Every failure collapses to the inert defaults, matching the pre-change
    behaviour where one bad stage did not take the whole load down.
    """
    try:
        index = build_animation_index(stage)
        return (
            index,
            build_playback_clock(stage, index),
            _up_axis_rt(str(UsdGeom.GetStageUpAxis(stage))),
            extract_skeletal_bindings(stage),
            extract_ui_controls(stage),
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[skinny] animation index build failed: {exc}")
        return (None, PlaybackClock(), None, None, [])


def _build_graph(stage, scene):
    from skinny.scene_graph import build_scene_graph
    try:
        return build_scene_graph(stage, scene)
    except Exception as exc:  # noqa: BLE001
        import traceback
        print(f"[skinny] scene graph build failed: {exc}")
        traceback.print_exc()
        return None


def read_stage(
    stage_path: Path,
    *,
    use_usd_mtlx_plugin: bool = False,
    build_graph: bool = True,
) -> SceneUpdate:
    """Open a stage from disk and return its metadata update, unbaked.

    Safe to call off the render thread: it touches no renderer state and the
    returned update is applied later, on the render thread. Meshes are left in
    `pending_prims` so the caller can bake them in a pool and stream the
    results in.
    """
    scene, prim_data, stage = _read_usd_stage(
        stage_path, use_usd_mtlx_plugin=use_usd_mtlx_plugin, keep_stage=True,
    )
    index, clock, rt, skeletal, controls = (
        _stage_state(stage) if stage is not None
        else (None, PlaybackClock(), None, None, [])
    )
    return SceneUpdate.streamed(
        scene,
        stage=stage,
        scene_graph=(
            _build_graph(stage, scene)
            if build_graph and stage is not None else None
        ),
        anim_index=index, clock=clock, up_axis_rt=rt,
        skeletal=skeletal, controls=controls,
        pending_prims=prim_data,
    )


def read_open_stage(
    stage,
    *,
    time: Optional["Usd.TimeCode"] = None,
    use_usd_mtlx_plugin: bool = False,
    allow_empty: bool = False,
    build_graph: bool = True,
    replaces: Optional[str] = None,
) -> SceneUpdate:
    """Re-read a stage the caller already owns, fully baked.

    This is the post-edit path: meshes are baked inline (the cache makes
    unchanged prims free) so the returned scene is complete.

    Pass `replaces=<model label>` when the stage is brand new rather than an
    edited version of the loaded one — a force-replace keeps no runtime state
    and adopts the new stage's scale.
    """
    scene, prim_data, _ = _read_open_stage(
        stage, time=time, use_usd_mtlx_plugin=use_usd_mtlx_plugin,
        allow_empty=allow_empty,
    )
    cache_index = load_cache_index()

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=4) as pool:
        scene.instances.extend(pool.map(
            lambda pd: bake_usd_prim(pd[0], pd[1], pd[2], cache_index),
            prim_data,
        ))
    graph = _build_graph(stage, scene) if build_graph else None
    if replaces is not None:
        index, clock, rt, skeletal, controls = _stage_state(stage)
        return SceneUpdate.replacing(
            scene, stage=stage, scene_graph=graph, label=replaces,
            anim_index=index, clock=clock, up_axis_rt=rt,
            skeletal=skeletal, controls=controls,
        )
    return SceneUpdate.resynced(scene, stage=stage, scene_graph=graph)


def adopt_scene(scene: Scene, *, stage=None) -> SceneUpdate:
    """Wrap an already-loaded `Scene` for synchronous adoption."""
    return SceneUpdate.adopted(scene, stage=stage)


def bake_pending(pending_prims, *, max_workers: int = 4):
    """Bake a `SceneUpdate`'s `pending_prims`, yielding instances as they land.

    Completion order, not authored order — the consumer streams each instance
    in as it arrives. One prim that fails to bake is reported and skipped; the
    rest of the scene still loads.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    cache_index = load_cache_index()
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(bake_usd_prim, src, xform, mat_id, cache_index): src.name
            for src, xform, mat_id in pending_prims
        }
        for fut in as_completed(futures):
            try:
                yield fut.result()
            except Exception as exc:  # noqa: BLE001
                print(f"[skinny] USD bake failed for {futures[fut]}: {exc}")


# ─── Reading a stage at a time code ───────────────────────────────────


def read_at_time(
    stage,
    time_code,
    *,
    up_axis_rt: Optional[np.ndarray] = None,
    xform_paths: Optional[list[str]] = None,
    want_lights: bool = True,
    want_camera: bool = True,
) -> TimeSample:
    """Re-read the animated part of a stage at `time_code`.

    Replaces the renderer's per-frame re-derivation from loader internals.
    `xform_paths` selects which prims to resolve a world transform for; pass
    the animated subset for playback, or every instance path for a live
    re-read after a raw USD edit.

    `time_code` is a frame number, or a `Usd.TimeCode` when the caller means
    the default time. It is not a float there: `Usd.TimeCode.Default()` is a
    sentinel whose `GetValue()` is NaN, so rounding it through a float would
    silently ask for frame NaN instead.

    Every extraction below resolves at `time_code`, emission included — see
    `_light_color_radiance`, which requires a time code precisely so a missed
    call site is a TypeError rather than a plausible-looking schema fallback
    (change `light-emission-time-sampling`).
    """
    time = (
        time_code if isinstance(time_code, Usd.TimeCode)
        else Usd.TimeCode(float(time_code))
    )
    rt = up_axis_rt
    rt4 = None
    if rt is not None:
        rt4 = np.eye(4, dtype=np.float32)
        rt4[:3, :3] = rt

    transforms: dict[str, np.ndarray] = {}
    for path in xform_paths or ():
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            continue
        m = _world_transform(prim, time)
        transforms[path] = (
            (m @ rt4).astype(np.float32) if rt4 is not None else m
        )

    lights_dir: list[LightDir] = []
    lights_sphere: list = []
    if want_lights:
        for prim in stage.Traverse():
            if not prim.IsActive() or prim.IsAbstract():
                continue
            if prim.IsA(UsdLux.DistantLight):
                ld = _extract_distant_light(prim, time)
                if ld is not None:
                    if rt is not None:
                        ld.direction = (ld.direction @ rt).astype(np.float32)
                    lights_dir.append(ld)
            elif prim.IsA(UsdLux.SphereLight):
                ls = _extract_sphere_light(prim, time)
                if ls is not None:
                    if rt is not None:
                        ls.position = (ls.position @ rt).astype(np.float32)
                    lights_sphere.append(ls)

    camera = None
    if want_camera:
        camera = _extract_camera(stage, time)
        if camera is not None and rt is not None:
            camera.position = (camera.position @ rt).astype(np.float32)
            camera.forward = (camera.forward @ rt).astype(np.float32)

    return TimeSample(
        time_code=time_code,
        instance_transforms=transforms,
        read_lights=want_lights,
        lights_dir=lights_dir,
        lights_sphere=lights_sphere,
        read_camera=want_camera,
        camera_override=camera,
    )


def deform_skinned_mesh(
    binding: SkinnedMeshBinding, source: MeshSource, time: float
) -> MeshSource:
    """Linear-blend-skin `source`'s rest points at `time`.

    Returns a copy of the rest `MeshSource` with deformed positions and
    re-smoothed normals, in the same (geomBind-relative) space as the rest
    points — so the instance's existing TLAS transform still places it.
    """
    mats = compute_joint_matrices(binding, time)
    deformed = lbs_points(
        binding.rest_points, binding.joint_indices, binding.joint_weights, mats,
    )
    return replace(
        source,
        positions=deformed.astype(np.float32),
        normals=_smooth_normals(deformed, source.tri_idx),
    )


def dome_light_intensity(prim, time) -> float:
    """Scalar intensity of a DomeLight prim at `time`: its colour × intensity
    × 2^exposure folded to luminance, matching `_extract_dome_light`.

    `time` is required for the same reason `_light_color_radiance` requires it:
    a dome whose intensity is time-sampled has no value at the default time
    code and would silently resolve to the schema fallback.
    """
    rad = _light_color_radiance(UsdLux.LightAPI(prim), time)
    return float(np.dot(rad, _LUMINANCE))


def read_lens_file(path: Path) -> Optional[LensSystem]:
    """Read a `.usda` lens definition and return its lens system.

    The file may be a bare lens prim (a top-level `Xform` with
    `skinny:lens:*` children) or any prim path; the first prim whose
    children carry a `skinny:lens:role` wins. Returns `None` when the file
    cannot be opened or holds no lens elements.
    """
    p = Path(path)
    if not p.is_file():
        print(f"[skinny] lens load failed: {p} not found", flush=True)
        return None
    try:
        stage = Usd.Stage.Open(str(p))
    except Exception as exc:  # noqa: BLE001
        print(f"[skinny] lens load failed to open stage: {exc}", flush=True)
        return None
    if stage is None:
        return None
    for prim in stage.Traverse():
        if not prim.IsActive() or prim.IsAbstract():
            continue
        ls = _extract_lens_system(prim, Usd.TimeCode.Default())
        if ls is not None:
            return ls
    return None


# ─── Control bindings ─────────────────────────────────────────────────


def _inert(reason: str) -> ControlBinding:
    import logging
    logging.getLogger(__name__).warning("skinny:ui control inert: %s", reason)
    return ControlBinding(kind="inert", reason=reason)


def resolve_control_binding(
    spec: ControlSpec, *, scene: Optional[Scene] = None, stage=None,
) -> ControlBinding:
    """Resolve `spec.target` to a description the renderer can apply.

    Intake does the lookup — which material index, which USD attribute — and
    returns it. It performs no write and holds no renderer reference; an
    unresolvable target returns an inert binding plus a warning rather than
    raising, so a bad declaration leaves the widget present-but-dead.
    """
    kind, _, rest = spec.target.partition(":")

    if kind == "renderer":
        return ControlBinding(kind="renderer", param_path=rest)

    if kind == "mtlx":
        return ControlBinding(kind="mtlx", param_path="mtlx." + rest)

    if kind == "material":
        mat_name, _, inp = rest.partition(":")
        if not inp:
            return _inert(f"material target {rest!r} missing input")
        mats = getattr(scene, "materials", None) or []
        mat_id = next(
            (i for i, m in enumerate(mats)
             if getattr(m, "name", None) == mat_name
             or getattr(m, "mtlx_target_name", None) == mat_name),
            None,
        )
        if mat_id is None:
            return _inert(f"material {mat_name!r} not found")
        return ControlBinding(
            kind="material", material_id=mat_id, input_name=inp,
        )

    if kind == "usd":
        prim_path, _, attr_name = rest.rpartition(".")
        if stage is None or not prim_path or not attr_name:
            return _inert(f"usd target {rest!r} unresolvable")
        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            return _inert(f"usd prim {prim_path!r} not found")
        attr = prim.GetAttribute(attr_name)
        if not attr or not attr.IsValid():
            return _inert(f"usd attr {attr_name!r} not found on {prim_path}")
        return ControlBinding(kind="usd", attribute=attr)

    return _inert(f"unknown target prefix in {spec.target!r}")
