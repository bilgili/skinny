"""Scene description for skinny — CPU-side data classes.

A `Scene` bundles the per-frame inputs the renderer needs:
mesh instances (geometry + transform + material), materials, lights,
environment, and the world-scale bridge. The renderer consumes one Scene
per frame; today's UI inputs (head index, env index, skin sliders, etc.)
materialize into a Scene through `build_default_scene()`.

Phase B-1 deliverable: data shape only. The renderer.py hookup, GPU TLAS
buffer, shader-side broad-phase, and per-triangle material_id lookup land
in subsequent Phase B steps. Phase C extends `Material` with a MaterialX
document reference and parameter overrides.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from skinny.environment import Environment
from skinny.mesh import Mesh, MeshSource
from skinny.tattoos import Tattoo


# ─── Materials ────────────────────────────────────────────────────────


@dataclass
class Material:
    """Material binding for one mesh instance.

    Phase B-1: a marker pointing at the active SkinParameters and an
    optional pigment overlay (today's tattoo). The renderer continues to
    look up SkinParameters off `Renderer.skin` directly; this field is the
    pipe through which Phase C will deliver MaterialX-driven values.

    Reserved for Phase C+:
    - `mtlx_document`: parsed MaterialX `Document` describing the surface
      and volume shaders.
    - `parameter_overrides`: per-instance scalars/colours that override
      defaults declared in the MaterialX nodedef (e.g. a melanin slider).
    - `texture_paths`: image references pulled from `<image>` nodes; the
      renderer resolves these into the bindless texture array.
    """

    name: str = "skin"
    # Phase B-1: pigment overlay carried as the existing Tattoo struct.
    # Phase C: replaced by a MaterialX `<image>` connected to the dermis
    # nodedef's `pigment` input.
    pigment: Optional[Tattoo] = None
    pigment_density: float = 1.0

    # Phase C placeholders (kept absent for now so we don't carry
    # half-implemented APIs).
    mtlx_document: Optional[object] = None  # MaterialX.Document when wired
    parameter_overrides: dict[str, object] = field(default_factory=dict)
    # Texture inputs by shader-input name. Phase D-3 fills this from
    # UsdUVTexture nodes connected to the surface shader's inputs;
    # Phase C-4 maps each entry to a slot in the bindless sampler array.
    texture_paths: dict[str, Path] = field(default_factory=dict)
    # When set, the renderer's per-scene-material gen path resolves this
    # name in the loaded MaterialLibrary (instead of authoring a fresh
    # standard_surface from `parameter_overrides`). Authors opt in via
    # USD: `customData = { string skinnyMaterialX = "M_skinny_skin_default" }`
    # on a UsdShade.Material prim. Lets a USD scene route a material
    # binding to a hand-authored MaterialX network shipped in skinny's
    # mtlx library.
    mtlx_target_name: Optional[str] = None


# ─── Geometry ─────────────────────────────────────────────────────────


@dataclass
class MeshInstance:
    """One renderable mesh: a (BLAS, transform, material) triple.

    `mesh` is a baked `Mesh` ready for upload (vertex_bytes, index_bytes,
    bvh_bytes). `transform` is a 4x4 model→world matrix. `material_id`
    indexes into `Scene.materials`.

    Today only one instance ever exists per Scene. Phase B-3 lifts the
    single-instance constraint and adds a TLAS over the instance world-
    space AABBs.
    """

    mesh: Mesh
    transform: np.ndarray = field(
        default_factory=lambda: np.eye(4, dtype=np.float32)
    )
    material_id: int = 0
    name: str = ""
    # CPU-side triangle source kept around so the renderer can enumerate
    # emissive triangles for area-light NEE without re-parsing the GPU
    # vertex buffer. Optional because non-USD instances (legacy SDF/OBJ)
    # don't flow through Scene yet.
    source: Optional[MeshSource] = None
    enabled: bool = True

    def __post_init__(self) -> None:
        if self.transform.dtype != np.float32:
            self.transform = self.transform.astype(np.float32)
        if self.transform.shape != (4, 4):
            raise ValueError(
                f"MeshInstance.transform must be 4x4, got {self.transform.shape}"
            )

    def local_aabb(self) -> tuple[np.ndarray, np.ndarray]:
        """Local-space AABB read off the mesh's BVH root node.

        BVH nodes are packed `fff i fff i` (32 bytes); the first three
        floats are aabb_min, the second three (after the int16/16 split
        at byte 12) are aabb_max. Returns zero-extent bounds at origin
        for empty meshes.
        """
        if not self.mesh.bvh_bytes or self.mesh.num_triangles == 0:
            zero = np.zeros(3, np.float32)
            return zero, zero.copy()
        aabb_min = np.array(struct.unpack("fff", self.mesh.bvh_bytes[0:12]), np.float32)
        aabb_max = np.array(struct.unpack("fff", self.mesh.bvh_bytes[16:28]), np.float32)
        return aabb_min, aabb_max

    def world_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """World-space AABB of this instance (used to build the TLAS).

        Walks the eight local-AABB corners through `transform` and returns
        the min/max envelope. The transform is stored in USD/row-vector
        convention (math-transpose), so we apply it as `p_world = p_local @ M`
        rather than `M @ p_local`.
        """
        amin, amax = self.local_aabb()
        corners = np.array([
            [amin[0], amin[1], amin[2], 1.0],
            [amax[0], amin[1], amin[2], 1.0],
            [amin[0], amax[1], amin[2], 1.0],
            [amax[0], amax[1], amin[2], 1.0],
            [amin[0], amin[1], amax[2], 1.0],
            [amax[0], amin[1], amax[2], 1.0],
            [amin[0], amax[1], amax[2], 1.0],
            [amax[0], amax[1], amax[2], 1.0],
        ], dtype=np.float32)
        world = corners @ self.transform
        wmin = world[:, :3].min(axis=0).astype(np.float32)
        wmax = world[:, :3].max(axis=0).astype(np.float32)
        return wmin, wmax


# ─── Lights ───────────────────────────────────────────────────────────


@dataclass
class LightDir:
    """Directional analytic light. Direction points from surface toward source."""

    direction: np.ndarray  # (3,) float32, expected unit length
    radiance: np.ndarray   # (3,) float32, color × intensity in linear HDR
    enabled: bool = True

    def __post_init__(self) -> None:
        self.direction = np.asarray(self.direction, np.float32).reshape(3)
        self.radiance = np.asarray(self.radiance, np.float32).reshape(3)


@dataclass
class LightSphere:
    """Spherical area light (UsdLux.SphereLight).

    Treated as a point light at `position` for direct shading today;
    proper area sampling lives in the upcoming NEE pass. `radius`
    is preserved so the future area-sampling code can disk-sample
    points on the sphere as seen from the shading point.
    """

    position: np.ndarray  # (3,) float32 world-space centre
    radius: float
    radiance: np.ndarray  # (3,) float32, color × intensity in linear HDR
    enabled: bool = True

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, np.float32).reshape(3)
        self.radiance = np.asarray(self.radiance, np.float32).reshape(3)


@dataclass
class CameraOverride:
    """Authored camera viewpoint extracted from a USD UsdGeom.Camera.

    `position` is the camera's world-space origin; `forward` is the unit
    direction the camera looks down (USD convention: local -Z transformed
    to world). `focal_length_mm` and `vertical_aperture_mm` come straight
    from the USD attributes and convert to a vertical FOV via
    `2·atan(0.5·va / fl)`. When `focus_distance` is None, the renderer
    picks a sensible distance from scene bounds.
    """

    position: np.ndarray            # (3,) float32, world space
    forward: np.ndarray             # (3,) float32, unit length
    focal_length_mm: float = 50.0
    vertical_aperture_mm: float = 24.0
    focus_distance: Optional[float] = None
    enabled: bool = True

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, np.float32).reshape(3)
        self.forward = np.asarray(self.forward, np.float32).reshape(3)
        n = float(np.linalg.norm(self.forward))
        if n > 1e-6:
            self.forward = self.forward / n


@dataclass
class LightEnvHDR:
    """HDR environment map (equirectangular RGBA32F)."""

    name: str
    data: np.ndarray  # (H, W, 4) float32
    intensity: float = 1.0
    enabled: bool = True


# ─── Scene ────────────────────────────────────────────────────────────


@dataclass
class Scene:
    """One frame's worth of renderable description.

    Lights are split into typed lists so the renderer can route each kind
    to its preferred estimator (analytic for `lights_dir`; importance-
    sampled IBL + miss radiance for `environment`). When `environment` is
    None, primary-ray misses go to black.
    """

    instances: list[MeshInstance] = field(default_factory=list)
    materials: list[Material] = field(default_factory=list)
    lights_dir: list[LightDir] = field(default_factory=list)
    lights_sphere: list[LightSphere] = field(default_factory=list)
    environment: Optional[LightEnvHDR] = None
    camera_override: Optional[CameraOverride] = None

    # Scene-scale bridge: 1 world unit = `mm_per_unit` millimetres. Used
    # to reconcile σ (mm⁻¹) with ray distances (world units) in the
    # volume march.
    mm_per_unit: float = 120.0

    # Furnace probe: when True the renderer swaps the scene for a unit
    # sphere + white-1 environment + analytic light off, and tints any
    # pixel whose accumulated radiance exceeds 1.0 per channel pink.
    furnace_mode: bool = False

    def primary_material(self) -> Optional[Material]:
        """First material in the scene, if any. Convenience accessor while
        the codebase is still single-material."""
        return self.materials[0] if self.materials else None

    def primary_instance(self) -> Optional[MeshInstance]:
        """First mesh instance, if any. Same convenience as above."""
        return self.instances[0] if self.instances else None

    def world_bounds(self) -> Optional[tuple[np.ndarray, np.ndarray]]:
        """Combined world-space AABB across all instances, or None if empty."""
        if not self.instances:
            return None
        mins: list[np.ndarray] = []
        maxs: list[np.ndarray] = []
        for inst in self.instances:
            wmin, wmax = inst.world_bounds()
            mins.append(wmin)
            maxs.append(wmax)
        amin = np.minimum.reduce(mins).astype(np.float32)
        amax = np.maximum.reduce(maxs).astype(np.float32)
        return amin, amax


# ─── Construction helpers ────────────────────────────────────────────


def build_default_scene(
    *,
    environment: Optional[Environment],
    env_intensity: float,
    mesh: Optional[Mesh],
    transform: Optional[np.ndarray] = None,
    light_direction: Optional[np.ndarray] = None,
    light_radiance: Optional[np.ndarray] = None,
    direct_light_enabled: bool = True,
    mm_per_unit: float = 120.0,
    furnace_mode: bool = False,
) -> Scene:
    """Materialize a Scene from the renderer's current UI state.

    Mirrors the inputs the legacy renderer.py reads off self directly. Pass
    None for `environment` or `mesh` to produce a degenerate scene (used
    for early init before assets are uploaded).

    The single-mesh / single-material / single-light constraint reflects
    Phase B-1's "Scene as a thin wrapper" goal. Multi-instance and multi-
    material scenes are mechanically supported by the data shape.
    """
    instances: list[MeshInstance] = []
    materials: list[Material] = [Material(name="default")]

    if mesh is not None:
        xform = transform if transform is not None else np.eye(4, dtype=np.float32)
        instances.append(MeshInstance(mesh=mesh, transform=xform, material_id=0,
                                       name=mesh.name))

    lights_dir: list[LightDir] = []
    if direct_light_enabled and light_direction is not None and light_radiance is not None:
        lights_dir.append(LightDir(direction=light_direction, radiance=light_radiance))

    env: Optional[LightEnvHDR] = None
    if environment is not None:
        env = LightEnvHDR(
            name=environment.name,
            data=environment.data,
            intensity=float(env_intensity),
        )

    return Scene(
        instances=instances,
        materials=materials,
        lights_dir=lights_dir,
        environment=env,
        mm_per_unit=float(mm_per_unit),
        furnace_mode=bool(furnace_mode),
    )
