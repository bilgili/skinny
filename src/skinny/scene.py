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
class TextureBinding:
    """USD UsdUVTexture node parameters resolved per shader input.

    Captures the per-connection authoring data that affects sampling:
    file path, scale/bias remap (encodes the OpenGL-vs-DirectX normal
    convention and unorm→signed remap for non-color data), the channel
    selector from `outputs:r|g|b|a|rgb`, sourceColorSpace ("sRGB" vs
    "raw"), and per-texture wrap modes from `inputs:wrapS/wrapT`.

    Defaults match the UsdPreviewSurface UsdUVTexture spec so callers
    that only have a path can still construct a sensible binding.
    """

    path: Path
    bias: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    scale: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    channel: str = "rgb"          # one of "rgb", "r", "g", "b", "a"
    source_color_space: str = "auto"  # "sRGB", "raw", "auto"
    wrap_s: str = "repeat"        # "repeat", "clamp", "mirror", "black"
    wrap_t: str = "repeat"
    # UsdTransform2d applied to `st` upstream of this UsdUVTexture, as
    # (scale_x, scale_y, translation_x, translation_y, rotation_degrees).
    # None means identity/absent. glTF-derived USD authors the V-flip
    # (1, -1, 0, 1, 0) here. The loader bakes this into mesh UVs at load
    # time (usd-texture-intake), so the renderer never reads it.
    uv_transform: Optional[
        tuple[float, float, float, float, float]
    ] = None


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
    # Per-input UsdUVTexture node settings (scale/bias/channel/wrap). Same
    # keys as `texture_paths`; loaders populating one should populate both
    # so the renderer can honour DirectX-style normal Y flips, alpha vs
    # red channel selection for opacity/roughness, and per-texture wrap
    # modes. Materials authored before this field existed continue to work
    # — missing entries fall back to the UsdUVTexture defaults.
    texture_bindings: dict[str, "TextureBinding"] = field(default_factory=dict)
    # When set, the renderer's per-scene-material gen path resolves this
    # name in the loaded MaterialLibrary (instead of authoring a fresh
    # standard_surface from `parameter_overrides`). Authors opt in via
    # USD: `customData = { string skinnyMaterialX = "M_skinny_skin_default" }`
    # on a UsdShade.Material prim. Lets a USD scene route a material
    # binding to a hand-authored MaterialX network shipped in skinny's
    # mtlx library.
    mtlx_target_name: Optional[str] = None

    # Author hint pointing at a Python-authored slangpile material module,
    # e.g. `customData = { string python_module = "python_materials.preview_surface_material" }`
    # on a UsdShade.Material prim. The Python material editor reads this
    # to know which source file to load for the active scene.
    python_module: Optional[str] = None

    # Editable-input descriptors for a synthesized/graph MaterialX material
    # (mcp-material-authoring, design D5): `{logical name: {uniforms, type,
    # default, range}}` (a legacy `{name: [uniforms]}` list is upgraded on load).
    # Read off the `<name>.json` sidecar beside the `.mtlx` file by
    # `_load_mtlx_materials`, or synthesized as identity descriptors for a curated
    # GRAPH preset. **Empty for constant-shader presets** (chrome/glass/jade) and
    # plain USD materials — those surface their `parameter_overrides` keys directly
    # on the scene-graph node instead (a constant preset advertises the flat-pack
    # keys `pack_flat_material` reads; finding B). The scene graph surfaces these
    # descriptor keys as editable properties and `scene_set` fans a single logical
    # edit out to every uniform the input controls.
    logical_inputs: dict[str, object] = field(default_factory=dict)

    # The bound material prim's full stage path (e.g. ``/ScopeA/Foo``), set at
    # load. A stable, globally-unique identity — unlike ``name`` (the prim leaf),
    # which collides across scopes (``/ScopeA/Foo`` vs ``/ScopeB/Foo``). Used to
    # key the override-preservation snapshot across a structural resync so live
    # edits are re-applied to the right same-pathed material, not cross-applied
    # to a same-named one (mcp-material-authoring, finding #7/D). ``None`` for the
    # flat fallback and any material loaded before this field existed.
    source_prim_path: Optional[str] = None


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
    # Full USD prim path this instance was baked from (e.g. "/World/Head").
    # Stable identity used by the runtime scene-graph editing API to target
    # add/remove/transform operations. Empty for non-USD (OBJ/SDF) instances.
    prim_path: str = ""
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
    # Originating USD prim path (e.g. "/World/Sun"). Stable identity used to
    # preserve the runtime `enabled` flag when lights are re-read from the stage
    # during a scene-graph edit. Empty for synthesized/non-USD lights.
    prim_path: str = ""
    # Authored illuminant SPD (spectral mode, Group 6.3): the resampled
    # 360-830/5 nm (95-sample) radiance from a pbrt `spectrum L`, preserved on
    # the light prim's `skinnyOverrides["spectral"]`. None for a plain-RGB light
    # (the spectral path then upsamples `radiance`). Only consumed under
    # `--spectral`; ignored by the RGB build.
    spectral_spd: "np.ndarray | None" = None

    def __post_init__(self) -> None:
        self.direction = np.asarray(self.direction, np.float32).reshape(3)
        self.radiance = np.asarray(self.radiance, np.float32).reshape(3)
        if self.spectral_spd is not None:
            self.spectral_spd = np.asarray(self.spectral_spd, np.float32).reshape(-1)


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
    # Authored chromaticity + scalar intensity kept alongside the combined
    # `radiance` so the scene-graph editor can adjust either independently
    # without losing the other (radiance alone collapses the color/intensity
    # split). ``radiance`` stays the source of truth at render time.
    color: np.ndarray = field(default_factory=lambda: np.ones(3, np.float32))
    intensity: float = 1.0
    # Originating USD prim path; see LightDir.prim_path.
    prim_path: str = ""

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, np.float32).reshape(3)
        self.radiance = np.asarray(self.radiance, np.float32).reshape(3)
        self.color = np.asarray(self.color, np.float32).reshape(3)


@dataclass
class LensElement:
    """One spherical interface in a thick-lens stack (PBRT § 6.4).

    Authored under a `UsdGeom.Camera` as a child `UsdGeom.Xform` with
    `skinny:lens:*` custom attributes. All distances are in millimetres
    in the lens's intrinsic frame; the renderer converts to world units
    via `Scene.mm_per_unit` before upload.

    Sign convention: `radius > 0` ⇒ centre of curvature on the scene
    side of the surface; `radius < 0` ⇒ centre on the film side;
    `radius == 0` ⇒ planar interface (used for the aperture stop).
    `thickness` is the axial distance from this surface to the **next**
    one toward the film (the rearmost element's `thickness` is the film
    distance). `ior` is the refractive index of the medium **after**
    this surface in the film direction (`1.0` for air gaps).
    """

    radius_mm: float
    thickness_mm: float
    ior: float
    aperture_mm: float            # clear aperture diameter
    is_aperture_stop: bool = False
    enabled: bool = True

    @property
    def half_aperture_mm(self) -> float:
        return 0.5 * float(self.aperture_mm)


@dataclass
class LensSystem:
    """Ordered stack of `LensElement`s describing one camera's optics.

    Elements are stored front-to-rear (index 0 = scene-side surface,
    index N-1 = film-side surface) per PBRT convention. Backward-compat
    with the existing pinhole pipeline: a `LensSystem` with no enabled
    elements collapses to pinhole (the renderer skips the lens path).
    """

    elements: list[LensElement] = field(default_factory=list)
    enabled: bool = True

    @property
    def active_elements(self) -> list[LensElement]:
        return [e for e in self.elements if e.enabled] if self.enabled else []

    @property
    def film_distance_mm(self) -> float:
        return sum(float(e.thickness_mm) for e in self.active_elements)

    @property
    def rear_aperture_mm(self) -> float:
        active = self.active_elements
        return float(active[-1].aperture_mm) if active else 0.0

    def signature(self) -> tuple:
        """Stable identity tuple for accumulation-reset detection."""
        if not self.enabled:
            return ("lens", "off")
        return (
            "lens",
            tuple(
                (
                    float(e.radius_mm),
                    float(e.thickness_mm),
                    float(e.ior),
                    float(e.aperture_mm),
                    bool(e.is_aperture_stop),
                    bool(e.enabled),
                )
                for e in self.elements
            ),
        )


@dataclass
class CameraOverride:
    """Authored camera viewpoint extracted from a USD UsdGeom.Camera.

    `position` is the camera's world-space origin; `forward` is the unit
    direction the camera looks down (USD convention: local -Z transformed
    to world). `focal_length_mm` and `vertical_aperture_mm` come straight
    from the USD attributes and convert to a vertical FOV via
    `2·atan(0.5·va / fl)`. When `focus_distance` is None, the renderer
    picks a sensible distance from scene bounds. `lens` carries an
    optional thick-lens stack authored as child prims; when present and
    enabled the renderer's pinhole ray-gen path is replaced with a
    PBRT-style realistic camera trace.

    `mirrored` marks an improper (orientation-reversing) pbrt camera — a
    camera-to-world transform with negative determinant, e.g. a `Scale -1 1 1`
    before `LookAt`. The importer reconstructs a right-handed camera and drops
    the reflection, so the renderer reproduces it as a horizontal screen-space
    mirror (negate `ndc.x` at ray-gen) to match the pbrt reference. It does not
    change `position`/`forward`/the camera basis — only the image is mirrored
    left↔right.
    """

    position: np.ndarray            # (3,) float32, world space
    forward: np.ndarray             # (3,) float32, unit length
    # Authored world-space up (camera local +Y → world). Honored by the renderer
    # so non-Y-up pbrt cameras (the Z-up convention) keep their roll. Defaults to
    # +Y, so Y-up cameras and any override built without an up are byte-identical.
    up: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0, 0.0], np.float32))
    focal_length_mm: float = 50.0
    vertical_aperture_mm: float = 24.0
    focus_distance: Optional[float] = None
    fstop: float = 0.0              # 0 ⇒ wide open; from USD `fStop` attr
    lens: Optional[LensSystem] = None
    enabled: bool = True
    mirrored: bool = False          # improper pbrt camera ⇒ horizontal screen mirror
    # pbrt film exposure controls (change pbrt-radiometric-parity). The imaging
    # ratio exposure_time·iso/100 is a live linear output scale applied by the
    # renderer (NOT baked into emitters at import), so ISO/exposure retune on the
    # fly. Defaults (iso=100, exposure_time=1) ⇒ ratio 1.0 ⇒ byte-identical render.
    iso: float = 100.0
    exposure_time: float = 1.0

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, np.float32).reshape(3)
        self.forward = np.asarray(self.forward, np.float32).reshape(3)
        n = float(np.linalg.norm(self.forward))
        if n > 1e-6:
            self.forward = self.forward / n
        self.up = np.asarray(self.up, np.float32).reshape(3)
        un = float(np.linalg.norm(self.up))
        self.up = self.up / un if un > 1e-6 else np.array([0.0, 1.0, 0.0], np.float32)


@dataclass
class LightEnvHDR:
    """HDR environment map (equirectangular RGBA32F)."""

    name: str
    data: np.ndarray  # (H, W, 4) float32
    intensity: float = 1.0
    enabled: bool = True


# ─── Volumes ──────────────────────────────────────────────────────────


@dataclass
class VolumeGrid:
    """One dense density grid decoded from a ``UsdVol.Volume`` prim
    (nanovdb-volume-rendering).

    ``density`` keeps the reader's ``(nx, ny, nz)`` layout (``density[a, b, c]``
    is the voxel at grid index ``index_min + (a, b, c)``); the renderer
    transposes to the GPU's ``(depth, height, width) = (nz, ny, nx)`` order at
    upload. ``world_to_uvw`` is the fully folded affine map — rows 0..2 of the
    math-convention (column-vector) matrix taking a skinny world-space point to
    normalized ``[0, 1]³`` texture coordinates of the density texture:
    ``uvw[r] = dot(world_to_uvw[r, :3], p) + world_to_uvw[r, 3]``. It already
    composes prim-xform⁻¹, the pbrt→USD axis flip B, the grid's index→medium
    map, the ``index_min`` origin shift, and the +0.5 voxel-center offset — the
    shader does a single affine transform per density lookup.
    """

    density: np.ndarray                  # float32 (nx, ny, nz), values >= 0
    value_max: float                     # decode-time max; normalizer for R16F upload
    index_min: tuple[int, int, int]      # grid ijk of density[0, 0, 0]
    dims: tuple[int, int, int]           # (nx, ny, nz) == density.shape
    world_to_uvw: np.ndarray             # (3, 4) float32, math (column-vector) rows
    asset_path: str = ""                 # resolved .nvdb path (state-hash identity)
    field_name: str = "density"
    prim_path: str = ""


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
    # True when the source USD stage contains an active, supported authored
    # lighting source, regardless of its current intensity/visibility. This is
    # source-authority metadata, not a "currently emits power" test: it
    # survives Rect/Disk conversion to emissive geometry and lets the renderer
    # decide whether its built-in DistantLight + IBL fallback pair is allowed.
    has_authored_lighting: Optional[bool] = None
    camera_override: Optional[CameraOverride] = None

    # Scene-scale bridge: 1 world unit = `mm_per_unit` millimetres. Used
    # to reconcile σ (mm⁻¹) with ray distances (world units) in the
    # volume march.
    mm_per_unit: float = 120.0

    # Film per-sample radiance clamp (pbrt `Film "maxcomponentvalue"`, change
    # film-maxcomponent-clamp). 0 = disabled. When >0 the renderer clamps each
    # sample's radiance proportionally before accumulation, matching pbrt's
    # RGBFilm firefly suppression so an imported pbrt scene reproduces its EXR.
    film_max_component: float = 0.0

    # Heterogeneous participating medium (nanovdb-volume-rendering): the ONE
    # density grid decoded from the stage's `UsdVol.Volume` prim, or None.
    # Exactly one grid per scene is supported for now (loader warns and keeps
    # the first when a stage authors several).
    volume_grid: Optional[VolumeGrid] = None

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


def scene_has_emissive_instances(scene) -> bool:
    """Whether any authored instance is bound to a non-zero emissive material."""
    materials = getattr(scene, "materials", []) or []
    for inst in getattr(scene, "instances", []) or []:
        material_id = getattr(inst, "material_id", -1)
        if not (0 <= material_id < len(materials)):
            continue
        value = materials[material_id].parameter_overrides.get("emissiveColor")
        if value is None:
            continue
        try:
            if any(float(value[i]) > 0.0 for i in range(3)):
                return True
        except (IndexError, TypeError, ValueError):
            continue
    return False


def scene_has_authored_lighting(scene) -> bool:
    """Source-authority predicate, independent of current emitted power."""
    authored = getattr(scene, "has_authored_lighting", None)
    if authored is not None:
        return bool(authored)
    return bool(
        (getattr(scene, "lights_dir", []) or [])
        or (getattr(scene, "lights_sphere", []) or [])
        or getattr(scene, "environment", None) is not None
        or scene_has_emissive_instances(scene)
    )


def scene_uses_default_lights(usd_scene, *, usd_active: bool) -> bool:
    """Whether Skinny's fallback pair owns lighting for the active scene."""
    return not (
        usd_active
        and usd_scene is not None
        and scene_has_authored_lighting(usd_scene)
    )


def scene_environment_for_authority(
    usd_scene,
    fallback_environment,
    *,
    uses_default_lights: bool,
):
    """Select the environment owned by the active lighting authority."""
    if uses_default_lights:
        return fallback_environment
    if usd_scene is None:
        return None
    return getattr(usd_scene, "environment", None)


def scene_auxiliary_lights_for_authority(
    usd_scene,
    *,
    uses_default_lights: bool,
) -> tuple[list, object | None]:
    """Select authored sphere and emissive sources for the active authority."""
    if uses_default_lights or usd_scene is None:
        return [], None
    return list(getattr(usd_scene, "lights_sphere", []) or []), usd_scene


def environment_contribution_intensity(environment) -> float:
    """Return an environment's live contribution, with no implicit fallback."""
    if environment is None or not getattr(environment, "enabled", True):
        return 0.0
    return float(getattr(environment, "intensity", 0.0))


def select_powered_distant_lights(lights, *, authority_enabled: bool = True) -> list:
    """Select distant lights that currently contribute under the active authority."""
    if not authority_enabled:
        return []

    def _has_power(light) -> bool:
        intensity = getattr(light, "intensity", None)
        if intensity is not None and float(intensity) == 0.0:
            return False
        radiance = np.asarray(
            getattr(light, "radiance", (0.0, 0.0, 0.0)),
            dtype=np.float32,
        )
        return bool(np.any(radiance > 0.0))

    return [
        light
        for light in lights
        if getattr(light, "enabled", True) and _has_power(light)
    ]


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
        has_authored_lighting=False,
        mm_per_unit=float(mm_per_unit),
        furnace_mode=bool(furnace_mode),
    )
