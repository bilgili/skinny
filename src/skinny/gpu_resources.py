"""Renderer GPU resource inventory — one declaration per resource.

Before this module the same inventory was described three times in
``renderer.py``, thousands of lines apart: ``_init_gpu`` allocated by attribute
name, ``_create_descriptors`` plus five ``_rebind_*_descriptors`` methods wrote
the Vulkan descriptor sets, ``_build_metal_binds`` re-stated the identical
mapping as Metal shader-global names, and ``cleanup`` destroyed by attribute
name again. Adding a resource meant editing all four and remembering the last
one; a missed line leaked silently because nothing without a device could see
it.

Here each resource is declared **once** (:class:`ResourceDecl`) and that single
declaration carries its allocation inputs, its binding identity on *both*
backends, and its destruction. Backend divergence is confined to one binding
step — :meth:`SceneResourceSet.bind_vulkan` versus
:meth:`SceneResourceSet.metal_binds` — consuming the same declaration list, so
the per-method ``is_metal`` early-returns that used to open every rebind helper
are gone.

Three distinct orders exist in the pre-change renderer and all three are
preserved verbatim, each recorded once here and pinned against a captured
golden by ``tests/test_gpu_resources.py``:

* :data:`DECLARATIONS` — allocation order.
* :data:`VULKAN_WRITE_SEQUENCE` — descriptor-write order (*not* allocation
  order: binding 26 is written between 11 and 12, binding 20 after 24, and
  bindings 1/30/31 last).
* :data:`DESTROY_SEQUENCE` — teardown order.

Binding numbers are literal here on purpose (design D2): ``docs/Architecture.md``
holds the authoritative binding map and ``bindings.slang`` declares the same
numbers. This module relocates them next to the allocation they belong to; it
does not derive them.

One exception, and it runs the other way: the **MLT chain bindings (52–57)** are
DERIVED, from ``wavefront_layout.MLT_CHAIN_BUFFERS`` (change
mlt-binding-declaration). Those buffers are pass-owned, not renderer-owned, and
their Vulkan binding travels with a Metal shader-global name and a size key that
this module never sees — so the pairing is declared once, beside the sizes, and
gated against the shader. See :data:`MLT_BINDINGS` below.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from skinny.wavefront_layout import mlt_binding_numbers

# Descriptor classes. `bindless` is binding 14's TexturePool array, written per
# filled slot rather than as a single descriptor.
UBO = "ubo"
IMAGE = "image"
SAMPLER = "sampler"
BUFFER = "buffer"
BINDLESS = "bindless"


@dataclass(frozen=True)
class ResourceDecl:
    """One GPU resource: how it is allocated, bound, and destroyed.

    ``attr`` is the renderer attribute name it has always had — the ~120
    ``self.<resource>`` reads inside ``renderer.py`` still resolve, through
    properties that forward to the set (design D4).
    """

    attr: str
    kind: str
    """Constructor name on the backend resource module (``vk_compute`` /
    ``metal_compute``), or ``"TexturePool"`` for the bindless pool."""

    args: Callable[["ResourceSizes"], tuple]
    """Positional constructor arguments after ``ctx``, from the size inputs."""

    binding: Optional[int] = None
    """Vulkan descriptor binding, or None when the resource is never bound
    through a descriptor set (the readback staging buffer)."""

    metal: Optional[str] = None
    """Metal shader-global name, or None when Metal reaches the resource some
    other way (the readback buffer; the bindless pool, passed as the separate
    ``bindless`` dispatch argument)."""

    descriptor: Optional[str] = None
    kwargs: Callable[["ResourceSizes"], dict] = lambda s: {}
    spectral: bool = False
    """Allocated only under ``--spectral``; absent from the RGB layout so the
    RGB descriptor set stays byte-identical."""

    size_dependent: bool = False
    """Recreated by ``resize()`` — the viewport-sized images plus readback."""

    metal_sampler: bool = False
    """Metal binds a discrete ``Texture*`` + its own ``SamplerState`` (combined
    ``Sampler2D``/``Sampler3D`` is unsupported there), so this declaration also
    contributes ``<metal>Sampler``."""

    def active(self, spectral: bool) -> bool:
        return spectral or not self.spectral


@dataclass
class ResourceSizes:
    """Allocation inputs the renderer computes and the declarations consume.

    The *formulas* live in the declarations; the *inputs* come from renderer
    state (mesh sources, USD scene extents, capacities) that is not GPU-resource
    business.
    """

    width: int
    height: int
    uniform_bytes: int
    material_capacity: int
    mtlx_skin_slot_bytes: int
    """256 on Metal (each ``float3`` pads to 16 B in MSL and the exact reflected
    stride is not known until the lazy pipeline build), the scalar record size
    on Vulkan."""

    instance_capacity: int
    emissive_tri_capacity: int
    gizmo_capacity: int
    gizmo_stride: int
    lens_element_capacity: int
    lens_element_stride: int
    lens_pupil_capacity: int
    lens_pupil_stride: int
    vertex_bytes: int
    index_bytes: int
    bvh_bytes: int
    env_width: int
    env_height: int
    tattoo_width: int
    tattoo_height: int
    detail_res: int
    neural_weight_bytes: int
    neural_bias_bytes: int
    neural_header_bytes: int
    neural_external: bool
    neural_shared: bool
    record_stride: int
    flat_material_stride: int
    std_surface_stride: int
    sphere_light_capacity: int
    sphere_light_stride: int
    distant_light_capacity: int
    distant_light_stride: int
    emissive_tri_stride: int
    instance_stride: int
    spectral_emitter_stride: int
    spectral_light_spd_stride: int
    spectral_scale_floats: int = 0
    spectral_data_floats: int = 0
    spectral_d65_floats: int = 0
    spectral_metals_floats: int = 0


# ── Declarations ────────────────────────────────────────────────────────
#
# Order IS allocation order, transcribed from the instrumented pre-change
# `_init_gpu` (both backends allocate the same 33 resources in the same order;
# `--spectral` interleaves 7 more, each at the point its RGB neighbour is
# allocated).

DECLARATIONS: tuple[ResourceDecl, ...] = (
    ResourceDecl(
        "uniform_buffer", "UniformBuffer", lambda s: (s.uniform_bytes,),
        binding=0, metal=None, descriptor=UBO,
    ),
    ResourceDecl(
        "mtlx_skin_buffer", "StorageBuffer",
        lambda s: (s.material_capacity * s.mtlx_skin_slot_bytes + 256,),
        binding=15, metal="mtlxSkin", descriptor=BUFFER,
    ),
    ResourceDecl(
        "accum_image", "StorageImage", lambda s: (s.width, s.height),
        kwargs=lambda s: {"transfer_src": True},
        binding=2, metal="accumBuffer", descriptor=IMAGE, size_dependent=True,
    ),
    ResourceDecl(
        "hud_overlay", "HudOverlay", lambda s: (s.width, s.height),
        binding=3, metal="hudMask", descriptor=IMAGE, size_dependent=True,
    ),
    ResourceDecl(
        "env_image", "SampledImage", lambda s: (s.env_width, s.env_height),
        binding=4, metal="envMap", descriptor=SAMPLER, metal_sampler=True,
    ),
    ResourceDecl(
        "volume_density_image", "SampledImage3D", lambda s: (1, 1, 1),
        binding=26, metal="volumeDensity", descriptor=SAMPLER,
        metal_sampler=True,
    ),
    ResourceDecl(
        "env_dist_buffer", "StorageBuffer",
        lambda s: (((s.env_height + 1)
                    + s.env_height * (s.env_width + 1)) * 4,),
        binding=31, metal="envDistCdf", descriptor=BUFFER,
    ),
    ResourceDecl(
        "_spectral_scale_buffer", "StorageBuffer",
        lambda s: (s.spectral_scale_floats * 4,),
        binding=45, metal="spectralScale", descriptor=BUFFER, spectral=True,
    ),
    ResourceDecl(
        "_spectral_data_buffer", "StorageBuffer",
        lambda s: (s.spectral_data_floats * 4,),
        binding=46, metal="spectralData", descriptor=BUFFER, spectral=True,
    ),
    ResourceDecl(
        "_spectral_d65_buffer", "StorageBuffer",
        lambda s: (s.spectral_d65_floats * 4,),
        binding=47, metal="spectralD65", descriptor=BUFFER, spectral=True,
    ),
    ResourceDecl(
        "_spectral_metals_buffer", "StorageBuffer",
        lambda s: (s.spectral_metals_floats * 4,),
        binding=48, metal="spectralMetals", descriptor=BUFFER, spectral=True,
    ),
    ResourceDecl(
        "neural_weights_buffer", "StorageBuffer",
        lambda s: (max(s.neural_weight_bytes, 4),),
        kwargs=lambda s: {"external": s.neural_external,
                          "shared": s.neural_shared},
        binding=33, metal="neuralWeights", descriptor=BUFFER,
    ),
    ResourceDecl(
        "neural_biases_buffer", "StorageBuffer",
        lambda s: (max(s.neural_bias_bytes, 4),),
        kwargs=lambda s: {"external": s.neural_external,
                          "shared": s.neural_shared},
        binding=34, metal="neuralBiases", descriptor=BUFFER,
    ),
    ResourceDecl(
        # Binding 35 (layer headers) is immutable after build, so it stays
        # device-local even under the Metal shared-storage handoff.
        "neural_layers_buffer", "StorageBuffer",
        lambda s: (max(s.neural_header_bytes, 4),),
        kwargs=lambda s: {"external": s.neural_external},
        binding=35, metal="neuralLayers", descriptor=BUFFER,
    ),
    ResourceDecl(
        "record_buffer", "StorageBuffer", lambda s: (s.record_stride,),
        binding=36, metal="recordBuf", descriptor=BUFFER,
    ),
    ResourceDecl(
        "record_counter", "StorageBuffer", lambda s: (8,),
        binding=37, metal="recordCounter", descriptor=BUFFER,
    ),
    ResourceDecl(
        "tattoo_image", "SampledImage",
        lambda s: (s.tattoo_width, s.tattoo_height),
        binding=8, metal="tattooMap", descriptor=SAMPLER, metal_sampler=True,
    ),
    ResourceDecl(
        "normal_image", "SampledImage", lambda s: (s.detail_res, s.detail_res),
        kwargs=lambda s: {"format": "rgba8_unorm", "bytes_per_pixel": 4},
        binding=9, metal="normalMap", descriptor=SAMPLER, metal_sampler=True,
    ),
    ResourceDecl(
        "roughness_image", "SampledImage",
        lambda s: (s.detail_res, s.detail_res),
        kwargs=lambda s: {"format": "rgba8_unorm", "bytes_per_pixel": 4},
        binding=10, metal="roughnessMap", descriptor=SAMPLER,
        metal_sampler=True,
    ),
    ResourceDecl(
        "displacement_image", "SampledImage",
        lambda s: (s.detail_res, s.detail_res),
        kwargs=lambda s: {"format": "rgba8_unorm", "bytes_per_pixel": 4},
        binding=11, metal="displacementMap", descriptor=SAMPLER,
        metal_sampler=True,
    ),
    ResourceDecl(
        "vertex_buffer", "StorageBuffer", lambda s: (s.vertex_bytes,),
        binding=5, metal="meshVertices", descriptor=BUFFER,
    ),
    ResourceDecl(
        "index_buffer", "StorageBuffer", lambda s: (s.index_bytes,),
        binding=6, metal="meshIndices", descriptor=BUFFER,
    ),
    ResourceDecl(
        "bvh_buffer", "StorageBuffer", lambda s: (s.bvh_bytes,),
        binding=7, metal="bvhNodes", descriptor=BUFFER,
    ),
    ResourceDecl(
        "instance_buffer", "StorageBuffer",
        lambda s: (s.instance_capacity * s.instance_stride + 256,),
        binding=12, metal="instances", descriptor=BUFFER,
    ),
    ResourceDecl(
        "flat_material_buffer", "StorageBuffer",
        lambda s: (s.material_capacity * s.flat_material_stride + 256,),
        binding=13, metal="flatMaterials", descriptor=BUFFER,
    ),
    ResourceDecl(
        "_spectral_mat_emission_buffer", "StorageBuffer",
        lambda s: (s.material_capacity * s.spectral_emitter_stride + 16,),
        binding=51, metal="spectralMatEmission", descriptor=BUFFER,
        spectral=True,
    ),
    ResourceDecl(
        "material_types_buffer", "StorageBuffer",
        lambda s: (s.material_capacity * 4 + 16,),
        binding=16, metal="materialTypes", descriptor=BUFFER,
    ),
    ResourceDecl(
        "sphere_lights_buffer", "StorageBuffer",
        lambda s: (s.sphere_light_capacity * s.sphere_light_stride + 16,),
        binding=17, metal="sphereLights", descriptor=BUFFER,
    ),
    ResourceDecl(
        "distant_lights_buffer", "StorageBuffer",
        lambda s: (s.distant_light_capacity * s.distant_light_stride + 16,),
        binding=20, metal="distantLights", descriptor=BUFFER,
    ),
    ResourceDecl(
        # Fixed capacity — distant lights never grow past the cap, so no
        # rebind path.
        "_spectral_light_spd_buffer", "StorageBuffer",
        lambda s: (s.distant_light_capacity * s.spectral_light_spd_stride + 16,),
        binding=50, metal="spectralLightSpd", descriptor=BUFFER, spectral=True,
    ),
    ResourceDecl(
        "emissive_tri_buffer", "StorageBuffer",
        lambda s: (s.emissive_tri_capacity * s.emissive_tri_stride + 16,),
        binding=18, metal="emissiveTriangles", descriptor=BUFFER,
    ),
    ResourceDecl(
        "_spectral_emitters_buffer", "StorageBuffer",
        lambda s: (s.emissive_tri_capacity * s.spectral_emitter_stride + 16,),
        binding=49, metal="spectralEmitters", descriptor=BUFFER, spectral=True,
    ),
    ResourceDecl(
        "std_surface_buffer", "StorageBuffer",
        lambda s: (s.material_capacity * s.std_surface_stride + 16,),
        binding=19, metal="stdSurfaceParams", descriptor=BUFFER,
    ),
    ResourceDecl(
        # Viewport-sized: the BDPT splat indexes it as
        # `(pixel.y * fc.width + pixel.x) * 3` with no bounds check against the
        # allocation (the only clamp is the camera's own frame test against
        # fc.width/fc.height), so a grow that left this at the old extent wrote
        # past the end.
        "light_splat_buffer", "StorageBuffer",
        lambda s: (s.width * s.height * 3 * 4,),
        binding=21, metal="lightSplatBuffer", descriptor=BUFFER,
        size_dependent=True,
    ),
    ResourceDecl(
        "gizmo_segments_buffer", "StorageBuffer",
        lambda s: (s.gizmo_capacity * s.gizmo_stride + 16,),
        binding=22, metal="gizmoSegments", descriptor=BUFFER,
    ),
    ResourceDecl(
        "lens_elements_buffer", "StorageBuffer",
        lambda s: (s.lens_element_capacity * s.lens_element_stride + 16,),
        binding=23, metal="lensElements", descriptor=BUFFER,
    ),
    ResourceDecl(
        "lens_pupil_buffer", "StorageBuffer",
        lambda s: (s.lens_pupil_capacity * s.lens_pupil_stride + 16,),
        binding=24, metal="exitPupilBounds", descriptor=BUFFER,
    ),
    ResourceDecl(
        # The only thing binding 1 ever points at, on every path: headless,
        # screenshot, and windowed render (which blits it into the acquired
        # swapchain image rather than rebinding).
        "_offscreen_output", "StorageImage", lambda s: (s.width, s.height),
        kwargs=lambda s: {"format": "rgba8_unorm", "transfer_src": True},
        binding=1, metal="outputBuffer", descriptor=IMAGE, size_dependent=True,
    ),
    ResourceDecl(
        "_readback", "ReadbackBuffer", lambda s: (s.width, s.height),
        binding=None, metal=None, descriptor=None, size_dependent=True,
    ),
    ResourceDecl(
        "tool_buffer", "HostStorageBuffer", lambda s: (128 * 64 * 16 + 4096,),
        binding=30, metal="toolBuffer", descriptor=BUFFER,
    ),
    ResourceDecl(
        # Bindless flat-material array. Slots fill lazily from each
        # Material.texture_paths entry; Metal passes the slot list as the
        # separate `bindless` dispatch argument rather than by name.
        "texture_pool", "TexturePool", lambda s: (),
        binding=14, metal=None, descriptor=BINDLESS,
    ),
)

BY_ATTR: dict[str, ResourceDecl] = {d.attr: d for d in DECLARATIONS}
_SPECTRAL_TAIL = frozenset(d.attr for d in DECLARATIONS if d.spectral)

# Descriptor-write order, transcribed from the instrumented pre-change
# `_create_descriptors`. NOT allocation order and not binding order.
VULKAN_WRITE_SEQUENCE: tuple[str, ...] = (
    "uniform_buffer",          # 0
    "accum_image",             # 2
    "hud_overlay",             # 3
    "env_image",               # 4
    "vertex_buffer",           # 5
    "index_buffer",            # 6
    "bvh_buffer",              # 7
    "tattoo_image",            # 8
    "normal_image",            # 9
    "roughness_image",         # 10
    "displacement_image",      # 11
    "volume_density_image",    # 26
    "instance_buffer",         # 12
    "flat_material_buffer",    # 13
    "mtlx_skin_buffer",        # 15
    "material_types_buffer",   # 16
    "sphere_lights_buffer",    # 17
    "emissive_tri_buffer",     # 18
    "std_surface_buffer",      # 19
    "light_splat_buffer",      # 21
    "gizmo_segments_buffer",   # 22
    "lens_elements_buffer",    # 23
    "lens_pupil_buffer",       # 24
    "distant_lights_buffer",   # 20
    "_offscreen_output",       # 1
    "tool_buffer",             # 30
    "env_dist_buffer",         # 31
    "neural_weights_buffer",   # 33
    "neural_biases_buffer",    # 34
    "neural_layers_buffer",    # 35
    "record_buffer",           # 36
    "record_counter",          # 37
    # Spectral tables last (45-51), only in the spectral layout.
    "_spectral_scale_buffer",
    "_spectral_data_buffer",
    "_spectral_d65_buffer",
    "_spectral_metals_buffer",
    "_spectral_emitters_buffer",
    "_spectral_light_spd_buffer",
    "_spectral_mat_emission_buffer",
)

# MLT chain buffers: only the wavefront (`scene_bindings_only`) layout declares
# them, and they are seeded with the record counter as a dummy until
# `_ensure_wavefront_mlt_pass` rebinds the real chain buffers — the 36/37
# record-dump precedent. Written after 37 and before the spectral tables.
#
# DERIVED, not stated (change mlt-binding-declaration): the numbers, the Metal
# global names and the sizes are one declaration in `wavefront_layout`, checked
# against the shader by `tests/test_mlt_binding_declaration.py`. This module is
# a consumer — it needs the numbers only, for the dummy writes and the pool
# count. `MLT_BINDINGS` keeps its name and shape (design D3).
MLT_BINDINGS: tuple[int, ...] = mlt_binding_numbers()
MLT_DUMMY_ATTR = "record_counter"

# Teardown order, transcribed from the instrumented pre-change `cleanup`.
DESTROY_SEQUENCE: tuple[str, ...] = (
    "std_surface_buffer",
    "emissive_tri_buffer",
    "sphere_lights_buffer",
    "distant_lights_buffer",
    "material_types_buffer",
    "flat_material_buffer",
    "instance_buffer",
    "bvh_buffer",
    "index_buffer",
    "vertex_buffer",
    "displacement_image",
    "roughness_image",
    "normal_image",
    "tattoo_image",
    "env_image",
    "volume_density_image",
    "env_dist_buffer",
    "_spectral_scale_buffer",
    "_spectral_data_buffer",
    "_spectral_d65_buffer",
    "_spectral_metals_buffer",
    "_spectral_emitters_buffer",
    "_spectral_light_spd_buffer",
    "_spectral_mat_emission_buffer",
    "hud_overlay",
    "accum_image",
    "_offscreen_output",
    "_readback",
    "uniform_buffer",
    "mtlx_skin_buffer",
    "neural_weights_buffer",
    "neural_biases_buffer",
    "neural_layers_buffer",
    "record_buffer",
    "record_counter",
    "light_splat_buffer",
    "gizmo_segments_buffer",
    "lens_elements_buffer",
    "lens_pupil_buffer",
    "tool_buffer",
)
# The texture pool is destroyed first of all, before the per-resource sequence
# above (it precedes them in `cleanup`, right after the descriptor pool).
DESTROY_FIRST: tuple[str, ...] = ("texture_pool",)


def _sequence(spectral: bool, order: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(a for a in order if BY_ATTR[a].active(spectral))


class SceneResourceSet:
    """The renderer's GPU resource inventory: allocate, bind, grow, destroy.

    One instance owns every resource declared above. The renderer reaches them
    by their historical attribute names through forwarding properties, so no
    call site outside this module changed.
    """

    def __init__(self, ctx, gpu, sizes: ResourceSizes, *, spectral: bool,
                 texture_pool_factory=None) -> None:
        self.ctx = ctx
        self._gpu = gpu
        self.sizes = sizes
        self.spectral = bool(spectral)
        self.is_metal = bool(getattr(ctx, "is_metal", False))
        self._resources: dict[str, object] = {}
        self._closed = False
        # The pool needs the resource module itself, not a constructor on it.
        self._texture_pool_factory = texture_pool_factory
        for decl in DECLARATIONS:
            if decl.active(self.spectral):
                self._resources[decl.attr] = self._allocate(decl)

    @classmethod
    def stub(cls, **resources) -> "SceneResourceSet":
        """A set holding the given resources and nothing else, allocating
        nothing.

        For tests that construct a `Renderer` via `__new__` and need one or two
        GPU resources faked. The renderer's resource attributes are read-only
        properties — assigning one would reallocate without rebinding, which is
        the split this module exists to prevent — so a test stubs the owner
        instead of the attribute.
        """
        rset = cls.__new__(cls)
        rset.ctx = None
        rset._gpu = None
        rset.sizes = None
        rset.spectral = False
        rset.is_metal = False
        rset._resources = dict(resources)
        rset._closed = False
        rset._texture_pool_factory = None
        return rset

    # ── allocation ──────────────────────────────────────────────────────

    def _allocate(self, decl: ResourceDecl, args: tuple | None = None):
        if decl.kind == "TexturePool":
            if self._texture_pool_factory is None:
                raise ValueError("texture_pool_factory required for TexturePool")
            return self._texture_pool_factory(self.ctx, self._gpu)
        ctor = getattr(self._gpu, decl.kind)
        if args is None:
            args = decl.args(self.sizes)
        return ctor(self.ctx, *args, **decl.kwargs(self.sizes))

    @property
    def declarations(self) -> tuple[ResourceDecl, ...]:
        """The active declarations, in allocation order."""
        return tuple(d for d in DECLARATIONS if d.active(self.spectral))

    def __getattr__(self, name):
        # Only reached for attributes not found normally, so the instance
        # attributes set in __init__ still win.
        try:
            return object.__getattribute__(self, "_resources")[name]
        except KeyError:
            raise AttributeError(name) from None

    def get(self, attr: str):
        """The live resource for ``attr``, or None when it is not active (a
        spectral-only buffer in an RGB session)."""
        return self._resources.get(attr)

    # ── growth ──────────────────────────────────────────────────────────

    def replace(self, allocs: dict[str, tuple], *,
                descriptor_sets=None) -> None:
        """Destroy and reallocate the named resources at new sizes, then
        rebind exactly their declarations.

        This is the single owner of "this resource grew": the callers that used
        to destroy, re-create and then remember to call the matching
        ``_rebind_*_descriptors`` now state only the new sizes.
        """
        # Destroy the whole group before allocating any of it, matching the
        # mesh-grow site this replaced (all three buffers freed, then all three
        # re-created) so a large grow does not transiently hold both copies.
        for attr in allocs:
            old = self._resources.get(attr)
            if old is not None:
                old.destroy()
                self._resources[attr] = None
        for attr, args in allocs.items():
            self._resources[attr] = self._allocate(BY_ATTR[attr], args)
        self.rebind(allocs.keys(), descriptor_sets=descriptor_sets)

    def regrow(self, *attrs: str, descriptor_sets=None) -> None:
        """Reallocate the named declarations at the sizes their own formulas
        now yield, after a capacity on :attr:`sizes` was bumped.

        The growth sites state the new *capacity*, never the new byte size —
        the ``cap * STRIDE + slack`` arithmetic stays in the declaration, so it
        cannot drift between the initial allocation and a later grow. Names
        that are not active (a spectral-only buffer in an RGB session) are
        skipped.
        """
        live = [a for a in attrs if self._resources.get(a) is not None]
        if live:
            self.replace({a: BY_ATTR[a].args(self.sizes) for a in live},
                         descriptor_sets=descriptor_sets)

    def adopt(self, attr: str, resource) -> None:
        """Put an externally-built resource into a declared slot, destroying
        whatever occupied it.

        The Metal record-drain path swaps a per-frame drain buffer into
        ``record_buffer`` so the bind-by-name table routes it to ``recordBuf``.
        Ownership transfers to the set, so the caller must not also destroy it
        — before this, ``cleanup`` freed the drain buffer *and* the
        record_buffer slot it had been aliased into, a latent double destroy.
        """
        old = self._resources.get(attr)
        if old is not None and old is not resource:
            old.destroy()
        self._resources[attr] = resource

    def owns(self, resource) -> bool:
        """True when the set holds this object in one of its slots."""
        return any(r is resource for r in self._resources.values())

    def resize(self, width: int, height: int, *, descriptor_sets=None) -> None:
        """Recreate the viewport-sized resources and rebind them."""
        self.sizes.width = int(width)
        self.sizes.height = int(height)
        self.replace(
            {d.attr: d.args(self.sizes)
             for d in DECLARATIONS
             if d.size_dependent and d.active(self.spectral)},
            descriptor_sets=descriptor_sets,
        )

    def rebind(self, attrs, *, descriptor_sets=None) -> None:
        """Rewrite the descriptor entries for ``attrs``.

        The one backend branch (design D3). Native Metal re-reads every
        resource reference fresh at each dispatch through :meth:`metal_binds`,
        so a reallocation is picked up automatically and there is nothing to
        rewrite — the Metal adapter simply has no descriptor step, rather than
        each rebind helper opting out with its own ``is_metal`` early return.
        """
        if self.is_metal or descriptor_sets is None:
            return
        import vulkan as vk

        writes = []
        for ds in descriptor_sets:
            for attr in attrs:
                decl = BY_ATTR[attr]
                res = self._resources.get(attr)
                if res is None or decl.binding is None:
                    continue
                if decl.descriptor == BINDLESS:
                    writes.extend(self._bindless_writes(ds, res))
                else:
                    writes.append(self._write(ds, decl, res))
        if writes:
            vk.vkUpdateDescriptorSets(
                self.ctx.device, len(writes), writes, 0, None)

    # ── binding: Vulkan ─────────────────────────────────────────────────

    def _write(self, ds, decl: ResourceDecl, res):
        import vulkan as vk

        if decl.descriptor == UBO:
            return vk.VkWriteDescriptorSet(
                dstSet=ds, dstBinding=decl.binding, dstArrayElement=0,
                descriptorCount=1,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                pBufferInfo=[vk.VkDescriptorBufferInfo(
                    buffer=res.buffer, offset=0, range=res.size)],
            )
        if decl.descriptor == IMAGE:
            return vk.VkWriteDescriptorSet(
                dstSet=ds, dstBinding=decl.binding, dstArrayElement=0,
                descriptorCount=1,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                pImageInfo=[vk.VkDescriptorImageInfo(
                    imageView=res.view,
                    imageLayout=vk.VK_IMAGE_LAYOUT_GENERAL)],
            )
        if decl.descriptor == SAMPLER:
            return vk.VkWriteDescriptorSet(
                dstSet=ds, dstBinding=decl.binding, dstArrayElement=0,
                descriptorCount=1,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                pImageInfo=[vk.VkDescriptorImageInfo(
                    sampler=res.sampler, imageView=res.view,
                    imageLayout=vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)],
            )
        return vk.VkWriteDescriptorSet(
            dstSet=ds, dstBinding=decl.binding, dstArrayElement=0,
            descriptorCount=1,
            descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            pBufferInfo=[vk.VkDescriptorBufferInfo(
                buffer=res.buffer, offset=0, range=res.size)],
        )

    def write_binding(self, binding: int, resource, descriptor_sets, *,
                      descriptor: str = BUFFER) -> None:
        """Point one descriptor binding at a resource the set does not own.

        Two resources have a scene-graph lifetime rather than the renderer's:
        the combined MaterialX graph-param buffer (rebuilt per scene graph) and
        the record-drain buffer (rebuilt per frame capacity). They are not
        declarations, but the *write* still goes through here so no call site
        hand-rolls a ``VkWriteDescriptorSet`` — the mechanics stay in one place
        even where the lifetime does not.
        """
        if self.is_metal or descriptor_sets is None:
            return
        import vulkan as vk

        decl = ResourceDecl(
            attr="<external>", kind="", args=lambda s: (),
            binding=binding, descriptor=descriptor,
        )
        for ds in descriptor_sets:
            write = self._write(ds, decl, resource)
            vk.vkUpdateDescriptorSets(self.ctx.device, 1, [write], 0, None)

    def _bindless_writes(self, ds, pool) -> list:
        """Binding 14, one write per filled slot. PARTIALLY_BOUND lets unfilled
        slots stay invalid — the shader gates reads behind a sentinel index."""
        import vulkan as vk

        return [
            vk.VkWriteDescriptorSet(
                dstSet=ds, dstBinding=14, dstArrayElement=slot_idx,
                descriptorCount=1,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                pImageInfo=[vk.VkDescriptorImageInfo(
                    sampler=sampled.sampler, imageView=sampled.view,
                    imageLayout=vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)],
            )
            for slot_idx, sampled in pool.filled_slots()
        ]

    def pool_sizes(self, *, frames_in_flight: int, graph_slot: int,
                   mlt_bindings: bool) -> list:
        """Descriptor-pool sizes, counted from the active declarations rather
        than a hand-maintained tally."""
        import vulkan as vk

        active = [d for d in self.declarations if d.binding is not None]
        images = sum(1 for d in active if d.descriptor == IMAGE)
        samplers = sum(1 for d in active if d.descriptor == SAMPLER)
        buffers = sum(1 for d in active if d.descriptor == BUFFER)
        buffers += graph_slot + (len(MLT_BINDINGS) if mlt_bindings else 0)
        return [
            vk.VkDescriptorPoolSize(
                type=vk.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                descriptorCount=frames_in_flight,
            ),
            vk.VkDescriptorPoolSize(
                type=vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                descriptorCount=frames_in_flight * images,
            ),
            vk.VkDescriptorPoolSize(
                type=vk.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                descriptorCount=frames_in_flight
                * (samplers + self._gpu.BINDLESS_TEXTURE_CAPACITY),
            ),
            vk.VkDescriptorPoolSize(
                type=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount=frames_in_flight * buffers,
            ),
        ]

    def bind_vulkan(self, descriptor_sets, *, mlt_bindings: bool = False) -> None:
        """Write every declared binding, in the recorded write order."""
        if self.is_metal or descriptor_sets is None:
            return
        import vulkan as vk

        seq = _sequence(self.spectral, VULKAN_WRITE_SEQUENCE)
        for ds in descriptor_sets:
            writes = [
                self._write(ds, BY_ATTR[a], self._resources[a])
                for a in seq
                if a not in _SPECTRAL_TAIL
            ]
            if mlt_bindings:
                dummy = self._resources[MLT_DUMMY_ATTR]
                writes.extend(
                    vk.VkWriteDescriptorSet(
                        dstSet=ds, dstBinding=b, dstArrayElement=0,
                        descriptorCount=1,
                        descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                        pBufferInfo=[vk.VkDescriptorBufferInfo(
                            buffer=dummy.buffer, offset=0, range=dummy.size)],
                    )
                    for b in MLT_BINDINGS
                )
            writes.extend(
                self._write(ds, BY_ATTR[a], self._resources[a])
                for a in seq if a in _SPECTRAL_TAIL
            )
            vk.vkUpdateDescriptorSets(
                self.ctx.device, len(writes), writes, 0, None)

    # ── binding: Metal ──────────────────────────────────────────────────

    def metal_binds(self) -> dict:
        """Map each Metal shader-global name to its native SlangPy resource,
        from the same declarations. The pipeline filters to the names the
        compiled module references, so binding an unused name is harmless and a
        dead-stripped one is simply skipped."""
        binds: dict = {}
        for decl in DECLARATIONS:
            if decl.metal is None or not decl.active(self.spectral):
                continue
            res = self._resources.get(decl.attr)
            if res is None:
                continue
            if decl.descriptor in (IMAGE, SAMPLER):
                binds[decl.metal] = res.texture
                if decl.metal_sampler:
                    binds[decl.metal + "Sampler"] = res.sampler
            else:
                binds[decl.metal] = res.buffer
        return binds

    def metal_bindless(self, name: str = "flatMaterialTextures") -> tuple:
        pool = self._resources["texture_pool"]
        return (name, [(s.texture if s is not None else None)
                       for s in pool._slots])

    # ── destruction ─────────────────────────────────────────────────────

    def close(self) -> None:
        """Destroy every allocated resource, in the recorded teardown order.

        Paired with allocation by construction: a declaration that is allocated
        is destroyed, so a resource can no longer be added in one place and
        forgotten in another.
        """
        if self._closed:
            return
        self._closed = True
        for attr in DESTROY_FIRST + _sequence(self.spectral, DESTROY_SEQUENCE):
            res = self._resources.pop(attr, None)
            if res is not None:
                res.destroy()
