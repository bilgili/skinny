"""Hostless gates for the renderer's GPU resource set (gpu_resources.py).

The set is checked against `tests/fixtures/gpu_resource_inventory.json`, which
was captured by instrumenting the live pre-change renderer on both backends —
recorded reality, never a transcription of the new module's own table. The
fixture is permanent: if a declaration legitimately changes, re-capture it,
don't edit the expectation to match the code.

Everything here runs without a GPU: the set is constructed against a recording
fake context whose resource constructors return sentinels.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from skinny.gpu_resources import (
    BINDLESS,
    BUFFER,
    BY_ATTR,
    DECLARATIONS,
    DESTROY_FIRST,
    DESTROY_SEQUENCE,
    IMAGE,
    SAMPLER,
    UBO,
    VULKAN_WRITE_SEQUENCE,
    ResourceSizes,
    SceneResourceSet,
)

FIXTURE = (Path(__file__).parent / "fixtures" / "gpu_resource_inventory.json")
GOLDEN = json.loads(FIXTURE.read_text())

# Vulkan descriptor-type codes, as recorded in the golden.
VK_TYPE = {UBO: 6, IMAGE: 3, SAMPLER: 1, BUFFER: 7, BINDLESS: 1}

# The size inputs the captured renderer used: 64x64, no scene, dummy mesh.
# These are the *recorded inputs*; the declarations' formulas are what is under
# test, so a stride constant changing in the renderer does not silently rewrite
# the expectation here.
CAPTURED_SIZES = dict(
    width=64, height=64, uniform_bytes=768,
    material_capacity=16, mtlx_skin_slot_bytes=164,
    instance_capacity=16, emissive_tri_capacity=256,
    gizmo_capacity=256, gizmo_stride=32,
    lens_element_capacity=32, lens_element_stride=16,
    lens_pupil_capacity=64, lens_pupil_stride=16,
    vertex_bytes=352, index_bytes=268, bvh_bytes=288,
    env_width=1024, env_height=512,
    tattoo_width=512, tattoo_height=512, detail_res=2048,
    neural_weight_bytes=412416, neural_bias_bytes=6360,
    neural_header_bytes=288,
    neural_external=False, neural_shared=False,
    record_stride=64,
    flat_material_stride=256, std_surface_stride=256,
    sphere_light_capacity=16, sphere_light_stride=32,
    distant_light_capacity=16, distant_light_stride=32,
    emissive_tri_stride=64, instance_stride=144,
    spectral_emitter_stride=8, spectral_light_spd_stride=380,
    spectral_scale_floats=64, spectral_data_floats=2359296,
    spectral_d65_floats=95, spectral_metals_floats=1330,
)


# ── recording fake backend ──────────────────────────────────────────────


class FakeResource:
    """Sentinel standing in for a GPU buffer/image."""

    def __init__(self, kind, args, kwargs, log):
        self.kind = kind
        self.args = args
        self.kwargs = kwargs
        self._log = log
        self.destroyed = False
        # Buffer-ish and image-ish handles, so both descriptor paths resolve.
        self.size = args[0] if args and isinstance(args[0], int) else 0
        self.buffer = f"buf<{kind}>"
        self.texture = f"tex<{kind}>"
        self.view = f"view<{kind}>"
        self.sampler = f"smp<{kind}>"

    def destroy(self):
        assert not self.destroyed, f"{self.kind} destroyed twice"
        self.destroyed = True
        self._log.append(self)

    def upload_sync(self, data):
        self.uploaded = data

    def fill_zero_sync(self):
        self.uploaded = b""


class FakeTexturePool:
    def __init__(self, ctx, gpu):
        self.destroyed = False
        self._slots = [None] * 4
        self._log = gpu.destroy_log

    def filled_slots(self):
        return [(i, s) for i, s in enumerate(self._slots) if s is not None]

    def destroy(self):
        self.destroyed = True
        self._log.append(self)


class FakeGpu:
    """Stands in for `vk_compute` / `metal_compute`, recording every
    construction in order."""

    BINDLESS_TEXTURE_CAPACITY = 128

    def __init__(self):
        self.alloc_log: list[dict] = []
        self.destroy_log: list = []

    def __getattr__(self, kind):
        if kind.startswith("_"):
            raise AttributeError(kind)

        def ctor(ctx, *args, **kwargs):
            res = FakeResource(kind, args, kwargs, self.destroy_log)
            self.alloc_log.append(
                {"kind": kind, "args": args, "kwargs": kwargs, "res": res})
            return res
        return ctor


class FakeCtx:
    device = "device"

    def __init__(self, is_metal=False):
        self.is_metal = is_metal


def build(spectral=False, is_metal=False, **overrides):
    sizes_kwargs = dict(CAPTURED_SIZES)
    if is_metal:
        sizes_kwargs["mtlx_skin_slot_bytes"] = 256
    sizes_kwargs.update(overrides)
    gpu = FakeGpu()
    ctx = FakeCtx(is_metal=is_metal)
    rset = SceneResourceSet(
        ctx, gpu, ResourceSizes(**sizes_kwargs), spectral=spectral,
        texture_pool_factory=FakeTexturePool,
    )
    return rset, gpu


def golden_resources(key):
    # The texture pool is not built through the resource module, so it is
    # absent from the captured constructor log.
    return GOLDEN[key]["resources"]


# ── inventory matches the pre-change renderer ───────────────────────────


@pytest.mark.parametrize(
    "key,spectral,is_metal",
    [("vulkan_rgb", False, False),
     ("metal_rgb", False, True),
     ("metal_spectral", True, True)],
)
def test_inventory_matches_golden(key, spectral, is_metal):
    """Scenario: inventory matches the pre-change renderer — name, kind, size
    inputs and allocation ORDER, entry for entry."""
    rset, gpu = build(spectral=spectral, is_metal=is_metal)
    # Allocation happens in declaration order, so the recorded constructor log
    # lines up with the declarations one for one (the texture pool is not built
    # through the resource module and so never appears in it).
    decls = [d for d in rset.declarations if d.kind != "TexturePool"]
    assert len(decls) == len(gpu.alloc_log)
    got = [
        (d.attr, rec["kind"], rec["args"], rec["kwargs"])
        for d, rec in zip(decls, gpu.alloc_log)
    ]
    want = [
        (r["attr"], r["kind"],
         tuple(eval(a) for a in r["args"]),             # noqa: S307 fixture
         {k: eval(v) for k, v in r["kwargs"].items()})  # noqa: S307 fixture
        for r in golden_resources(key)
    ]
    assert got == want


# ── one declaration feeds both backends ─────────────────────────────────


def test_vulkan_bindings_unique():
    seen = {}
    for d in DECLARATIONS:
        if d.binding is None:
            continue
        assert d.binding not in seen, (
            f"binding {d.binding}: {d.attr} collides with {seen[d.binding]}")
        seen[d.binding] = d.attr


def test_metal_names_unique():
    names = [d.metal for d in DECLARATIONS if d.metal is not None]
    assert len(names) == len(set(names))


def test_both_targets_cover_the_same_declarations():
    """Scenario: one declaration feeds both backends — the Vulkan bindings and
    the Metal names cover the same declarations, modulo the ones explicitly
    marked absent on one target."""
    vulkan = {d.attr for d in DECLARATIONS if d.binding is not None}
    metal = {d.attr for d in DECLARATIONS if d.metal is not None}
    # `_readback` is a host staging buffer, bound on neither target.
    # `texture_pool` is binding 14 on Vulkan; Metal passes the slot list as a
    # separate dispatch argument, not by name. `uniform_buffer` is binding 0 on
    # Vulkan; Metal takes the frame constants as a `uniform_blob` byte payload.
    assert vulkan - metal == {"texture_pool", "uniform_buffer"}
    assert metal - vulkan == set()
    assert {d.attr for d in DECLARATIONS} - vulkan - metal == {"_readback"}


@pytest.mark.parametrize("key,spectral", [("metal_rgb", False),
                                          ("metal_spectral", True)])
def test_metal_binds_match_golden(key, spectral):
    rset, _ = build(spectral=spectral, is_metal=True)
    want = GOLDEN[key]["metal_binds"]
    got = rset.metal_binds()
    # The shared bindless-pool sampler (binding 38) is owned by the renderer,
    # not the resource set; everything else comes from the declarations.
    assert set(got) == set(want) - {"commonSampler"}
    for name, target in want.items():
        if name == "commonSampler":
            continue
        attr, sub = target.rsplit(".", 1)
        assert got[name] == getattr(rset.get(attr), sub)


def test_vulkan_write_sequence_matches_golden():
    """Scenario: descriptor-write order, binding for binding, in order."""
    frames = 2  # MAX_FRAMES_IN_FLIGHT at capture time
    expected = [
        {"binding": BY_ATTR[a].binding, "array_elem": 0,
         "type": VK_TYPE[BY_ATTR[a].descriptor]}
        for a in VULKAN_WRITE_SEQUENCE if not BY_ATTR[a].spectral
    ] * frames
    assert GOLDEN["vulkan_rgb"]["descriptor_writes"] == expected


def test_write_sequence_covers_every_bound_declaration():
    bound = {d.attr for d in DECLARATIONS
             if d.binding is not None and d.descriptor != BINDLESS}
    assert set(VULKAN_WRITE_SEQUENCE) == bound


# ── every allocated resource is destroyed ───────────────────────────────


@pytest.mark.parametrize("spectral", [False, True])
def test_close_destroys_exactly_what_was_allocated(spectral):
    """Scenario: every allocated resource is destroyed — the destroyed set
    equals the allocated set, nothing allocated twice, nothing left over."""
    rset, gpu = build(spectral=spectral)
    allocated = [r["res"] for r in gpu.alloc_log]
    rset.close()
    destroyed = gpu.destroy_log
    # The pool is not built through the resource module but is owned by the set.
    assert len(destroyed) == len(allocated) + 1
    assert all(r.destroyed for r in allocated)
    assert len(set(map(id, destroyed))) == len(destroyed)


def test_close_is_idempotent():
    rset, gpu = build()
    rset.close()
    n = len(gpu.destroy_log)
    rset.close()
    assert len(gpu.destroy_log) == n


@pytest.mark.parametrize("key,spectral", [("vulkan_rgb", False),
                                          ("metal_spectral", True)])
def test_destroy_order_matches_golden(key, spectral):
    rset, gpu = build(spectral=spectral, is_metal=key.startswith("metal"))
    decls = [d for d in rset.declarations if d.kind != "TexturePool"]
    by_res = {id(rec["res"]): d.attr for d, rec in zip(decls, gpu.alloc_log)}
    rset.close()
    got = [by_res[id(r)] for r in gpu.destroy_log if id(r) in by_res]
    assert got == GOLDEN[key]["destroyed"]


def test_destroy_sequence_covers_every_declaration():
    assert set(DESTROY_SEQUENCE) | set(DESTROY_FIRST) == {
        d.attr for d in DECLARATIONS}
    assert len(DESTROY_SEQUENCE) + len(DESTROY_FIRST) == len(DECLARATIONS)


# ── growth reflows through the set ──────────────────────────────────────


def test_replace_destroys_the_old_resource_and_allocates_the_new():
    rset, gpu = build()
    old = rset.vertex_buffer
    rset.replace({"vertex_buffer": (4096,)})
    assert old.destroyed
    assert rset.vertex_buffer is not old
    assert rset.vertex_buffer.args == (4096,)


def test_resize_replaces_every_size_dependent_resource():
    """Scenario: growth reflows bindings through the set — a viewport resize
    recreates exactly the size-dependent declarations."""
    rset, gpu = build()
    before = {d.attr: rset.get(d.attr)
              for d in DECLARATIONS if d.active(False)}
    rset.resize(128, 256)
    changed = {a for a, r in before.items() if rset.get(a) is not r}
    assert changed == {d.attr for d in DECLARATIONS if d.size_dependent}
    assert rset.accum_image.args == (128, 256)
    assert rset.sizes.width == 128 and rset.sizes.height == 256


def test_resize_regrows_the_bdpt_light_splat_buffer():
    """The BDPT splat is indexed `(pixel.y * fc.width + pixel.x) * 3` with no
    bounds check against the allocation, so leaving binding 21 at the old
    extent wrote past the end of the buffer after any viewport grow."""
    rset, _ = build()
    old = rset.light_splat_buffer
    rset.resize(128, 256)
    assert rset.light_splat_buffer is not old
    assert old.destroyed
    assert rset.light_splat_buffer.args == (128 * 256 * 3 * 4,)


def test_adopt_destroys_the_displaced_occupant_exactly_once():
    """The Metal record-drain grow path: `adopt` is the SINGLE destroyer of the
    slot it displaces.

    `FakeResource.destroy` asserts on a second call, so a double free fails here
    rather than on a device. This is the shape codex found: the caller also
    freed the outgoing drain buffer, which by then WAS the `record_buffer` slot
    occupant, so `adopt` destroyed it again.
    """
    rset, gpu = build()
    dummy = rset.record_buffer
    drain1 = FakeResource("StorageBuffer", (4096,), {}, gpu.destroy_log)
    rset.adopt("record_buffer", drain1)
    assert dummy.destroyed and not drain1.destroyed
    assert rset.record_buffer is drain1
    assert rset.owns(drain1)

    # Grow: the caller must NOT free drain1 itself — `owns()` tells it not to.
    drain2 = FakeResource("StorageBuffer", (8192,), {}, gpu.destroy_log)
    rset.adopt("record_buffer", drain2)
    assert drain1.destroyed and not drain2.destroyed
    assert rset.record_buffer is drain2

    # Teardown frees the adopted buffer once, and only once.
    rset.close()
    assert drain2.destroyed


def test_adopt_of_the_same_object_is_not_a_self_destroy():
    rset, gpu = build()
    drain = FakeResource("StorageBuffer", (4096,), {}, gpu.destroy_log)
    rset.adopt("record_buffer", drain)
    rset.adopt("record_buffer", drain)
    assert not drain.destroyed
    assert rset.record_buffer is drain


def test_owns_reports_slot_membership():
    rset, gpu = build()
    stranger = FakeResource("StorageBuffer", (16,), {}, gpu.destroy_log)
    assert rset.owns(rset.vertex_buffer)
    assert not rset.owns(stranger)


def test_metal_record_drain_grow_does_not_double_free():
    """Drives the real renderer path, not just `adopt`'s contract.

    `_ensure_wf_record_drain` on Metal allocates a drain buffer and adopts it
    into the `record_buffer` slot. On a later grow the caller must leave the
    outgoing buffer to `adopt` — freeing it itself frees the slot occupant, and
    `adopt` then frees it a second time. Reachable only with Metal + the record
    drain + a capacity increase, which no image gate exercises.

    Reverting the `owns()` guard in `_ensure_wf_record_drain` makes this fail
    with "StorageBuffer destroyed twice".
    """
    try:
        from skinny.renderer import Renderer
    except OSError as exc:  # libvulkan not on the dylib path
        pytest.skip(f"needs the Vulkan SDK on the dylib path: {exc}")

    rset, gpu = build()
    r = Renderer.__new__(Renderer)  # bypass __init__ — no GPU/context needed
    r.is_metal = True
    r.ctx = FakeCtx(is_metal=True)
    r._gpu = gpu
    r._gpu_set = rset
    r._drain_buffer = None
    r.width, r.height = 64, 64

    first = r._ensure_wf_record_drain(max_records_per_frame=64)
    adopted = r.record_buffer
    assert adopted is r._drain_buffer
    assert first == 64

    # Grow: a bigger capacity forces a fresh allocation.
    r._ensure_wf_record_drain(max_records_per_frame=4096)
    assert adopted.destroyed, "the displaced drain buffer must be freed once"
    assert r.record_buffer is r._drain_buffer
    assert not r.record_buffer.destroyed

    rset.close()  # FakeResource.destroy asserts on a second free


def test_metal_bindless_names_the_pool_slots():
    """Binding 14's Metal identity belongs to the declaration list, not to each
    dispatch site — the megakernel and the preview dock both take it from
    here."""
    rset, _ = build(is_metal=True)
    name, slots = rset.metal_bindless()
    assert name == "flatMaterialTextures"
    assert len(slots) == len(rset.texture_pool._slots)
    assert all(s is None for s in slots)


def test_rebind_is_a_no_op_on_metal():
    """The Metal adapter has no descriptor step at all — the per-method
    `is_metal` early-returns the rebind helpers used to open with are gone."""
    rset, _ = build(is_metal=True)
    rset.rebind(["vertex_buffer"], descriptor_sets=["ds"])  # must not raise


def test_spectral_declarations_absent_from_the_rgb_layout():
    rgb, _ = build(spectral=False)
    spec, _ = build(spectral=True)
    only_spectral = {d.attr for d in DECLARATIONS if d.spectral}
    assert {d.attr for d in spec.declarations} - {
        d.attr for d in rgb.declarations} == only_spectral
    for attr in only_spectral:
        assert rgb.get(attr) is None


# ── no descriptor write survives outside the set ────────────────────────


RENDERER_SRC = (
    Path(__file__).parent.parent / "src" / "skinny" / "renderer.py").read_text()


def test_renderer_hand_rolls_no_descriptor_write():
    """Source gate, same shape as the shader-variant-key grep gate.

    Every descriptor write for the renderer's inventory goes through
    `SceneResourceSet`. A `VkWriteDescriptorSet` reappearing in `renderer.py`
    means the inventory is being bound from two places again — exactly the
    split this change removed, and the one that let binding 49 keep pointing at
    a freed buffer while 18 was rewritten. For a resource whose lifetime is not
    the renderer's, use `SceneResourceSet.write_binding`.
    """
    assert "VkWriteDescriptorSet" not in RENDERER_SRC
    assert "vkUpdateDescriptorSets" not in RENDERER_SRC


def test_no_per_method_backend_guard_returns_remain():
    """The five `is_metal` / `descriptor_sets is None` early-returns that used
    to open each `_rebind_*` helper must not come back: backend divergence is
    decided once, at the set's binding step."""
    for gone in ("_rebind_scene_descriptors", "_rebind_aux_material_descriptors",
                 "_rebind_emissive_descriptors", "_rebind_mesh_descriptors",
                 "_rebind_volume_descriptor",
                 "_rewrite_size_dependent_descriptors"):
        assert gone not in RENDERER_SRC, f"{gone} came back"


def test_capacities_have_exactly_one_home():
    """A capacity is an allocation input, so it lives on `_gpu_set.sizes` and
    nowhere else.

    A growth site that bumped a renderer attribute AND a `sizes` field would be
    the same two-copy invariant this module removes for buffers, reintroduced
    one level up: the second line goes missing, `regrow()` re-derives against a
    stale capacity, and the packer writes past the end of the buffer it just
    allocated.
    """
    try:
        from skinny.renderer import Renderer
    except OSError as exc:  # libvulkan not on the dylib path
        pytest.skip(f"needs the Vulkan SDK on the dylib path: {exc}")

    rset, _ = build()
    r = Renderer.__new__(Renderer)
    r._gpu_set = rset

    for field in ("material_capacity", "instance_capacity",
                  "emissive_tri_capacity"):
        # Reading the renderer reads the set.
        setattr(rset.sizes, field, 123)
        assert getattr(r, field) == 123
        # Writing the renderer writes the set — no second copy to forget.
        setattr(r, field, 456)
        assert getattr(rset.sizes, field) == 456
        assert field not in r.__dict__


def test_capacities_are_settable_before_the_set_exists():
    """`__init__` assigns capacities before `_init_gpu` builds the set;
    construction order must be unchanged."""
    try:
        from skinny.renderer import Renderer
    except OSError as exc:
        pytest.skip(f"needs the Vulkan SDK on the dylib path: {exc}")

    r = Renderer.__new__(Renderer)
    r.material_capacity = 16
    assert r.material_capacity == 16
    rset, _ = build(material_capacity=99)
    r._gpu_set = rset
    assert r.material_capacity == 99  # the set takes over once it exists


# `test_mlt_binding_numbers_agree_with_the_wavefront_pass` lived here: it
# compared `MLT_BINDINGS` against `WavefrontMltPass._BINDINGS`. Both now derive
# from `wavefront_layout.MLT_CHAIN_BUFFERS` (change mlt-binding-declaration), so
# it would compare a value with itself — a test that cannot fail is worse than
# no test, because it reads as coverage. Replaced by, and strictly weaker than,
# `tests/test_mlt_binding_declaration.py`, which checks the declaration against
# the shader's own `[[vk::binding(N)]] … <name>` pairing.


def test_pool_sizes_counted_from_declarations():
    """The descriptor-pool tally is derived, not hand-maintained — it must
    still reproduce the pre-change counts (22 storage buffers, 3 storage
    images, 6 samplers + the bindless array, 1 UBO)."""
    rset, _ = build()
    counts = {
        d.descriptor: sum(
            1 for x in rset.declarations
            if x.binding is not None and x.descriptor == d.descriptor)
        for d in DECLARATIONS
    }
    assert counts[BUFFER] == 22
    assert counts[IMAGE] == 3
    assert counts[SAMPLER] == 6
    assert counts[UBO] == 1
