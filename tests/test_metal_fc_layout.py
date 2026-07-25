"""gpu-marked MSL ground truth for the derived `fc` layout
(change reflection-owned-byte-layouts, task 2.4).

``slang_layout`` computes the Metal (MSL) offsets of ``FrameConstants`` from the
parsed declaration, using size/alignment rules that ``FrameConstants`` is the
first struct to need (``float4x4``, ``uint2``, ``uint3``, nested-struct
flattening). This locks those rules to what Slang's Metal target *actually*
emits, by reflecting the real ``fc`` uniform block out of a compiled probe
program — the same reflection ``metal_compute.ComputePipeline`` consumes.

RGB and MLT variants both, because the MLT tail changes the struct.

Run (guarded runner, one Metal process — see CLAUDE.md dispatch hygiene):

    PYTHONPATH=src SKINNY_BACKEND=metal ./bin/python3.13 -m pytest \\
        tests/test_metal_fc_layout.py -m gpu -q
"""

from __future__ import annotations

from pathlib import Path

import pytest

from skinny import slang_layout as sl

_SHADERS = Path(__file__).resolve().parent.parent / "src" / "skinny" / "shaders"

pytestmark = pytest.mark.gpu


def _reflect_fc(*, mlt: bool):
    """Reflect ``{field: (offset, size)}`` + struct size of the ``fc`` uniform
    block under Slang's Metal target. Returns None when no Metal device exists."""
    spy = pytest.importorskip("slangpy")
    from skinny.backend_select import metal_available

    ok, _reason = metal_available()
    if not ok:
        return None
    opts = spy.SlangCompilerOptions()
    opts.include_paths = [_SHADERS, _SHADERS.parent / "mtlx" / "genslang"]
    defines = {"SKINNY_COMPUTE_PIPELINE": "1", "SKINNY_METAL": "1"}
    if mlt:
        defines["SKINNY_MLT"] = "1"
    opts.defines = defines
    # Match the megakernel's compile (metal_compute._build) — matrix layout does
    # not move a field offset, but keep the probe faithful.
    opts.matrix_layout = spy.SlangMatrixLayout.column_major
    dev = spy.create_device(
        type=spy.DeviceType.metal,
        include_paths=[str(_SHADERS), str(_SHADERS.parent / "mtlx" / "genslang")],
    )
    try:
        # Touch enough of `fc` that nothing is dead-stripped from the block.
        touch = "fc.mltSigma + " if mlt else ""
        src = (
            "import bindings;\n"
            "RWStructuredBuffer<float> probe_out;\n"
            '[shader("compute")] [numthreads(1, 1, 1)]\n'
            "void m(uint3 t : SV_DispatchThreadID) {\n"
            f"    probe_out[0] = {touch}fc.time + fc.camera.fov "
            "+ float(fc.tileOriginY) + fc.exposure + fc.proposalAlpha.x\n"
            "        + fc.focusPlaneOrigin.x + fc.zoomMin.x + float(fc.pickPixel.x)\n"
            "        + float(fc.sppmGridRes.x) + fc.camera.viewInverse[0][0];\n"
            "}\n"
        )
        session = dev.create_slang_session(compiler_options=opts)
        module = session.load_module_from_source("fc_layout_probe", src)
        program = session.link_program([module], [module.entry_point("m")])
        p = next(x for x in program.layout.parameters if x.name == "fc")
        tl = p.type_layout
        layout: dict[str, tuple[int, int]] = {}

        def walk(type_layout, base: int, prefix: str) -> None:
            for f in getattr(type_layout, "fields", None) or []:
                off = base + int(f.offset)
                ftl = f.type_layout
                name = f"{prefix}{f.name}"
                layout[name] = (off, int(getattr(ftl, "size", 0)))
                if getattr(ftl, "fields", None):
                    walk(ftl, off, f"{name}.")

        walk(tl, 0, "")
        return layout, int(tl.size)
    finally:
        dev.close()


@pytest.mark.parametrize("mlt", [False, True], ids=["rgb", "mlt"])
def test_derived_msl_fc_layout_matches_live_reflection(mlt):
    refl = _reflect_fc(mlt=mlt)
    if refl is None:
        pytest.skip("no Metal device for MSL reflection")
    live, live_size = refl
    derived = sl.msl_layout("FrameConstants", mlt=mlt, metal=True)
    assert derived.stride == live_size, (derived.stride, live_size)
    # Every field the reflection reports (parents included) must match the
    # derived offset AND size — this is what validates the float4x4 / uint2 /
    # uint3 / nested-struct MSL rules the module introduced.
    mismatched = {
        name: (derived.offsets.get(name), off_size)
        for name, off_size in live.items()
        if derived.offsets.get(name) != off_size
    }
    assert not mismatched, mismatched
    # …and nothing the module derives is absent from the reflection.
    assert set(derived.offsets) == set(live)
