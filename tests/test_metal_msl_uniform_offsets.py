"""Task 6.3 — MSL uniform-offset / size invariants.

Two halves:

* **Host invariants (always run, no GPU):** the Vulkan *scalar* ``FrameConstants``
  blob is exactly 544 B — ``_FC_SCALAR_FIELDS`` (the relocation table the MSL packer
  walks) covers it with no gap or overlap — and ``SkinParameters.pack()`` (the
  ``std140`` skin UBO) is byte-stable. These pin the Vulkan side "byte-unchanged".

* **Metal MSL pin (guarded, off by default):** when run on a Metal device under
  ``scripts/guarded_metal.sh`` it asserts the reflected ``fc`` block is 592 B with the
  float3 fields at their 16-aligned MSL offsets and that ``_pack_uniforms_msl`` packs to
  exactly that size. In normal CI this equality is *already* enforced live: every Metal
  frame runs the ``_pack_uniforms_msl`` drift guards (``assert off == len(scalar)`` and
  ``assert len(out) == size``, task 3.3) and the shaded-parity / head-render tests exercise
  them — so the host invariants below are the everyday regression net.

Run the Metal half:

    export VULKAN_SDK=/Users/ahmetbilgili/VulkanSDK/1.4.341.1/macOS
    export DYLD_LIBRARY_PATH=$VULKAN_SDK/lib
    RUN_METAL_MEGAKERNEL_COMPILE=1 PYTHONPATH=$PWD/src \
        scripts/guarded_metal.sh -- \
        <repo>/bin/python3.13 -m pytest tests/test_metal_msl_uniform_offsets.py -q
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

_COMPILE_GATE = "RUN_METAL_MEGAKERNEL_COMPILE"
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SHADER_DIR = _PROJECT_ROOT / "src" / "skinny" / "shaders"

# Importing skinny.renderer pulls in `import vulkan` unconditionally; skip cleanly
# when the Vulkan SDK is not on the dylib path (the host invariants still need the
# module for the field table + packers).
try:
    from skinny.renderer import (
        _FC_SCALAR_FIELDS,
        _VK_UNIFORM_BUFFER_BYTES,
        SkinParameters,
    )
except OSError as exc:  # pragma: no cover - environment dependent
    pytest.skip(f"needs the Vulkan SDK on the dylib path: {exc}",
                allow_module_level=True)

# Expected sizes — the Vulkan scalar FrameConstants blob and the std140 skin UBO.
# If either struct grows, update the packer AND this pin in the same change.
_VK_SCALAR_FC_BYTES = 544  # 516 + 28 SPPM tail (changes photon-mapping-sppm +
#   sppm-glossy-final-gather): sppmInitialRadius + sppmCellSize + sppmGridRes(float3)
#   + sppmPhotonsEmitted + sppmGlossyContinueRoughness
_SKIN_PARAMS_BYTES = 80
# Reflected MSL `fc` size on Metal (float3 padded to 16 B; design D3 / task 1.2).
# 592 -> 640 B (change sppm-glossy-final-gather): unlike recordMode/cameraMirror —
# which landed in the struct's existing trailing padding and kept it at 592 — the
# new 4-byte float sppmGlossyContinueRoughness (scalar blob 540 -> 544) tips the
# reflected struct past that padding to the next alignment multiple. Verified live
# under guarded Metal (RUN_METAL_MEGAKERNEL_COMPILE=1): MetalContext main_pass
# reflects 640 B and _pack_uniforms_msl packs to exactly that (it sizes from the
# reflection, so the Metal megakernel self-adapts; only this pin is hand-tracked).
_MSL_FC_BYTES = 640


def test_vulkan_scalar_fc_blob_is_544():
    """`_FC_SCALAR_FIELDS` covers the whole 544 B scalar blob, no gap/overlap — this
    is the table `_pack_uniforms_msl` walks to relocate each field into MSL."""
    total = sum(sz for _, sz in _FC_SCALAR_FIELDS)
    assert total == _VK_SCALAR_FC_BYTES, (
        f"_FC_SCALAR_FIELDS covers {total} B, expected {_VK_SCALAR_FC_BYTES}")


def test_vulkan_ubo_covers_scalar_blob():
    """The Vulkan FrameConstants UBO must be ≥ the scalar blob or `upload` silently
    truncates the tail (this dropped cameraMirror on Vulkan; Metal was fine)."""
    total = sum(sz for _, sz in _FC_SCALAR_FIELDS)
    assert _VK_UNIFORM_BUFFER_BYTES >= total, (
        f"Vulkan UBO is {_VK_UNIFORM_BUFFER_BYTES} B but the scalar blob is {total} B")


def test_skin_params_std140_pack_is_byte_stable():
    """The Vulkan std140 skin UBO must stay byte-unchanged (task 6.3 / 3.2)."""
    blob = SkinParameters().pack()
    assert len(blob) == _SKIN_PARAMS_BYTES, (
        f"SkinParameters.pack() is {len(blob)} B, expected {_SKIN_PARAMS_BYTES}")


@pytest.mark.skipif(
    os.environ.get(_COMPILE_GATE) != "1",
    reason=(
        f"builds the megakernel on Metal (MTLCompilerService RAM spike); set "
        f"{_COMPILE_GATE}=1 and run under scripts/guarded_metal.sh"
    ),
)
def test_metal_msl_fc_size_matches_reflection():
    """On a live Metal device the reflected `fc` block is 592 B with float3 fields
    16-aligned, and `_pack_uniforms_msl` packs to exactly that size."""
    from skinny.backend_select import metal_available
    ok, reason = metal_available()
    if not ok:
        pytest.skip(f"native Metal unavailable: {reason}")
    from skinny.metal_compute import ComputePipeline
    from skinny.metal_context import MetalContext

    ctx = MetalContext(window=None, width=8, height=8)
    try:
        pipe = ComputePipeline(ctx, _SHADER_DIR, entry_module="main_pass",
                               entry_point="mainImage", graph_fragments=[])
        assert pipe.uniform_size == _MSL_FC_BYTES, (
            f"reflected MSL fc size {pipe.uniform_size}, expected {_MSL_FC_BYTES}")
        layout = pipe.uniform_layout
        # float3 fields land at 16-aligned MSL offsets (task 1.2 / 3.1).
        assert layout["focusPlaneOrigin"][0] == 416, layout["focusPlaneOrigin"]
        assert layout["focusPlaneNormal"][0] == 432, layout["focusPlaneNormal"]
        assert layout["camera.position"][0] == 256, layout["camera.position"]
    finally:
        ctx.destroy()
