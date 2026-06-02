"""Regression guards for the wavefront bdpt kernels: SPIR-V compile size + the
GPU pipeline build.

Why: MoltenVK's SPIR-V->Metal compile (done at vkCreateComputePipelines time)
becomes flaky for oversized compute kernels. The path tracer's catch-all shade
kernel hit that at ~2.83 MB and motivated the 5.4-A per-type split. The bdpt
pass is now fully staged (eye/light walks into per-bounce extend kernels, the
connection split into NEE/FULL); the heavy material-tree kernels are the two
connect kernels (~1.15 MiB) and the bounce-extend kernels (≤0.49 MiB), all well
under the danger zone. These guards fail loudly if a future change pushes a
kernel toward it (re-measure / split before shipping) or breaks the pipeline
build outright.

The size guard compiles the *scene-independent baseline* of each kernel with
slangc: the flat-BSDF + bdpt machinery with a no-op `evalSceneGraph` (zero
MaterialX nodegraphs). That is the renderer-code floor; per-scene graph
injection (~+0.15 MB for the 3-graph demo) stacks on top at runtime. The
baseline is what a renderer-code change regresses, needs only slangc (no GPU
device), and is skipped where slangc is unavailable.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SHADER_DIR = PROJECT_ROOT / "src" / "skinny" / "shaders"
MTLX_GENSLANG = PROJECT_ROOT / "src" / "skinny" / "mtlx" / "genslang"
HDR_DIR = PROJECT_ROOT / "hdrs"
TATTOO_DIR = PROJECT_ROOT / "tattoos"
DEMO_SCENE = PROJECT_ROOT / "assets" / "three_materials_demo.usda"

# Per-kernel baseline-compile budgets (bytes) for the heavy (material-tree)
# staged bdpt kernels. Measured baseline sizes (no graph injection) 2026-06-01,
# after the walk/connect staging:
#   wfBdptBounceEye      513,296  (~0.49 MiB)   eye-walk extend
#   wfBdptBounceLight    250,400  (~0.24 MiB)   light-walk extend
#   wfBdptConnectNee   1,206,464  (~1.15 MiB)   emissive + connectT1
#   wfBdptConnectFull  1,206,464  (~1.15 MiB)   + generic + MIS
# (The small kernels — gen/classify/build_args/scatter/splat/resolve — carry no
# material tree and are KB-scale; not budgeted.) Danger reference: the path
# catch-all (wfPathShade) was MoltenVK-flaky at ~2,833,808 B. Budgets keep
# headroom over the baseline yet stay below that known-bad size. If a kernel
# trips its budget, re-measure compile flakiness (or split it further) before
# raising the number.
BDPT_SPV_BUDGETS = {
    "wfBdptBounceEye": 1_500_000,
    "wfBdptBounceLight": 1_200_000,
    "wfBdptConnectNee": 2_000_000,
    "wfBdptConnectFull": 2_000_000,
}

WIDTH = HEIGHT = 96


def _compile_baseline_spv(entry: str, out: Path) -> Path:
    """Compile one bdpt kernel against a no-op (zero-graph) generated_materials,
    mirroring the renderer's runtime slangc invocation. Returns the .spv path."""
    from skinny.vk_compute import emit_megakernel_aggregator

    inc = out.parent
    (inc / "generated_materials.slang").write_text(emit_megakernel_aggregator([], 0))
    cmd = [
        "slangc", str(SHADER_DIR / "wavefront" / "wavefront_bdpt.slang"),
        "-target", "spirv", "-entry", entry, "-stage", "compute",
        "-I", str(inc), "-I", str(SHADER_DIR), "-I", str(MTLX_GENSLANG),
        "-D", "SKINNY_COMPUTE_PIPELINE=1", "-fvk-use-scalar-layout", "-o", str(out),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise AssertionError(f"slangc failed for {entry}:\n{result.stderr}")
    return out


@pytest.mark.parametrize("entry", sorted(BDPT_SPV_BUDGETS))
def test_wavefront_bdpt_spv_under_budget(entry, tmp_path):
    """The bdpt kernel's baseline compiles to SPIR-V under its MoltenVK-safe
    size budget."""
    if shutil.which("slangc") is None:
        pytest.skip("slangc not on PATH — cannot compile the kernel to measure it")

    spv = _compile_baseline_spv(entry, tmp_path / f"{entry}.spv")
    size = spv.stat().st_size
    budget = BDPT_SPV_BUDGETS[entry]
    assert size <= budget, (
        f"{entry} baseline compiles to {size} B (> budget {budget} B). The bdpt "
        f"kernel grew toward MoltenVK's Metal-compile danger zone (the path "
        f"catch-all was flaky at ~2.83 MB). Re-measure compile flakiness or split "
        f"the kernel (path 5.4-A staged-bounce style) before raising the budget."
    )


# Kernels each walk_mode builds (shared connect + resolve in every mode).
_SHARED = (
    "wfBdptClassify", "wfBdptBuildArgs", "wfBdptScatter",
    "wfBdptConnectNee", "wfBdptConnectFull", "wfBdptResolve",
)
_MODE_ENTRIES = {
    "fused": ("wfBdptWalk",) + _SHARED,
    "eye": ("wfBdptGenEye", "wfBdptWalkClassify", "wfBdptBounceEye",
            "wfBdptLightTail") + _SHARED,
    "eye_light": ("wfBdptGenEye", "wfBdptWalkClassify", "wfBdptBounceEye",
                  "wfBdptGenLight", "wfBdptBounceLight", "wfBdptSplat") + _SHARED,
}


@pytest.mark.gpu
@pytest.mark.parametrize("walk_mode", sorted(_MODE_ENTRIES))
def test_wavefront_bdpt_pipelines_build(walk_mode):
    """Each bdpt walk_mode's compute pipelines build on this driver -- a focused
    vkCreateComputePipelines guard, lighter than the A/B image parity test. Only
    the active mode's kernels are compiled, so each mode is checked separately."""
    from skinny.renderer import Renderer
    from skinny.vk_context import VulkanContext

    ctx = VulkanContext(window=None, width=WIDTH, height=HEIGHT)
    renderer = Renderer(
        vk_ctx=ctx, shader_dir=SHADER_DIR, hdr_dir=HDR_DIR,
        tattoo_dir=TATTOO_DIR, usd_scene_path=DEMO_SCENE,
        execution_mode="wavefront", bdpt_walk=walk_mode,
    )
    try:
        deadline = 200
        while deadline > 0 and (
            renderer._usd_scene is None
            or len(renderer._usd_scene.instances) < 3
            or renderer._scene_bindings is None
        ):
            renderer.update(0.025)
            deadline -= 1
        assert renderer._scene_bindings is not None, "scene bindings not built (scene load)"
        assert renderer.pipeline is None, "wavefront must not build the megakernel"

        renderer.integrator_index = 1  # bdpt
        assert renderer.WAVEFRONT_BDPT_SUPPORTED, "wavefront bdpt gated off"
        renderer._material_version += 1
        renderer.update(0.04)
        renderer.render_headless()  # builds + dispatches the bdpt pass

        assert renderer.effective_execution_mode_index == 1, (
            "wavefront not active for bdpt (capability gate fell back)"
        )
        bdpt = renderer._wavefront_bdpt_pass
        assert bdpt is not None, "bdpt pass not built after a wavefront-bdpt render"
        assert bdpt.walk_mode == walk_mode
        for entry in _MODE_ENTRIES[walk_mode]:
            assert bdpt._pipelines.get(entry) is not None, (
                f"{entry} pipeline did not build for walk_mode={walk_mode}"
            )
    finally:
        renderer.cleanup()
        ctx.destroy()
