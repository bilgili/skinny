"""Regression guards for the wavefront bdpt kernels: SPIR-V compile size + the
GPU pipeline build.

Why: MoltenVK's SPIR-V->Metal compile (done at vkCreateComputePipelines time)
becomes flaky for oversized compute kernels. The path tracer's catch-all shade
kernel hit that at ~2.83 MB and motivated the 5.4-A per-type split. The bdpt
walk kernel (wfBdptWalk) imports the full flat-BSDF + bdpt subpath/connection
tree and is the largest bdpt kernel. It was measured 10/10 clean, deterministic
to compile (shaderball scene, MoltenVK + Vulkan SDK 1.4.341.1, 2026-06-01), so
no split is warranted now. These guards fail loudly if a future change pushes
the kernel toward the danger zone (re-measure / split before shipping) or breaks
the pipeline build outright.

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

# Per-kernel baseline-compile budgets (bytes). Measured baseline sizes (no graph
# injection) 2026-06-01:
#   wfBdptWalk     1,722,112  (~1.64 MiB)   demo-injected: 1,884,528
#   wfBdptConnect  1,204,760  (~1.15 MiB)   demo-injected: 1,328,340
# Danger reference: the path catch-all (wfPathShade) was MoltenVK-flaky at
# ~2,833,808 B. Budgets keep headroom over the baseline yet stay below that
# known-bad size. If a kernel trips its budget, re-measure compile flakiness (or
# split the kernel, path 5.4-A staged-bounce style) before raising the number.
BDPT_SPV_BUDGETS = {
    "wfBdptWalk": 2_400_000,
    "wfBdptConnect": 2_000_000,
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


@pytest.mark.gpu
def test_wavefront_bdpt_pipelines_build():
    """The three bdpt compute pipelines (walk/connect/resolve) build on this
    driver -- a focused vkCreateComputePipelines guard, lighter than the A/B
    image parity test."""
    from skinny.renderer import Renderer
    from skinny.vk_context import VulkanContext

    ctx = VulkanContext(window=None, width=WIDTH, height=HEIGHT)
    renderer = Renderer(
        vk_ctx=ctx, shader_dir=SHADER_DIR, hdr_dir=HDR_DIR,
        tattoo_dir=TATTOO_DIR, usd_scene_path=DEMO_SCENE,
    )
    try:
        deadline = 200
        while deadline > 0 and (
            renderer._usd_scene is None
            or len(renderer._usd_scene.instances) < 3
            or renderer.pipeline is None
        ):
            renderer.update(0.025)
            deadline -= 1
        assert renderer.pipeline is not None, "megakernel pipeline not built (scene load)"

        renderer.integrator_index = 1  # bdpt
        assert renderer.WAVEFRONT_BDPT_SUPPORTED, "wavefront bdpt gated off"
        renderer.set_execution_mode(1)  # wavefront
        renderer._material_version += 1
        renderer.update(0.04)
        renderer.render_headless()  # builds + dispatches the bdpt pass

        assert renderer.effective_execution_mode_index == 1, (
            "wavefront not active for bdpt (capability gate fell back)"
        )
        bdpt = renderer._wavefront_bdpt_pass
        assert bdpt is not None, "bdpt pass not built after a wavefront-bdpt render"
        for entry in ("wfBdptWalk", "wfBdptConnect", "wfBdptResolve"):
            assert bdpt._pipelines.get(entry) is not None, (
                f"{entry} pipeline did not build"
            )
    finally:
        renderer.cleanup()
        ctx.destroy()
