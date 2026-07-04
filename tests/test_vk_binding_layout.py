"""Hostless audit: every shared scene-set binding declared in bindings.slang
(Vulkan branch) must be declared in the Vulkan set-0 descriptor-set layout
(change fix-vulkan-volume-density-binding).

A ``[[vk::binding(N)]]`` the megakernel SPIR-V references but the pipeline
layout omits is undefined behaviour on Vulkan and a hard pipeline-build
failure on MoltenVK (``SPIR-V to MSL conversion error: nullptr`` — MoltenVK
derives its SPIR-V→MSL resource map from the pipeline layout). That is exactly
how nanovdb-volume-rendering's ``volumeDensity`` (binding 26) silently broke
every raw-``VulkanContext`` render on macOS: the shader declaration, the
descriptor write, and the pool sizing all landed, but the hand-maintained
layout list in ``ComputePipeline._create_descriptor_set_layout`` did not.

Pure text analysis (no ``import vulkan``, no GPU): a mini-preprocessor walks
``bindings.slang`` with ``SKINNY_METAL``/``SKINNY_METAL_*`` undefined and
collects the active single-argument (= set 0) binding indices; a source scan
of ``_create_descriptor_set_layout`` collects the layout's ``binding=N``
entries. Shader ⊆ layout must hold. Scope note: standalone pipelines
(``skin.slang``, ``bvh_refit.slang``, ``preview_pass.slang``, …) own separate
layouts and are out of scope; ``bindings.slang`` is the one shared surface
where new scene bindings land.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BINDINGS_SLANG = PROJECT_ROOT / "src" / "skinny" / "shaders" / "bindings.slang"
VK_COMPUTE_PY = PROJECT_ROOT / "src" / "skinny" / "vk_compute.py"

# The single combined MaterialX graph-param buffer: declared in the layout only
# when the scene carries at least one nodegraph (GRAPH_BINDING_BASE), and its
# shader declaration lives in generated code, not bindings.slang.
GRAPH_BINDING_BASE = 25


def _eval_preproc_condition(cond: str) -> bool:
    """Evaluate a bindings.slang ``#if`` condition with no macros defined
    (the Vulkan / non-Metal build: SKINNY_METAL, SKINNY_METAL_RECORDS,
    SKINNY_METAL_NEURAL … all undefined)."""
    py = re.sub(r"defined\s*\(\s*\w+\s*\)", "False", cond)
    py = py.replace("&&", " and ").replace("||", " or ").replace("!", " not ")
    return bool(eval(py, {"__builtins__": {}}, {}))  # noqa: S307 - fixed grammar


def vulkan_branch_bindings(slang_path: Path) -> set[int]:
    """Single-arg ``[[vk::binding(N)]]`` indices active in the Vulkan build."""
    active = [True]  # condition stack; declarations count when all(active)
    out: set[int] = set()
    for raw in slang_path.read_text().splitlines():
        line = raw.strip()
        if line.startswith("#if"):
            active.append(_eval_preproc_condition(line.split(None, 1)[1]))
        elif line.startswith("#else"):
            active[-1] = not active[-1]
        elif line.startswith("#endif"):
            active.pop()
        elif all(active):
            for m in re.finditer(r"\[\[vk::binding\((\d+)\)\]\]", line):
                out.add(int(m.group(1)))
    assert len(active) == 1, "unbalanced #if/#endif in bindings.slang"
    return out


def layout_bindings(vk_compute_path: Path) -> set[int]:
    """``binding=N`` entries inside ``_create_descriptor_set_layout``."""
    src = vk_compute_path.read_text()
    m = re.search(
        r"def _create_descriptor_set_layout\(self\).*?\n(    def |\Z)", src, re.S)
    assert m, "_create_descriptor_set_layout not found in vk_compute.py"
    body = m.group(0)
    found = {int(n) for n in re.findall(r"\bbinding=(\d+)\b", body)}
    if "GRAPH_BINDING_BASE" in body:
        found.add(GRAPH_BINDING_BASE)
    return found


def test_bindings_slang_vulkan_branch_covered_by_set0_layout():
    shader = vulkan_branch_bindings(BINDINGS_SLANG)
    layout = layout_bindings(VK_COMPUTE_PY)
    missing = sorted(shader - layout)
    assert not missing, (
        f"bindings.slang declares Vulkan set-0 bindings {missing} that "
        "_create_descriptor_set_layout does not declare — every Vulkan "
        "pipeline build will fail on MoltenVK (VUID-07988 → SPIR-V to MSL "
        "conversion error: nullptr). Add the matching "
        "VkDescriptorSetLayoutBinding entries in vk_compute.py."
    )


def test_audit_parsers_see_expected_shape():
    """Guard the audit itself: the parsers must keep finding the known
    surface (a regex gone stale would make the main test vacuously green)."""
    shader = vulkan_branch_bindings(BINDINGS_SLANG)
    # Anchor points that exist in the Vulkan branch today.
    assert {1, 2, 13, 14, 26, 30}.issubset(shader), shader
    # Metal-only declarations must be excluded by the mini-preprocessor.
    assert 38 not in shader and 44 not in shader, shader
    layout = layout_bindings(VK_COMPUTE_PY)
    assert {0, 14, 24, 31, 37, GRAPH_BINDING_BASE}.issubset(layout), layout
    assert len(layout) > 20, layout


def test_audit_uses_current_sources():
    """The audit reads the real files, not fixtures."""
    assert BINDINGS_SLANG.exists() and VK_COMPUTE_PY.exists()
    assert "volumeDensity" in BINDINGS_SLANG.read_text()
    assert "_create_descriptor_set_layout" in inspect.cleandoc(
        VK_COMPUTE_PY.read_text())
