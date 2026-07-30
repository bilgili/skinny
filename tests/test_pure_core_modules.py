"""The renderer's device-free core must import with no GPU package present.

Change ``renderer-pure-core-extraction``. Before the split, every symbol above
the ``Renderer`` class — the material packers, the camera math, the film
writers, the SPPM photon budget — sat behind ``renderer.py``'s module-scope
``import vulkan``. On a Metal-only host that import raises ``OSError``, so nine
test files guarded themselves with a ``pytest.skip`` and went **green without
running**. The packers that produce the bytes the Metal backend uploads were
therefore only ever checked where the Vulkan SDK happened to be installed.

This file makes "device-free" a property instead of a comment. Each module is
imported in a subprocess in which every GPU package is made unimportable; the
import must still succeed. Same shape as the no-Qt-import check in
``tests/test_render_session_module.py``.

Hostless: plain ``pytest`` runs it.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

# Modules whose names `renderer.py` re-exports, so pre-split call sites still
# resolve them through `skinny.renderer`. The re-export gate below covers these.
RE_EXPORTED_MODULES = (
    "skinny.camera",
    "skinny.film_io",
    "skinny.material_pack",
    "skinny.renderer_helpers",
    "skinny.skin_params",
    "skinny.sppm_budget",
    "skinny.texture_pool",
)

# Device-free modules `renderer.py` consumes as MODULES (`frame_plan.derive`,
# `mlt_chain.run_bootstrap`, …) rather than by re-exported name. They carry the
# same device-free obligation and none of the re-export one.
MODULE_IMPORTED_MODULES = (
    "skinny.frame_derive",
    "skinny.frame_plan",
    "skinny.mlt_chain",
)

# Every module the pure-core extraction produced. Adding a module to the
# device-free side means adding it here — that is the whole gate.
PURE_MODULES = RE_EXPORTED_MODULES + MODULE_IMPORTED_MODULES

# Packages that mean "this module reached a GPU". `vulkan` is the one
# `renderer.py` imports at module scope; `slangpy` is the Metal backend's
# equivalent. Neither may be needed to import — or be pulled in by — a pure
# module.
GPU_PACKAGES = ("vulkan", "slangpy")

_BLOCKER = """
import sys

class _Blocked:
    def find_module(self, name, path=None):
        return self.find_spec(name, path)

    def find_spec(self, name, path=None, target=None):
        root = name.split(".")[0]
        if root in {gpu!r}:
            raise ImportError(
                "blocked by the pure-core import gate: " + name)
        return None

sys.meta_path.insert(0, _Blocked())
"""


def _run_blocked(body: str) -> subprocess.CompletedProcess:
    code = _BLOCKER.format(gpu=set(GPU_PACKAGES)) + textwrap.dedent(body)
    return subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False,
    )


@pytest.mark.parametrize("module", PURE_MODULES)
def test_module_imports_with_no_gpu_package(module: str) -> None:
    """The import succeeds in a process where no GPU package can be imported."""
    result = _run_blocked(f"""
        import importlib
        importlib.import_module({module!r})
        print("ok")
    """)
    assert result.returncode == 0, (
        f"{module} could not be imported without a GPU package:\n{result.stderr}")
    assert "ok" in result.stdout


@pytest.mark.parametrize("module", PURE_MODULES)
def test_module_pulls_in_no_gpu_package(module: str) -> None:
    """Nor does importing it drag one in transitively.

    The blocker above raises on a GPU import, so a transitive one would already
    fail the previous test — but only where the package is installed at all.
    This asserts on ``sys.modules`` so the property holds on a host where the
    package is simply absent and the blocker never fires.
    """
    result = _run_blocked(f"""
        import importlib, sys
        importlib.import_module({module!r})
        leaked = sorted(
            m for m in sys.modules
            if m.split(".")[0] in {set(GPU_PACKAGES)!r}
        )
        assert not leaked, leaked
        print("ok")
    """)
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_pure_modules_do_not_import_the_renderer() -> None:
    """No cycle back through ``renderer.py`` (task 3.4).

    ``renderer`` imports all seven to re-export them. If any of them imported
    ``renderer`` back, the cycle would resolve by accident of import order — and
    would also re-introduce the GPU dependency the split removed.
    """
    result = _run_blocked(f"""
        import importlib, sys
        for name in {PURE_MODULES!r}:
            importlib.import_module(name)
        assert "skinny.renderer" not in sys.modules, "cycle back into renderer"
        print("ok")
    """)
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_renderer_re_exports_every_moved_name() -> None:
    """Source call sites keep working through ``skinny.renderer`` (design D2).

    Parsed from source rather than imported: this test is hostless, and
    importing ``skinny.renderer`` is exactly the thing that needs the SDK.
    """
    import ast
    import pathlib

    pkg = pathlib.Path(__file__).resolve().parents[1] / "src" / "skinny"
    tree = ast.parse((pkg / "renderer.py").read_text())
    re_exported = {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module in PURE_MODULES
        for alias in node.names
    }

    for module in RE_EXPORTED_MODULES:
        mod_tree = ast.parse((pkg / f"{module.split('.')[1]}.py").read_text())
        declared: set[str] = set()
        for node in mod_tree.body:  # top level only — an import is not a declaration
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                declared.add(node.name)
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                declared.add(node.target.id)
            elif isinstance(node, ast.Assign):
                declared.update(
                    t.id for t in node.targets if isinstance(t, ast.Name))
        missing = declared - re_exported
        assert not missing, (
            f"{module}: not re-exported from skinny.renderer: {sorted(missing)}")


class _FakeSampledImage:
    """Recording stand-in for the backend's ``SampledImage`` (design D4)."""

    def __init__(self, ctx, width, height, **kwargs):
        self.size = (width, height)
        self.kwargs = kwargs
        self.uploaded: bytes | None = None
        self.destroyed = False

    def upload_sync(self, data: bytes) -> None:
        self.uploaded = data

    def destroy(self) -> None:
        self.destroyed = True


class _FakeGpu:
    """Stand-in for ``vk_compute`` / ``metal_compute``."""

    BINDLESS_TEXTURE_CAPACITY = 2
    SampledImage = _FakeSampledImage


def test_texture_pool_dedupes_and_destroys_without_a_device(tmp_path) -> None:
    """``TexturePool`` runs against a fake resource module — no GPU (design D4)."""
    from PIL import Image

    from skinny.texture_pool import TexturePool

    png = tmp_path / "t.png"
    Image.new("RGBA", (4, 4), (255, 0, 0, 255)).save(png)

    pool = TexturePool(object(), _FakeGpu())

    assert pool.add_or_get(tmp_path / "missing.png") == TexturePool.SENTINEL
    first = pool.add_or_get(png)
    assert first == 0
    assert pool.add_or_get(png) == first, "same key must reuse the slot"
    # A different wrap mode owns its own sampler, so it takes a fresh slot.
    assert pool.add_or_get(png, wrap_s="clamp") == 1
    # Capacity is 2 — the third distinct key has nowhere to go.
    assert pool.add_or_get(png, linear=True) == TexturePool.SENTINEL

    slots = pool.filled_slots()
    assert [i for i, _ in slots] == [0, 1]

    pool.destroy()
    assert all(img.destroyed for _, img in slots)
