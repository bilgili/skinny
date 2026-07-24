"""Hostless guards for the parity_core/parity split (change parity-pure-core-split).

Two invariants, both green before the split and after it:

1. **Surface compatibility** — every name the tests, ``furnace.py`` and the docs
   consume still resolves from ``skinny.pbrt.parity`` (public *and* the three
   consumed private names), and the facade's ``combo_is_valid`` /
   ``self_consistency_tol`` still behave through it.
2. **Tolerance-table pinning** — the RGB and spectral self-consistency tables
   equal their pre-split literals at full precision, so deriving the spectral
   table as an overlay over the RGB one can never silently move a value.
"""

from __future__ import annotations

import pytest

from skinny.pbrt import parity

# The full surface grepped from tests/ + src/ before the split.
_PUBLIC_SURFACE = (
    "SceneSpec",
    "RenderCombo",
    "ParityResult",
    "ANCHOR",
    "SPECTRAL_ANCHOR",
    "INTEGRATORS",
    "EXECUTION_MODES",
    "PROPOSAL_AXES",
    "REUSE_AXES",
    "all_combos",
    "combo_is_valid",
    "combo_axis_class",
    "enumerate_combos",
    "spectral_envelope",
    "spectral_selfconsistency_assertable",
    "self_consistency_anchor",
    "self_consistency_tol",
    "load_manifest",
    "materialx_specs",
    "reference_exists",
    "pbrt_truth_result",
    "absolute_radiance_result",
    "self_consistency_result",
    "authoring_equivalence_result",
    "render_log_path",
    "render_linear",
    "render_combo",
    "evaluate",
    "scene_has_environment",
)

# Private names consumed by tests/furnace.py — ``import *`` would skip these, so
# they are pinned explicitly (design D2).
_PRIVATE_SURFACE = (
    "_DEFAULT_SELF_CONSISTENCY",
    "_DEFAULT_SPECTRAL_SELF_CONSISTENCY",
    "_scene_source",
    "_render_log",
)


@pytest.mark.parametrize("name", _PUBLIC_SURFACE + _PRIVATE_SURFACE)
def test_parity_surface_intact(name):
    assert getattr(parity, name, None) is not None, f"skinny.pbrt.parity lost {name}"


def test_parity_surface_importable_by_from_import():
    """``from skinny.pbrt.parity import <name>`` resolves for the whole surface."""
    mod = __import__(
        "skinny.pbrt.parity",
        fromlist=list(_PUBLIC_SURFACE + _PRIVATE_SURFACE),
    )
    missing = [n for n in _PUBLIC_SURFACE + _PRIVATE_SURFACE if not hasattr(mod, n)]
    assert not missing


def test_parity_imports_without_pxr():
    """Neither module drags USD in at import time (the ``import_pbrt`` import is
    lazy inside ``render_linear``)."""
    import subprocess
    import sys

    code = (
        "import sys\n"
        "class _Block:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name == 'pxr' or name.startswith('pxr.'):\n"
        "            raise ImportError('pxr blocked')\n"
        "        return None\n"
        "sys.meta_path.insert(0, _Block())\n"
        "import skinny.pbrt.parity_core as core\n"
        "import skinny.pbrt.parity as p\n"
        "assert 'pxr' not in sys.modules\n"
        "assert 'skinny.pbrt.api' not in sys.modules\n"
        "assert core.combo_is_valid and p.combo_is_valid is core.combo_is_valid\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_combo_is_valid_through_facade():
    """The validity oracle still answers through the ``parity`` facade."""
    flat = parity.SceneSpec(name="s", file="s.pbrt", ref="s.exr", width=8, height=8,
                            spp=1, relmse_tol=0.1, flip_tol=0.1)
    ok, reason = parity.combo_is_valid(parity.ANCHOR, flat)
    assert ok and reason == ""
    # A recorded refusal still carries a reason (nothing dropped silently).
    bad = parity.RenderCombo(integrator="sppm", execution_mode="megakernel")
    ok, reason = parity.combo_is_valid(bad, flat)
    assert not ok and reason


def test_self_consistency_tol_through_facade():
    flat = parity.SceneSpec(name="s", file="s.pbrt", ref="s.exr", width=8, height=8,
                            spp=1, relmse_tol=0.1, flip_tol=0.1)
    wave_path = parity.RenderCombo(integrator="path", execution_mode="megakernel")
    assert parity.self_consistency_tol(wave_path, flat) == (0.02, 0.03)
    spec_path = parity.RenderCombo(integrator="path", execution_mode="megakernel",
                                   spectral=True)
    assert parity.self_consistency_tol(spec_path, flat) == (0.03, 0.03)


# ─── tolerance-table pinning ──────────────────────────────────────────────
#
# Full-precision literals copied from the pre-split module. The spectral table is
# derived from the RGB one by an overlay; these assertions are what make that
# derivation safe.

_RGB_TABLE = {
    "mode": {"relmse": 0.02, "flip": 0.03},
    "integrator": {"relmse": 0.06, "flip": 0.06},
    "sppm": {"relmse": 0.15, "flip": 0.12},
    "mlt": {"relmse": 0.15, "flip": 0.12},
    "unbiased": {"relmse": 0.05, "flip": 0.05},
}

_SPECTRAL_TABLE = {
    "mode": {"relmse": 0.03, "flip": 0.03},
    "integrator": {"relmse": 0.09, "flip": 0.06},
    "sppm": {"relmse": 0.15, "flip": 0.12},
    "mlt": {"relmse": 0.15, "flip": 0.12},
    "unbiased": {"relmse": 0.05, "flip": 0.05},
}


def test_rgb_self_consistency_table_pinned():
    assert parity._DEFAULT_SELF_CONSISTENCY == _RGB_TABLE


def test_spectral_self_consistency_table_pinned():
    assert parity._DEFAULT_SPECTRAL_SELF_CONSISTENCY == _SPECTRAL_TABLE


def test_spectral_table_widens_only_two_rows():
    """The spectral table differs from the RGB table in exactly mode/integrator relmse."""
    diff = {
        cls: {k: v for k, v in row.items() if _RGB_TABLE[cls][k] != v}
        for cls, row in _SPECTRAL_TABLE.items()
    }
    assert {c: d for c, d in diff.items() if d} == {
        "mode": {"relmse": 0.03},
        "integrator": {"relmse": 0.09},
    }
