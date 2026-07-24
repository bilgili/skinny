"""Key-equality unit tests for the wavefront pass factories (change
renderer-module-carveout, Stage C, task 3.3).

The `_ensure_*` construction bodies moved into `vk_wavefront.ensure_pass` /
`metal_wavefront.ensure_pass`. Their rebuild keys gate pass reconstruction on
reuse/neural/record/dims changes — a wrong key means a stale pass = wrong image
only after a runtime toggle, which the parity matrix may not exercise. These
tests pin the key TUPLES the factories compute against the pre-carve-out
values, using a stub renderer with no GPU: the factory returns the cached pass
untouched when the key matches, so we drive it through the cache-hit path and
assert the stored `_wf_*_pass_dims` equals the hand-computed pre-carve-out key.
"""

from __future__ import annotations

import pytest

vk = pytest.importorskip("skinny.vk_wavefront")
mw = pytest.importorskip("skinny.metal_wavefront")


class _Reuse:
    def __init__(self, mode):
        self.reuse_mode = mode


class _StubRenderer:
    """Minimal duck-typed renderer exposing only what the key computation +
    cache-hit early-return read. No GPU, no real passes."""

    def __init__(self, *, is_metal, width=128, height=96, reuse_mode=0,
                 neural=False, wf_record=False, has_nonflat=False,
                 restir_regime=0, restir_config=None, walk_mode=0,
                 graph_sig="g0", heavy=False, mlt_chains=16384,
                 mlt_bootstrap=8192):
        self.is_metal = is_metal
        self.width, self.height = width, height
        self._reuse_mode = reuse_mode
        self._neural = neural
        self._wf_record_active = wf_record
        self.restir_regime_index = restir_regime
        self._restir_config = restir_config
        self.bdpt_walk_mode = walk_mode
        self._graph_sig = graph_sig
        self._heavy = heavy
        self.mlt_num_chains = mlt_chains
        self.mlt_bootstrap_samples = mlt_bootstrap
        self._material_types = (5,) if has_nonflat else (1,)
        # Pretend a pass is already built and cached under `cached_key`.
        self._wavefront_path_pass = object()
        self._wavefront_bdpt_pass = object()
        self._wavefront_sppm_pass = object()
        self._wavefront_mlt_pass = object()
        self._wf_path_pass_dims = None
        self._wf_bdpt_pass_dims = None
        self._wf_sppm_pass_dims = None
        self._wf_mlt_pass_dims = None
        # Metal factories gate on scene bindings existing; the Vulkan MLT
        # factory also reads `mlt_bindings`.
        self._scene_bindings = type("_SB", (), {"mlt_bindings": True})()
        # cache-hit restir refresh path: no restir pass.
        self._restir_pass = None

    # --- methods the factories call during key computation ---
    def _active_reuse(self):
        return _Reuse(self._reuse_mode)

    def _neural_active(self):
        return self._neural

    def _graph_set_signature(self):
        return self._graph_sig

    def _has_heavy_nonflat(self):
        return self._heavy

    def _mlt_pass_key(self):
        return (self.width, self.height,
                int(self.mlt_num_chains), int(self.mlt_bootstrap_samples))

    # Vulkan gate reads this attr.
    class _Ctx:
        compute_queue = object()
    ctx = _Ctx()
    descriptor_sets = (object(),)


# Expected pre-carve-out key formulas (verbatim from the former _ensure_* bodies).

def _expect_vk_path(r):
    _rcfg = r._restir_config
    return (r.width, r.height, r._material_types != (1,),
            r._reuse_mode,
            r.restir_regime_index if r._reuse_mode == 1 else None,
            tuple(sorted(_rcfg.items())) if _rcfg else None,
            r._neural, r._wf_record_active)


def _expect_metal_path(r):
    _rcfg = r._restir_config
    return (r.width, r.height, r._material_types != (1,),
            r._graph_sig, r._reuse_mode,
            r.restir_regime_index if r._reuse_mode == 1 else None,
            tuple(sorted(_rcfg.items())) if _rcfg else None,
            r._neural, r._wf_record_active)


def _drive_cache_hit(r, integrator, mod, expected):
    """Seed the cache with `expected`, then assert the factory returns the
    cached pass — which happens iff the key it computes equals `expected`."""
    attr = {"path": "_wf_path_pass_dims", "bdpt": "_wf_bdpt_pass_dims",
            "sppm": "_wf_sppm_pass_dims", "mlt": "_wf_mlt_pass_dims"}[integrator]
    pass_attr = {"path": "_wavefront_path_pass", "bdpt": "_wavefront_bdpt_pass",
                 "sppm": "_wavefront_sppm_pass", "mlt": "_wavefront_mlt_pass"}[integrator]
    setattr(r, attr, expected)
    cached = getattr(r, pass_attr)
    got = mod.ensure_pass(r, integrator)
    assert got is cached, \
        f"{integrator}: factory recomputed a DIFFERENT key than {expected!r}"


# ── Vulkan ───────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("reuse,neural,rec,nonflat", [
    (0, False, False, False),
    (1, False, False, False),   # ReSTIR
    (0, True, False, False),    # neural
    (0, False, True, False),    # record drain
    (0, False, False, True),    # non-flat
])
def test_vk_path_key_matches_precarveout(reuse, neural, rec, nonflat):
    r = _StubRenderer(is_metal=False, reuse_mode=reuse, neural=neural,
                      wf_record=rec, has_nonflat=nonflat)
    _drive_cache_hit(r, "path", vk, _expect_vk_path(r))


def test_vk_bdpt_key_is_dims_only():
    r = _StubRenderer(is_metal=False)
    _drive_cache_hit(r, "bdpt", vk, (r.width, r.height))


def test_vk_sppm_key_is_dims_only():
    r = _StubRenderer(is_metal=False)
    _drive_cache_hit(r, "sppm", vk, (r.width, r.height))


def test_vk_mlt_key_is_dims_and_chain_config():
    r = _StubRenderer(is_metal=False, mlt_chains=4096, mlt_bootstrap=2048)
    _drive_cache_hit(r, "mlt", vk, (r.width, r.height, 4096, 2048))


# ── Metal ────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("reuse,neural,rec,nonflat,graph", [
    (0, False, False, False, "g0"),
    (1, False, False, False, "g0"),
    (0, True, False, False, "g0"),
    (0, False, True, False, "g0"),
    (0, False, False, True, "gX"),
])
def test_metal_path_key_matches_precarveout(reuse, neural, rec, nonflat, graph):
    r = _StubRenderer(is_metal=True, reuse_mode=reuse, neural=neural,
                      wf_record=rec, has_nonflat=nonflat, graph_sig=graph)
    _drive_cache_hit(r, "path", mw, _expect_metal_path(r))


def test_metal_bdpt_key_includes_walk_graph_heavy():
    r = _StubRenderer(is_metal=True, walk_mode=2, graph_sig="gB", heavy=True)
    _drive_cache_hit(r, "bdpt", mw,
                     (r.width, r.height, 2, "gB", True))


def test_metal_sppm_key_includes_graph_heavy():
    r = _StubRenderer(is_metal=True, graph_sig="gS", heavy=True)
    _drive_cache_hit(r, "sppm", mw, (r.width, r.height, "gS", True))


def test_metal_mlt_key_is_dims_and_chain_config():
    r = _StubRenderer(is_metal=True, mlt_chains=8192, mlt_bootstrap=1024)
    _drive_cache_hit(r, "mlt", mw, (r.width, r.height, 8192, 1024))
