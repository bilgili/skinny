"""Hostless tests for the MLT host wiring (change mlt-integrator).

Two halves, no GPU:
- the pure bootstrap resample (``skinny.mlt_bootstrap.resample_chain_seeds``:
  b math, weight-proportional seeding, zero-weight refusal, determinism);
- source-level guards that the renderer wiring exists on BOTH backends (the
  ensure/destroy/dispatch seam, the MLT uniform tail, the Metal adapter) — the
  ``test_mlt_selection`` grep-style pattern, since constructing a Renderer
  needs a GPU — plus the pure MSL field-table math, which is importable.
"""

from __future__ import annotations

import inspect
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from skinny.mlt_bootstrap import resample_chain_seeds

_SRC = Path(__file__).resolve().parents[1] / "src" / "skinny"


def _read(rel: str) -> str:
    return (_SRC / rel).read_text()


# ── resample_chain_seeds ─────────────────────────────────────────────────────

def test_b_is_mean_of_weights():
    w = np.array([0.0, 1.0, 3.0, 4.0], dtype=np.float32)
    b, _ = resample_chain_seeds(w, 8, seed=1)
    assert b == pytest.approx(float(w.mean()))


def test_seeds_proportional_to_weights():
    # Index 2 carries 3× index 1's weight; index 0 must never be drawn.
    w = np.array([0.0, 1.0, 3.0], dtype=np.float32)
    _, seeds = resample_chain_seeds(w, 40_000, seed=7)
    assert seeds.dtype == np.uint32 and seeds.shape == (40_000,)
    counts = np.bincount(seeds, minlength=3)
    assert counts[0] == 0
    assert counts[2] / counts[1] == pytest.approx(3.0, rel=0.05)


def test_deterministic_per_seed():
    w = np.array([1.0, 2.0, 3.0])
    _, a = resample_chain_seeds(w, 128, seed=42)
    _, b = resample_chain_seeds(w, 128, seed=42)
    _, c = resample_chain_seeds(w, 128, seed=43)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_all_zero_weights_raise():
    with pytest.raises(RuntimeError, match="no light-carrying paths"):
        resample_chain_seeds(np.zeros(16), 4, seed=0)


def test_nonfinite_only_weights_raise():
    w = np.array([np.nan, np.inf, -1.0, 0.0])
    with pytest.raises(RuntimeError, match="no light-carrying paths"):
        resample_chain_seeds(w, 4, seed=0)


def test_nonfinite_entries_zeroed_not_drawn():
    w = np.array([np.nan, 2.0, np.inf])
    b, seeds = resample_chain_seeds(w, 1000, seed=3)
    assert b == pytest.approx(2.0 / 3.0)  # non-finite zeroed before the mean
    assert set(np.unique(seeds)) == {1}


# ── renderer wiring (source-level, no GPU) ───────────────────────────────────

def test_renderer_has_mlt_pass_lifecycle():
    src = _read("renderer.py")
    for sym in ("_ensure_wavefront_mlt_pass", "_destroy_wavefront_mlt_pass",
                "_run_wavefront_mlt_bootstrap", "resample_chain_seeds",
                "WavefrontMltPass"):
        assert sym in src, f"renderer.py must reference {sym}"


def test_renderer_dispatch_branch_selects_mlt():
    src = _read("renderer.py")
    start = src.index("def _record_wavefront_dispatch")
    body = src[start:start + 4000]
    assert "integrator_index == 3" in body, \
        "_record_wavefront_dispatch must branch on the MLT integrator index"
    assert "_ensure_wavefront_mlt_pass" in body
    assert "record_frame" in body


def test_renderer_packs_mlt_uniform_tail_in_shader_order():
    # The SKINNY_MLT FrameConstants tail: 4 floats then 4 uints. It must be
    # packed BEFORE the trailing tileOriginY filler u32 — `tileOriginY` is
    # SKINNY_METAL-gated, so in the Vulkan MLT SPIR-V `mltSigma` sits at
    # offset 564 right after sppmGroupPmfEnv (spirv-cross-verified); a tail
    # packed after the filler would shift every MLT field by +4 bytes.
    src = _read("renderer.py")
    start = src.index("def _pack_uniforms(")
    body = src[start:]
    for field in ("mlt_sigma", "mlt_large_step_prob", "mlt_max_depth",
                  "_mlt_seed"):
        assert field in body, f"_pack_uniforms must pack {field}"
    # 4f then 4I — the exact common.slang field order.
    m_f = re.search(r'"ffff",\s*float\(self\.mlt_sigma\)', body)
    m_i = re.search(r'"IIII",\s*chains', body)
    assert m_f and m_i and m_f.start() < m_i.start()
    filler = body.index("tileOriginY (Metal band loop patches)")
    assert m_i.end() < filler, \
        "MLT uniform tail must precede the tileOriginY filler u32"


def test_metal_dispatch_selects_mlt_pass():
    # Task 5.6 replaced the "adapter pending" NotImplementedError with the real
    # Metal branch. MLT must be selected BEFORE the SPPM/BDPT/path branches,
    # exactly like the Vulkan `_record_wavefront_dispatch` order.
    src = _read("renderer.py")
    start = src.index("def _render_scene_metal")
    body = src[start:start + 3000]
    assert "integrator_index == 3" in body
    assert "_ensure_wavefront_mlt_pass_metal" in body
    assert body.index("integrator_index == 3") < body.index("integrator_index == 2"), \
        "the MLT branch must precede SPPM's"


def test_metal_mlt_pass_and_recorder_exist():
    src = _read("metal_wavefront.py")
    assert "class MetalWavefrontMltPass" in src
    assert "class _MetalMltRecorder" in src
    assert '"SKINNY_MLT": "1"' in src, "MLT kernels must compile under SKINNY_MLT"
    for entry in ("wfMltBootstrap", "wfMltInit", "wfMltMutate", "wfMltResolve"):
        assert entry in src


def test_metal_mlt_recorder_flushes_every_sub_batch():
    # Design D7 / the Metal dispatch-hygiene rule: one mutation dispatch runs a
    # complete BDPT sample per chain, so each breadth-tiled sub-batch must be
    # committed + drained or the macOS GPU watchdog wedges the device.
    src = _read("metal_wavefront.py")
    start = src.index("class _MetalMltRecorder")
    body = src[start:src.index("class MetalWavefrontMltPass")]
    assert "def flush" in body and "self._enc.flush()" in body
    assert "def push_window" in body, \
        "the MLT driver sequences push a chain window, not a lane stream"


def test_metal_mlt_binds_the_chain_buffers_by_name():
    # Metal has no descriptor sets: the Vulkan bindings 52–56 become Slang
    # global names merged into the per-dispatch bind map.
    src = _read("metal_wavefront.py")
    start = src.index("class MetalWavefrontMltPass")
    body = src[start:]
    for name in ("mltPrimarySamples", "mltChainMeta", "mltCurrentRecords",
                 "mltBootstrapWeights", "mltChainSeeds"):
        assert name in body, f"Metal MLT pass must bind {name}"


def test_renderer_metal_mlt_bootstrap_round_trip():
    src = _read("renderer.py")
    start = src.index("def _run_wavefront_mlt_bootstrap_metal")
    body = src[start:start + 2500]
    # Same host round-trip as Vulkan, in order: bootstrap → readback →
    # resample → seed upload → init. Anchored on the CALL sites (the
    # `resample_chain_seeds` import sits above them all).
    order = ["mlt.dispatch_bootstrap(", "mlt.read_bootstrap_weights(",
             "resample_chain_seeds(weights", "mlt.upload_chain_seeds(",
             "mlt.dispatch_init("]
    at = [body.index(sym) for sym in order]
    assert at == sorted(at), f"Metal MLT bootstrap steps out of order: {order}"


def _renderer_module():
    # renderer.py imports `vulkan` at module load, which raises OSError (NOT
    # ImportError) when the SDK is not on the dynamic-library path — so
    # importorskip does not catch it.
    try:
        from skinny import renderer
    except (ImportError, OSError) as exc:  # pragma: no cover - env-dependent
        pytest.skip(f"renderer import unavailable: {exc}")
    return renderer


def test_mlt_msl_field_table_inserts_tail_before_tile_origin():
    # `_pack_uniforms_msl` walks a field table over the scalar blob, so the
    # table must describe the MLT pack EXACTLY: the 32 B tail sits where the
    # Vulkan filler word would be, and tileOriginY stays last.
    R = _renderer_module()
    base = [n for n, _ in R._FC_SCALAR_FIELDS]
    mlt = [n for n, _ in R._FC_SCALAR_FIELDS_MLT]
    assert base[-1] == "tileOriginY" and mlt[-1] == "tileOriginY"
    assert mlt == base[:-1] + [n for n, _ in R._FC_MLT_FIELDS] + ["tileOriginY"]
    assert sum(s for _, s in R._FC_SCALAR_FIELDS_MLT) == \
        sum(s for _, s in R._FC_SCALAR_FIELDS) + 32


def test_mlt_tail_starts_where_vulkan_spirv_expects_it():
    # In the Vulkan MLT SPIR-V `tileOriginY` does not exist (SKINNY_METAL-gated),
    # so `mltSigma` must land at 564, immediately after sppmGroupPmfEnv.
    R = _renderer_module()
    off = 0
    for name, sz in R._FC_SCALAR_FIELDS_MLT:
        if name == "mltSigma":
            break
        off += sz
    assert off == R._TILE_ORIGIN_Y_OFFSET == 564


def test_mlt_seed_is_stable_across_processes():
    # The MLT replay seed must be reproducible in a FRESH interpreter: the
    # parity gate re-renders in a new process and compares against a recorded
    # tolerance (design D6). Seeding from `_current_state_hash()` broke this —
    # that hash covers tuples containing str, so PYTHONHASHSEED randomizes it
    # per process and the same scene scored relMSE 0.17 / 0.25 / 1.10 across
    # three runs. Assert the derivation is hash()-free and str-free.
    R = _renderer_module()
    src = inspect.getsource(R.Renderer._next_mlt_seed)
    assert "zlib.crc32" in src
    assert "_current_state_hash" not in src.split('"""')[-1], \
        "_next_mlt_seed must not derive from the randomized change-detection hash"

    # Same frame_index → same seed, in a subprocess with a DIFFERENT hash seed.
    probe = (
        "import struct, zlib;"
        "print(zlib.crc32(struct.pack('<i', 7)) & 0xFFFFFFFF)"
    )
    outs = set()
    for hashseed in ("0", "1", "12345"):
        env = {**os.environ, "PYTHONHASHSEED": hashseed}
        outs.add(subprocess.run([sys.executable, "-c", probe], capture_output=True,
                                text=True, env=env, check=True).stdout.strip())
    assert len(outs) == 1, f"seed derivation varies with PYTHONHASHSEED: {outs}"

    # And the randomized hash it replaced really is unstable — the bug is real,
    # not hypothetical (guards against someone "simplifying" it back).
    bad = "print(hash(('orbit', 1.0)))"
    bad_outs = {subprocess.run([sys.executable, "-c", bad], capture_output=True,
                               text=True, env={**os.environ, "PYTHONHASHSEED": hs},
                               check=True).stdout.strip()
                for hs in ("0", "1", "12345")}
    assert len(bad_outs) > 1


def test_mlt_uniform_tail_gated_on_active_consumer():
    # codex pre-merge review: the SKINNY_MLT fc tail must be packed by
    # _mlt_uniform_tail_active(), NOT bare `integrator_index == 3`. On Metal the
    # blob length must equal the dispatched pipeline's reflected fc, so the tail
    # is packed only when the MLT wavefront pass is the real consumer —
    # otherwise a runtime switch to path, or a megakernel-fallback MLT
    # selection, desyncs the blob and trips the drift-guard assertion.
    R = _renderer_module()
    tail = inspect.getsource(R.Renderer._mlt_uniform_tail_active)
    assert "integrator_index != 3" in tail
    assert "EXECUTION_WAVEFRONT" in tail and "self.is_metal" in tail
    assert "_wavefront_mlt_pass is not None" in tail

    # The layout source routes through the predicate; the scalar packer's tail
    # is driven either by the predicate (Vulkan direct call) or by the target
    # layout (the MSL path passes an explicit mlt_tail).
    src = _read("renderer.py")
    assert "if self._mlt_uniform_tail_active():\n            return self._wavefront_mlt_pass" in src
    assert "self._mlt_uniform_tail_active() if mlt_tail is None" in src
    # The dropped naive gate must not linger anywhere.
    assert "integrator_index == 3 and not self.is_metal" not in src


def test_mlt_msl_pack_matches_target_layout_not_session():
    # codex pre-merge review: an explicit non-MLT layout_source (the material
    # preview's PreviewPipelineMetal) must pack the base 568 B blob even while
    # MLT is the active integrator — _pack_uniforms_msl keys the tail off
    # `"mltSigma" in layout` and passes that to _pack_uniforms(mlt_tail=…), so
    # the blob and the field table always agree with the target layout.
    src = _read("renderer.py")
    msl = src[src.index("def _pack_uniforms_msl"):src.index("def _build_metal_binds")]
    assert 'has_tail = "mltSigma" in layout' in msl
    assert "self._pack_uniforms(mlt_tail=has_tail)" in msl
    assert "_FC_SCALAR_FIELDS_MLT if has_tail else _FC_SCALAR_FIELDS" in msl


def test_mlt_metal_chain_batch_defaults_to_one_batch_at_default_chains():
    # codex pre-merge review / design D7: the Metal MLT phases breadth-tile so a
    # large --chains can't wedge the GPU. The default batch equals the default
    # nChains, so the GPU-validated single-dispatch path is unchanged.
    R = _renderer_module()
    assert R._MLT_METAL_CHAIN_BATCH_DEFAULT == 16384
    src = _read("renderer.py")
    assert "chain_batch=self._mlt_metal_chain_batch()" in src
    assert "chain_batch=batch" in src  # bootstrap + init in the reseed path
    ms = _read("metal_wavefront.py")
    # All three Metal dispatch entries thread chain_batch into the driver.
    assert ms.count("chain_batch=int(chain_batch)") == 3


def test_mlt_seed_masks_frame_index_to_u32():
    # codex pre-merge review: a signed "<i" pack raises past 2**31; mltSeed is a
    # u32 shader field, so the pack must mask to 32 bits.
    R = _renderer_module()
    src = inspect.getsource(R.Renderer._next_mlt_seed)
    assert 'struct.pack("<I"' in src and "& 0xFFFFFFFF" in src
    assert 'struct.pack("<i"' not in src


def test_both_backends_share_one_seed_derivation():
    # Vulkan and Metal must seed identically or the backends' chains diverge
    # for reasons unrelated to the backend (this is what made the int_caustic
    # gate look Metal-specific when it was not).
    src = _read("renderer.py")
    assert src.count("self._mlt_seed = self._next_mlt_seed()") == 2, \
        "both bootstrap paths must route through _next_mlt_seed"
    assert "_mlt_seed = hash(" not in src, "no call site may re-introduce hash()"


def test_vk_wavefront_has_mlt_pass_and_recorder():
    src = _read("vk_wavefront.py")
    assert "class WavefrontMltPass" in src
    assert "class _VkMltRecorder" in src
    assert '"SKINNY_MLT=1"' in src, "MLT kernels must compile under -DSKINNY_MLT=1"
    assert 'tag="_mlt"' in src, "MLT .spv names must never alias the RGB kernels"
    for entry in ("wfMltBootstrap", "wfMltInit", "wfMltMutate", "wfMltResolve"):
        assert entry in src


def test_scene_layout_declares_mlt_bindings_for_wavefront():
    src = _read("vk_compute.py")
    assert "mlt_bindings" in src
    assert "(52, 53, 54, 55, 56, 57)" in src, \
        "the wavefront scene set-0 layout must declare MLT bindings 52-57 " \
        "(57 = mltProposalRecords, change spectral-mlt)"
