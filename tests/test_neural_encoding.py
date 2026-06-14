"""Conditioner positional encoding parity + invariants (change
renderer-conditioner-encoding).

Axis 2 of the flow parameterization: the renderer applies a NeRF-γ feature map to
the neural-proposal condition *before* the conditioner MLP. The map is part of the
canonical, byte-for-byte trainer↔renderer contract (``neuralCondition``); a
mismatch raises variance silently rather than biasing, so the encoding the shader
applies (``neural_flow.slang`` ``nf_encode``) MUST equal the trainer's
``spline_flow`` ``make_cond_encoding(regime="path")``.

The shader ``nf_encode`` is a transliteration of the host reference
``skinny.sampling.neural_weights.encode_condition``; this file pins that host
reference three ways, none needing a GPU:

  1. **E0 identity** — ``E0`` is the raw passthrough and the default config emits
     NO ``-D`` flags, so the shipped net is byte-identical (task 3.2).
  2. **Byte-for-byte parity** — ``encode_condition`` equals a self-contained numpy
     oracle re-derived from ``spline_flow/train.py`` (always), and the REAL
     ``make_cond_encoding(regime="path")`` when torch + spline_flow import
     (task 3.1).
  3. **Jacobian-free / conditioner-side only** — the encoding changes only the
     first conditioner layer's input width; the rest of the NFW1 layout (the
     spline-parameter head, the measure transform) is identical across
     ``E0/E1/E3`` (task 3.3).

Plus the host guards: ``_layout``'s first header follows the encoding (task 2.2)
and the loader refuses an encoding/first-layer-width mismatch (task 2.3).

Run (no torch / no GPU)::

    PYTHONPATH=src .venv/bin/python -m pytest tests/test_neural_encoding.py -q
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from skinny.sampling.neural_weights import (
    NF_COND,
    NF_L_POS,
    Encoding,
    NeuralBuildConfig,
    bake_dummy_weights,
    deserialize_neural_weights,
    encode_condition,
    encoded_cond_dim,
    fourier_gamma,
    load_neural_weights,
    mlp_in_dim,
    serialize_neural_weights,
    _layout,
)

SPLINE_FLOW = Path("/Users/ahmetbilgili/projects/spline_flow")


# ---------------------------------------------------------------------------
# Self-contained numpy oracle — re-derived directly from spline_flow/train.py
# (fourier_gamma + make_cond_encoding(regime="path")), NOT calling the host
# encode_condition, so the parity assertion compares two independent paths.
# ---------------------------------------------------------------------------

def _oracle_gamma(p: np.ndarray, L: int) -> np.ndarray:
    """γ(s) = (sin(2⁰πs),cos(2⁰πs),…,sin(2^{L-1}πs),cos(2^{L-1}πs)) per scalar."""
    out = []
    for col in range(p.shape[1]):
        s = p[:, col]
        for band in range(L):
            f = (2.0 ** band) * np.pi
            out.append(np.sin(f * s))
            out.append(np.cos(f * s))
    return np.stack(out, axis=-1) if out else np.zeros((p.shape[0], 0))


def _oracle_make_cond_encoding(cond: np.ndarray, preset: str,
                               l_pos: int = NF_L_POS) -> np.ndarray:
    """spline_flow make_cond_encoding(regime="path") for cond_dim == NF_COND,
    no temporal axis: every condition scalar is a position scalar banded with
    L_pos; E3 additionally appends the raw condition."""
    b, dim = cond.shape
    if preset == "E0":
        return cond.copy()
    feats = [_oracle_gamma(cond[:, i:i + 1], l_pos) for i in range(dim)]
    out = np.concatenate(feats, axis=-1)
    if preset == "E3":
        out = np.concatenate([out, cond], axis=-1)
    return out


def _rand_cond(seed: int, n: int = 7) -> np.ndarray:
    # Conditions are normalised-AABB position / unit normal / unit dir → O(1).
    return np.random.RandomState(seed).uniform(-1.0, 1.0, size=(n, NF_COND))


# ---------------------------------------------------------------------------
# 3.2 — E0 identity + byte-identical default
# ---------------------------------------------------------------------------

def test_e0_is_raw_passthrough():
    c = _rand_cond(0)
    enc = encode_condition(c, Encoding.E0)
    assert enc.shape == (c.shape[0], NF_COND)
    assert np.array_equal(enc, c)


def test_default_config_emits_no_defines_and_e0_width():
    """The default build (E0) emits no `-D NF_ENCODING`, so the neural SPIR-V is
    byte-identical to the shipped proposal, and the first-layer width is unchanged."""
    cfg = NeuralBuildConfig()
    assert cfg.encoding is Encoding.E0
    assert "NF_ENCODING" not in " ".join(cfg.slang_defines())
    assert cfg.mlp_in == 1 + NF_COND
    assert _layout()[0][0][2] == 1 + NF_COND   # first header inDim


# ---------------------------------------------------------------------------
# 3.1 — byte-for-byte parity with the trainer's encoding
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("preset", ["E0", "E1", "E3"])
@pytest.mark.parametrize("l_pos", [1, 4, NF_L_POS])
def test_encode_matches_numpy_oracle(preset, l_pos):
    enc = Encoding(preset)
    c = _rand_cond(hash((preset, l_pos)) & 0xFFFF)
    got = encode_condition(c, enc, l_pos=l_pos)
    want = _oracle_make_cond_encoding(c, preset, l_pos=l_pos)
    assert got.shape == want.shape == (c.shape[0], encoded_cond_dim(enc, NF_COND, l_pos))
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-12)


def test_fourier_gamma_interleave_order():
    """γ ordering is (sin0,cos0,sin1,cos1,…) with freqs 2^l·π — matches the shader
    nf_encode loop and spline_flow fourier_gamma."""
    s = np.array([[0.3], [-0.7]])
    g = fourier_gamma(s, 3)
    assert g.shape == (2, 6)
    for band in range(3):
        f = (2.0 ** band) * np.pi
        np.testing.assert_allclose(g[:, 2 * band], np.sin(f * s[:, 0]), atol=1e-12)
        np.testing.assert_allclose(g[:, 2 * band + 1], np.cos(f * s[:, 0]), atol=1e-12)


def test_e3_is_e1_plus_raw_tail():
    c = _rand_cond(3)
    e1 = encode_condition(c, Encoding.E1)
    e3 = encode_condition(c, Encoding.E3)
    np.testing.assert_array_equal(e3[:, : e1.shape[1]], e1)
    np.testing.assert_array_equal(e3[:, e1.shape[1]:], c)


def test_encode_matches_real_spline_flow():
    """The authoritative byte-for-byte gate: host encode_condition equals
    spline_flow's real make_cond_encoding(regime="path"). Skipped where torch /
    spline_flow are unavailable (they live on the training box)."""
    pytest.importorskip("torch")
    import sys
    if not (SPLINE_FLOW / "train.py").exists():
        pytest.skip("spline_flow checkout not found")
    sys.path.insert(0, str(SPLINE_FLOW))
    try:
        import importlib
        train = importlib.import_module("train")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"spline_flow.train import failed: {exc!r}")
    import torch

    c = _rand_cond(11)
    for preset in ("E0", "E1", "E3"):
        enc = train.make_cond_encoding(preset, NF_COND, "path", L_pos=NF_L_POS)
        if enc is None:                      # E0 → identity passthrough
            ref = c
        else:
            ref = enc(torch.tensor(c, dtype=torch.float64)).numpy()
        got = encode_condition(c, Encoding(preset), l_pos=NF_L_POS)
        np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-12,
                                   err_msg=f"encoding {preset} differs from trainer")


# ---------------------------------------------------------------------------
# 3.3 — Jacobian-free: encoding touches ONLY the first conditioner layer
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("preset", ["E1", "E3"])
def test_only_conditioner_input_layer_widens(preset):
    """Across encodings the NFW1 layout changes ONLY each coupling's conditioner
    INPUT layer (the first of its three Linears, ``mlp_in→hidden``). The
    hidden→hidden and hidden→n_params layers — the spline-parameter head feeding
    the measure transform — are byte-identical, so |J| and NF_LOG2PI cannot move
    (the encoding is conditioner-side only; axis 2 is Jacobian-free)."""
    base, _, _ = _layout(encoding=Encoding.E0)
    enc, _, _ = _layout(encoding=Encoding(preset))
    assert len(base) == len(enc)
    want_in = mlp_in_dim(Encoding(preset))
    for k, ((_, _, bi, bo), (_, _, ei, eo)) in enumerate(zip(base, enc)):
        if k % 3 == 0:                       # conditioner input layer of a coupling
            assert eo == bo                  # output (hidden) unchanged
            assert ei == want_in > bi        # input widened to the encoded width
        else:                                # hidden→hidden / hidden→n_params head
            assert (ei, eo) == (bi, bo)      # untouched — the pdf path is invariant


# ---------------------------------------------------------------------------
# 2.2 — _layout first header follows the encoding
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("preset,exp", [("E0", 9), ("E1", 9 * 2 * 10),
                                        ("E3", 9 * 2 * 10 + 9)])
def test_layout_first_header_in_dim(preset, exp):
    enc = Encoding(preset)
    headers, _, _ = _layout(encoding=enc)
    assert headers[0][2] == 1 + exp == mlp_in_dim(enc)


# ---------------------------------------------------------------------------
# 2.3 — loader refuses an encoding / first-layer-width mismatch
# ---------------------------------------------------------------------------

def test_loader_accepts_matching_encoding(tmp_path):
    cfg = NeuralBuildConfig(encoding=Encoding.E1)
    p = tmp_path / "e1.nfw"
    bake_dummy_weights(p, cfg)
    nw = load_neural_weights(p, expect=cfg.arch, expect_mlp_in=cfg.mlp_in)
    assert int(nw.headers[0][2]) == cfg.mlp_in


def test_loader_refuses_encoding_mismatch(tmp_path):
    """An E1-trained net loaded with the E0 build (or vice versa) has the same
    (layers,bins,hidden,cond) arch tuple but a different first-layer width — the
    encoding guard catches it rather than rendering mis-conditioned."""
    e1 = NeuralBuildConfig(encoding=Encoding.E1)
    e0 = NeuralBuildConfig(encoding=Encoding.E0)
    p = tmp_path / "e1.nfw"
    bake_dummy_weights(p, e1)
    # arch tuple matches (encoding-independent) — only the mlp_in guard fires.
    assert e1.arch == e0.arch
    with pytest.raises(ValueError, match="first-layer in_dim"):
        load_neural_weights(p, expect=e0.arch, expect_mlp_in=e0.mlp_in)
    # The reverse direction too (E0 net, E1 build).
    p0 = tmp_path / "e0.nfw"
    bake_dummy_weights(p0, e0)
    with pytest.raises(ValueError, match="first-layer in_dim"):
        load_neural_weights(p0, expect=e1.arch, expect_mlp_in=e1.mlp_in)


def test_deserialize_roundtrip_preserves_encoding_width():
    cfg = NeuralBuildConfig(encoding=Encoding.E3)
    from skinny.sampling.neural_weights import make_dummy_weights
    nw = make_dummy_weights(cfg)
    blob = serialize_neural_weights(nw)
    rt = deserialize_neural_weights(blob, expect=cfg.arch, expect_mlp_in=cfg.mlp_in)
    assert int(rt.headers[0][2]) == cfg.mlp_in
