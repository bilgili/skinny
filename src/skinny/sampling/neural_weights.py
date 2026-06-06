"""Flat neural-flow weight format (``NFW1``) — loader + dummy baker.

The renderer's Slang inference (``shaders/sampling/neural_flow.slang``) reads a
frozen network as three flat GPU buffers — ``weights[]``, ``biases[]`` and a
``NfLayerHeader[]`` table. The offline trainer bakes that file via
``spline_flow/export_weights.py``; this module is the host side that parses it
back, validates the architecture against the compiled-in Slang constants, and
can bake an untrained ("dummy") net with no PyTorch dependency for the 1a
plumbing bring-up (prove the mixture is unbiased independent of training).

Binary layout (little-endian), identical to ``export_weights.py``::

    uint32  magic = 0x4E465731 ("NFW1")
    uint32  version = 1
    uint32  layers, bins, hidden, cond     # architecture (must match the Slang consts)
    uint32  nHeaders
    nHeaders * (uint32 weightOffset, biasOffset, inDim, outDim)
    uint32  nWeights ; nWeights * float32
    uint32  nBiases  ; nBiases  * float32
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MAGIC = 0x4E465731
VERSION = 1

# Architecture — MUST match shaders/sampling/neural_flow.slang (NF_* consts).
NF_LAYERS = 6
NF_BINS = 24
NF_HIDDEN = 96
NF_COND = 9
NF_PARAMS = 3 * NF_BINS + 1          # widths(K) + heights(K) + derivs(K+1)
NF_MLP_IN = 1 + NF_COND              # one pass-through coord + condition
# One NfLayerHeader = 4 × uint32 (weightOffset, biasOffset, inDim, outDim).
HEADER_STRIDE = 16


@dataclass
class NeuralWeights:
    """Parsed ``NFW1`` network ready to upload to bindings 33/34/35."""

    layers: int
    bins: int
    hidden: int
    cond: int
    headers: np.ndarray   # uint32 [nHeaders, 4] — (weightOffset, biasOffset, inDim, outDim)
    weights: np.ndarray   # float32 [nWeights]
    biases: np.ndarray    # float32 [nBiases]

    @property
    def header_bytes(self) -> bytes:
        return self.headers.astype("<u4").tobytes()

    @property
    def weight_bytes(self) -> bytes:
        return self.weights.astype("<f4").tobytes()

    @property
    def bias_bytes(self) -> bytes:
        return self.biases.astype("<f4").tobytes()

    def assert_matches_shader(self) -> None:
        """Raise if the file's architecture differs from the Slang constants —
        a mismatch would index the flat buffers wrong and corrupt inference."""
        got = (self.layers, self.bins, self.hidden, self.cond)
        want = (NF_LAYERS, NF_BINS, NF_HIDDEN, NF_COND)
        if got != want:
            raise ValueError(
                f"neural weights architecture {got} != shader "
                f"(layers, bins, hidden, cond)={want}; rebake with the matching net"
            )


def _layout() -> tuple[list[tuple[int, int, int, int]], int, int]:
    """Header table + total (nWeights, nBiases) for the fixed architecture.

    Three Linear layers per coupling: (NF_MLP_IN→hidden), (hidden→hidden),
    (hidden→NF_PARAMS). Row-major [out, in], matching the exporter + the Slang
    ``W[weightOffset + o*inDim + i]`` indexing.
    """
    dims = [(NF_MLP_IN, NF_HIDDEN), (NF_HIDDEN, NF_HIDDEN), (NF_HIDDEN, NF_PARAMS)]
    headers: list[tuple[int, int, int, int]] = []
    w_off = b_off = 0
    for _ in range(NF_LAYERS):
        for in_dim, out_dim in dims:
            headers.append((w_off, b_off, in_dim, out_dim))
            w_off += out_dim * in_dim
            b_off += out_dim
    return headers, w_off, b_off


def load_neural_weights(path: str | Path) -> NeuralWeights:
    """Parse an ``NFW1`` file. Raises on bad magic / version / truncation."""
    data = Path(path).read_bytes()
    o = 0
    magic, ver = struct.unpack_from("<II", data, o); o += 8
    if magic != MAGIC:
        raise ValueError(f"{path}: bad magic 0x{magic:08X} (want 0x{MAGIC:08X})")
    if ver != VERSION:
        raise ValueError(f"{path}: unsupported version {ver}")
    layers, bins, hidden, cond = struct.unpack_from("<IIII", data, o); o += 16
    (n_headers,) = struct.unpack_from("<I", data, o); o += 4
    headers = np.frombuffer(data, dtype="<u4", count=n_headers * 4,
                            offset=o).reshape(n_headers, 4).copy()
    o += n_headers * HEADER_STRIDE
    (n_weights,) = struct.unpack_from("<I", data, o); o += 4
    weights = np.frombuffer(data, dtype="<f4", count=n_weights, offset=o).copy()
    o += n_weights * 4
    (n_biases,) = struct.unpack_from("<I", data, o); o += 4
    biases = np.frombuffer(data, dtype="<f4", count=n_biases, offset=o).copy()
    nw = NeuralWeights(int(layers), int(bins), int(hidden), int(cond),
                       headers, weights, biases)
    nw.assert_matches_shader()
    return nw


def make_dummy_weights() -> NeuralWeights:
    """In-memory all-zero net for the fixed architecture (no file I/O).

    Used to seed bindings 33/34/35 so the inline flow inverse in proposal.slang
    always has valid, correctly-sized descriptors even when neural is inactive.
    """
    headers, n_weights, n_biases = _layout()
    return NeuralWeights(
        NF_LAYERS, NF_BINS, NF_HIDDEN, NF_COND,
        np.array(headers, dtype="<u4"),
        np.zeros(n_weights, dtype="<f4"),
        np.zeros(n_biases, dtype="<f4"),
    )


def bake_dummy_weights(path: str | Path) -> NeuralWeights:
    """Write a valid all-zero ``NFW1`` net (no PyTorch).

    An untrained / zero net still produces a normalised, finite-pdf distribution
    over the hemisphere (the RQ spline degenerates to a near-identity map), so it
    is a correct *dummy* for the 1a bring-up: the mixture-MIS estimator stays
    unbiased regardless of the proposal's quality, which is exactly what the
    plumbing milestone proves before any training.
    """
    headers, n_weights, n_biases = _layout()
    weights = np.zeros(n_weights, dtype="<f4")
    biases = np.zeros(n_biases, dtype="<f4")
    buf = bytearray()
    buf += struct.pack("<II", MAGIC, VERSION)
    buf += struct.pack("<IIII", NF_LAYERS, NF_BINS, NF_HIDDEN, NF_COND)
    buf += struct.pack("<I", len(headers))
    for h in headers:
        buf += struct.pack("<IIII", *h)
    buf += struct.pack("<I", n_weights)
    buf += weights.tobytes()
    buf += struct.pack("<I", n_biases)
    buf += biases.tobytes()
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(bytes(buf))
    return NeuralWeights(NF_LAYERS, NF_BINS, NF_HIDDEN, NF_COND,
                         np.array(headers, dtype="<u4"), weights, biases)
