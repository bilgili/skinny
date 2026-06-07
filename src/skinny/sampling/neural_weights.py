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

import enum
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MAGIC = 0x4E465731
VERSION = 1

# Default architecture — the shipped net + the shader's `#ifndef` defaults
# (shaders/sampling/neural_flow.slang: NF_LAYERS/NF_BINS/NF_HIDDEN). The size is
# build-time configurable (study change neural-precision-size-study); these are
# the defaults, NF_COND is fixed by the renderer's condition encoding.
NF_LAYERS = 6
NF_BINS = 24
NF_HIDDEN = 96
NF_COND = 9
NF_PARAMS = 3 * NF_BINS + 1          # widths(K) + heights(K) + derivs(K+1)
NF_MLP_IN = 1 + NF_COND              # one pass-through coord + condition
# One NfLayerHeader = 4 × uint32 (weightOffset, biasOffset, inDim, outDim).
HEADER_STRIDE = 16


class NeuralPrecision(enum.Enum):
    """MLP inference precision (study change neural-precision-size-study).

    Drives both the host upload (which dtype the weight bytes are cast to) and
    the shader compile (the NF_WT/NF_CT `-D` aliases). The RQ-spline math + the
    returned pdf are always full precision; only the linear-layer GEMMs change.
    """

    FP32 = "fp32"                    # NF_WT=float NF_CT=float — the shipped net
    FP16_STORAGE = "fp16-storage"    # NF_WT=half  NF_CT=float — ½-byte weights
    FP16_COMPUTE = "fp16-compute"    # NF_WT=half  NF_CT=half  — + half GEMM

    @property
    def weight_half(self) -> bool:
        """True when the weight *storage* (bindings 33/34) is half — both fp16
        modes upload half bytes; only fp32 stays 4-byte float."""
        return self is not NeuralPrecision.FP32

    @property
    def compute_half(self) -> bool:
        """True when the MLP GEMM *accumulate* is half (fp16-compute only)."""
        return self is NeuralPrecision.FP16_COMPUTE

    @property
    def needs_device_fp16_compute(self) -> bool:
        """fp16-compute needs `shaderFloat16` (half ALU); the storage mode only
        needs 16-bit SSBO access. The renderer gates fallback on these."""
        return self is NeuralPrecision.FP16_COMPUTE

    @property
    def needs_device_fp16_storage(self) -> bool:
        return self.weight_half


@dataclass(frozen=True)
class NeuralBuildConfig:
    """A point in the size×precision grid: the shader dims + the inference
    precision. The single source of truth threaded into every neural ``.spv``
    compile (its `-D` flags) and the weight upload (its dtype). The default
    config emits NO `-D` flags, so its compiles are byte-identical to the
    shipped proposal (study change neural-precision-size-study)."""

    layers: int = NF_LAYERS
    bins: int = NF_BINS
    hidden: int = NF_HIDDEN
    precision: NeuralPrecision = NeuralPrecision.FP32

    @property
    def arch(self) -> tuple[int, int, int, int]:
        """(layers, bins, hidden, cond) — the NFW1 architecture tuple the loader
        validates a baked net against."""
        return (self.layers, self.bins, self.hidden, NF_COND)

    @property
    def is_default_size(self) -> bool:
        return (self.layers, self.bins, self.hidden) == (NF_LAYERS, NF_BINS, NF_HIDDEN)

    def slang_defines(self) -> tuple[str, ...]:
        """Flat `-D name=value` tokens for slangc. Only non-default dims/precision
        emit a flag, so the default config → empty tuple → unchanged cache key →
        byte-identical SPIR-V."""
        d: list[str] = []
        if self.layers != NF_LAYERS:
            d += ["-D", f"NF_LAYERS={self.layers}"]
        if self.bins != NF_BINS:
            d += ["-D", f"NF_BINS={self.bins}"]
        if self.hidden != NF_HIDDEN:
            d += ["-D", f"NF_HIDDEN={self.hidden}"]
        if self.precision.weight_half:
            d += ["-D", "NF_WT=half"]
        if self.precision.compute_half:
            d += ["-D", "NF_CT=half"]
        return tuple(d)

    @property
    def cache_tag(self) -> str:
        """Short slug uniquely identifying this config — folded into the wavefront
        `.spv` out-name so distinct configs never clobber each other's module."""
        return f"L{self.layers}B{self.bins}H{self.hidden}_{self.precision.value}"


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

    def weight_bytes_for(self, precision: NeuralPrecision) -> bytes:
        """Weight bytes in the storage dtype for ``precision`` — fp32 stays
        4-byte float; the fp16 modes cast fp32→half (``<f2``) at upload, halving
        the GPU footprint. NFW1 on disk is always fp32; this is the upload-time
        cast (study change neural-precision-size-study)."""
        dt = "<f2" if precision.weight_half else "<f4"
        return self.weights.astype(dt).tobytes()

    def bias_bytes_for(self, precision: NeuralPrecision) -> bytes:
        dt = "<f2" if precision.weight_half else "<f4"
        return self.biases.astype(dt).tobytes()

    def assert_matches_shader(self, expect: tuple[int, int, int, int] | None = None) -> None:
        """Raise if the file's architecture differs from the built dimensions — a
        mismatch would index the flat buffers wrong and corrupt inference. With no
        argument it checks the default (shipped) architecture; the renderer passes
        the active NeuralBuildConfig.arch for an off-default size."""
        got = (self.layers, self.bins, self.hidden, self.cond)
        want = expect if expect is not None else (NF_LAYERS, NF_BINS, NF_HIDDEN, NF_COND)
        if got != want:
            raise ValueError(
                f"neural weights architecture {got} != shader "
                f"(layers, bins, hidden, cond)={want}; rebake with the matching net"
            )


def _layout(layers: int = NF_LAYERS, bins: int = NF_BINS,
            hidden: int = NF_HIDDEN) -> tuple[list[tuple[int, int, int, int]], int, int]:
    """Header table + total (nWeights, nBiases) for the given architecture.

    Three Linear layers per coupling: (mlp_in→hidden), (hidden→hidden),
    (hidden→n_params). Row-major [out, in], matching the exporter + the Slang
    ``W[weightOffset + o*inDim + i]`` indexing. Defaults to the shipped size.
    """
    n_params = 3 * bins + 1
    mlp_in = 1 + NF_COND
    dims = [(mlp_in, hidden), (hidden, hidden), (hidden, n_params)]
    headers: list[tuple[int, int, int, int]] = []
    w_off = b_off = 0
    for _ in range(layers):
        for in_dim, out_dim in dims:
            headers.append((w_off, b_off, in_dim, out_dim))
            w_off += out_dim * in_dim
            b_off += out_dim
    return headers, w_off, b_off


def load_neural_weights(path: str | Path,
                        expect: tuple[int, int, int, int] | None = None) -> NeuralWeights:
    """Parse an ``NFW1`` file. Raises on bad magic / version / truncation, and on
    an architecture mismatch against ``expect`` (defaults to the shipped size when
    None) — the loader assert is the size-mismatch guard for the configurable
    build (study change neural-precision-size-study)."""
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
    nw.assert_matches_shader(expect)
    return nw


def make_dummy_weights(config: NeuralBuildConfig | None = None) -> NeuralWeights:
    """In-memory all-zero net for the given architecture (no file I/O).

    Used to seed bindings 33/34/35 so the inline flow inverse in proposal.slang
    always has valid, correctly-sized descriptors even when neural is inactive.
    Defaults to the shipped size; pass a config to size the dummy for an
    off-default build.
    """
    cfg = config or NeuralBuildConfig()
    headers, n_weights, n_biases = _layout(cfg.layers, cfg.bins, cfg.hidden)
    return NeuralWeights(
        cfg.layers, cfg.bins, cfg.hidden, NF_COND,
        np.array(headers, dtype="<u4"),
        np.zeros(n_weights, dtype="<f4"),
        np.zeros(n_biases, dtype="<f4"),
    )


def bake_dummy_weights(path: str | Path,
                       config: NeuralBuildConfig | None = None) -> NeuralWeights:
    """Write a valid all-zero ``NFW1`` net (no PyTorch) at the given architecture.

    An untrained / zero net still produces a normalised, finite-pdf distribution
    over the hemisphere (the RQ spline degenerates to a near-identity map), so it
    is a correct *dummy* for the 1a bring-up: the mixture-MIS estimator stays
    unbiased regardless of the proposal's quality, which is exactly what the
    plumbing milestone proves before any training. The header stores fp32 on disk
    in every precision mode — the half cast happens at upload, not in the file.
    """
    cfg = config or NeuralBuildConfig()
    headers, n_weights, n_biases = _layout(cfg.layers, cfg.bins, cfg.hidden)
    weights = np.zeros(n_weights, dtype="<f4")
    biases = np.zeros(n_biases, dtype="<f4")
    buf = bytearray()
    buf += struct.pack("<II", MAGIC, VERSION)
    buf += struct.pack("<IIII", cfg.layers, cfg.bins, cfg.hidden, NF_COND)
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
    return NeuralWeights(cfg.layers, cfg.bins, cfg.hidden, NF_COND,
                         np.array(headers, dtype="<u4"), weights, biases)


def write_neural_weights(path: str | Path, nw: NeuralWeights) -> NeuralWeights:
    """Serialise a ``NeuralWeights`` to an ``NFW1`` file (inverse of
    ``load_neural_weights``). The online trainer (change ``neural-online-training``)
    uses this to publish updated weights that the renderer hot-reloads via the
    file-double-buffer handoff. NFW1 on disk is always fp32."""
    buf = bytearray()
    buf += struct.pack("<II", MAGIC, VERSION)
    buf += struct.pack("<IIII", nw.layers, nw.bins, nw.hidden, nw.cond)
    buf += struct.pack("<I", len(nw.headers))
    for h in nw.headers:
        buf += struct.pack("<IIII", *(int(x) for x in h))
    w = np.ascontiguousarray(nw.weights, dtype="<f4")
    b = np.ascontiguousarray(nw.biases, dtype="<f4")
    buf += struct.pack("<I", len(w))
    buf += w.tobytes()
    buf += struct.pack("<I", len(b))
    buf += b.tobytes()
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(bytes(buf))
    return nw
