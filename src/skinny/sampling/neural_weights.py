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
TAG_MAGIC = 0x50544731   # "PTG1" — optional parametrization-tag trailer (export_weights)

# Default architecture — the shipped net + the shader's `#ifndef` defaults
# (shaders/sampling/neural_flow.slang: NF_LAYERS/NF_BINS/NF_HIDDEN). The size is
# build-time configurable (study change neural-precision-size-study); these are
# the defaults, NF_COND is fixed by the renderer's condition encoding.
NF_LAYERS = 6
NF_BINS = 24
NF_HIDDEN = 96
NF_COND = 9
NF_PARAMS = 3 * NF_BINS + 1          # widths(K) + heights(K) + derivs(K+1)
NF_MLP_IN = 1 + NF_COND              # one pass-through coord + condition (E0 default)
NF_L_POS = 10                        # position-group Fourier bands (spline_flow L_pos)
# One NfLayerHeader = 4 × uint32 (weightOffset, biasOffset, inDim, outDim).
HEADER_STRIDE = 16


class Encoding(enum.Enum):
    """Conditioner positional encoding (axis 2; change renderer-conditioner-encoding).

    Maps to the shader's ``NF_ENCODING`` build define. The encoding is a NeRF-γ
    feature map applied to the condition BEFORE the conditioner MLP — a side input
    only, never the u→z transform, so it is Jacobian-free. ``E0`` is the raw
    passthrough (byte-identical to the pre-encoding net)."""

    E0 = "E0"   # raw passthrough — no feature map (default, byte-identical)
    E1 = "E1"   # per-scalar γ(s, L_pos); path regime ⇒ every condition scalar banded
    E3 = "E3"   # E1 + the full raw condition appended (include_raw tail)

    @property
    def nf_define(self) -> int:
        """The integer value the shader's ``-D NF_ENCODING=`` expects."""
        return {Encoding.E0: 0, Encoding.E1: 1, Encoding.E3: 3}[self]


def encoded_cond_dim(encoding: Encoding, cond: int = NF_COND,
                     l_pos: int = NF_L_POS) -> int:
    """Width of the encoded condition (the shader's ``NF_MLP_ENC``).

    Byte-for-byte with ``neural_flow.slang`` and ``spline_flow``
    ``make_cond_encoding(regime="path")``: ``E0`` is raw (``cond``); ``E1`` bands
    every scalar (``cond·2·L_pos``); ``E3`` appends the raw condition tail."""
    if encoding is Encoding.E0:
        return cond
    banded = cond * 2 * l_pos
    return banded + cond if encoding is Encoding.E3 else banded


def mlp_in_dim(encoding: Encoding, cond: int = NF_COND,
               l_pos: int = NF_L_POS) -> int:
    """First conditioner-layer input width = 1 (pass-through coord) + encoded
    condition. Equals the shader's ``NF_MLP_IN`` and the NFW1 first header's
    ``inDim`` — the value the loader's encoding/arch guard validates."""
    return 1 + encoded_cond_dim(encoding, cond, l_pos)


# Per-coordinate conditioner output width by coupling (the last Linear's outDim,
# shader NF_PARAMS): rqs = 3K+1 (widths+heights+derivs), nis-pq = 2K+1
# (widths + K+1 vertices), nis-pl = K (bin masses). Matches spline_flow
# make_coupling_transform's n_params and the shader's NF_PARAMS define.
def coupling_n_params(coupling: str, bins: int = NF_BINS) -> int:
    if coupling == "rqs":
        return 3 * bins + 1
    if coupling == "nis-pq":
        return 2 * bins + 1
    if coupling == "nis-pl":
        return bins
    raise ValueError(f"unknown coupling {coupling!r}")


# Hemisphere chart -> shader NF_CHART define value (change renderer-chart-selection).
# V1 is the shipped default (no -D flag). V2 is reserved but NOT implemented in the
# shader (needs flow-local wo absent from the .nrec schema) — selecting it errors.
NF_CHART_CODE = {"V0": 0, "V1": 1, "V5": 5}


def chart_code(chart: str) -> int:
    if chart not in NF_CHART_CODE:
        raise ValueError(
            f"unknown/unimplemented chart {chart!r}; renderer implements "
            f"{sorted(NF_CHART_CODE)} (V2 reserved, not buildable)")
    return NF_CHART_CODE[chart]


def fourier_gamma(p: np.ndarray, L: int) -> np.ndarray:
    """NeRF positional encoding — host mirror of ``spline_flow.train.fourier_gamma``
    and ``neural_flow.slang`` ``nf_encode``.

    ``p: [..., k] -> [..., k·2L]``, per-scalar order ``(sin0,cos0,sin1,cos1,…)``
    with frequencies ``2^l · π`` (``l = 0..L-1``)."""
    p = np.asarray(p, np.float64)
    freqs = (2.0 ** np.arange(L)) * np.pi
    ang = p[..., None] * freqs
    enc = np.stack([np.sin(ang), np.cos(ang)], axis=-1)
    return enc.reshape(*p.shape[:-1], p.shape[-1] * 2 * L)


def encode_condition(cond: np.ndarray, encoding: Encoding,
                     l_pos: int = NF_L_POS) -> np.ndarray:
    """Host reference for the shader ``nf_encode`` — apply the conditioner encoding
    to a ``[B, NF_COND]`` condition batch, byte-for-byte with the trainer's
    ``make_cond_encoding(regime="path")``.

    ``E0`` is the identity. Path regime: every condition scalar is banded (groups
    emitted in condition order); ``E3`` appends the raw condition as the tail."""
    cond = np.asarray(cond, np.float64)
    if cond.ndim != 2:
        raise ValueError(f"cond must be [B, NF_COND], got shape {cond.shape}")
    if encoding is Encoding.E0:
        return cond.copy()
    feats = [fourier_gamma(cond[:, i:i + 1], l_pos) for i in range(cond.shape[1])]
    out = np.concatenate(feats, axis=-1)
    if encoding is Encoding.E3:
        out = np.concatenate([out, cond], axis=-1)
    return out

# e4m3 (OCP "E4M3FN") 8-bit float: 1 sign, 4 exponent (bias 7), 3 mantissa; no
# inf, max finite magnitude 448. Used for the fp8 weight-STORAGE inference mode
# (NeuralPrecision.FP8_STORAGE): the host encodes fp32→e4m3 here and the shader
# decodes byte→float in the scalar GEMM (neural_flow.slang nf_decode_e4m3). NFW1
# on disk stays fp32 in every precision mode; this is an upload-time cast.
E4M3_MAX = 448.0


def f32_to_e4m3(x: np.ndarray) -> np.ndarray:
    """Round fp32 → e4m3 bytes (uint8), round-to-nearest-even, saturating to
    ±448. Matches the in-shader :func:`e4m3_to_f32` decode bit-for-bit."""
    x = np.asarray(x, np.float64)
    shape = x.shape
    x = x.ravel()
    sign = np.signbit(x).astype(np.uint32)
    ax = np.abs(x)
    nz = np.isfinite(ax) & (ax > 0.0)
    axc = np.where(np.isfinite(ax), np.minimum(ax, E4M3_MAX), E4M3_MAX)

    e = np.floor(np.log2(np.where(nz, axc, 1.0))).astype(np.int64)
    # normal: e >= -6
    en = np.clip(e, -6, 8)
    mant = axc / np.exp2(en.astype(np.float64)) - 1.0
    q = np.rint(mant * 8.0).astype(np.int64)
    carry = q >= 8
    en = np.where(carry, en + 1, en)
    q = np.where(carry, 0, q)
    over = en > 8
    en = np.where(over, 8, en)
    q = np.where(over, 6, q)
    exp_n = (en + 7).astype(np.uint32)
    man_n = q.astype(np.uint32)
    # never emit the e4m3 NaN slot (exp=15, man=7) → clamp to 448 (exp15, man6)
    nan_slot = (exp_n == 15) & (man_n == 7)
    man_n = np.where(nan_slot, 6, man_n)

    # subnormal: e < -6 → value = m * 2^-9, m in 1..7 (m==8 rolls to normal e=-6)
    qs = np.clip(np.rint(axc / (2.0 ** -9)).astype(np.int64), 0, 8)
    exp_s = np.where(qs >= 8, 1, 0).astype(np.uint32)
    man_s = np.where(qs >= 8, 0, qs).astype(np.uint32)

    normal = nz & (e >= -6)
    sub = nz & (e < -6)
    exp_f = np.where(normal, exp_n, np.where(sub, exp_s, 0)).astype(np.uint32)
    man_f = np.where(normal, man_n, np.where(sub, man_s, 0)).astype(np.uint32)
    out = (sign << 7) | (exp_f << 3) | man_f
    out = np.where(nz, out, sign << 7).astype(np.uint8)   # ±0 → 0x00 / 0x80
    return out.reshape(shape)


def e4m3_to_f32(b: np.ndarray) -> np.ndarray:
    """Decode e4m3 bytes (uint8) → fp32, matching neural_flow.slang
    nf_decode_e4m3 (and the GPU scalar GEMM)."""
    b = np.asarray(b, np.uint32)
    s = (b >> 7) & 0x1
    e = (b >> 3) & 0xF
    m = b & 0x7
    val = np.where(e == 0,
                   m.astype(np.float64) / 512.0,                       # 2^-9 * m
                   (1.0 + m.astype(np.float64) * 0.125) * np.exp2(e.astype(np.float64) - 7.0))
    return np.where(s != 0, -val, val).astype(np.float32)


def _pack_e4m3(arr: np.ndarray) -> bytes:
    """Encode an fp32 array to e4m3 bytes, padded to a multiple of 4 so the
    shader can bind it as ``StructuredBuffer<uint>`` (4 weights / word)."""
    enc = f32_to_e4m3(np.ascontiguousarray(arr, np.float32)).reshape(-1)
    pad = (-enc.size) % 4
    if pad:
        enc = np.concatenate([enc, np.zeros(pad, np.uint8)])
    return enc.tobytes()


class NeuralPrecision(enum.Enum):
    """MLP inference precision (study change neural-precision-size-study,
    extended by change neural-trainer-backends).

    Drives both the host upload (which dtype the weight bytes are cast to) and
    the shader compile (the NF_WT/NF_CT/NF_FP8 `-D` aliases). The RQ-spline math
    + the returned pdf are always full precision; only the linear-layer GEMMs
    change.
    """

    FP32 = "fp32"                    # NF_WT=float NF_CT=float — the shipped net
    FP16_STORAGE = "fp16-storage"    # NF_WT=half  NF_CT=float — ½-byte weights
    FP16_COMPUTE = "fp16-compute"    # NF_WT=half  NF_CT=half  — + half GEMM
    FP8_STORAGE = "fp8-storage"      # e4m3 weights decoded to float in the GEMM
    #                                  — ¼-byte weights, no device feature needed

    @property
    def weight_half(self) -> bool:
        """True when the weight *storage* (bindings 33/34) is fp16 half — both
        fp16 modes upload half bytes; fp32 stays 4-byte float and fp8 is a
        separate quarter-byte path (see :attr:`weight_fp8`)."""
        return self in (NeuralPrecision.FP16_STORAGE, NeuralPrecision.FP16_COMPUTE)

    @property
    def weight_fp8(self) -> bool:
        """True when the weight storage is 8-bit e4m3, decoded to float in the
        scalar GEMM (manual decode → no device feature required)."""
        return self is NeuralPrecision.FP8_STORAGE

    @property
    def compute_half(self) -> bool:
        """True when the MLP GEMM *accumulate* is half (fp16-compute only)."""
        return self is NeuralPrecision.FP16_COMPUTE

    @property
    def storage_bytes(self) -> int:
        """Bytes per weight scalar in the GPU buffer: 4 (fp32), 2 (fp16), 1 (fp8)."""
        if self.weight_fp8:
            return 1
        return 2 if self.weight_half else 4

    @property
    def needs_device_fp16_compute(self) -> bool:
        """fp16-compute needs `shaderFloat16` (half ALU); the storage mode only
        needs 16-bit SSBO access. The renderer gates fallback on these. fp8
        decodes from `uint` words and needs neither."""
        return self is NeuralPrecision.FP16_COMPUTE

    @property
    def needs_device_fp16_storage(self) -> bool:
        return self.weight_half


@dataclass(frozen=True)
class NeuralBuildConfig:
    """A point in the size×precision×encoding grid: the shader dims, the inference
    precision, and the conditioner encoding (axis 2). The single source of truth
    threaded into every neural ``.spv`` compile (its `-D` flags) and the weight
    upload (its dtype). The default config emits NO `-D` flags, so its compiles are
    byte-identical to the shipped proposal (study change neural-precision-size-study;
    encoding axis by change renderer-conditioner-encoding)."""

    layers: int = NF_LAYERS
    bins: int = NF_BINS
    hidden: int = NF_HIDDEN
    precision: NeuralPrecision = NeuralPrecision.FP32
    encoding: Encoding = Encoding.E0
    l_pos: int = NF_L_POS
    coupling: str = "rqs"          # rqs (default) | nis-pq (change neural-nis-baseline)
    chart: str = "V1"              # V1 (default) | V0 | V5 (change renderer-chart-selection)

    @property
    def n_params(self) -> int:
        """Per-coordinate conditioner output width (last Linear outDim = shader
        NF_PARAMS) for this coupling: rqs 3K+1, nis-pq 2K+1. The loader's
        coupling guard validates a baked net's last-layer outDim against this."""
        return coupling_n_params(self.coupling, self.bins)

    @property
    def arch(self) -> tuple[int, int, int, int]:
        """(layers, bins, hidden, cond) — the NFW1 architecture tuple the loader
        validates a baked net against. Encoding-independent (cond is the RAW
        condition dim); the encoding is validated separately via :attr:`mlp_in`."""
        return (self.layers, self.bins, self.hidden, NF_COND)

    @property
    def mlp_in(self) -> int:
        """First conditioner-layer input width for this config's encoding — the
        NFW1 first header's ``inDim``. The encoding/arch guard (the loader's
        ``expect_mlp_in``) refuses a net whose first-layer width differs."""
        return mlp_in_dim(self.encoding, NF_COND, self.l_pos)

    @property
    def is_default_size(self) -> bool:
        return (self.layers, self.bins, self.hidden) == (NF_LAYERS, NF_BINS, NF_HIDDEN)

    def slang_defines(self) -> tuple[str, ...]:
        """Flat `-D name=value` tokens for slangc. Only non-default dims/precision/
        encoding emit a flag, so the default config → empty tuple → unchanged cache
        key → byte-identical SPIR-V."""
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
        if self.precision.weight_fp8:
            # e4m3 weights packed in uint words; NF_WT becomes the storage word
            # type and nf_fetch decodes byte→float (NF_CT stays float).
            d += ["-D", "NF_FP8=1", "-D", "NF_WT=uint"]
        if self.encoding is not Encoding.E0:
            # NF_ENCODING selects the in-shader NeRF-γ feature map; NF_L_POS sets
            # the per-scalar band count (only meaningful when encoding is active).
            d += ["-D", f"NF_ENCODING={self.encoding.nf_define}"]
            if self.l_pos != NF_L_POS:
                d += ["-D", f"NF_L_POS={self.l_pos}"]
        if self.coupling != "rqs":
            # NF_COUPLING selects the in-shader warp: 1 = nis-pq (piecewise-quadratic).
            d += ["-D", f"NF_COUPLING={ {'nis-pq': 1}[self.coupling] }"]
        if self.chart != "V1":
            # NF_CHART selects the square<->direction map: 0 = V0 (cylindrical),
            # 5 = V5 (equirectangular). V1 (default) emits no flag → byte-identical.
            d += ["-D", f"NF_CHART={chart_code(self.chart)}"]
        return tuple(d)

    @property
    def cache_tag(self) -> str:
        """Short slug uniquely identifying this config — folded into the wavefront
        `.spv` out-name so distinct configs never clobber each other's module."""
        enc = "" if self.encoding is Encoding.E0 else (
            f"_{self.encoding.value}"
            + (f"P{self.l_pos}" if self.l_pos != NF_L_POS else ""))
        cpl = "" if self.coupling == "rqs" else f"_{self.coupling}"
        cht = "" if self.chart == "V1" else f"_{self.chart}"
        return f"L{self.layers}B{self.bins}H{self.hidden}{enc}{cpl}{cht}_{self.precision.value}"


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
    chart: str | None = None   # stamped chart from the PTG1 trailer (None if absent)

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
        4-byte float; the fp16 modes cast fp32→half (``<f2``); fp8 encodes
        fp32→e4m3 (¼ the bytes) packed for a ``StructuredBuffer<uint>``. NFW1 on
        disk is always fp32; this is the upload-time cast (study change
        neural-precision-size-study, fp8 by neural-trainer-backends)."""
        if precision.weight_fp8:
            return _pack_e4m3(self.weights)
        dt = "<f2" if precision.weight_half else "<f4"
        return self.weights.astype(dt).tobytes()

    def bias_bytes_for(self, precision: NeuralPrecision) -> bytes:
        if precision.weight_fp8:
            return _pack_e4m3(self.biases)
        dt = "<f2" if precision.weight_half else "<f4"
        return self.biases.astype(dt).tobytes()

    def assert_matches_shader(self, expect: tuple[int, int, int, int] | None = None,
                              *, expect_mlp_in: int | None = None,
                              expect_n_params: int | None = None,
                              expect_chart: str | None = None) -> None:
        """Raise if the file's architecture differs from the built dimensions — a
        mismatch would index the flat buffers wrong and corrupt inference. With no
        argument it checks the default (shipped) architecture; the renderer passes
        the active NeuralBuildConfig.arch for an off-default size.

        ``expect_mlp_in`` is the encoding/arch guard (change
        renderer-conditioner-encoding): the conditioner encoding only changes the
        first Linear's input width, which is encoding-independent in the
        ``(layers, bins, hidden, cond)`` tuple, so a net trained under a different
        ``--encoding`` is caught here — its first header's ``inDim`` differs from
        the built ``NeuralBuildConfig.mlp_in``."""
        got = (self.layers, self.bins, self.hidden, self.cond)
        want = expect if expect is not None else (NF_LAYERS, NF_BINS, NF_HIDDEN, NF_COND)
        if got != want:
            raise ValueError(
                f"neural weights architecture {got} != shader "
                f"(layers, bins, hidden, cond)={want}; rebake with the matching net"
            )
        if expect_mlp_in is not None:
            got_in = int(self.headers[0][2]) if len(self.headers) else -1
            if got_in != expect_mlp_in:
                raise ValueError(
                    f"neural weights first-layer in_dim {got_in} != built encoding's "
                    f"mlp_in {expect_mlp_in}; the network was trained with a different "
                    f"--encoding (load it with the matching encoding, or rebake)"
                )
        if expect_n_params is not None:
            got_out = int(self.headers[-1][3]) if len(self.headers) else -1
            if got_out != expect_n_params:
                raise ValueError(
                    f"neural weights last-layer out_dim {got_out} != built coupling's "
                    f"n_params {expect_n_params}; the network was trained with a "
                    f"different --coupling (rqs=3K+1 vs nis-pq=2K+1) — load it with the "
                    f"matching coupling, or rebake"
                )
        if expect_chart is not None and self.chart is not None and self.chart != expect_chart:
            # The chart is a post-coupling measure transform — it does NOT change the
            # layer dims, so a chart mismatch is invisible to the arch/encoding/coupling
            # guards above and would render a silently-wrong distribution (change
            # renderer-chart-selection). Caught here via the PTG1 trailer's chart tag.
            raise ValueError(
                f"neural weights chart {self.chart!r} != built NF_CHART {expect_chart!r}; "
                f"the network was trained on a different hemisphere chart — load it with "
                f"the matching chart, or rebake"
            )


def _layout(layers: int = NF_LAYERS, bins: int = NF_BINS,
            hidden: int = NF_HIDDEN, encoding: Encoding = Encoding.E0,
            l_pos: int = NF_L_POS, coupling: str = "rqs",
            ) -> tuple[list[tuple[int, int, int, int]], int, int]:
    """Header table + total (nWeights, nBiases) for the given architecture.

    Three Linear layers per coupling: (mlp_in→hidden), (hidden→hidden),
    (hidden→n_params). Row-major [out, in], matching the exporter + the Slang
    ``W[weightOffset + o*inDim + i]`` indexing. Defaults to the shipped size.

    The first layer's ``mlp_in`` follows the conditioner ``encoding`` (change
    renderer-conditioner-encoding): ``E0`` keeps ``1 + NF_COND``; ``E1``/``E3``
    widen it to ``1 + NF_MLP_ENC``. The rest of the table is encoding-independent.
    """
    n_params = coupling_n_params(coupling, bins)
    mlp_in = mlp_in_dim(encoding, NF_COND, l_pos)
    dims = [(mlp_in, hidden), (hidden, hidden), (hidden, n_params)]
    headers: list[tuple[int, int, int, int]] = []
    w_off = b_off = 0
    for _ in range(layers):
        for in_dim, out_dim in dims:
            headers.append((w_off, b_off, in_dim, out_dim))
            w_off += out_dim * in_dim
            b_off += out_dim
    return headers, w_off, b_off


def _read_chart_tag(data: bytes, offset: int) -> str | None:
    """Best-effort read of the chart string from the optional PTG1 trailer that
    follows the bias payload (export_weights writes chart/encoding/jacobian). Returns
    None when the trailer is absent (older nets) — the chart guard then no-ops."""
    o = offset
    if o + 4 > len(data):
        return None
    (tag,) = struct.unpack_from("<I", data, o); o += 4
    if tag != TAG_MAGIC or o + 4 > len(data):
        return None
    (n,) = struct.unpack_from("<I", data, o); o += 4
    if o + n > len(data):
        return None
    return data[o:o + n].decode("utf-8", errors="replace")


def deserialize_neural_weights(data: bytes,
                               expect: tuple[int, int, int, int] | None = None,
                               *, src: str = "<bytes>",
                               expect_mlp_in: int | None = None,
                               expect_n_params: int | None = None,
                               expect_chart: str | None = None) -> NeuralWeights:
    """Parse ``NFW1`` bytes into a ``NeuralWeights`` (inverse of
    ``serialize_neural_weights``). The in-memory core shared by
    ``load_neural_weights`` (disk) and the ``shared`` weight-handoff backend
    (process memory, change shared-neural-handoff) — no filesystem access.
    ``src`` names the source in error messages. Raises on bad magic / version /
    truncation, and on an architecture mismatch against ``expect`` (defaults to
    the shipped size when None) or — when ``expect_mlp_in`` is given — a
    first-layer input-width mismatch (the conditioner encoding guard)."""
    o = 0
    magic, ver = struct.unpack_from("<II", data, o)
    o += 8
    if magic != MAGIC:
        raise ValueError(f"{src}: bad magic 0x{magic:08X} (want 0x{MAGIC:08X})")
    if ver != VERSION:
        raise ValueError(f"{src}: unsupported version {ver}")
    layers, bins, hidden, cond = struct.unpack_from("<IIII", data, o)
    o += 16
    (n_headers,) = struct.unpack_from("<I", data, o)
    o += 4
    headers = np.frombuffer(data, dtype="<u4", count=n_headers * 4,
                            offset=o).reshape(n_headers, 4).copy()
    o += n_headers * HEADER_STRIDE
    (n_weights,) = struct.unpack_from("<I", data, o)
    o += 4
    weights = np.frombuffer(data, dtype="<f4", count=n_weights, offset=o).copy()
    o += n_weights * 4
    (n_biases,) = struct.unpack_from("<I", data, o)
    o += 4
    biases = np.frombuffer(data, dtype="<f4", count=n_biases, offset=o).copy()
    o += n_biases * 4
    chart = _read_chart_tag(data, o)        # optional PTG1 trailer (None if absent)
    nw = NeuralWeights(int(layers), int(bins), int(hidden), int(cond),
                       headers, weights, biases, chart=chart)
    nw.assert_matches_shader(expect, expect_mlp_in=expect_mlp_in,
                             expect_n_params=expect_n_params,
                             expect_chart=expect_chart)
    return nw


def load_neural_weights(path: str | Path,
                        expect: tuple[int, int, int, int] | None = None,
                        *, expect_mlp_in: int | None = None,
                        expect_n_params: int | None = None,
                        expect_chart: str | None = None) -> NeuralWeights:
    """Parse an ``NFW1`` file. Raises on bad magic / version / truncation, and on
    an architecture mismatch against ``expect`` (defaults to the shipped size when
    None) — the loader assert is the size-mismatch guard for the configurable
    build (study change neural-precision-size-study). ``expect_mlp_in`` adds the
    conditioner-encoding guard (change renderer-conditioner-encoding): a net whose
    first-layer input width differs from the built ``--encoding`` is refused.
    ``expect_chart`` adds the chart guard (change renderer-chart-selection): a net
    stamped with a different hemisphere chart is refused (the chart is invisible to
    the dim guards)."""
    return deserialize_neural_weights(Path(path).read_bytes(), expect, src=str(path),
                                      expect_mlp_in=expect_mlp_in,
                                      expect_n_params=expect_n_params,
                                      expect_chart=expect_chart)


def make_dummy_weights(config: NeuralBuildConfig | None = None) -> NeuralWeights:
    """In-memory all-zero net for the given architecture (no file I/O).

    Used to seed bindings 33/34/35 so the inline flow inverse in proposal.slang
    always has valid, correctly-sized descriptors even when neural is inactive.
    Defaults to the shipped size; pass a config to size the dummy for an
    off-default build.
    """
    cfg = config or NeuralBuildConfig()
    headers, n_weights, n_biases = _layout(cfg.layers, cfg.bins, cfg.hidden,
                                           cfg.encoding, cfg.l_pos, cfg.coupling)
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
    headers, n_weights, n_biases = _layout(cfg.layers, cfg.bins, cfg.hidden,
                                           cfg.encoding, cfg.l_pos, cfg.coupling)
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


def serialize_neural_weights(nw: NeuralWeights) -> bytes:
    """Serialise a ``NeuralWeights`` to ``NFW1`` bytes (inverse of
    ``deserialize_neural_weights``). The in-memory core shared by
    ``write_neural_weights`` (disk) and the ``shared`` weight-handoff backend
    (process memory, change shared-neural-handoff). Applies the canonical
    ``<f4`` / ``<u4`` casts, so a round-trip through serialize→deserialize yields
    exactly the bytes the renderer would consume from a file publish. NFW1 is
    always fp32."""
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
    return bytes(buf)


def write_neural_weights(path: str | Path, nw: NeuralWeights) -> NeuralWeights:
    """Serialise a ``NeuralWeights`` to an ``NFW1`` file (inverse of
    ``load_neural_weights``). The online trainer (change ``neural-online-training``)
    uses this to publish updated weights that the renderer hot-reloads via the
    file-double-buffer handoff. NFW1 on disk is always fp32."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(serialize_neural_weights(nw))
    return nw
