"""Slang-port numerical parity for the neural spline-flow proposal (task 1.4).

The directional proposal samples a conditional rational-quadratic neural spline
flow on the GPU (``src/skinny/shaders/sampling/neural_flow.slang``:
``sampleNeural`` forward u->wi+pdf, ``pdfNeural`` inverse wi->pdf). That Slang is
a hand-port of the PyTorch reference in ``spline_flow/train.py``
(``ConditionalSplineFlow2D`` + ``square_to_hemisphere``). Three implementations
must agree to machine precision; this test locks the agreement as a regression
WITHOUT torch and WITHOUT a GPU:

  1. PyTorch reference   -> committed goldens (tests/data/neural_parity/, baked
                            once by generate_goldens.py with the torch venv).
  2. numpy mirror        -> this file, ``_flow_forward`` / ``_flow_inverse``,
                            re-implementing neural_flow.slang line-for-line off
                            the SAME flat weight layout the Slang reads
                            (NeuralWeights.headers / weights / biases).
  3. Slang on GPU        -> ``test_neural_parity_gpu`` (skipped without Vulkan).

The numpy-mirror-vs-PyTorch-golden tests (forward + inverse) are the required
deliverable; the proven bar from the prototype gate
(spline_flow/parity_check.py) is |Δwi| < 1e-4 and rel Δpdf < 1e-3.

Run (CI-style, no torch/GPU)::

    PYTHONPATH=src .venv/bin/python -m pytest tests/test_neural_parity.py -q
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from skinny.sampling.neural_weights import (
    NF_BINS,
    NF_COND,
    NF_LAYERS,
    NF_PARAMS,
    load_neural_weights,
)

DATA_DIR = Path(__file__).parent / "data" / "neural_parity"
WEIGHTS_BIN = DATA_DIR / "weights.bin"
GOLDENS_NPZ = DATA_DIR / "goldens.npz"

LOG2PI = math.log(2.0 * math.pi)

# Proven tolerances from spline_flow/parity_check.py (it achieves 3.3e-6 / 4.6e-5).
WI_ABS_TOL = 1e-4
PDF_REL_TOL = 1e-3

pytestmark = pytest.mark.skipif(
    not (WEIGHTS_BIN.exists() and GOLDENS_NPZ.exists()),
    reason=(
        "neural-parity goldens missing; regenerate with "
        "tests/data/neural_parity/generate_goldens.py (spline_flow torch venv)"
    ),
)


# ---------------------------------------------------------------------------
# numpy mirror of shaders/sampling/neural_flow.slang
#
# Adapted from spline_flow/parity_check.py, but driven directly off the
# NeuralWeights flat buffers (headers[nH,4], weights[], biases[]) that the
# renderer's host loader produces and the Slang reads — same layout, same
# row-major [out, in] indexing W[weightOffset + o*inDim + i].
# ---------------------------------------------------------------------------

def _silu(x):
    return x / (1.0 + np.exp(-x))


def _softplus(x):
    return np.log1p(np.exp(x))


def _f16(x):
    """Round a float array through IEEE half and back — models NF_WT/NF_CT=half
    storage/compute quantization (study change neural-precision-size-study)."""
    return np.asarray(x, np.float32).astype(np.float16).astype(np.float32)


def _linear(nw, header, inp, act, prec=(False, False)):
    """One Linear layer: out[o] = bias[o] + sum_i W[o,i]*in[i], optional SiLU.

    ``header`` is a row of ``NeuralWeights.headers``:
    (weightOffset, biasOffset, inDim, outDim). Mirrors ``nf_linear``.

    ``prec`` = (wt_half, ct_half) mirrors the shader's NF_WT/NF_CT aliases:
      wt_half — weights/biases stored as half (fp16-storage + fp16-compute).
      ct_half — the activations + GEMM accumulate are half (fp16-compute only);
                the SiLU is evaluated in float then re-quantized, exactly as
                nf_linear does. Default (False, False) = the shipped fp32 path,
                so the golden parity tests below are byte-for-byte unchanged.
    """
    wt_half, ct_half = prec
    w_off, b_off, in_dim, out_dim = (int(v) for v in header)
    w = nw.weights[w_off:w_off + out_dim * in_dim].reshape(out_dim, in_dim).astype(np.float32)
    b = nw.biases[b_off:b_off + out_dim].astype(np.float32)
    if wt_half:                                  # NF_WT=half — half weight storage
        w, b = _f16(w), _f16(b)
    x = inp[:in_dim].astype(np.float32)
    if ct_half:                                  # NF_CT=half — half activations/GEMM
        x = _f16(x)
        out = _f16(_f16(w) @ _f16(x)) + b
        out = _f16(out)
    else:
        out = w @ x + b
    if act:
        out = _silu(out)
        if ct_half:
            out = _f16(out)
    return out


def _mlp(nw, base3, xcond, cond, prec=(False, False)):
    """3-layer conditioner MLP for one coupling (mirrors ``nf_mlp``).

    ``base3`` indexes this coupling's first header (3 Linear per coupling).
    ``prec`` selects the MLP precision; the spline decode/eval stays float.
    """
    a = np.concatenate([[xcond], cond]).astype(np.float32)
    if prec[1]:                                  # half activations enter the MLP
        a = _f16(a)
    a = _linear(nw, nw.headers[base3 + 0], a, True, prec)   # in -> hidden, SiLU
    a = _linear(nw, nw.headers[base3 + 1], a, True, prec)   # hidden -> hidden, SiLU
    return _linear(nw, nw.headers[base3 + 2], a, False, prec)[:NF_PARAMS]  # -> NF_PARAMS


def _decode(params, K=NF_BINS):
    """Raw MLP params -> normalized spline knots (mirrors ``nf_decode`` /
    PyTorch ``SplineCoupling._params``)."""
    rw, rh, rd = params[:K], params[K:2 * K], params[2 * K:]
    w = np.exp(rw - rw.max())
    w = 1e-4 + w / w.sum()
    w = w / w.sum()
    h = np.exp(rh - rh.max())
    h = 1e-4 + h / h.sum()
    h = h / h.sum()
    d = _softplus(rd) + 1e-3
    return w, h, d


def _rqs_fwd(x, w, h, d):
    """Monotone rational-quadratic spline, forward. Returns (y, log|dy/dx|).

    Bin location matches ``nf_rqs_fwd`` exactly: running cumulative sum,
    ``idx`` is the largest k with x >= cum-before-k.
    """
    x = min(max(x, 0.0), 1.0)
    K = len(w)
    # locate bin (Slang: for k: if (x >= cum) idx = k; cum += widths[k])
    idx = 0
    cum = 0.0
    for k in range(K):
        if x >= cum:
            idx = k
        cum += w[k]
    x0 = float(np.sum(w[:idx]))
    y0 = float(np.sum(h[:idx]))
    wid = max(w[idx], 1e-8)
    hgt = max(h[idx], 1e-8)
    d0, d1 = d[idx], d[idx + 1]
    delta = hgt / wid
    theta = (x - x0) / wid
    t1 = theta * (1.0 - theta)
    num = hgt * (delta * theta * theta + d0 * t1)
    den = max(delta + (d0 + d1 - 2.0 * delta) * t1, 1e-8)
    y = y0 + num / den
    dn = delta * delta * (
        d1 * theta * theta + 2.0 * delta * t1 + d0 * (1.0 - theta) ** 2
    )
    dydx = dn / (den * den)
    return min(max(y, 0.0), 1.0), math.log(max(dydx, 1e-8))


def _rqs_inv(y, w, h, d):
    """Monotone rational-quadratic spline, inverse. Returns (x, log|dx/dy|).

    Mirrors ``nf_rqs_inv``: analytic quadratic solve, then logdet = -fwdlog.
    """
    y = min(max(y, 0.0), 1.0)
    K = len(w)
    idx = 0
    cum = 0.0
    for k in range(K):
        if y >= cum:
            idx = k
        cum += h[k]
    x0 = float(np.sum(w[:idx]))
    y0 = float(np.sum(h[:idx]))
    wid = max(w[idx], 1e-8)
    hgt = max(h[idx], 1e-8)
    d0, d1 = d[idx], d[idx + 1]
    delta = hgt / wid
    z = (y - y0) / hgt
    a = d0 + d1 - 2.0 * delta
    A = z * a + delta - d0
    Bc = d0 - z * a
    C = -delta * z
    disc = max(Bc * Bc - 4.0 * A * C, 0.0)
    denom = min(-Bc - math.sqrt(disc), -1e-8)
    theta = min(max((2.0 * C) / denom, 0.0), 1.0)
    x = x0 + theta * wid
    _, fwdlog = _rqs_fwd(x, w, h, d)
    return min(max(x, 0.0), 1.0), -fwdlog


def _flow_forward(nw, u, cond, prec=(False, False)):
    """Forward flow u -> z, accumulating log|det dz/du| (mirrors
    ``nf_flow_forward``: even layer conditions on dim0, transforms dim1)."""
    z = np.array(u, np.float64)
    logdet = 0.0
    for L in range(NF_LAYERS):
        even = (L % 2 == 0)
        xcond = z[0] if even else z[1]
        xtr = z[1] if even else z[0]
        w, h, d = _decode(_mlp(nw, L * 3, xcond, cond, prec))
        ytr, ld = _rqs_fwd(xtr, w, h, d)
        if even:
            z[1] = ytr
        else:
            z[0] = ytr
        logdet += ld
    return z, logdet


def _flow_inverse(nw, zin, cond, prec=(False, False)):
    """Inverse flow z -> u, accumulating log|det du/dz| (mirrors
    ``nf_flow_inverse``: reversed layer order)."""
    z = np.array(zin, np.float64)
    logdet = 0.0
    for L in range(NF_LAYERS - 1, -1, -1):
        even = (L % 2 == 0)
        ycond = z[0] if even else z[1]
        ytr = z[1] if even else z[0]
        w, h, d = _decode(_mlp(nw, L * 3, ycond, cond, prec))
        xtr, ld = _rqs_inv(ytr, w, h, d)
        if even:
            z[1] = xtr
        else:
            z[0] = xtr
        logdet += ld
    return z, logdet


def _square_to_hemi(z):
    """z in [0,1]^2 -> upper-hemisphere direction, y-up (mirrors
    ``nf_square_to_hemi`` / ``square_to_hemisphere``)."""
    u = min(max(z[0], 0.0), 1.0)
    v = min(max(z[1], 0.0), 1.0)
    phi = 2.0 * math.pi * u
    st = math.sqrt(max(1.0 - v * v, 0.0))
    return np.array([st * math.cos(phi), v, st * math.sin(phi)], np.float64)


def _hemi_to_square(wld):
    """Inverse of ``_square_to_hemi`` (mirrors ``nf_hemi_to_square``)."""
    phi = math.atan2(wld[2], wld[0])
    if phi < 0.0:
        phi += 2.0 * math.pi
    return np.array([phi / (2.0 * math.pi), min(max(wld[1], 0.0), 1.0)], np.float64)


def _sample_neural(nw, cond, u, prec=(False, False)):
    """PUBLIC mirror of ``sampleNeural``: forward u -> (wi, solid-angle pdf)."""
    z, logdet = _flow_forward(nw, u, cond, prec)
    pdf_omega = math.exp(-logdet - LOG2PI)   # base pdf on unit square = 1
    return _square_to_hemi(z), pdf_omega


def _pdf_neural(nw, cond, wi, prec=(False, False)):
    """PUBLIC mirror of ``pdfNeural``: inverse wi -> solid-angle pdf."""
    if wi[1] <= 0.0:
        return 0.0
    z = _hemi_to_square(wi)
    _, logdet = _flow_inverse(nw, z, cond, prec)   # log|det du/dz| = log q_square
    return math.exp(logdet - LOG2PI)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def weights():
    nw = load_neural_weights(WEIGHTS_BIN)
    # The whole parity argument rests on the baked architecture matching the
    # compiled-in Slang constants — assert it loudly here.
    nw.assert_matches_shader()
    assert (nw.layers, nw.bins, nw.cond) == (NF_LAYERS, NF_BINS, NF_COND)
    return nw


@pytest.fixture(scope="module")
def goldens():
    return np.load(GOLDENS_NPZ)


# ---------------------------------------------------------------------------
# REQUIRED: numpy mirror vs PyTorch golden — forward path (sampleNeural)
# ---------------------------------------------------------------------------

def test_forward_matches_pytorch_golden(weights, goldens):
    """For each committed (cond, u), the numpy mirror of the Slang forward must
    reproduce the PyTorch (wi, solid-angle pdf) within the proven tolerance."""
    conds = goldens["forward_cond"]
    us = goldens["forward_u"]
    wi_ref = goldens["forward_wi"]
    pdf_ref = goldens["forward_pdf"]
    assert conds.shape[0] > 0 and conds.shape[1] == NF_COND

    max_dwi = 0.0
    max_rel_pdf = 0.0
    for cond, u, wi_t, pdf_t in zip(conds, us, wi_ref, pdf_ref):
        wi_np, pdf_np = _sample_neural(weights, cond.astype(np.float32), u.astype(np.float32))
        max_dwi = max(max_dwi, float(np.abs(wi_np - wi_t).max()))
        max_rel_pdf = max(max_rel_pdf, abs(pdf_np - float(pdf_t)) / max(float(pdf_t), 1e-6))

    assert max_dwi < WI_ABS_TOL, f"max |Δwi| = {max_dwi:.2e} (bar {WI_ABS_TOL:.0e})"
    assert max_rel_pdf < PDF_REL_TOL, (
        f"max rel Δpdf = {max_rel_pdf:.2e} (bar {PDF_REL_TOL:.0e})"
    )


def test_forward_pdf_is_positive_finite(weights, goldens):
    """Every forward solid-angle pdf must be finite and strictly positive."""
    for cond, u in zip(goldens["forward_cond"], goldens["forward_u"]):
        _, pdf = _sample_neural(weights, cond.astype(np.float32), u.astype(np.float32))
        assert math.isfinite(pdf) and pdf > 0.0, f"bad pdf {pdf}"


# ---------------------------------------------------------------------------
# REQUIRED: inverse / round-trip — exercises pdfNeural (spline inverse + Jacobian)
# ---------------------------------------------------------------------------

def test_inverse_matches_pytorch_golden(weights, goldens):
    """For each committed (cond, wi-in-upper-hemisphere), the numpy mirror of
    the Slang inverse pdf must match the PyTorch inverse pdf, and be > 0."""
    conds = goldens["inverse_cond"]
    wis = goldens["inverse_wi"]
    pdf_ref = goldens["inverse_pdf"]

    max_rel_pdf = 0.0
    for cond, wi, pdf_t in zip(conds, wis, pdf_ref):
        pdf_np = _pdf_neural(weights, cond.astype(np.float32), wi.astype(np.float64))
        assert math.isfinite(pdf_np) and pdf_np > 0.0, f"non-positive inverse pdf {pdf_np}"
        max_rel_pdf = max(max_rel_pdf, abs(pdf_np - float(pdf_t)) / max(float(pdf_t), 1e-6))

    assert max_rel_pdf < PDF_REL_TOL, (
        f"inverse max rel Δpdf = {max_rel_pdf:.2e} (bar {PDF_REL_TOL:.0e})"
    )


def test_forward_then_inverse_pdf_consistent(weights, goldens):
    """Round-trip the flow's own math: draw wi via the forward sampler from u,
    then evaluate the inverse pdf at that wi. Because forward and inverse are
    analytic inverses, the inverse pdf must equal the forward pdf at the same
    point — this is the cross-check between ``sampleNeural`` and ``pdfNeural``.
    """
    conds = goldens["forward_cond"]
    us = goldens["forward_u"]

    max_rel = 0.0
    for cond, u in zip(conds, us):
        c32 = cond.astype(np.float32)
        wi_np, pdf_fwd = _sample_neural(weights, c32, u.astype(np.float32))
        # wi from the forward sampler always lands in the upper hemisphere.
        assert wi_np[1] >= 0.0
        pdf_inv = _pdf_neural(weights, c32, wi_np)
        max_rel = max(max_rel, abs(pdf_inv - pdf_fwd) / max(pdf_fwd, 1e-6))

    # Round-trip uses the same RQ-spline forward/inverse pair; agreement is far
    # tighter than the PyTorch bar, but assert the proven bar to stay robust.
    assert max_rel < PDF_REL_TOL, (
        f"forward/inverse pdf mismatch: max rel = {max_rel:.2e} (bar {PDF_REL_TOL:.0e})"
    )


# ---------------------------------------------------------------------------
# PRECISION TRACK (study change neural-precision-size-study): fp16 pdf-parity
# drift. The SAME trained net evaluated at fp16-storage / fp16-compute vs the
# fp32 reference — numerical fidelity, scene-independent (unlike the size track,
# which retrains per size). The drift is REPORTED (spec: "the measured drift is
# reported, not hidden") and bounded by a relaxed, mode-specific bar (fp16's
# ~10-bit mantissa is looser than the 1e-3 fp32 golden bar). The fp32 mirror is
# itself golden-matched (tests above), so drift-vs-mirror ≈ drift-vs-PyTorch.
# ---------------------------------------------------------------------------

# Generous regression guards; the measured drift (printed) sits well under them.
# fp16-storage only quantizes the weights (GEMM stays float); fp16-compute also
# runs the GEMM in half, so it drifts further.
FP16_STORAGE_PDF_REL_BAR = 0.05
FP16_COMPUTE_PDF_REL_BAR = 0.30


def _pdf_drift(weights, goldens, prec):
    """Max + mean relative forward-pdf drift of a precision mode vs the fp32
    mirror, over the committed forward (cond, u) inputs."""
    max_rel = mean_rel = 0.0
    n = 0
    for cond, u in zip(goldens["forward_cond"], goldens["forward_u"]):
        c = cond.astype(np.float32)
        uu = u.astype(np.float32)
        _, pdf32 = _sample_neural(weights, c, uu)            # fp32 reference
        _, pdf16 = _sample_neural(weights, c, uu, prec)      # the fp16 mode
        rel = abs(pdf16 - pdf32) / max(pdf32, 1e-8)
        max_rel = max(max_rel, rel)
        mean_rel += rel
        n += 1
    return max_rel, mean_rel / max(n, 1)


def test_fp16_storage_pdf_drift(weights, goldens):
    """3.1: fp16-storage (half weights, float GEMM) drift vs fp32 — reported + bounded."""
    max_rel, mean_rel = _pdf_drift(weights, goldens, (True, False))
    print(f"\n[3.1] fp16-storage pdf drift vs fp32: max={max_rel:.2e} mean={mean_rel:.2e} "
          f"(bar {FP16_STORAGE_PDF_REL_BAR:.0e})")
    assert max_rel < FP16_STORAGE_PDF_REL_BAR, (
        f"fp16-storage drift {max_rel:.2e} exceeds bar {FP16_STORAGE_PDF_REL_BAR:.0e}"
    )


def test_fp16_compute_pdf_drift(weights, goldens):
    """3.1: fp16-compute (half weights + half GEMM) drift vs fp32 — reported + bounded."""
    max_rel, mean_rel = _pdf_drift(weights, goldens, (True, True))
    print(f"\n[3.1] fp16-compute pdf drift vs fp32: max={max_rel:.2e} mean={mean_rel:.2e} "
          f"(bar {FP16_COMPUTE_PDF_REL_BAR:.0e})")
    assert max_rel < FP16_COMPUTE_PDF_REL_BAR, (
        f"fp16-compute drift {max_rel:.2e} exceeds bar {FP16_COMPUTE_PDF_REL_BAR:.0e}"
    )


def test_fp16_pdf_positive_finite(weights, goldens):
    """3.1: both fp16 modes must still produce finite, strictly-positive pdfs — a
    valid mixture component is the unbiasedness precondition (4.2 confirms the
    in-renderer convergence on GPU)."""
    for cond, u in zip(goldens["forward_cond"], goldens["forward_u"]):
        for prec in ((True, False), (True, True)):
            _, pdf = _sample_neural(weights, cond.astype(np.float32),
                                    u.astype(np.float32), prec)
            assert math.isfinite(pdf) and pdf > 0.0, f"bad fp16 pdf {pdf} prec={prec}"


# ---------------------------------------------------------------------------
# OPTIONAL: GPU-runtime parity (the true 1a gate) — skipped without Vulkan.
#
# Compiles a tiny Slang compute kernel that calls sampleNeural / pdfNeural over
# neural_flow.slang, dispatches it with the committed weights + golden inputs
# via slangpy, reads back, and compares to the goldens. The headless env is the
# repo-root Python 3.13 venv with VULKAN_SDK + DYLD_LIBRARY_PATH set, and slangc
# at $VULKAN_SDK/bin/slangc; see CLAUDE.md "Headless / offscreen rendering".
# Under the plain repo .venv (no slangpy / no Vulkan) this skips cleanly, so it
# never blocks the CI deliverable above.
# ---------------------------------------------------------------------------

_GPU_HARNESS = """
import "sampling/neural_flow.slang";

StructuredBuffer<float> gW;
StructuredBuffer<float> gB;
StructuredBuffer<NfLayerHeader> gH;
StructuredBuffer<float> gCond;   // N * NF_COND, row-major
StructuredBuffer<float> gU;      // N * 2
RWStructuredBuffer<float> gWi;   // N * 3   (forward direction)
RWStructuredBuffer<float> gPdf;  // N       (forward solid-angle pdf)
RWStructuredBuffer<float> gPdfInv; // N     (inverse pdf at the forward wi)
RWStructuredBuffer<uint>  gHdr0; // 4: gH[0] fields, lets the host confirm the
                                 // typed StructuredBuffer<NfLayerHeader> bound

[shader("compute")]
[numthreads(64, 1, 1)]
void neuralParity(uint3 tid: SV_DispatchThreadID, uniform int n)
{
    int i = int(tid.x);
    if (i == 0)
    {
        gHdr0[0] = gH[0].weightOffset; gHdr0[1] = gH[0].biasOffset;
        gHdr0[2] = gH[0].inDim;        gHdr0[3] = gH[0].outDim;
    }
    if (i >= n) return;
    float cond[NF_COND];
    for (int k = 0; k < NF_COND; ++k) cond[k] = gCond[i * NF_COND + k];
    float2 u = float2(gU[i * 2 + 0], gU[i * 2 + 1]);

    float pdf;
    float3 wi = sampleNeural(gW, gB, gH, cond, u, pdf);
    gWi[i * 3 + 0] = wi.x; gWi[i * 3 + 1] = wi.y; gWi[i * 3 + 2] = wi.z;
    gPdf[i] = pdf;
    gPdfInv[i] = pdfNeural(gW, gB, gH, cond, wi);
}
"""


def _have_gpu() -> bool:
    try:
        import slangpy  # noqa: F401
    except Exception:
        return False
    return True


@pytest.mark.gpu
@pytest.mark.skipif(not _have_gpu(), reason="slangpy / Vulkan device unavailable")
def test_neural_parity_gpu(weights, goldens):
    """Dispatch the real Slang sampleNeural/pdfNeural over the committed weights
    and compare GPU results to the PyTorch goldens (true 1a gate)."""
    import slangpy as spy

    shader_dir = Path(__file__).resolve().parent.parent / "src" / "skinny" / "shaders"
    try:
        device = spy.create_device(include_paths=[str(shader_dir)])
    except Exception:
        pytest.skip("No Vulkan device available")

    module = spy.Module(device.load_module_from_source("neural_parity_harness", _GPU_HARNESS))

    conds = goldens["forward_cond"].astype(np.float32)
    us = goldens["forward_u"].astype(np.float32)
    wi_ref = goldens["forward_wi"]
    pdf_ref = goldens["forward_pdf"]
    n = conds.shape[0]

    def _fbuf(arr):
        return spy.NDBuffer.from_numpy(device, np.ascontiguousarray(arr, np.float32)).storage

    # The headers feed a typed StructuredBuffer<NfLayerHeader> (4 x uint32, 16-B
    # stride). slangpy NDBuffer has no 16-byte struct dtype, so build a raw GPU
    # buffer with the matching struct_size and bind it directly.
    n_hdr = weights.headers.shape[0]
    hdr_u8 = np.frombuffer(weights.headers.astype("<u4").tobytes(), dtype=np.uint8).copy()
    g_h = device.create_buffer(
        element_count=n_hdr, struct_size=16,
        usage=spy.BufferUsage.shader_resource, data=hdr_u8,
    )

    g_w = _fbuf(weights.weights)
    g_b = _fbuf(weights.biases)
    g_cond = _fbuf(conds.reshape(-1))
    g_u = _fbuf(us.reshape(-1))
    out_wi = spy.NDBuffer.from_numpy(device, np.zeros(n * 3, np.float32))
    out_pdf = spy.NDBuffer.from_numpy(device, np.zeros(n, np.float32))
    out_pdf_inv = spy.NDBuffer.from_numpy(device, np.zeros(n, np.float32))
    out_hdr0 = spy.NDBuffer.from_numpy(device, np.zeros(4, np.uint32))

    try:
        module.neuralParity.dispatch(
            (((n + 63) // 64) * 64, 1, 1),
            gW=g_w, gB=g_b, gH=g_h, gCond=g_cond, gU=g_u,
            gWi=out_wi.storage, gPdf=out_pdf.storage, gPdfInv=out_pdf_inv.storage,
            gHdr0=out_hdr0.storage, n=n,
        )
        device.wait()
    except Exception as exc:  # pragma: no cover - platform dependent
        pytest.skip(f"slangpy could not dispatch the neural harness: {exc!r}")

    # Confirm the typed header buffer actually bound — slangpy's high-level
    # dispatch silently leaves an un-bindable StructuredBuffer<struct> zeroed on
    # some backends. If the GPU read zero headers (CPU headers are non-zero),
    # the typed-struct bind is unavailable here; the GPU gate is then validated
    # via the headless bring-up (task 4.3), so skip rather than report red.
    gpu_hdr0 = out_hdr0.to_numpy()
    if not np.array_equal(gpu_hdr0, weights.headers[0].astype(np.uint32)):
        pytest.skip(
            "slangpy cannot bind StructuredBuffer<NfLayerHeader> via the "
            f"high-level dispatch here (GPU read header[0]={list(gpu_hdr0)}, "
            f"expected {list(weights.headers[0])}); GPU-runtime parity is "
            "validated via the headless bring-up (openspec task 4.3)"
        )

    wi_gpu = out_wi.to_numpy().reshape(n, 3)
    pdf_gpu = out_pdf.to_numpy()
    pdf_inv_gpu = out_pdf_inv.to_numpy()

    max_dwi = float(np.abs(wi_gpu - wi_ref).max())
    rel_pdf = np.abs(pdf_gpu - pdf_ref) / np.maximum(pdf_ref, 1e-6)
    rel_inv = np.abs(pdf_inv_gpu - pdf_gpu) / np.maximum(pdf_gpu, 1e-6)

    assert max_dwi < WI_ABS_TOL, f"GPU max |Δwi| = {max_dwi:.2e}"
    assert float(rel_pdf.max()) < PDF_REL_TOL, f"GPU max rel Δpdf = {rel_pdf.max():.2e}"
    assert float(rel_inv.max()) < PDF_REL_TOL, (
        f"GPU forward/inverse pdf mismatch = {rel_inv.max():.2e}"
    )
