"""pbrt classic-Perlin / CloudMedium::Density parity oracle
(pbrt-cloud-procedural-medium, tasks 1.3 + 4.4).

Hostless: a float32 numpy mirror of the EXACT pbrt algorithm (`util/noise.cpp`
Noise/DNoise + `media.h` CloudMedium::Density) validates the Slang port's
constants (the 256-entry permutation table is embedded here independently, so
editing cloud_noise.slang cannot silently drift) and the density formula's
analytic properties (lattice zeros, altitude falloff endpoints, fBm structure).

gpu-marked: a tiny slangpy Metal probe kernel evaluates the ACTUAL
cloud_noise.slang `pbrtNoise`/`cloudDensity` at a grid of points and compares
against the numpy mirror to float tolerance -- the ported-noise-matches-pbrt
scenario of the heterogeneous-media spec delta.

Run (gpu part, guarded -- one Metal process at a time):

    export VULKAN_SDK=/Users/ahmetbilgili/VulkanSDK/1.4.341.1/macOS
    export DYLD_LIBRARY_PATH=$VULKAN_SDK/lib
    PYTHONPATH=$PWD/src <repo>/bin/python3.13 -m pytest \
        tests/test_cloud_noise.py -m gpu -q
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

_SLANG_PATH = (
    Path(__file__).resolve().parents[1]
    / "src/skinny/shaders/materials/subsurface/cloud_noise.slang"
)

# pbrt util/noise.cpp NoisePerm base table (256 entries), transcribed from the
# pbrt-v4 source INDEPENDENTLY of cloud_noise.slang -- the .slang table is
# checked against this, so neither copy can drift alone.
_PBRT_NOISE_PERM = [
    151, 160, 137, 91, 90, 15, 131, 13, 201, 95, 96, 53, 194, 233, 7, 225,
    140, 36, 103, 30, 69, 142, 8, 99, 37, 240, 21, 10, 23, 190, 6, 148,
    247, 120, 234, 75, 0, 26, 197, 62, 94, 252, 219, 203, 117, 35, 11, 32,
    57, 177, 33, 88, 237, 149, 56, 87, 174, 20, 125, 136, 171, 168, 68, 175,
    74, 165, 71, 134, 139, 48, 27, 166, 77, 146, 158, 231, 83, 111, 229, 122,
    60, 211, 133, 230, 220, 105, 92, 41, 55, 46, 245, 40, 244, 102, 143, 54,
    65, 25, 63, 161, 1, 216, 80, 73, 209, 76, 132, 187, 208, 89, 18, 169,
    200, 196, 135, 130, 116, 188, 159, 86, 164, 100, 109, 198, 173, 186, 3, 64,
    52, 217, 226, 250, 124, 123, 5, 202, 38, 147, 118, 126, 255, 82, 85, 212,
    207, 206, 59, 227, 47, 16, 58, 17, 182, 189, 28, 42, 223, 183, 170, 213,
    119, 248, 152, 2, 44, 154, 163, 70, 221, 153, 101, 155, 167, 43, 172, 9,
    129, 22, 39, 253, 19, 98, 108, 110, 79, 113, 224, 232, 178, 185, 112, 104,
    218, 246, 97, 228, 251, 34, 242, 193, 238, 210, 144, 12, 191, 179, 162, 241,
    81, 51, 145, 235, 249, 14, 239, 107, 49, 192, 214, 31, 181, 199, 106, 157,
    184, 84, 204, 176, 115, 121, 50, 45, 127, 4, 150, 254, 138, 236, 205, 93,
    222, 114, 67, 29, 24, 72, 243, 141, 128, 195, 78, 66, 215, 61, 156, 180,
]

_PERM = np.array(_PBRT_NOISE_PERM + _PBRT_NOISE_PERM, dtype=np.int64)

_F = np.float32


# --------------------------------------------------------------------------- #
# float32 numpy mirror of the pbrt algorithm
# --------------------------------------------------------------------------- #

def _noise_weight(t: np.ndarray) -> np.ndarray:
    t3 = t * t * t
    t4 = t3 * t
    return _F(6) * t4 * t - _F(15) * t4 + _F(10) * t3


def _grad(ix, iy, iz, dx, dy, dz):
    h = _PERM[_PERM[_PERM[ix] + iy] + iz] & 15
    u = np.where((h < 8) | (h == 12) | (h == 13), dx, dy)
    v = np.where((h < 4) | (h == 12) | (h == 13), dy, dz)
    return np.where(h & 1, -u, u) + np.where(h & 2, -v, v)


def pbrt_noise(p: np.ndarray) -> np.ndarray:
    """pbrt Noise(Point3f) over an (N, 3) float32 array."""
    q = np.fmod(p.astype(_F), _F(1 << 30))
    ip = np.floor(q).astype(np.int64)
    d = (q - ip.astype(_F)).astype(_F)
    ix, iy, iz = ip[:, 0] & 255, ip[:, 1] & 255, ip[:, 2] & 255
    dx, dy, dz = d[:, 0], d[:, 1], d[:, 2]
    one = _F(1)
    w000 = _grad(ix, iy, iz, dx, dy, dz)
    w100 = _grad(ix + 1, iy, iz, dx - one, dy, dz)
    w010 = _grad(ix, iy + 1, iz, dx, dy - one, dz)
    w110 = _grad(ix + 1, iy + 1, iz, dx - one, dy - one, dz)
    w001 = _grad(ix, iy, iz + 1, dx, dy, dz - one)
    w101 = _grad(ix + 1, iy, iz + 1, dx - one, dy, dz - one)
    w011 = _grad(ix, iy + 1, iz + 1, dx, dy - one, dz - one)
    w111 = _grad(ix + 1, iy + 1, iz + 1, dx - one, dy - one, dz - one)
    wx, wy, wz = _noise_weight(dx), _noise_weight(dy), _noise_weight(dz)
    x00 = w000 + wx * (w100 - w000)
    x10 = w010 + wx * (w110 - w010)
    x01 = w001 + wx * (w101 - w001)
    x11 = w011 + wx * (w111 - w011)
    y0 = x00 + wy * (x10 - x00)
    y1 = x01 + wy * (x11 - x01)
    return (y0 + wz * (y1 - y0)).astype(_F)


def pbrt_dnoise(p: np.ndarray) -> np.ndarray:
    """pbrt DNoise: forward difference with delta = 0.01, (N, 3) -> (N, 3)."""
    delta = _F(0.01)
    n = pbrt_noise(p)
    out = np.empty_like(p, dtype=_F)
    for axis in range(3):
        dp = p.astype(_F).copy()
        dp[:, axis] += delta
        out[:, axis] = (pbrt_noise(dp) - n) / delta
    return out


def cloud_density(p: np.ndarray, density: float, wispiness: float,
                  frequency: float) -> np.ndarray:
    """pbrt CloudMedium::Density over an (N, 3) float32 array of
    medium-local points (unclipped, exactly like pbrt's Density)."""
    p = p.astype(_F)
    pp = _F(frequency) * p
    if wispiness > 0:
        vomega, vlambda = _F(0.05) * _F(wispiness), _F(10)
        for _ in range(2):
            pp = pp + vomega * pbrt_dnoise(vlambda * pp)
            vomega *= _F(0.5)
            vlambda *= _F(1.99)
    d = np.zeros(p.shape[0], dtype=_F)
    omega, lam = _F(0.5), _F(1)
    for _ in range(5):
        d += omega * pbrt_noise(lam * pp)
        omega *= _F(0.5)
        lam *= _F(1.99)
    d = np.clip((_F(1) - p[:, 1]) * _F(4.5) * _F(density) * d, _F(0), _F(1))
    d += _F(2) * np.maximum(_F(0), _F(0.5) - p[:, 1])
    return np.clip(d, _F(0), _F(1)).astype(_F)


def _grid_points(n_side: int = 10, lo: float = -1.5, hi: float = 2.5) -> np.ndarray:
    """Deterministic (n^3, 3) float32 grid covering negatives + off-lattice."""
    ax = np.linspace(lo, hi, n_side, dtype=_F) + _F(0.137)
    g = np.stack(np.meshgrid(ax, ax, ax, indexing="ij"), axis=-1).reshape(-1, 3)
    return g.astype(_F)


# --------------------------------------------------------------------------- #
# hostless: constants + analytic properties
# --------------------------------------------------------------------------- #

def test_slang_perm_table_matches_pbrt():
    """The 512-entry table in cloud_noise.slang is pbrt's base permutation
    duplicated -- checked against this file's independent transcription."""
    src = _SLANG_PATH.read_text()
    m = re.search(r"NOISE_PERM\[512\] = \{(.*?)\};", src, re.S)
    assert m, "NOISE_PERM table not found in cloud_noise.slang"
    vals = [int(x) for x in re.findall(r"\d+", m.group(1))]
    assert len(vals) == 512
    assert vals[:256] == _PBRT_NOISE_PERM
    assert vals[256:] == _PBRT_NOISE_PERM


def test_perm_table_is_permutation():
    assert sorted(_PBRT_NOISE_PERM) == list(range(256))


def test_noise_zero_on_integer_lattice():
    """Classic Perlin is exactly 0 at lattice points (all gradient offsets 0)."""
    pts = np.array([[0, 0, 0], [1, 2, 3], [-4, 5, -6], [17, -23, 200]], dtype=_F)
    assert np.all(pbrt_noise(pts) == 0.0)


def test_noise_bounded_and_structured():
    n = pbrt_noise(_grid_points())
    assert np.all(np.abs(n) <= 1.2)         # classic Perlin stays within ~[-1, 1]
    assert float(np.std(n)) > 0.05          # not a constant field


def test_cloud_density_altitude_endpoints():
    """y=0 floor term forces density 1; y=1 kills both terms -> 0."""
    xz = np.linspace(0.1, 0.9, 5, dtype=_F)
    base = np.stack([xz, np.zeros_like(xz), xz], axis=-1)
    top = np.stack([xz, np.ones_like(xz), xz], axis=-1)
    d_base = cloud_density(base, density=2.0, wispiness=1.0, frequency=5.0)
    d_top = cloud_density(top, density=2.0, wispiness=1.0, frequency=5.0)
    assert np.all(d_base == 1.0)            # += 2*max(0, 0.5-0) == 1, clamped
    assert np.all(d_top == 0.0)             # (1-1)*... == 0, no floor term


def test_cloud_density_in_unit_range_and_structured():
    p = (_grid_points(8, 0.0, 1.0) % _F(1.0)).astype(_F)
    d = cloud_density(p, density=2.0, wispiness=1.0, frequency=5.0)
    assert np.all((d >= 0.0) & (d <= 1.0))
    mid = d[(p[:, 1] > 0.55) & (p[:, 1] < 0.95)]
    assert float(np.std(mid)) > 0.05        # fBm structure, not homogeneous


def test_wispiness_warps_domain():
    p = (_grid_points(6, 0.0, 1.0) % _F(1.0)).astype(_F)
    d0 = cloud_density(p, density=2.0, wispiness=0.0, frequency=5.0)
    d1 = cloud_density(p, density=2.0, wispiness=1.0, frequency=5.0)
    changed = d0 != d1
    assert changed.mean() > 0.3             # the warp visibly moves the field


def test_zero_density_scale_zeroes_fbm_term():
    p = (_grid_points(5, 0.0, 1.0) % _F(1.0)).astype(_F)
    d = cloud_density(p, density=0.0, wispiness=1.0, frequency=5.0)
    floor = np.clip(_F(2) * np.maximum(_F(0), _F(0.5) - p[:, 1]), 0, 1)
    assert np.allclose(d, floor, atol=0.0)  # only the altitude floor survives


# --------------------------------------------------------------------------- #
# gpu: the ACTUAL Slang port vs the mirror (Metal probe kernel)
# --------------------------------------------------------------------------- #

@pytest.mark.gpu
def test_slang_cloud_density_matches_numpy_mirror():
    try:
        from skinny.backend_select import metal_available
    except OSError as exc:  # pragma: no cover
        pytest.skip(f"needs the Vulkan SDK on the dylib path: {exc}")
    ok, reason = metal_available()
    if not ok:
        pytest.skip(f"native Metal unavailable: {reason}")
    spy = pytest.importorskip("slangpy")

    pts = _grid_points(10)                              # 1000 points, negatives too
    n_pts = pts.shape[0]
    kernel_src = _SLANG_PATH.read_text() + f"""
StructuredBuffer<float4> pts;
RWStructuredBuffer<float> outNoise;
RWStructuredBuffer<float> outDensity;
[shader("compute")][numthreads(64,1,1)]
void computeMain(uint3 t : SV_DispatchThreadID) {{
    if (t.x >= {n_pts}) return;
    float3 p = pts[t.x].xyz;
    outNoise[t.x] = pbrtNoise(p);
    // clouds.pbrt parameters: density 2, wispiness 1 (default), frequency 5.
    outDensity[t.x] = cloudDensity(p, 2.0, 1.0, 5.0);
}}
"""
    dev = spy.create_device(type=spy.DeviceType.metal)
    sess = dev.create_slang_session(compiler_options=spy.SlangCompilerOptions())
    mod = sess.load_module_from_source("cloud_probe", kernel_src, "cloud_probe.slang")
    prog = sess.link_program([mod], [mod.entry_point("computeMain")])
    kernel = dev.create_compute_kernel(prog)

    pts4 = np.zeros((n_pts, 4), np.float32)
    pts4[:, :3] = pts
    buf = dev.create_buffer(
        element_count=n_pts, struct_size=16,
        usage=spy.BufferUsage.shader_resource, data=pts4,
        memory_type=spy.MemoryType.device_local, label="pts")
    out_n = dev.create_buffer(
        element_count=n_pts, struct_size=4,
        usage=spy.BufferUsage.unordered_access | spy.BufferUsage.shader_resource,
        memory_type=spy.MemoryType.device_local, label="n")
    out_d = dev.create_buffer(
        element_count=n_pts, struct_size=4,
        usage=spy.BufferUsage.unordered_access | spy.BufferUsage.shader_resource,
        memory_type=spy.MemoryType.device_local, label="d")
    kernel.dispatch(thread_count=[((n_pts + 63) // 64) * 64, 1, 1],
                    vars={"pts": buf, "outNoise": out_n, "outDensity": out_d})
    dev.wait_for_idle()
    gpu_n = out_n.to_numpy().view(np.float32)[:n_pts]
    gpu_d = out_d.to_numpy().view(np.float32)[:n_pts]

    ref_n = pbrt_noise(pts)
    ref_d = cloud_density(pts, density=2.0, wispiness=1.0, frequency=5.0)
    # float32 GPU vs float32 numpy: only FMA-contraction noise expected.
    np.testing.assert_allclose(gpu_n, ref_n, atol=1e-4)
    np.testing.assert_allclose(gpu_d, ref_d, atol=5e-3)
