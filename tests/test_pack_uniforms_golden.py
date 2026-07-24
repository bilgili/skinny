"""Golden byte-equality gate for `_pack_uniforms` / `_pack_uniforms_msl`
(change renderer-module-carveout, Stage B).

Stronger than image parity: it pins the packed `fc` blob byte-for-byte across a
state matrix that drives every value the frame-constant derivation extraction
touches — the detail-flag bitfield, the lens FOV-framing ratio, the
exposure/imaging-ratio fold, and the proposal-mask/reuse capability folding
(incl. the neural-on-megakernel strip). Any drift in `frame_derive`'s output
shifts a byte and fails here.

NOT hostless: it constructs `Renderer`s (GPU sessions), and the execution mode
is fixed per session — so it is gpu-marked and runs one Metal context at a time
under the metal-dispatch-hygiene rules (the two modes are sequential contexts,
not parallel processes). Regenerate the golden with

    SKINNY_CAPTURE_GOLDEN=1 PYTHONPATH=src SKINNY_BACKEND=metal ./bin/python3.13 \
        -m pytest tests/test_pack_uniforms_golden.py -m gpu -q

on the pre-refactor tree; the extraction must then leave it green.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.gpu

_GOLDEN = Path(__file__).resolve().parent / "golden" / "pack_uniforms_golden.npz"
_SUITE = Path(__file__).resolve().parent / "assets" / "suite" / "int_caustic" / \
    "int_caustic.usda"


def _msl(r):
    """Reflected MSL blob, or a sentinel when the active pipeline has no MSL
    layout to reflect (Vulkan, or a wavefront session mid-build). Pre and post
    hit the sentinel identically, so equality still holds."""
    try:
        return r._pack_uniforms_msl()
    except Exception as exc:  # pragma: no cover - mode-dependent
        return f"<no-msl: {type(exc).__name__}>".encode()


def _capture(mode: str) -> dict[str, bytes]:
    """Build one renderer in *mode*, force pipeline compilation, then pack the
    fc blob across the state matrix. Returns {label: pre-packed bytes}."""
    import skinny
    from skinny.backend_select import make_context, select_backend
    from skinny.headless import HeadlessRenderer, RenderOptions

    backend = select_backend()
    blobs: dict[str, bytes] = {}
    with HeadlessRenderer(96, 96, backend=backend, execution_mode=mode) as h:
        r = h.renderer
        h._prepare(str(_SUITE), RenderOptions(samples=1))
        for _ in range(60):
            if r._usd_scene is not None and getattr(
                    r._usd_scene, "instances", None):
                break
            r.update(0.02)
        r.update(0.02)
        r.render_headless()  # forces the megakernel/wavefront pipeline build

        def snap(label):
            blobs[f"{mode}/{label}"] = bytes(r._pack_uniforms())
            blobs[f"{mode}/{label}#msl"] = bytes(_msl(r))

        # 1. defaults
        snap("base")

        # 2. detail flags — master on, all maps present + baked normals
        r.detail_maps_index = 0
        r._detail_available = (True, True, True)
        r._baked_normals = True
        r.normal_map_strength = 0.7
        r.displacement_scale_mm = 1.3
        snap("detail_all")

        # 3. detail master on but every map missing → masked to off
        r._detail_available = (False, False, False)
        r._baked_normals = False
        snap("detail_missing")
        r._detail_available = (True, True, True)  # restore

        # 4. exposure + imaging-ratio fold
        r.exposure = 1.7
        r.film.exposure_time = 2.0
        r.film.iso = 200.0
        snap("exposure_ratio")
        r.exposure = 0.0
        r.film.exposure_time = 1.0
        r.film.iso = 100.0

        # 5. lens framing ratio (stub the lens sync so the fields survive; the
        #    pack's consumption of them is what this gate covers).
        _orig_sync = r._sync_lens_buffer
        r._sync_lens_buffer = lambda: None
        r._lens_active_count = 2
        r._lens_film_distance_world = 0.052
        r.camera.focal_length_mm = 85.0
        r.camera.vertical_aperture_mm = 24.0
        snap("lens_framing")
        r._sync_lens_buffer = _orig_sync
        r._lens_active_count = 0

        # 6. integrator variations
        r.integrator_index = 1  # BDPT
        snap("bdpt")
        r.integrator_index = 0

        # 7. neural-on-megakernel strip / renormalise. Arm the neural proposal;
        #    on the megakernel the bit is stripped and the mixture renormalised.
        try:
            r.proposal_preset_index = r.proposal_preset_from_token("bsdf,neural")
            snap("neural_proposal")
            r.proposal_preset_index = r.proposal_preset_from_token("bsdf")
        except Exception:  # pragma: no cover - preset table variant
            pass

    return blobs


@pytest.mark.parametrize("mode", ["megakernel", "wavefront"])
def test_pack_uniforms_bytes_match_golden(mode):
    blobs = _capture(mode)

    if os.environ.get("SKINNY_CAPTURE_GOLDEN") == "1":
        _GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        existing = dict(np.load(_GOLDEN)) if _GOLDEN.exists() else {}
        existing.update({k: np.frombuffer(v, dtype=np.uint8)
                         for k, v in blobs.items()})
        np.savez(_GOLDEN, **existing)
        pytest.skip(f"captured golden for {mode} ({len(blobs)} blobs)")

    assert _GOLDEN.exists(), \
        "golden missing — regenerate with SKINNY_CAPTURE_GOLDEN=1 (see module docstring)"
    golden = np.load(_GOLDEN)
    mine = {k: v for k, v in blobs.items() if k.startswith(f"{mode}/")}
    assert mine, f"no blobs captured for {mode}"
    for label, packed in mine.items():
        assert label in golden.files, f"golden has no entry for {label}"
        want = golden[label].tobytes()
        assert packed == want, \
            f"{label}: packed {len(packed)}B != golden {len(want)}B — " \
            f"first diff at byte {next((i for i in range(min(len(packed), len(want))) if packed[i] != want[i]), 'len')}"
