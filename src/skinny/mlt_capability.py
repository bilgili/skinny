"""Single source of truth for whether the MLT integrator is wired.

This constant has exactly one consumer: :mod:`skinny.render_envelope`, the shared
render-envelope predicate, which reads it **live** at evaluation time (never
captured at import, so a test monkeypatch takes effect). ``--integrator mlt``'s
startup refusals (``cli_common``) and the parity matrix's integrator axis
(``pbrt.parity``) key off it through that predicate — the same pattern as
:mod:`skinny.spectral_capability`.

The Metropolis Light Transport integrator (change ``mlt-integrator``) is
PSSMLT — Kelemen primary-sample-space Metropolis driving the existing
wavefront BDPT strategies, matching pbrt-v4's ``MLTIntegrator``. The
transport is wired on **both** backends: the SKINNY_MLT kernels
(``shaders/wavefront/wavefront_mlt.slang`` + the common.slang PSS sampler),
the ``vk_wavefront.WavefrontMltPass`` / ``metal_wavefront.MetalWavefrontMltPass``
host passes, the bootstrap resample (``mlt_bootstrap.resample_chain_seeds``),
and the renderer dispatch/uniform seam.

``MLT_IMPLEMENTED = True`` admits the in-envelope combos (``mlt`` ×
``wavefront`` × RGB or spectral × flat, layer-free); out-of-envelope combos are
still refused.
"""

from __future__ import annotations

MLT_IMPLEMENTED = True
