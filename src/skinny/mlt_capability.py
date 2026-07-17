"""Single source of truth for whether the MLT integrator is wired.

``--integrator mlt``, its startup refusals (``cli_common``), and the parity
matrix's integrator axis (``pbrt.parity``) all key off this one constant —
the same pattern as :mod:`skinny.spectral_capability`.

The Metropolis Light Transport integrator (change ``mlt-integrator``) is
PSSMLT — Kelemen primary-sample-space Metropolis driving the existing
wavefront BDPT strategies, matching pbrt-v4's ``MLTIntegrator``. The
transport is now wired on the Vulkan backend: the SKINNY_MLT kernels
(``shaders/wavefront/wavefront_mlt.slang`` + the common.slang PSS sampler),
the ``vk_wavefront.WavefrontMltPass`` host pass, the bootstrap resample
(``mlt_bootstrap.resample_chain_seeds``), and the renderer dispatch/uniform
seam. **Native-Metal adapter status: pending** — a Metal MLT session raises
``NotImplementedError`` at dispatch (``--backend vulkan`` runs MLT); the
Metal chain-state pass is the recorded follow-up.

``MLT_IMPLEMENTED = True`` admits the in-envelope combos (``mlt`` ×
``wavefront`` × RGB × flat, layer-free) through ``reject_mlt_unsupported``
and ``parity.combo_is_valid``; out-of-envelope combos are still refused.
"""

from __future__ import annotations

MLT_IMPLEMENTED = True
