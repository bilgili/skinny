"""Single source of truth for whether spectral rendering is wired.

The `--spectral` flag, its startup refusals (``cli_common``), and the parity
matrix's spectral axis (``pbrt.parity``) all key off this one constant.

The hero-wavelength spectral megakernel transport (change ``spectral-rendering``
Groups 4–6) is **wired and GPU-validated**: the path+megakernel+flat v1 envelope
renders under ``--spectral`` on both backends — spectral NEE, the sigmoid/D65
upsampling model, exact named-conductor Fresnel (6.2), authored/blackbody
illuminant SPDs (6.1/6.3), and hero-λ glass dispersion (6.4). So
``SPECTRAL_IMPLEMENTED`` is ``True``: the flag is accepted for an in-envelope run
(the other startup refusals — non-path integrator, wavefront, ReSTIR reuse,
neural proposal, skin/subsurface/volume scene — still apply), and the parity
matrix admits ``(path, megakernel, spectral)`` into the rendered set.
"""

from __future__ import annotations

SPECTRAL_IMPLEMENTED = True
