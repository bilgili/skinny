"""Single source of truth for whether spectral rendering is wired.

The `--spectral` flag, its startup refusals (``cli_common``), and the parity
matrix's spectral axis (``pbrt.parity``) all key off this one constant.

The hero-wavelength spectral megakernel transport (change ``spectral-rendering``
Groups 4–6) is **wired and GPU-validated**: the path+megakernel+flat v1 envelope
renders under ``--spectral`` on both backends — spectral NEE, the sigmoid/D65
upsampling model, exact named-conductor Fresnel (6.2), authored/blackbody
illuminant SPDs (6.1/6.3), and hero-λ glass dispersion (6.4). So
``SPECTRAL_IMPLEMENTED`` is ``True``: the flag is accepted for an in-envelope run.

The envelope now also spans the **wavefront** execution mode and the **SPPM**
integrator (change ``spectral-wavefront``): ``--spectral`` accepts ``path``,
``bdpt``, and ``sppm`` under either ``megakernel`` or ``wavefront`` (SPPM under
wavefront only, as in RGB — its photon pass has no megakernel path). The startup
refusals that remain are ReSTIR reuse, the neural directional proposal, a
non-BSDF proposal, and skin/subsurface/heterogeneous-volume scenes. The parity
matrix admits ``(path|bdpt|sppm, megakernel|wavefront, spectral)`` into the
rendered set within that envelope.
"""

from __future__ import annotations

SPECTRAL_IMPLEMENTED = True
