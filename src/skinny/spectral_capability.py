"""Single source of truth for whether spectral rendering is wired.

The `--spectral` flag, its startup refusals (``cli_common``), and the parity
matrix's spectral axis (``pbrt.parity``) all key off this one constant. The
CPU/data/shader-source foundation for hero-wavelength spectral rendering is
merged, but the megakernel transport that actually consumes the spectral flag
(change ``spectral-rendering`` Group 5) is not wired yet — so the renderer would
silently produce an ordinary RGB frame for a ``--spectral`` run.

While ``SPECTRAL_IMPLEMENTED`` is ``False`` the flag is refused at startup and
spectral parity combos are recorded "not yet wired" skips. Flip it to ``True``
together with the megakernel spectral transport so both the CLI and the matrix
go live at once.
"""

from __future__ import annotations

SPECTRAL_IMPLEMENTED = False
