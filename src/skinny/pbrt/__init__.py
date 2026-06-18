"""pbrt v4 scene importer for skinny.

Reads a pbrt v4 text scene and emits a USD stage loadable by ``skinny``'s
existing ``usd_loader``. Direction is import-only (pbrt -> skinny); the
user-facing "exporter" is this converter.

The public entry point is :func:`import_pbrt`. It is imported lazily so that
lightweight submodules (``tokenizer``, ``parser``, ``spectra``, ``metrics``)
can be used without pulling in USD (``pxr``).
"""

from __future__ import annotations

__all__ = ["import_pbrt", "PbrtError", "PbrtParseError"]

from .errors import PbrtError, PbrtParseError


def __getattr__(name: str):  # PEP 562 lazy attribute access
    if name == "import_pbrt":
        from .api import import_pbrt

        return import_pbrt
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
