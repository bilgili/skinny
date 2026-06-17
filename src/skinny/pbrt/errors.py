"""Error types for the pbrt importer."""

from __future__ import annotations


class PbrtError(Exception):
    """Base class for all pbrt-import errors."""


class PbrtParseError(PbrtError):
    """Raised on malformed pbrt input.

    Carries an optional source location so the caller can see *where* the
    failure happened rather than getting a partial scene.
    """

    def __init__(self, message: str, *, file: str | None = None, line: int | None = None):
        self.file = file
        self.line = line
        loc = ""
        if file is not None:
            loc = file
            if line is not None:
                loc += f":{line}"
            loc += ": "
        elif line is not None:
            loc = f"line {line}: "
        super().__init__(loc + message)
