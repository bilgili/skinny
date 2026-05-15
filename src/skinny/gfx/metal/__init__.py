"""Metal backend stubs.

Step 7 fills these out with PyObjC + the Metal framework. Until then, any
attempt to construct a MetalBackend raises NotImplementedError so the
backend selection wiring (``SKINNY_BACKEND=metal``) is testable end-to-end.
"""

from __future__ import annotations

from skinny.gfx.backend import Backend


class MetalBackend(Backend):
    name = "metal"

    @classmethod
    def create(cls, **kwargs) -> "MetalBackend":
        raise NotImplementedError(
            "Metal backend not implemented. Use SKINNY_BACKEND=vulkan "
            "(MoltenVK provides this on macOS) until the native Metal "
            "backend lands."
        )

    @property
    def caps(self):  # pragma: no cover - stub
        raise NotImplementedError

    @property
    def device(self):  # pragma: no cover - stub
        raise NotImplementedError

    @property
    def presenter(self):  # pragma: no cover - stub
        raise NotImplementedError

    def shader_target(self):  # pragma: no cover - stub
        return "metal"

    def destroy(self) -> None:  # pragma: no cover - stub
        return None


__all__ = ["MetalBackend"]
