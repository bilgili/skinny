"""Bindless flat-material texture pool — device-free module.

Split out of ``renderer.py`` (change renderer-pure-core-extraction). The pool
*holds* GPU objects but never imports a GPU package: its constructor takes the
backend's resource module (``vk_compute`` / ``metal_compute``) and calls
``SampledImage`` through it, so a hostless test can pass a fake (design D4).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image

if TYPE_CHECKING:
    from skinny.vk_compute import SampledImage

class TexturePool:
    """Bindless flat-material texture pool (binding 14 in main_pass.slang).

    Owns up to BINDLESS_TEXTURE_CAPACITY SampledImage slots. Materials
    point at slots by index; unused slots stay None and are gated off by
    PARTIALLY_BOUND on the descriptor binding plus a sentinel index in
    the material record.

    Deduplication is by file path: two materials referencing the same
    PNG share one slot. Allocation is monotonic; we don't free slots
    mid-session because materials don't change after scene load.
    """

    SENTINEL = 0xFFFFFFFF

    def __init__(self, ctx, gpu) -> None:
        self.ctx = ctx
        # GPU-resource module (vk_compute / metal_compute) — the pool's bindless
        # capacity follows the active backend's cap (Metal trims to fit its
        # 128-texture / 16-sampler argument limit, design D8).
        self._gpu = gpu
        self._capacity = int(gpu.BINDLESS_TEXTURE_CAPACITY)
        self._slots = [None] * self._capacity
        self._by_path: dict[str, int] = {}
        self._next_slot = 0

    # Backend-neutral wrap tokens (resolved per backend inside SampledImage).
    _WRAP_TOKENS = {
        "repeat": "repeat", "clamp": "clamp", "mirror": "mirror",
        "black": "black", "useMetadata": "repeat",
    }

    def add_or_get(
        self,
        path: Path,
        *,
        linear: bool = False,
        wrap_s: str = "repeat",
        wrap_t: str = "repeat",
    ) -> int:
        """Decode the file at `path` and return the array slot it lives in.

        Subsequent calls with the same (path, linear, wrap_s, wrap_t) tuple
        return the cached slot. Returns SENTINEL when the file can't be
        loaded (missing/corrupt).

        `linear=True` uploads as VK_FORMAT_R8G8B8A8_UNORM (no gamma decode) —
        use for normal, roughness, metallic, and other non-colour data textures.
        `wrap_s` / `wrap_t` come from USD's per-texture
        `inputs:wrapS` / `inputs:wrapT`. Two materials referencing the same
        file with different wrap modes get distinct slots (each owns its
        own sampler).
        """
        key = str(path.resolve()) if path.is_absolute() else str(path)
        if linear:
            key += ":linear"
        key += f":{wrap_s}/{wrap_t}"
        cached = self._by_path.get(key)
        if cached is not None:
            return cached
        try:
            img = Image.open(path).convert("RGBA")
        except (FileNotFoundError, OSError):
            return self.SENTINEL
        if self._next_slot >= self._capacity:
            return self.SENTINEL
        w, h = img.size
        fmt = "rgba8_unorm" if linear else "rgba8_srgb"
        addr_u = self._WRAP_TOKENS.get(wrap_s, "repeat")
        addr_v = self._WRAP_TOKENS.get(wrap_t, "repeat")
        slot = self._gpu.SampledImage(
            self.ctx, w, h,
            format=fmt,
            bytes_per_pixel=4,
            address_mode_u=addr_u,
            address_mode_v=addr_v,
        )
        slot.upload_sync(img.tobytes())
        idx = self._next_slot
        self._slots[idx] = slot
        self._by_path[key] = idx
        self._next_slot += 1
        return idx

    def filled_slots(self) -> list[tuple[int, SampledImage]]:
        """(slot_index, SampledImage) pairs for every populated slot."""
        return [(i, s) for i, s in enumerate(self._slots) if s is not None]

    def destroy(self) -> None:
        for slot in self._slots:
            if slot is not None:
                slot.destroy()
        self._slots = []
