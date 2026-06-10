#!/usr/bin/env python3
"""Render the head on Vulkan and Metal, save tonemapped PNGs for visual diff.
Run under scripts/guarded_metal.sh. Writes /tmp/cmp_vulkan.png + /tmp/cmp_metal.png.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np


def _root():
    import skinny
    r = Path(skinny.__file__).resolve().parent
    return r / "shaders", r.parent.parent, r


def _pump(rdr):
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        rdr.update(0.016)
        if rdr._backend_render_ready and rdr._num_instances >= 1 and rdr._baked_source_idx >= 0:
            return True
        time.sleep(0.02)
    return False


def _render(ctx, shader_dir, head, hdr, samples=64):
    from skinny.renderer import Renderer
    r = Renderer(vk_ctx=ctx, shader_dir=shader_dir, execution_mode="megakernel",
                 hdr_dir=hdr if hdr.is_dir() else None)
    r.load_model_from_path(head)
    assert _pump(r), "not ready"
    r.accum_frame = 0
    for _ in range(samples):
        r.update(0.016)
        r.render_headless()
    raw = r.render_headless()
    w, h = r.width, r.height
    img = np.frombuffer(raw, np.uint8).reshape(h, w, 4)[..., :3]
    return img


def _save(img, path):
    # Minimal PNG writer via numpy + zlib (no PIL dependency).
    import struct, zlib
    h, w, _ = img.shape
    raw = b"".join(b"\x00" + img[y].tobytes() for y in range(h))
    def chunk(typ, data):
        c = typ + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xffffffff)
    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw, 9))
    png += chunk(b"IEND", b"")
    open(path, "wb").write(png)
    print(f"[cmp] wrote {path} ({w}x{h})", flush=True)


def main() -> int:
    shader_dir, proj, _ = _root()
    head = proj / "heads" / "head.obj"
    hdr = proj / "hdrs"
    if not head.exists():
        head = Path.cwd() / "heads" / "head.obj"
        hdr = Path.cwd() / "hdrs"
    res = 128

    from skinny.vk_context import VulkanContext
    vctx = VulkanContext(window=None, width=res, height=res)
    try:
        vimg = _render(vctx, shader_dir, head, hdr)
        _save(vimg, "/tmp/cmp_vulkan.png")
        print(f"[cmp] vulkan mean={vimg.mean():.1f} nonzero={np.count_nonzero(vimg.sum(2))}", flush=True)
    finally:
        vctx.destroy()

    from skinny.metal_context import MetalContext
    mctx = MetalContext(window=None, width=res, height=res)
    try:
        mimg = _render(mctx, shader_dir, head, hdr)
        _save(mimg, "/tmp/cmp_metal.png")
        print(f"[cmp] metal  mean={mimg.mean():.1f} nonzero={np.count_nonzero(mimg.sum(2))}", flush=True)
    finally:
        mctx.destroy()
    return 0


if __name__ == "__main__":
    sys.exit(main())
