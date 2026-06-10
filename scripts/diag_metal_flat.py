#!/usr/bin/env python3
"""Diagnostic: localize the Metal flat-render bug (6.2). Build a Metal megakernel
renderer, load the head, render one frame, and dump the spatial variation of the
offscreen output and the accumulation image. Run ONLY under scripts/guarded_metal.sh.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np


def main() -> int:
    import skinny
    root = Path(skinny.__file__).resolve().parent
    shader_dir = root / "shaders"
    head = root.parent.parent / "heads" / "head.obj"
    hdr = root.parent.parent / "hdrs"
    if not head.exists():
        # worktree layout: <wt>/heads
        head = Path.cwd() / "heads" / "head.obj"
        hdr = Path.cwd() / "hdrs"

    from skinny.metal_context import MetalContext
    from skinny.renderer import Renderer

    ctx = MetalContext(window=None, width=64, height=64)
    r = Renderer(vk_ctx=ctx, shader_dir=shader_dir, execution_mode="megakernel",
                 hdr_dir=hdr if hdr.is_dir() else None)
    r.load_model_from_path(head)
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        r.update(0.016)
        if r._backend_render_ready and r._num_instances >= 1 and r._baked_source_idx >= 0:
            break
        time.sleep(0.02)
    print(f"[diag] ready={r._backend_render_ready} num_instances={r._num_instances} "
          f"baked={r._baked_source_idx} dims={r.width}x{r.height}", flush=True)

    # Dump the MSL-packed uniform values the shader will read.
    import struct as _struct
    blob = r._pack_uniforms_msl()
    lay = r.pipeline.uniform_layout
    print(f"[diag] msl uniform_size={r.pipeline.uniform_size} blob_len={len(blob)}", flush=True)
    for key in ("width", "height", "numInstances", "useMesh", "frameIndex",
                "numLensElements", "furnaceMode"):
        if key in lay:
            off, sz = lay[key]
            v = _struct.unpack_from("<I", blob, off)[0]
            print(f"[diag]   fc.{key:16s}@{off:<4} = {v}", flush=True)
    for key in ("camera.fov", "envIntensity"):
        if key in lay:
            off, sz = lay[key]
            v = _struct.unpack_from("<f", blob, off)[0]
            print(f"[diag]   fc.{key:16s}@{off:<4} = {v:.5f}", flush=True)
    # camera.viewInverse first row (should be a real matrix, not zeros)
    if "camera.viewInverse" in lay:
        off, _ = lay["camera.viewInverse"]
        row0 = _struct.unpack_from("<4f", blob, off)
        print(f"[diag]   fc.camera.viewInverse[0]@{off} = {[round(x,4) for x in row0]}", flush=True)

    r.accum_frame = 0
    r.update(0.016)
    r.render_headless()

    off = np.asarray(r._offscreen_output.read_rgba(), dtype=np.float32)
    acc, samples = r.read_accumulation_hdr()
    acc = acc[..., :3]

    def stats(name, a):
        a = np.asarray(a, dtype=np.float64)
        rgb = a[..., :3]
        print(f"[diag] {name}: shape={a.shape} min={rgb.min():.5f} max={rgb.max():.5f} "
              f"mean={rgb.mean():.5f} std={rgb.std():.6f} "
              f"contrast={rgb.max()/(rgb.mean()+1e-9):.3f}", flush=True)
        H, W = a.shape[:2]
        for (y, x, tag) in [(0, 0, "TL"), (H//2, W//2, "C"), (H-1, W-1, "BR"),
                            (H//4, W//2, "top"), (3*H//4, W//2, "bot")]:
            print(f"[diag]    {tag}({y},{x}) = {np.round(a[y, x, :3], 5)}", flush=True)

    # What does toolBuffer[0].x decode to? Non-zero hijacks mainImage into the
    # BXDF/BSSRDF tool branch. Read it back via the buffer's native to_numpy.
    try:
        tb = np.asarray(r.tool_buffer.buffer.to_numpy()).view(np.uint32)
        print(f"[diag] toolBuffer[0..3] uint = {tb[:4].tolist()}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[diag] toolBuffer read failed: {exc}", flush=True)

    # to_numpy fidelity: write a known per-pixel gradient into a fresh texture and
    # read it back. If it comes back uniform, the readback (not the render) is the
    # bug — the windowed blit would still show the correct image.
    try:
        import slangpy as spy
        grad = np.zeros((64, 64, 4), np.float32)
        yy, xx = np.mgrid[0:64, 0:64]
        grad[..., 0] = xx / 63.0
        grad[..., 1] = yy / 63.0
        tex = ctx.device.create_texture(
            type=spy.TextureType.texture_2d, format=spy.Format.rgba32_float,
            width=64, height=64,
            usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access
            | spy.TextureUsage.copy_source | spy.TextureUsage.copy_destination,
            memory_type=spy.MemoryType.device_local, label="diag.grad")
        tex.copy_from_numpy(grad)
        ctx.device.wait_for_idle()
        back = np.asarray(tex.to_numpy(), np.float32).reshape(64, 64, 4)
        print(f"[diag] gradient roundtrip: TL={np.round(back[0,0,:2],3).tolist()} "
              f"BR={np.round(back[63,63,:2],3).tolist()} "
              f"distinct={len(np.unique(back[...,0]))} (expect ~64)", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[diag] gradient roundtrip failed: {exc}", flush=True)

    stats("offscreen", off)
    stats("accum(samples=%d)" % samples, acc)

    # Move the camera and re-render: if the output is byte-identical, the camera
    # rays are degenerate (don't depend on view); if it changes, rays work and the
    # flat look is a constant-env / all-miss issue.
    try:
        r.camera.azimuth += 1.2
        r.camera.elevation += 0.3
    except Exception:  # noqa: BLE001
        try:
            r.camera.yaw += 1.2
        except Exception:  # noqa: BLE001
            pass
    r.accum_frame = 0
    r.update(0.016)
    r.render_headless()
    off2 = np.asarray(r._offscreen_output.read_rgba(), dtype=np.float32)[..., :3]
    same = bool(np.allclose(off[..., :3], off2, atol=1e-6))
    print(f"[diag] after camera move: identical={same} "
          f"newmean={off2.mean():.5f} newval_TL={np.round(off2[0,0],5)}", flush=True)

    # Count distinct rows/cols to detect a broadcast or single-thread pattern.
    off3 = off[..., :3]
    distinct_pixels = len(np.unique(off3.reshape(-1, 3), axis=0))
    print(f"[diag] offscreen distinct pixel values: {distinct_pixels} "
          f"of {off3.shape[0]*off3.shape[1]}", flush=True)

    r.cleanup()
    ctx.destroy()
    return 0


if __name__ == "__main__":
    sys.exit(main())
