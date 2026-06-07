"""Benchmark harness for online neural training (change neural-online-training, task 7.3).

Measures the costs that make up the online loop and frames what is validated vs
what remains an NVIDIA-box task:

  * VALIDATED HERE — file-handoff swap latency: the per-frame cost of
    publish → frame-end swap → re-upload weights to bindings 33/34/35
    (`bench_file_handoff_swap`, runnable on any GPU box, no torch needed).
  * VALIDATED ELSEWHERE — CUDA trainer cycle cost: one warm-started
    contribution-weighted MLE cycle on the 4090 (see
    `tests/test_neural_trainer_torch.py`, task 2.2). Time it with
    `NeuralTrainer.train_cycle` under the torch venv.
  * PENDING (NVIDIA box, needs interop CUDA-write + torch in the *renderer* venv):
    the file-vs-interop swap comparison, CUDA training concurrent with the Vulkan
    render in one process, and frames-to-recover on a moving object. The interop
    CUDA-import seam is `neural_handoff_interop._import_external_memory` (task 5.2);
    the Vulkan export side is wired (task 5.1, `StorageBuffer.export_handle`).

Run the file-handoff swap benchmark (skinny renderer venv)::

  PYTHONUTF8=1 PYTHONPATH=src <py> tests/bench_neural_online.py --frames 64
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SHADER_DIR = PROJECT_ROOT / "src" / "skinny" / "shaders"
HDR_DIR = PROJECT_ROOT / "hdrs"
TATTOO_DIR = PROJECT_ROOT / "tattoos"
SCENE = PROJECT_ROOT / "assets" / "cornell_box_emissive.usda"


def _build_neural_renderer(width: int, height: int):
    from skinny.renderer import Renderer
    from skinny.vk_context import VulkanContext

    ctx = VulkanContext(window=None, width=width, height=height)
    r = Renderer(vk_ctx=ctx, shader_dir=SHADER_DIR, hdr_dir=HDR_DIR,
                 tattoo_dir=TATTOO_DIR, usd_scene_path=SCENE, execution_mode="wavefront")
    r.proposal_preset_index = r.proposal_preset_from_token("bsdf,neural")
    for _ in range(300):
        r.update(0.025)
        if (r._usd_scene is not None and len(r._usd_scene.instances) >= 1
                and r._scene_bindings is not None):
            break
    assert r._scene_bindings is not None, "scene bindings never built"
    r.update(0.04)
    r.render_headless()
    return ctx, r


def bench_file_handoff_swap(width: int = 96, height: int = 96, frames: int = 64,
                            weights_dir: str = "_bench_neural_handoff") -> dict:
    """Time the file-handoff online swap: each frame stage a (placeholder) update,
    render, and let the frame-end swap promote + re-upload bindings 33/34/35.
    Returns per-frame timings (ms). Runs on any GPU box (no torch)."""
    from skinny.sampling.path_records import RECORD_DTYPE

    ctx, r = _build_neural_renderer(width, height)
    try:
        r.enable_online_training(handoff="file", weights_dir=weights_dir)
        recs = np.zeros(1024, dtype=RECORD_DTYPE)
        recs["wi_local"] = [0.0, 1.0, 0.0]
        recs["contrib"] = 1.0
        r._neural_replay.add(recs)

        swap_ms, frame_ms = [], []
        for _ in range(frames):
            t0 = time.perf_counter()
            r.online_train_and_publish()              # stage a new version
            t1 = time.perf_counter()
            r.update(0.04)
            r.render_headless()                        # frame-end swap re-uploads weights
            t2 = time.perf_counter()
            swap_ms.append((t1 - t0) * 1e3)
            frame_ms.append((t2 - t1) * 1e3)
        ver = r._neural_network_version
    finally:
        r.cleanup()
        ctx.destroy()

    return {
        "frames": frames, "final_version": ver,
        "publish_ms_med": float(np.median(swap_ms)),
        "render+swap_ms_med": float(np.median(frame_ms)),
        "render+swap_ms_p90": float(np.percentile(frame_ms, 90)),
    }


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames", type=int, default=64)
    ap.add_argument("--width", type=int, default=96)
    ap.add_argument("--height", type=int, default=96)
    args = ap.parse_args()

    res = bench_file_handoff_swap(args.width, args.height, args.frames)
    print("[7.3] file-handoff online swap (VALIDATED on this GPU):")
    for k, v in res.items():
        print(f"        {k:22s} {v}")
    print("[7.3] PENDING (NVIDIA box): file-vs-interop comparison, CUDA train "
          "concurrent with Vulkan render, frames-to-recover — need the interop "
          "CUDA-write seam (5.2) + torch in the renderer venv.")


if __name__ == "__main__":
    main()
