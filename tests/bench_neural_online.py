"""Benchmark harness for online neural training (change neural-online-training, task 7.3).

Measures the costs that make up the online loop, across BOTH weight-handoff
backends, and (with torch present) the real CUDA trainer cycle interleaved with
the Vulkan render in one process:

  * file-handoff swap    - publish (NFW1 write + reload) -> frame-end swap ->
    re-upload bindings 33/34/35 (`bench_handoff_swap("file")`, any GPU box).
  * interop-handoff swap  - publish (CUDA cudaMemcpy into the exported buffers +
    timeline signal, NO CPU round-trip) -> frame-end swap (timeline host-wait, no
    re-upload) (`bench_handoff_swap("interop")`, CUDA + external-memory Vulkan;
    skips cleanly otherwise).
  * CUDA trainer cycle    - one warm-started contribution-weighted MLE cycle on
    the shipped flow (`bench_trainer_cycle`), real when torch + spline_flow are
    importable in this venv (CUDA + autocast-fp16 on the NVIDIA box), else a
    placeholder. When torch is active, `bench_concurrent_train_render` runs the
    real trainer each frame - CUDA training concurrent with the Vulkan render in
    one process, the task-7.3 integration.
  * moving-object render   - `bench_moving_object_render` translates one instance
    each frame (real TLAS re-upload) with the online loop live; confirms the
    moving-object scene renders stably under geometry motion + concurrent training.
  * frames-to-recover      - `bench_frames_to_recover` converges the real flow on
    one radiance lobe, "moves the object" (mirror lobe at the same condition fed
    into the recency-weighted replay), and counts cycles until the NLL recovers -
    the online-adaptation metric. Records are synthesized-but-pose-grounded since
    the live GPU drain device-losts under the 2 s TDR here; the trainer + replay
    under test are real.

Run (skinny renderer venv; the real trainer needs torch - the CUDA build, e.g.
``pip install torch --index-url https://download.pytorch.org/whl/cu126`` - plus
``matplotlib`` (a top-level import in ``spline_flow/train.py``) and the
``spline_flow`` sibling repo on the path; without them the trainer falls back to
the placeholder and the swap/handoff numbers still run)::

  PYTHONUTF8=1 PYTHONPATH=src <py> tests/bench_neural_online.py --frames 64

Measured on an RTX 4090: file publish ~29 ms vs interop publish ~0.5 ms (~54x,
the NFW1 CPU round-trip interop removes); a real trainer cycle ~2.8 s (64 Adam
steps, autocast-fp16) runs concurrent with the Vulkan render in one process while
the weighted-NLL keeps dropping.
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


def _build_neural_renderer(width: int, height: int, handoff: str = "file"):
    from skinny.renderer import Renderer
    from skinny.vk_context import VulkanContext

    ctx = VulkanContext(window=None, width=width, height=height)
    r = Renderer(vk_ctx=ctx, shader_dir=SHADER_DIR, hdr_dir=HDR_DIR,
                 tattoo_dir=TATTOO_DIR, usd_scene_path=SCENE,
                 execution_mode="wavefront", neural_handoff=handoff)
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


def _feed_records(r, n: int = 4096) -> None:
    """Stage a recency-weighted batch of valid (upper-hemisphere, positive-weight)
    records so the real trainer's `build_dataset` keeps them (wi y-up, contrib>0)."""
    from skinny.sampling.path_records import RECORD_DTYPE

    rng = np.random.default_rng(0)
    recs = np.zeros(n, dtype=RECORD_DTYPE)
    v = rng.normal(size=(n, 3)).astype(np.float32)
    v[:, 1] = np.abs(v[:, 1]) + 0.1                       # force y-up hemisphere
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    recs["wi_local"] = v
    recs["pos"] = rng.random((n, 3)).astype(np.float32)   # spread the condition
    recs["contrib"] = 1.0
    r._neural_replay.add(recs)


def _interop_skip_reason(ctx) -> str | None:
    from skinny.sampling.neural_handoff_interop import interop_available

    ok, reason = interop_available()
    if not ok:
        return reason
    if not (getattr(ctx, "supports_external_memory", False)
            and getattr(ctx, "supports_external_semaphore", False)):
        return "device lacks external memory + timeline semaphore"
    return None


def bench_handoff_swap(handoff: str, width: int = 96, height: int = 96,
                       frames: int = 64,
                       weights_dir: str = "_bench_neural_handoff") -> dict:
    """Isolate the *handoff* cost for one backend: each frame ``publish``es the
    current weights (NO training, so the timing is the handoff alone - file's NFW1
    write+reload vs interop's CUDA memcpy + timeline signal), renders, and lets the
    frame-end swap commit it. The clean file-vs-interop comparison. `interop` skips
    cleanly where CUDA / external memory is absent."""
    ctx, r = _build_neural_renderer(width, height, handoff=handoff)
    try:
        if handoff == "interop":
            reason = _interop_skip_reason(ctx)
            if reason:
                return {"skipped": reason}

        kw = {"weights_dir": weights_dir} if handoff == "file" else {}
        r.enable_online_training(handoff=handoff, **kw)
        weights = r._neural_trainer.weights          # publish these as-is, no train

        # Warm up: first publish does the lazy CUDA import / first NFW1 write.
        r._neural_publisher.publish(weights)
        r.update(0.04)
        r.render_headless()

        publish_ms, frame_ms = [], []
        for _ in range(frames):
            t0 = time.perf_counter()
            r._neural_publisher.publish(weights)       # handoff cost only
            t1 = time.perf_counter()
            r.update(0.04)
            r.render_headless()                        # frame-end swap commits it
            t2 = time.perf_counter()
            publish_ms.append((t1 - t0) * 1e3)
            frame_ms.append((t2 - t1) * 1e3)
        ver = r._neural_network_version
    finally:
        r.cleanup()
        ctx.destroy()

    return {
        "handoff": handoff, "frames": frames, "final_version": ver,
        "publish_ms_med": float(np.median(publish_ms)),
        "publish_ms_p90": float(np.percentile(publish_ms, 90)),
        "render+swap_ms_med": float(np.median(frame_ms)),
        "render+swap_ms_p90": float(np.percentile(frame_ms, 90)),
    }


def bench_concurrent_train_render(handoff: str = "interop", width: int = 96,
                                  height: int = 96, frames: int = 12) -> dict:
    """CUDA training concurrent with the Vulkan render in ONE process (task 7.3):
    each frame runs a real warm-started trainer cycle (CUDA), publishes the result
    through the handoff, and renders - interleaved on the same GPU. Reports the
    per-frame train vs publish vs render split and the loss trajectory. Needs torch
    + spline_flow (the real trainer); `interop` needs CUDA + external-memory Vulkan."""
    ctx, r = _build_neural_renderer(width, height, handoff=handoff)
    try:
        if handoff == "interop":
            reason = _interop_skip_reason(ctx)
            if reason:
                return {"skipped": reason}
        r.enable_online_training(handoff=handoff)
        if not r._neural_trainer.torch_active:
            return {"skipped": "torch/spline_flow unavailable - real trainer inactive"}
        _feed_records(r)
        rng = np.random.default_rng(0)

        r.online_train_and_publish(rng)               # warm-up (build + warm-start)
        r.update(0.04)
        r.render_headless()

        train_ms, pub_ms, frame_ms, losses = [], [], [], []
        for _ in range(frames):
            t0 = time.perf_counter()
            new_w = r._neural_trainer.train_cycle(r._neural_replay, rng)
            t1 = time.perf_counter()
            r._neural_publisher.publish(new_w)
            t2 = time.perf_counter()
            r.update(0.04)
            r.render_headless()
            t3 = time.perf_counter()
            train_ms.append((t1 - t0) * 1e3)
            pub_ms.append((t2 - t1) * 1e3)
            frame_ms.append((t3 - t2) * 1e3)
            losses.append(r._neural_trainer.last_loss)
        ver = r._neural_network_version
    finally:
        r.cleanup()
        ctx.destroy()

    return {
        "handoff": handoff, "frames": frames, "final_version": ver,
        "train_ms_med": float(np.median(train_ms)),
        "publish_ms_med": float(np.median(pub_ms)),
        "render+swap_ms_med": float(np.median(frame_ms)),
        "loss_first": losses[0], "loss_last": losses[-1],
    }


def _peaked_records(direction, n: int, pos, spread: float = 0.18,
                    rng=None) -> np.ndarray:
    """Records whose `wi_local` cluster around `direction` (upper hemisphere) at a
    fixed condition `pos` - a stand-in for the radiance lobe a light/object at one
    pose produces. Moving the object = re-pointing `direction` at the same `pos`,
    which is the hardest recovery (overwrite the same conditional)."""
    from skinny.sampling.path_records import RECORD_DTYPE

    rng = rng or np.random.default_rng(0)
    d = np.asarray(direction, np.float32)
    d = d / np.linalg.norm(d)
    recs = np.zeros(n, dtype=RECORD_DTYPE)
    w = d[None, :] + rng.normal(scale=spread, size=(n, 3)).astype(np.float32)
    w[:, 1] = np.abs(w[:, 1]) + 0.05                  # keep y-up (build_dataset filter)
    w /= np.linalg.norm(w, axis=1, keepdims=True)
    recs["wi_local"] = w
    recs["pos"] = np.asarray(pos, np.float32)
    recs["contrib"] = 1.0
    return recs


def bench_frames_to_recover(steps_per_cycle: int = 32, batch: int = 2048,
                            converge_cycles: int = 12, max_cycles: int = 30,
                            recover_eps: float = 0.03) -> dict:
    """Frames-to-recover after a moving object (task 7.3). Train the real flow to
    convergence on radiance lobe A, then 'move the object' (lobe B at the SAME
    condition, fed each cycle into the recency-weighted replay), and count cycles
    until the training NLL returns to the A-converged level - the recency-weighting
    adaptation the online loop relies on. Needs torch + spline_flow (else skip)."""
    from skinny.sampling.neural_replay import ReplayBuffer
    from skinny.sampling.neural_trainer import NeuralTrainer, TrainerConfig

    trainer = NeuralTrainer(TrainerConfig(batch=batch, steps_per_cycle=steps_per_cycle))
    if not trainer.torch_active:
        return {"skipped": "torch/spline_flow unavailable - real trainer inactive"}

    replay = ReplayBuffer(capacity=400_000)
    rng = np.random.default_rng(0)
    pos = np.array([0.5, 0.5, 0.5], np.float32)
    dir_a = np.array([0.6, 0.7, 0.0], np.float32)
    dir_b = np.array([-0.6, 0.7, 0.0], np.float32)     # mirror lobe -> same converged loss

    # Converge on A (drain a batch of A each cycle, like the renderer per frame).
    for _ in range(converge_cycles):
        replay.add(_peaked_records(dir_a, batch, pos, rng=rng))
        trainer.train_cycle(replay, rng)
    loss_a = float(trainer.last_loss)

    # Move the object: lobe B now, same condition. Recency must demote A.
    traj, frames_to_recover = [], None
    for k in range(1, max_cycles + 1):
        replay.add(_peaked_records(dir_b, batch, pos, rng=rng))
        trainer.train_cycle(replay, rng)
        loss = float(trainer.last_loss)
        traj.append(round(loss, 4))
        if frames_to_recover is None and loss <= loss_a + recover_eps:
            frames_to_recover = k

    return {
        "steps_per_cycle": steps_per_cycle, "batch": batch,
        "loss_converged_A": round(loss_a, 4),
        "loss_spike_after_move": max(traj) if traj else None,
        "frames_to_recover": frames_to_recover if frames_to_recover is not None
        else f">{max_cycles}",
        "recover_eps": recover_eps,
        "loss_trajectory": traj,
    }


def bench_moving_object_render(handoff: str = "interop", frames: int = 12,
                               width: int = 96, height: int = 96) -> dict:
    """A literally-moving object rendered headless with the online loop live
    (task 7.3): translate one instance each frame (TLAS re-upload), run a trainer
    cycle + handoff publish, render. Confirms the moving-object scene runs and the
    per-frame cost stays stable with geometry motion + concurrent training."""
    ctx, r = _build_neural_renderer(width, height, handoff=handoff)
    try:
        if handoff == "interop":
            reason = _interop_skip_reason(ctx)
            if reason:
                return {"skipped": reason}
        r.enable_online_training(handoff=handoff)
        torch_active = bool(r._neural_trainer.torch_active)
        _feed_records(r)

        insts = r._usd_scene.instances
        idx = len(insts) - 1                           # last loaded (the sphere here)
        base = np.array(insts[idx].transform, np.float32).copy()

        def _center():
            lo, hi = insts[idx].world_bounds()
            return (np.asarray(lo) + np.asarray(hi)) * 0.5

        c_base = _center()
        # Detect which slot holds translation in this transform convention.
        T = base.copy()
        T[3, 0] += 1.0
        insts[idx].transform = T
        r._reupload_instance_transforms()
        slot_row = float(np.linalg.norm(_center() - c_base)) > 1e-4
        insts[idx].transform = base.copy()
        r._reupload_instance_transforms()

        rng = np.random.default_rng(0)
        train_ms, pub_ms, frame_ms = [], [], []
        max_disp = 0.0
        for f in range(frames):
            off = 0.4 * float(np.sin(f * 0.5))
            T = base.copy()
            if slot_row:
                T[3, 0] += off
            else:
                T[0, 3] += off
            insts[idx].transform = T
            r._reupload_instance_transforms()
            max_disp = max(max_disp, float(np.linalg.norm(_center() - c_base)))

            t0 = time.perf_counter()
            new_w = r._neural_trainer.train_cycle(r._neural_replay, rng)
            t1 = time.perf_counter()
            r._neural_publisher.publish(new_w)
            t2 = time.perf_counter()
            r.update(0.04)
            r.render_headless()
            t3 = time.perf_counter()
            train_ms.append((t1 - t0) * 1e3)
            pub_ms.append((t2 - t1) * 1e3)
            frame_ms.append((t3 - t2) * 1e3)
        ver = r._neural_network_version
    finally:
        r.cleanup()
        ctx.destroy()

    return {
        "handoff": handoff, "frames": frames, "moved_instance": idx,
        "object_world_displacement": round(max_disp, 4),
        "torch_active": torch_active, "final_version": ver,
        "train_ms_med": float(np.median(train_ms)),
        "publish_ms_med": float(np.median(pub_ms)),
        "render+swap_ms_med": float(np.median(frame_ms)),
        "render+swap_ms_p90": float(np.percentile(frame_ms, 90)),
    }


def bench_trainer_cycle(cycles: int = 8, batch: int = 4096,
                        steps_per_cycle: int = 64) -> dict:
    """Time one warm-started trainer cycle in isolation (real CUDA loop when torch
    + spline_flow are importable, else the placeholder). Reports the loss
    trajectory so a torch run shows the net actually learning - the concurrent
    train-while-render proxy for frames-to-recover."""
    from skinny.sampling.neural_replay import ReplayBuffer
    from skinny.sampling.neural_trainer import NeuralTrainer, TrainerConfig

    replay = ReplayBuffer(capacity=200_000)
    trainer = NeuralTrainer(TrainerConfig(batch=batch, steps_per_cycle=steps_per_cycle))

    # Reuse the same valid-record generator the swap bench uses.
    class _Shim:
        _neural_replay = replay
    _feed_records(_Shim(), n=batch * 2)

    torch_active = bool(trainer.torch_active)
    rng = np.random.default_rng(0)
    trainer.train_cycle(replay, rng)                  # warm-up (build + warm-start)

    cycle_ms, losses = [], []
    for _ in range(cycles):
        t0 = time.perf_counter()
        trainer.train_cycle(replay, rng)
        cycle_ms.append((time.perf_counter() - t0) * 1e3)
        losses.append(trainer.last_loss)
    return {
        "torch_active": torch_active, "cycles": cycles,
        "steps_per_cycle": steps_per_cycle, "batch": batch,
        "cycle_ms_med": float(np.median(cycle_ms)),
        "loss_first": losses[0], "loss_last": losses[-1],
    }


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames", type=int, default=64)
    ap.add_argument("--concurrent-frames", type=int, default=12)
    ap.add_argument("--moving-frames", type=int, default=12)
    ap.add_argument("--max-recover-cycles", type=int, default=30)
    ap.add_argument("--width", type=int, default=96)
    ap.add_argument("--height", type=int, default=96)
    ap.add_argument("--cycles", type=int, default=8)
    args = ap.parse_args()

    def _show(title, res):
        print(f"[7.3] {title}:")
        for k, v in res.items():
            print(f"        {k:22s} {v}")

    file_res = bench_handoff_swap("file", args.width, args.height, args.frames)
    _show("file-handoff cost (publish only)", file_res)

    interop_res = bench_handoff_swap("interop", args.width, args.height, args.frames)
    _show("interop-handoff cost (publish only)", interop_res)

    if "publish_ms_med" in file_res and "publish_ms_med" in interop_res:
        f, i = file_res["publish_ms_med"], interop_res["publish_ms_med"]
        speedup = f / i if i > 0 else float("inf")
        print(f"[7.3] publish latency  file={f:.2f}ms  interop={i:.3f}ms  "
              f"=> interop {speedup:.0f}x faster (no NFW1 CPU round-trip)")
    elif "skipped" in interop_res:
        print(f"[7.3] interop skipped: {interop_res['skipped']}")

    conc_res = bench_concurrent_train_render(
        "interop", args.width, args.height, args.concurrent_frames)
    _show("concurrent CUDA-train + Vulkan-render (interop, one process)", conc_res)

    move_res = bench_moving_object_render(
        "interop", args.moving_frames, args.width, args.height)
    _show("moving-object render (interop, geometry animated per frame)", move_res)

    recover_res = bench_frames_to_recover(max_cycles=args.max_recover_cycles)
    _show("frames-to-recover after a moving object (recency replay)", recover_res)

    trainer_res = bench_trainer_cycle(cycles=args.cycles)
    _show("CUDA trainer cycle (isolated)", trainer_res)


if __name__ == "__main__":
    main()
