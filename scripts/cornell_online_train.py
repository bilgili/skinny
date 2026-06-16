"""Drive a live Cornell render with online neural training to bake a
scene-adapted guiding net (nis-baseline-comparison render go/no-go).

Renders cornell_box_emissive with the bsdf,neural proposal, enables online
training (real spline_flow trainer via torch CPU), and pumps frames so the
renderer drains real path records -> replay -> trainer -> publishes weights to
.skinny_neural/. Prints replay size + net version so we can see whether the GPU
record drain works on this Mac/MoltenVK (the bench notes it can TDR).

Run (Vulkan SDK sourced):
  PYTHONPATH=src:/Users/ahmetbilgili/projects/spline_flow \
    ./bin/python3.13 scripts/cornell_online_train.py --seconds 90
"""
import argparse
import sys
import time
from pathlib import Path

# real spline_flow trainer on the path
sys.path.insert(0, "/Users/ahmetbilgili/projects/spline_flow")

ROOT = Path("/Users/ahmetbilgili/projects/skinny")
SHADER_DIR = ROOT / "src" / "skinny" / "shaders"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=90.0)
    ap.add_argument("--res", type=int, default=96)
    ap.add_argument("--scene", default="cornell_box_emissive.usda",
                    help="USD scene under assets/ (name or path)")
    ap.add_argument("--trainer", default="auto",
                    help="training-compute backend: auto|cpu|cuda|mlx. 'auto' picks "
                         "MLX (Apple GPU) on this Mac; 'cpu' is the slow numpy oracle.")
    args = ap.parse_args()
    SCENE = Path(args.scene) if "/" in args.scene else ROOT / "assets" / args.scene
    assert SCENE.exists(), f"scene not found: {SCENE}"
    print(f"[drive] scene: {SCENE.name}")

    from skinny.renderer import Renderer
    from skinny.vk_context import VulkanContext

    ctx = VulkanContext(window=None, width=args.res, height=args.res)
    r = Renderer(vk_ctx=ctx, shader_dir=SHADER_DIR, hdr_dir=ROOT / "hdrs",
                 tattoo_dir=ROOT / "tattoos", usd_scene_path=SCENE,
                 execution_mode="wavefront", neural_handoff="file")
    r.proposal_preset_index = r.proposal_preset_from_token("bsdf,neural")

    # pump until the USD scene + bindings are ready
    for _ in range(500):
        r.update(0.025)
        if (r._usd_scene is not None and r._scene_bindings is not None):
            break
    assert r._scene_bindings is not None, "scene bindings never built"
    r.update(0.04)
    r.render_headless()
    print(f"[drive] scene ready; enabling online training (torch cpu)")

    print(f"[drive] online training backend: {args.trainer}")
    r.enable_online_training(handoff="file", trainer_backend=args.trainer,
                             weights_dir=str(ROOT / ".skinny_neural"))

    t0 = time.time()
    frames = 0
    total_rec = 0
    last = 0.0
    while time.time() - t0 < args.seconds:
        r.update(0.04)
        # per-frame record drain (render thread) — THIS feeds the replay buffer;
        # the background trainer thread consumes it. Missing this was why replay=0.
        try:
            total_rec += int(r.online_training_tick())
        except Exception as e:
            print(f"[drive] online_training_tick raised: {type(e).__name__}: {e}")
            break
        r.render_headless()
        frames += 1
        now = time.time()
        if now - last > 5.0:
            last = now
            replay = getattr(r, "_neural_replay", None)
            rsize = len(replay) if replay is not None and hasattr(replay, "__len__") else "?"
            ver = getattr(r, "_neural_network_version", "?")
            print(f"[drive] t={now-t0:5.1f}s frames={frames} drained={total_rec} "
                  f"replay={rsize} net_version={ver}")
    ver = getattr(r, "_neural_network_version", "?")
    print(f"[drive] DONE frames={frames} final net_version={ver}")
    # list newest checkpoints
    nets = sorted((ROOT / ".skinny_neural").glob("weights_v*.nfw1"))
    for p in nets[-3:]:
        print(f"[drive] checkpoint: {p.name} ({p.stat().st_size} B)")


if __name__ == "__main__":
    raise SystemExit(main())
