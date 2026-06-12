"""Equal-time bench: Metal wavefront vs Metal megakernel (task 7.1).

Renders the three-materials demo headless on a MetalContext in ONE
(integrator, execution-mode) configuration per invocation, measures steady-
state frames/second over a fixed wall-clock window, and prints samples/sec.
One case per process keeps each run to a single Metal pipeline compile (the
megakernel main_pass compile is the slow, memory-spiking step) and lets the
cases be executed and re-run independently; combine the printed numbers into
the wavefront/megakernel ratio by hand or via the docs table. The change
keeps mode selection user-driven — there is NO silent fallback if wavefront
is slower; this script records the number the docs report.

Run each case under scripts/guarded_metal.sh:

    PYTHONPATH=$PWD/src TIMEOUT_S=900 scripts/guarded_metal.sh -- \
        <repo>/bin/python3.13 scripts/bench_metal_equal_time.py path megakernel
    ... path wavefront | bdpt megakernel | bdpt wavefront
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHADER_DIR = ROOT / "src" / "skinny" / "shaders"

RES = 256
WARMUP_FRAMES = 8
WINDOW_S = 6.0


def _bench(execution_mode: str, integrator_index: int) -> float:
    """Build a renderer in `execution_mode`, converge to steady state, return
    accumulation frames/sec over the measurement window."""
    from skinny.metal_context import MetalContext
    from skinny.renderer import Renderer

    ctx = MetalContext(window=None, width=RES, height=RES)
    r = Renderer(
        vk_ctx=ctx, shader_dir=SHADER_DIR, hdr_dir=ROOT / "hdrs",
        tattoo_dir=ROOT / "tattoos",
        usd_scene_path=ROOT / "assets" / "three_materials_demo.usda",
        execution_mode=execution_mode,
    )
    r.integrator_index = integrator_index
    deadline = time.monotonic() + 120.0
    while time.monotonic() < deadline:
        r.update(0.016)
        if r._backend_render_ready and r._num_instances >= 3:
            break
        time.sleep(0.02)
    assert r._backend_render_ready, f"{execution_mode} never became ready"
    for _ in range(WARMUP_FRAMES):  # pipeline compiles + first-dispatch costs
        r.update(0.016)
        r.render_headless()
    n = 0
    t0 = time.monotonic()
    while time.monotonic() - t0 < WINDOW_S:
        r.update(0.016)
        r.render_headless()
        n += 1
    dt = time.monotonic() - t0
    r.cleanup()
    ctx.destroy()
    return n / dt


INTEGRATORS = {"path": 0, "bdpt": 1}
MODES = ("megakernel", "wavefront")


def main() -> int:
    args = sys.argv[1:]
    if len(args) != 2 or args[0] not in INTEGRATORS or args[1] not in MODES:
        print(f"usage: bench_metal_equal_time.py {{{'|'.join(INTEGRATORS)}}} "
              f"{{{'|'.join(MODES)}}}", file=sys.stderr)
        return 2
    name, mode = args
    fps = _bench(mode, INTEGRATORS[name])
    print(f"[bench] {name} {mode} @ {RES}x{RES}: {fps:.2f} fps "
          f"({fps * RES * RES / 1e6:.1f} Mspp/s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
