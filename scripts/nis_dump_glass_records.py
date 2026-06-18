"""Dump glass-scene path records (.nrec) for offline RQ-vs-PQ training.
Direct light off (emissive-panel only) to match the no-direct comparison.

Run: PYTHONPATH=src ./bin/python3.13 scripts/nis_dump_glass_records.py
"""
import sys
from pathlib import Path
sys.path.insert(0, "src")

ROOT = Path("/Users/ahmetbilgili/projects/skinny")
SHADER_DIR = ROOT / "src" / "skinny" / "shaders"
SCENE = ROOT / "assets" / "glass_caustics_test.usda"
OUT = ROOT / ".skinny_neural" / "glass_nodirect.nrec"
RES, FRAMES = 128, 600


def main():
    from skinny.vk_context import VulkanContext   # dump uses Vulkan descriptor sets
    from skinny.renderer import Renderer
    ctx = VulkanContext(window=None, width=RES, height=RES)
    r = Renderer(vk_ctx=ctx, shader_dir=SHADER_DIR, hdr_dir=ROOT / "hdrs",
                 tattoo_dir=ROOT / "tattoos", usd_scene_path=SCENE,
                 execution_mode="megakernel")
    r.proposal_preset_index = r.proposal_preset_from_token("bsdf")
    for _ in range(400):
        r.update(0.025)
        if (r._usd_scene is not None and len(r._usd_scene.instances) >= 1
                and r._scene_bindings is not None):
            break
    assert r._scene_bindings is not None
    r.update(0.04); r.render_headless()   # build descriptor sets / pipelines
    r.light_intensity = 0.0          # direct off (emissive panel only)
    r.direct_light_index = 1
    print(f"[dump] glass direct-off: {FRAMES} frames @ {RES}^2 -> {OUT.name}", flush=True)
    n = r.dump_path_records(str(OUT), num_frames=FRAMES, frame_seed_base=0)
    print(f"[dump] wrote {n} records to {OUT}", flush=True)
    r.cleanup(); ctx.destroy()


if __name__ == "__main__":
    main()
