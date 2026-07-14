# Repro — SPPM env-direct under-count

Tight, deterministic loop (goes red on the bug, green when fixed).

## Isolation scene (env-only)

A diffuse ground + box under ONLY the env dome — a flat plane's value is almost
pure env DIRECT (analytic ≈ albedo·L), so the deficit is unambiguous. See
`/tmp` scratch `env_ground.usda` (ground albedo 0.8, no analytic light).

```bash
export VULKAN_SDK=/Users/ahmetbilgili/VulkanSDK/1.4.341.1/macOS
export DYLD_LIBRARY_PATH=$VULKAN_SDK/lib SKINNY_BACKEND=metal
WT=<worktree>; MAIN=/Users/ahmetbilgili/projects/skinny
for I in path sppm; do
  PYTHONPATH=$WT/src $MAIN/bin/python3.13 -m skinny.headless env_ground.usda \
    -o /tmp/${I}.hdr --integrator $I --backend metal --width 256 --height 256 \
    --samples 1024 --tonemap linear --format hdr
done
# flat-ground median(sppm)/median(path): 0.735 before, ~1.0 after.
```

## Confirmation probe (root cause)

Force the env NEE to full weight in `nee.slang`
(`float w = powerHeuristic(es.pdf, misPdf);` → `float w = 1.0;`) and render sppm:
env-only flat ground 0.735 → **0.998**. Proves the deficit is the MIS-weighted env
NEE with no BSDF companion (the eye terminates at the VP). Revert the probe.

## Methodology guards (learned the hard way)

- Compare integrators at the SAME spp with a MEDIAN (firefly-robust). A low-spp box
  MEAN is firefly-dominated (path reads 263× at 1 spp), and `--format hdr
  --tonemap linear` has a per-spp export scaling common to both — cross-spp linear
  comparison is invalid.
- `--no-direct` is INVALID for SPPM: the eye stores the VP with NEE-based direct
  and returns, so NEE-off strips SPPM's direct entirely (not just NEE).
- `--env-intensity 0` reads the authored env-dome intensity (renderer.py ~9320),
  not the flag — do not use it as a clean env-off control.
