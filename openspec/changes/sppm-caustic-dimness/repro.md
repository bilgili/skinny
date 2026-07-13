# Repro / measurement methodology — SPPM vs path dimness

## Correct methodology (avoids the two artifacts)

1. Render path AND sppm at the **same, high** sample count (≥2048) so fireflies
   have averaged out.
2. Read the linear `.hdr`, compare **per-region MEDIAN** (firefly-robust), path
   vs sppm **at the same spp** (a per-spp export scaling cancels).

```bash
export VULKAN_SDK=/Users/ahmetbilgili/VulkanSDK/1.4.341.1/macOS
export DYLD_LIBRARY_PATH=$VULKAN_SDK/lib SKINNY_BACKEND=metal
WT=<worktree>; MAIN=/Users/ahmetbilgili/projects/skinny
for I in path sppm; do
  PYTHONPATH=$WT/src $MAIN/bin/python3.13 -m skinny.headless \
    $WT/assets/glass_caustics_test.usda -o /tmp/${I}.hdr --integrator $I \
    --backend metal --width 256 --height 256 --samples 2048 --tonemap linear --format hdr
done
# median(sppm)/median(path) per region; ~0.75-0.84 = the real dimness.
```

## Two artifacts that fooled the first pass (do NOT repeat)

- **Low-spp box MEAN** is firefly-dominated: path reads 263× its converged value
  at 1 spp. Never use a low-spp mean — use median or high spp.
- **Cross-spp linear-HDR comparison** is invalid: `--format hdr --tonemap linear`
  has a per-spp scaling common to path AND sppm (path's flat-ground median halves
  per spp doubling). Compare at the SAME spp only.

An earlier "1/N energy / 58× direct-term" conclusion was these artifacts —
retracted. The real signal is a spp-invariant ~0.75-0.84 same-spp ratio.
