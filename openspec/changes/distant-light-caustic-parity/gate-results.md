# Recorded gate results (Metal, 256², worktree)

## Gate A — phantom policy (assets/glass_caustics_spectral.usda, authored SphereLight only, 1024 spp)
- numDistantLights: path=0 bdpt=0 sppm=0 (phantom suppressed by policy, no direct_light_index hack)
- bdpt/path: full = 1.0013, shadow-box = 1.0140
- SPPM fireflies at default radius: maxL = 2.04, px>3 = 0  (was maxL 12.7, 62 px before the change)
- Honest SPPM baselines vs bdpt (env-indirect follow-up must LOWER these):
  caustic-mask sppm/bdpt = 0.9258, shadow-box sppm/bdpt = 0.7700

## Gate B — authored DistantLight over glass (scratchpad/caustics_sun.usda, 2048 spp, accum + splat composite)
- bdpt renders the distant-light caustic; (bdpt−path) ≡ (sppm−path) spatially (284/300 top-excess px agree)
- bdpt/sppm: full = 1.0095, ground = 1.0140, caustic-mask mean = 0.92 / median = 1.015 (splat noise at this budget)
- distant DIRECT no-double-count: bdpt/path = 1.0016 (after the t=2 DIR-splat NEE-ownership carve-out;
  1.0601 before it — the z1.pdfFwd=1 camera convention cannot partition the direct splat at grazing pdfs)
- path = recorded per-component exclusion (delta-delta SDS): bdpt/path full = 1.0272 — entirely the caustic
- megakernel ≡ wavefront bdpt: full = 1.0000, caustic = 1.0000 (lane budgeting fine, no watchdog wedge)
