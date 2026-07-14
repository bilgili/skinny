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

## Full parity matrix (task 3.3, Metal, worktree src)
- 17 passed, 2 skipped, 1 xfailed, 35 deselected — 875 s
- 2 failures (disney_cloud relMSE 0.5840 vs measured 0.075; subsurface_infinite) are
  PRE-EXISTING on main: identical to the digit with unmodified main src (path-only /
  env-only scenes that never enter the changed code). Filed as a separate follow-up
  task; NOT regressed by this change.

## Codex pre-merge review (gpt-5.5) — 2 findings, both fixed
- P2-1: connectT1's s==2 distant MIS still counted the REMOVED t=2 splat strategy →
  scene-scale-dependent darkening of distant direct (visible when bounds radius ~1).
  Fixed: s==2 distant NEE takes full weight (its only 3-vertex partner is skipped by
  design); s>=3 keeps the partition (those alternatives exist). All 3 shader files.
- P2-2: a zero-power-only authored lights_dir entered the authored mirror branch by
  list truthiness → zero records uploaded, slider fallback silently dropped. Fixed:
  branch on _scene_has_powered_dir; zero-power-dir-only scenes keep the default light.
- Both gates re-run after the fixes: Gate A PASS (identical), Gate B identical
  (bdpt/sppm 1.0095 full, direct 1.0016, mega≡wave 1.0000). 19 hostless policy tests.

## Follow-ups filed (task 4.4)
- SPPM env-indirect dimness (no env photon emission; shadow-box 0.77 baseline recorded).
- SPPM deposit vp.beta omission (through-glass visible points).
- Pre-existing disney_cloud + subsurface_infinite parity failures (bisect task chip filed).
- Optional: shrink the distant emission disk toward the specular geometry's projected
  extent (variance); spectral dispersion on the light-side specular chain.
