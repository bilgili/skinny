## Context

Wavefront path shade today is a fixed 2-slot split, classified by material
*type*:

```
slot 0  wfPathShadeFlat   flat + MaterialX graphs        ~1.4 MB   (safe)
slot 1  wfPathShade       skin + python×N + debug (UNION) ~2.8 MB   (flaky compile)
```

`wf_shade_common.slang` hardcodes `WF_NUM_SLOTS = 2u`; `wfSlotForType(matType)`
maps flat→0, everything-else→1. The counting-sort infra
(`material_count/offset/indirect_args`, `wfBuildArgs`/`wfScatter`/indirect
dispatch) is already sized to `num_materials` and slot-generic — only the static
slot count and the type classifier collapse it to two.

The catch-all is large because it is a union of independent materials. Splitting
it so each kernel imports only its own material makes every kernel small. The
goal (B) is purely to keep every shade `.spv` under MoltenVK's ~2.83 MB Metal
compile danger line, killing the observed flaky compile.

## Goals / Non-Goals

**Goals**
- No wavefront path shade kernel approaches the MoltenVK compile limit.
- Identical rendered image for every grouping (same estimator).
- Grouping decided from real measured `.spv` sizes (conservatively safe).
- Foundation reusable by the later perf-tune (goal A) change.

**Non-Goals**
- bdpt shade/connect (different axis), megakernel (untouched).
- Dispatch-overhead / shade-coherence tuning (goal A — follow-up, opt-in).
- Shrinking a single monolithic kernel (e.g. skin) that alone exceeds the limit.

## Decisions

### 1. Measured two-pass bin-pack (not modeled)
A static cost model can under-split if wrong, re-introducing the flaky compile.
Measuring the real isolated `.spv` size is exact and driver-accurate; the
existing content-hash `.spv` cache amortizes the per-material isolate-compiles
after the first frame. Packing by the **sum** of isolated member sizes is
conservatively safe: a group's shared imports are counted once, so its real size
is ≤ the sum — bin-packing by the sum never under-splits past the threshold.

```
on material-set change (scene load / add material / live python edit):
  members = [skin?, python_0..python_{N-1}, debug?]          # flat/graphs → slot 0
  for m in members:                                          # PASS 1
      size[m] = bytes(compile_isolated_shade(m))             # cache-keyed
  groups  = ffd_bin_pack(members, size, THRESHOLD,           # PASS 2
                         pin_alone={skin})                   # skin its own group
  for g in groups: compile_group_shade(g)                    # one pipeline / group
  materialSlot[matId] = slot_of(group containing matId)      # flat/graph → 0
```

`THRESHOLD ≈ 2.4 MB` — a margin under 2.83 MB absorbing the union-vs-sum slack
in the wrong direction (real group < sum, so the effective headroom is larger).
First-fit-decreasing: sort members by size desc, place each into the first group
it fits, else open a new group. Skin is pre-pinned to its own group.

### 2. Fixed `MAX_SLOTS` padding for the dynamic slot count
`WF_NUM_SLOTS` becomes `MAX_SLOTS = 32`, a compile-time constant. Padding (vs a
specialization constant) avoids recompiling the classify / build-args / clear
kernels every time the material set changes. The cost of a large cap is **not**
dispatch count — the CPU records exactly `used` indirect shade dispatches by
reading the per-slot group-count — only:
- shader loops `0..MAX_SLOTS` in clear-counts / build-args (32 iterations, trivial);
- counting-sort buffers sized to `MAX_SLOTS` (`32 × {count,offset,cursor}` uints
  + `32 × indirect_args` — a few KB).

`MAX_SLOTS = 32` makes overflow effectively impossible for real scenes (a skin
renderer has a handful of distinct materials). A headroom test asserts a
representative heavy scene stays well under 32 groups.

### 3. `materialId → slot` lookup replaces the type classifier
`wfSlotForType(matType)` → `wfSlotForMaterial(matId)` reading a small CPU-built
`materialSlot[]` storage buffer (new set-1 binding), one `uint` per material.
Flat and MaterialX-graph materials map to slot 0. The classify kernel writes
`wfLaneSlot[i] = wfSlotForMaterial(hit.materialId)` and atomically counts the
slot — otherwise the counting-sort path is unchanged.

### 4. Grouped shade entry bodies
Each group's shade kernel imports only its members and dispatches over its slot
queue (existing `wfQueueLane` idiom). A single-member group is a straight-line
shade (no switch — the common case after splitting). A multi-member group (small
materials packed together) switches on `materialTypes[matId] & 0xFF` /
`pythonMaterialId` among its members. Slot 0 (flat + graphs) is unchanged.

### 5. `WavefrontPathPass` takes the grouping
Constructor argument changes from `build_catchall: bool` to a `grouping`
(ordered list of groups, each a list of member descriptors, + the `materialSlot`
array). The pass builds one shade pipeline per group and records `len(groups)`
indirect dispatches. `renderer._ensure_wavefront_path_pass` computes the grouping
from `_material_types` (+ python ids), uploads `materialSlot[]`, and keys the
pass-rebuild on the grouping signature (so an unchanged material set reuses the
pass; the per-material `.spv` cache makes re-grouping cheap).

## Risks / Trade-offs

- **Skin alone over the limit.** If pass-1 measures skin's isolated shade above
  2.83 MB, grouping cannot help — surfaced immediately as a separate skin-shrink
  effort. The change still lands (skin gets its own group, as small as it can
  be).
- **Cold-start compiles.** First frame compiles each member isolated + each
  group — more `slangc` invocations than the old 2 kernels. Amortized by the
  content-hash cache; only the common-case (flat-only) scene is unaffected (it
  builds slot 0 alone, exactly as today).
- **New `materialSlot[]` binding** must stay consistent with the grouping across
  the classify + every shade dispatch — covered by the slot-map/classify test.

## Migration

No persisted state, no checked-in `.spv` change, `main_pass.spv` byte-unchanged.
A flat-only scene compiles only slot 0 (identical to today). Multi-material
scenes that previously built the flaky catch-all now build size-bounded groups —
same image, reliable compile.

## Implementation Prerequisite (found while scoping the code)

PASS-1 measurement requires compiling a **skin-only** and a **python-only** shade
kernel — each importing only its own material subtree. Today only
`flat_bounce.evaluateFlatBounce` exists; skin/python/debug all live inside
`integrators/path.slang::evaluateBounce` (one switch over the whole material
tree, importing `materials.skin.*` + `python_materials_dispatcher` together).

So the first implementation step is to **extract per-type bounce evaluators** the
way `flat_bounce` already was: `skin_bounce.slang` (`evaluateSkinBounce`, imports
only `materials.skin.skin_material` → `evalSkinRadiance`), `python_bounce.slang`
(`evaluatePythonBounce`, imports only the generated `python_materials_dispatcher`),
and a trivial `debug_bounce.slang` (`evaluateDebugBounce`, no heavy import — the
`0.5 + 0.5*N` default case). Each mirrors one `case` of `evaluateBounce`'s switch
verbatim. `evaluateBounce` itself stays for the megakernel (unchanged).

**Python is one member, not N (revised during implementation).**
`python_materials_dispatcher.slang` is generated by
`ComputePipeline._emit_python_dispatcher()` as a single switch over *all* python
materials, importing every one — so a kernel importing it pulls in the whole
python set. Single-python-material isolation would need a new
`_emit_python_dispatcher(subset_ids)` subset emitter. Since **skin** is the heavy
member (BSSRDF), pulling skin into its own kernel almost certainly drops
`python_all + debug` well under the threshold, so the first cut treats the members
as `{skin, python_all, debug}`. Per-single-python-material isolation (the subset
emitter) is deferred to the goal-A perf-tune follow-up — same overhead/coherence
territory. The packer, MAX_SLOTS, classifier, and measurement are unchanged; only
the member enumeration is coarser.

With these three members and skin pinned alone, the measured packer's grouping is
`{skin}` + `{python_all, debug}` unless `python_all + debug` alone exceeds the
threshold (which would then require the deferred subset emitter). A single-member
group calls that member's evaluator directly; the `{python_all, debug}` group
switches on `materialTypes[matId] & 0xFF` between the python and debug evaluators.

## Per-group shade kernel generation

A group's shade entry is **generated** from its member set (like the flat kernel
is hand-written, but composed per group): it imports `wf_shade_common` + only its
members' evaluator modules, and its body is

```
i = wfQueueLane(shadeSlot, tid.x); if (i == SENTINEL) return;
s = wfState[i]; h = wfHits[i]; rng = s.rngState; r = {s.rayOrigin, s.rayDir};
BounceResult br = <single member: evaluateXBounce(...)               >
                  <multi  member: switch(matType){ case ...: ... }   >
wfFinishShade(i, s, rng, h, br);
```

Slot 0 stays the hand-written `wfPathShadeFlat`. The generated group entries reuse
the existing per-material `.spv` content-hash cache (`_compile_shade`), keyed by
the member set, so identical groupings across frames/scenes hit the cache.
