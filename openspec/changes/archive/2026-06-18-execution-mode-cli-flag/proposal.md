## Why

The wavefront-execution-backend change made the execution mode (`megakernel` |
`wavefront`) a **runtime, GUI-selectable, persisted** parameter — surfaced in
`ALL_PARAMS` and every front-end, switchable mid-session with an accumulation
reset. Two problems with that, surfaced once the backend was complete:

1. **It's not really a runtime knob — it's a build-time backend choice**, like
   `--backend metal|vulkan`. Switching it mid-session is rarely wanted and
   forces *both* backends' machinery to be live at once.
2. **Both backends compile even when only one is used.** The megakernel
   `ComputePipeline` (a multi-second `slangc` compile of `main_pass.slang`) is
   built unconditionally at scene load, even in `wavefront` mode — because the
   wavefront passes currently *reuse* the megakernel's descriptor-set layout,
   per-frame descriptor sets, `generated_materials` emission, and
   `graph_bindings`. So wavefront mode pays the full megakernel compile + its
   pipeline VRAM for nothing, and (vice-versa) the lazy wavefront passes add
   their own pipelines on top in mixed use.

Selecting the backend on the command line and compiling **only** the chosen
backend removes the wasted compile/VRAM and matches how `--backend` already
works.

## What Changes

- **Select the execution mode on the command line**, mirroring `--backend`:
  `--execution-mode megakernel|wavefront` (+ `SKINNY_EXECUTION_MODE` env),
  default `megakernel`. The renderer takes it as a constructor argument fixed
  for the session.
- **Remove the GUI/runtime path**: drop the `execution_mode_index` entry from
  `STATIC_PARAMS`/`ALL_PARAMS`, the front-end Combo, the settings-snapshot
  persistence, and the runtime `set_execution_mode`/`cycle_execution_mode`
  switching (and the `9.6` front-end-parity test that asserted the GUI toggle).
  The capability gate (`effective_execution_mode`) and the Metal pin stay, now
  applied to the CLI value at construction.
- **Mutually-exclusive compilation.** In `megakernel` mode, build the megakernel
  `ComputePipeline` and never build the wavefront stage pipelines. In
  `wavefront` mode, **do not** compile `main_pass.slang` / the megakernel
  pipeline at all; build only the wavefront stage pipelines.
- **Decouple the wavefront from the megakernel pipeline** (the enabler for the
  above): factor the shared scene plumbing the wavefront currently borrows from
  the megakernel `ComputePipeline` — the set-0 descriptor-set layout, the
  per-frame descriptor sets, `generated_materials`/per-graph emission, and the
  `graph_bindings` map — into a backend-independent owner the wavefront builds
  standalone. (This also subsumes the deferred task 6.4 "skip the megakernel
  rebuild on a wavefront-mode add".)

## Capabilities

### Modified Capabilities
- `wavefront-execution`: the execution-mode axis becomes a command-line/session
  selection rather than a runtime GUI toggle, and the renderer compiles only the
  selected backend (mutually-exclusive megakernel vs wavefront).

## Impact

- **Code:** `src/skinny/app.py` (`--execution-mode` arg + env + pass to
  `Renderer`); `src/skinny/renderer.py` (constructor arg; gate
  `_build_pipeline_for_current_graphs` / the wavefront-pass build on the fixed
  mode; decouple the scene descriptor layout + `generated_materials` emission +
  graph bindings from the megakernel `ComputePipeline`); `src/skinny/params.py`
  + `src/skinny/ui/*` (remove the `execution_mode_index` ParamSpec + Combo +
  persistence). The Metal pin + `effective_execution_mode` gate stay.
- **Tests:** replace the GUI/runtime tests (`9.6` front-end parity,
  `test_execution_mode` persistence/snapshot, the runtime-switch state-hash
  reset) with CLI-selection + mutual-exclusion tests: assert `wavefront` mode
  builds no megakernel pipeline (`renderer.pipeline is None`) and renders, and
  `megakernel` mode builds no wavefront stage pipelines.
- **Behaviour:** switching execution mode now requires a restart (like
  `--backend`); the A/B parity tests select the mode per `Renderer` instance
  instead of toggling at runtime.
- **Risk:** the decoupling is the bulk — the wavefront's set-0 layout must be
  built independently (reflected from the wavefront kernels or defined
  explicitly) and stay consistent across all stage pipelines + the descriptor
  writes; the megakernel path must be unaffected when selected.

## Notes

Authored as a proposal to implement later (the wavefront-execution-backend
change lands first). The MODIFIED requirement below targets the post-archive
baseline of that change.
