## Why

`skinny-gui` (Qt) already owns proposal selection at runtime: the **Proposals**
combobox writes `renderer.proposal_preset_index` live (same attribute the CLI
`--proposals` flag seeds at startup), and accumulation resets on change. So the
GUI's `--proposals` flag is redundant *except* for one coupling: the
`--online-training` gate checks `can_online_train()` **once** after the scene
builds and gives up for good if a neural proposal is not yet active
(`viewport.py` sets `_online_training_requested = False` on refusal). The flag
exists to seed the neural preset *before* that one-shot check fires.

This makes the GUI confusing: the combobox can switch to a neural proposal at
runtime, but online training never starts because the gate already gave up — and
two surfaces (flag + combobox) control the same state.

## What Changes

- **Remove `--proposals` from `skinny-gui`** (`add_render_flags(parser,
  proposals=False)`; drop the startup override and the `proposals` parameter on
  `MainWindow`). The Proposals combobox is the single proposal control in the Qt
  front-end. `--proposals` stays on the other front-ends (`skinny` GLFW,
  `skinny-web`, `skinny-render`) where there is no combobox.
- **Make the online-training gate lazy** instead of one-shot: the
  *neural-proposal* prerequisite is runtime-satisfiable, so the worker keeps
  polling and enables training the moment a neural proposal becomes active (via
  the combobox) rather than refusing permanently. The *wavefront* prerequisite is
  fixed for the session, so a non-wavefront mode still refuses clearly once and
  gives up (unchanged).
- Add a small renderer helper that reports whether the **permanent** prerequisite
  (wavefront execution mode) holds, so the worker can distinguish a permanent
  refusal (give up + message) from a transient one (keep polling silently).

No behavior change for the non-GUI front-ends or for `--online-training`
semantics there. With `--online-training` and `--execution-mode wavefront`, the
GUI user now simply picks a neural proposal in the combobox to start training.

## Capabilities

### Modified Capabilities
- `online-training-control`: the prerequisite gate becomes lazy for the
  runtime-selectable neural-proposal prerequisite on interactive front-ends with
  a proposal combobox; the wavefront prerequisite stays a permanent clear
  refusal. `skinny-gui` drops its `--proposals` flag in favor of the combobox.
