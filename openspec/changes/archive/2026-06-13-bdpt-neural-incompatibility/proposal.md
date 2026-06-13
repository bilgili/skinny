## Why

`skinny --integrator bdpt … --online-training` crashes mid-frame on the Metal
backend:

```
RuntimeError: the megakernel record source is unavailable on the Metal backend
— use the wavefront path integrator (record source 'wavefront') for online
training there
```

The crash is the visible symptom of a deeper mismatch. The neural directional
proposal — and the online training that exists to improve it — is consumed
**only by the path integrator**: `integrators/path.slang` imports
`sampling.proposal` and builds a `ProposalContext` at each bounce, while
`integrators/bdpt.slang` and `wavefront_bdpt.slang` never import the proposal
seam and sample directions with native flat-material BSDF sampling. This is
backend-independent (shaders are shared), so **bdpt + neural is unsupported on
Vulkan too** — the combo trains/selects weights the bdpt render never reads.

Today the renderer's `can_online_train()` only checks the execution mode and an
active neural proposal, so the combo passes the gate and dies later in the
record drain (no wavefront record source exists for bdpt; the megakernel
fallback is absent on Metal). The user wants a clear up-front error, not a
mid-frame crash or a silent no-op.

## What Changes

- **CLI startup validation, error + exit.** A shared
  `cli_common.validate_render_flags(args)` rejects `--integrator bdpt` combined
  with `--online-training` or a neural proposal (`--proposals …,neural`) by
  raising `SystemExit` with a clear message naming the fix (`--integrator
  path`). Called from every front-end's `main()` right after `parse_args`
  (`skinny`, `skinny-gui`, `skinny-web`, `skinny-render`). Only an explicit
  `--integrator bdpt` trips it; `integrator=None` (persisted/default path) does
  not. Tolerant of front-ends that suppress `--proposals` (GUI/web).
- **Renderer runtime refusal (defensive).** `can_online_train()` also refuses
  when the integrator is bdpt, so the interactive GUI — where the integrator is
  selected after startup, past the CLI gate — reports it cleanly (the config
  matrix's online-training row shows `REFUSED (requires --integrator path …)`)
  and the record drain never runs. This makes the mid-frame crash impossible.

No change to rendering or to the supported `--integrator path` neural/online
combos.

## Capabilities

### Modified Capabilities
- `render-cli`: adds a startup validation that errors-and-exits on render-flag
  combinations that cannot work, starting with bdpt × neural / online-training.
- `online-training-control`: the prerequisite gate also refuses the bdpt
  integrator (it does not consume the neural proposal), so the runtime/GUI path
  refuses cleanly instead of crashing in the record drain.

## Impact

- **Code:** `src/skinny/cli_common.py` (`validate_render_flags`); the four
  front-end `main()`s (`app.py`, `ui/qt/app.py`, `web_app.py`, `headless.py`)
  call it; `src/skinny/renderer.py` (`can_online_train` bdpt refusal + the
  config-matrix online-training row).
- **Tests:** `tests/test_cli_common.py` (validator: exits on bdpt+online /
  bdpt+neural, OK for bdpt alone / bdpt+non-neural / path+neural / default
  integrator / missing `proposals` attr); `tests/test_online_training_observability.py`
  (`can_online_train` refuses bdpt; matrix row REFUSED under bdpt).
- **Docs:** `README.md` + `docs/NeuralGuiding.md` note bdpt is incompatible
  with the neural proposal / online training.
- **Behaviour:** the documented crash becomes a one-line error + exit on the CLI
  front-ends and a clean `REFUSED` in the GUI. `--integrator path` is unchanged.
- **Risk:** low — pure validation/guarding, no render-path change; covered by
  unit tests and a live repro of the previously-crashing command.
