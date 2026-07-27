# Change: choice-table-owners

## Why

`render_envelope.py` gave the *validity* of a combo one owner. The human-facing
**names, labels and indices** that go with it did not get one, and several
copies have already drifted.

- **Integrator index**: 4 copies — `cli_common.INTEGRATOR_INDEX` (`:36`),
  `headless._INTEGRATORS` (`:27`, an identical dict, second copy),
  `cli_common.py:536` argparse `choices`, `render_envelope.INTEGRATORS` (`:48`).
- **Integrator labels**: `renderer.py:1631` has `["Path","BDPT","SPPM","MLT"]`;
  `render_session.py:218` has `["Path","BDPT","SPPM"]` — **MLT missing**.
- **Tonemap**: 4 copies — `renderer.py:1839` `["ACES","Reinhard","Hable",
  "Linear"]`, `headless._TONEMAPS` (`:28`), `headless.py:359` argparse choices,
  `render_session.py:226` `["Filmic"]`. Two disagree.
- `render_session._default_choice_names()` (`:212-231`) re-lists **17 choice
  sources** as GUI-thread placeholders; at least 6 disagree with the renderer's
  real lists — `reuse_modes` `["Off"]` vs `["None","ReSTIR DI"]`,
  `detail_maps_modes` `["Off"]` vs `["On","Off"]`, `restir_combination_modes`
  `["Unbiased","Biased"]` vs `["Unbiased (GRIS)","Biased (ΣM)"]`, and more.
- `render_session._default_values()` (`:303-324`) hardcodes 8 override defaults
  on top of `STATIC_PARAMS` — a third defaults authority beside `params.py` and
  `Renderer.__init__`.

The same shape appears in the wavefront layer: **33 of the 34 kernel entry-point
names are written in three files** (`wavefront_driver.py`, `vk_wavefront.py`,
`metal_wavefront.py`), and 14 class constants — `MAX_BOUNCES`, `STREAM_CAP`,
`BDPT_MAX_VERTS`, `VERTEX_STRIDE`, `AUX_STRIDE`, `EYE_BOUNCES`,
`LIGHT_BOUNCES`, `WALK_MODES`, `RESERVOIR_STRIDE`, the ReSTIR `DEFAULT_CONFIG`
— are duplicated verbatim between the Vulkan and Metal pass classes with **no
test pinning them equal**.

By the deletion test these mirrors are shallow: removing them concentrates the
lists rather than moving complexity anywhere. What they currently buy is a
GUI-thread placeholder and a couple of avoided imports — at the cost of labels
that are already wrong in shipped UI.

## What Changes

- One owner per axis for names, labels and indices — integrators, tonemaps,
  reuse modes, detail-map modes, ReSTIR combination modes, proposal presets,
  execution modes — read by the CLI's `choices`, the headless tables, the
  renderer's display lists, and the GUI-thread proxy defaults.
- `render_session._default_choice_names` and `_default_values` read the owner
  instead of restating it; the placeholder need is met by importing a
  toolkit-free table, not by retyping the lists.
- Kernel entry-point names get one table, imported by the driver and both
  backends, so a rename breaks the build rather than a render.
- The 14 duplicated wavefront constants get one home, or a test that pins the
  two copies equal where they must legitimately differ.
- Fix the drifted values found along the way: the missing MLT label, the
  disagreeing tonemap list, and the six divergent placeholder lists.

## Capabilities

### New Capabilities

- `choice-table-ownership`: one owner per enumerated axis for its values,
  labels and indices — consumed by the CLI's `choices`, the headless tables,
  the renderer's display lists and the GUI-thread proxy defaults — plus one
  owner for the wavefront kernel entry-point names and shared pass constants.
  `render-cli` and `wavefront-execution` behaviour is unchanged except where a
  currently-drifted label or list is corrected.

## Impact

- Modified: `src/skinny/cli_common.py`, `src/skinny/headless.py`,
  `src/skinny/render_envelope.py`, `src/skinny/renderer.py` (display lists),
  `src/skinny/render_session.py` (placeholders and defaults),
  `src/skinny/wavefront_driver.py`, `src/skinny/vk_wavefront.py`,
  `src/skinny/metal_wavefront.py`.
- **User-visible**: the MLT integrator label and the tonemap list become
  correct in surfaces that currently show the wrong ones.
- Hostless, no GPU work, no shader change, no risk to the parity matrix. Lands
  piecemeal — one axis at a time.
- Docs: `README.md` where flag choices are listed; `docs/Wavefront.md` for the
  kernel-name table.
