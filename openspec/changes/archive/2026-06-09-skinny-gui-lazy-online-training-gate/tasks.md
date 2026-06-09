## 1. Renderer: split the permanent prerequisite

- [x] 1.1 Add `Renderer.online_train_execution_supported()` returning
      `self.effective_execution_mode_index == EXECUTION_WAVEFRONT`, with a
      docstring noting the neural-proposal half is runtime-selectable (transient).

## 2. skinny-gui: drop --proposals, lazy gate

- [x] 2.1 `ui/qt/app.py`: call `add_render_flags(parser, proposals=False)`.
- [x] 2.2 `ui/qt/app.py`: drop `proposals` from the `MainWindow(...)` call and
      from `MainWindow.__init__` signature; remove the startup `--proposals`
      override block.
- [x] 2.3 `ui/qt/viewport.py` `_maybe_online_training`: on `can_online_train()`
      refusal, give up only when `online_train_execution_supported()` is False
      (print once, set `_online_training_requested = False`); otherwise keep
      polling. Emit a one-time armed-and-waiting message on the first transient
      miss.

## 3. Verify

- [x] 3.1 `.venv/bin/python -m py_compile` on the touched files.
- [x] 3.2 `.venv/bin/ruff check src/`.
- [ ] 3.3 `skinny-gui --proposals bsdf,neural` errors with unrecognized argument;
      `skinny-gui` launches and the Proposals combobox still switches state.
      (MANUAL — not run; needs a GPU + Qt display. `proposals=False` suppression
      is covered by `test_flags_can_be_suppressed`; combobox→state was confirmed
      by code inspection.)
- [ ] 3.4 With `--execution-mode wavefront --online-training`, launching without
      a neural proposal then selecting one in the combobox starts training (log
      shows the trainer config); `--execution-mode megakernel --online-training`
      refuses once and stops. (MANUAL — interactive GPU run, not executed here.)

## 4. Docs

- [x] 4.1 `README.md`: note `skinny-gui` has no `--proposals` (combobox owns it);
      adjust the `--online-training` example that uses `skinny-gui --proposals …`.
- [x] 4.2 `docs/NeuralGuiding.md` (running online training): GUI users select the
      neural proposal in the combobox rather than passing `--proposals`.
- [x] 4.3 Regenerate embedded code if any marked shader region changed (none
      expected) — `node docs/diagrams/embed_code.cjs --check`.
