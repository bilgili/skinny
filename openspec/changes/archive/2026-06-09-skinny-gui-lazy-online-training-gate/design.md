## Context

`can_online_train()` bundles two prerequisites with very different lifetimes:

1. **Execution mode = wavefront** — fixed for the whole session (the execution
   axis is not runtime-switchable). A miss here is *permanent*.
2. **A neural proposal active in the mixture** — runtime-selectable via the
   Proposals combobox on `skinny-gui`. A miss here is *transient*.

The current worker treats both the same: one check after the scene builds, then
`_online_training_requested = False` forever. That defeats runtime proposal
selection.

## Decision

Split the lifetime, not the public gate API. Keep `can_online_train()` as the
single full gate (used by every front-end and the docs' contract). Add a narrow
helper for the permanent half:

```python
def online_train_execution_supported(self) -> bool:
    """Whether this session's execution mode permits online training at all
    (wavefront-only). Distinct from can_online_train(), which also requires a
    neural proposal to be *currently* active — that half is runtime-selectable
    via the Proposals combobox, so a False there is transient, not permanent."""
    return self.effective_execution_mode_index == EXECUTION_WAVEFRONT
```

Worker logic (Qt `_RenderWorker._maybe_online_training`):

- wait until scene built (`descriptor_sets is not None`) — unchanged;
- `can_online_train()` ok → enable, done;
- not ok and `not online_train_execution_supported()` → permanent: print the
  refusal once, set `_online_training_requested = False` (give up) — matches
  today's behavior for the wavefront miss;
- not ok but execution supported → transient (no neural proposal yet): return and
  retry next frame. Print a one-time informational line so the user knows
  training is armed and waiting for a neural proposal.

The per-frame cost while waiting is a bool compare + a `_neural_active()` check —
negligible.

Only `skinny-gui` has the combobox, so only it benefits from the lazy path. The
GLFW/web/render front-ends still pass through `can_online_train()` with their
startup proposal selection; the lazy gate is harmless there (their proposal is
also runtime-selectable on GLFW, so they get the same benefit for free, but this
change does not remove their `--proposals` flag).

## Why not keep `--proposals` on the GUI too

Two surfaces for one piece of state (`proposal_preset_index`) is the confusion
the user flagged. The combobox already drives it live and is persisted; the flag
only ever seeded the now-removed one-shot timing dependency. Removing it leaves
exactly one control.

## Risks

- A user scripting `skinny-gui --proposals …` will now get an "unrecognized
  argument" error. Acceptable: the flag is being intentionally retired on this
  front-end; the combobox + persisted settings replace it. Documented in README.
