## Context

Online neural-proposal training (change `neural-online-training`) hands trainer
weights to the renderer through a `NeuralWeightPublisher` double-buffer selected
by `--neural-handoff`. Two backends ship:

- `file` (`neural_handoff_file.py`) — `publish()` writes an NFW1 file and reads
  it back via `load_neural_weights`; `swap()` promotes pending→render at the
  frame boundary. Portable, but a disk round-trip per publish.
- `interop` (`neural_handoff_interop.py` / `neural_handoff_interop_metal.py`) —
  writes the GPU-shared weight/bias buffers in place (CUDA↔Vulkan external
  memory, or Metal UMA shared storage). Zero CPU copy, but needs one of those
  GPU paths and is guarded off elsewhere.

The trainer runs as a **same-process daemon thread** (`_start_trainer_thread`);
the publisher double-buffer is the only trainer→render handoff, and no extra
locking is used because `publish`/`swap` hand off whole buffers. So on a host
with no `interop` path, `file` writes weights to disk only to read them back
into the same address space — pure overhead.

`NeuralWeights` is a mutable `@dataclass` of three numpy arrays (`headers`,
`weights`, `biases`). The publisher contract: `publish(weights)->version`
(trainer, any time), `swap()->bool` (renderer, frame end), `acquire_for_render()
->(weights,version)` (renderer), `current_version()->int`.

## Goals / Non-Goals

**Goals:**
- A `shared` handoff backend that skips the disk and needs no CUDA/UMA: an
  in-process CPU double-buffer of `NeuralWeights`.
- Exact same publisher contract and frame-boundary swap / version-increment
  semantics as `file` and `interop` — drop-in selectable at runtime.
- Bytes a renderer would consume under `shared` are identical to a `file` publish
  of the same weights (no quantization or transport drift).
- Runs on every platform with zero new dependencies.

**Non-Goals:**
- Writing GPU buffers directly (that is `interop`; `shared` uploads through the
  renderer's existing post-swap GPU upload path, exactly as `file` does).
- Cross-process / shared-memory IPC. The trainer is in-process; OS shared memory
  (`multiprocessing.shared_memory`) is explicitly out of scope until/unless the
  trainer moves out of process.
- Any change to `file`, `interop`, the trainer backends, the replay buffer, the
  record drain, or the NFW1 on-disk format.
- Changing the default (`file` stays default).

## Decisions

### D1 — Flag value name `shared` (not `cpu`)
`--neural-trainer` already has a `cpu` value (the numpy reference oracle, a
*compute* backend). Naming the handoff `cpu` too would read ambiguously on a
command line that sets both (`--neural-trainer mlx --neural-handoff cpu`).
`shared` names the property that matters — shared in-process buffers, no disk —
and reads cleanly beside `file`/`interop`. *Alternative `memory`* was equivalent;
`shared` was chosen for the "shared buffer" connotation that distinguishes it
from the per-publish allocation of `file`.

### D2 — Deep-copy at publish, not at swap
`publish()` stores a deep copy of the staged `NeuralWeights` (copying the three
numpy arrays) as pending. This preserves the frozen-render-buffer invariant: the
trainer thread may keep mutating its working `NeuralWeights` in place after
publishing without racing the render buffer. `swap()` then just rebinds the
pending reference to render (no copy needed — pending is already private). This
mirrors what `file` gets for free (serialize then parse yields a fresh object)
without the disk hit. *Alternative* — copy at swap — was rejected: it would force
the trainer to not touch its buffer between publish and the next frame, a
contract the other backends don't impose.

### D3 — Reuse the byte-faithful round-trip for parity, in memory
To guarantee D-goal byte-parity with `file`, the copy goes through the same
serializer the file path uses: `load_neural_weights` applied to
`write_neural_weights`'s bytes — but to an in-memory `io.BytesIO`, never the
filesystem. This catches serialise/parse drift identically to `file` (the stated
reason `file` round-trips through disk) while costing only RAM. If
`write/load_neural_weights` lack a file-like overload, add a minimal
bytes/`BytesIO` entry point; otherwise a direct numpy `.copy()` of the three
arrays is the fallback (still byte-faithful, since the arrays are the canonical
content). The implementation picks the cheapest of the two that preserves
parity; tests assert parity regardless.

### D4 — Factory + no interop kwargs
`make_publisher("shared", initial=…, expect_arch=…)` constructs
`SharedWeightPublisher`. Unlike `interop`, `shared` needs none of the
`weights_buffer`/`biases_buffer`/`timeline_semaphore`/`precision` kwargs;
`enable_online_training` only adds those for `kind == "interop"`, so `shared`
flows through unchanged. The factory's unknown-kind error message widens to list
all three values.

### D5 — Persistence and CLI surface unchanged in shape
`--neural-handoff` choices widen to `(file, interop, shared)`; the
`SKINNY_NEURAL_HANDOFF` env and the front-end persistence of
`_neural_handoff_kind` (`app.py`, `ui/qt/app.py`) already round-trip an arbitrary
string, so only the argparse `choices` tuple and help text change.

## Risks / Trade-offs

- **[Stale requirement title]** The base spec requirement is named "Two
  selectable weight-handoff backends." → Generalize its body to three backends;
  if the delta tooling supports a clean rename it is renamed, otherwise the body
  is authoritative and the count in the title is corrected at archive.
- **[Parity drift between `shared` and `file`]** A future weight-format change
  could make a naive numpy-copy diverge from the serialized bytes. → D3 routes
  the copy through the same serializer; a parity test publishes the same weights
  both ways and asserts byte-equal `weights`/`biases`/`headers`.
- **[Confusion with `interop`'s "no disk" claim]** Both `shared` and `interop`
  avoid disk; they differ in whether the GPU buffer is written directly. →
  Docs/help state `shared` = CPU double-buffer + normal GPU upload, `interop` =
  direct GPU-buffer write.
- **[Trainer mutates published buffer]** If publish did not copy, a swap could
  expose a half-mutated array. → D2 copies at publish; a test mutates the
  trainer's source after publish and asserts the acquired render weights are
  unchanged.

## Migration Plan

Additive and opt-in. Default remains `file`; existing `file`/`interop` runs are
byte-identical. Rollback is removing the `shared` choice — no persisted state or
format depends on it. No data migration.

## Open Questions

None. Naming (`shared`) and scope (in-process only) were resolved before
authoring.
