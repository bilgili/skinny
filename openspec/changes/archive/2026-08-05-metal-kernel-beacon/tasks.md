# Tasks — metal-kernel-beacon

Build order is bottom-up. Each group depends only on earlier groups. Every task
traces to a frozen interface in `design.md` and a spec requirement. Do NOT change
`design.md`, any spec, or an interface. On any interface friction, STOP and report
to the orchestrator.

Note: the worktree `.gitignore` is `*`. Add every new file with `git add -f`.

## 1. Host module `metal_beacon.py`

Owner of the mmap cell format, the writer, the reader, and the kernel-identity
table. Device-free, so it imports NO GPU package. New file — `git add -f`.

Traces: "Host module `metal_beacon.py`", "A static table maps kernel ids to
entry-point names", the byte-layout and seqlock contract.

- [x] Declare the module constants: `BEACON_BYTES = 256`, `BEACON_MAGIC =
  0xB0AC0001`, `KERNEL_ID_NONE = 0`, `PHASE_ENTRY = 0`.
- [x] Add the frozen dataclass `BeaconReport(kernel_id, kernel_name, phase, trip,
  seq)` with `@dataclass(frozen=True)`.
- [x] Build `KERNEL_NAMES: dict[int, str]` as the single id → entry-point-name
  source. Reserve id 0 for `"<none>"`. Assign ids append-only from 1.
- [x] Cover `mainImage`, the 35 wavefront entry names in `vk_wavefront.py`, the
  preview entry, and the debug-raster entry. Read the wavefront names from the
  `_ENTRIES` lists (`wfPath*`, `wfSppm*`, `wfMlt*`, `wfBdpt*`, `wfNeuralProposal`,
  `wfIndirectPaint`). Do NOT invent or reorder ids.
- [x] Add `kernel_id_for(entry: str) -> int`; raise `KeyError` for an unknown
  entry.
- [x] Add `kernel_name_for(kernel_id: int) -> str`; return `"<unknown:N>"` for an
  id absent from the table; never raise.
- [x] Add `BeaconWriter(path)` with `stamp(kernel_id)` and `close()`. `stamp`
  bumps `seq`, writes `kernel_id` with `phase=0` and `trip=0`, writes `seq_check =
  seq`, then `msync`. Raise `ValueError` when `kernel_id` is not in `KERNEL_NAMES`.
  Pack the cell little-endian at the frozen offsets (magic 0, seq 4, kernel_id 8,
  phase 12, trip 16, seq_check 20).
- [x] Add `BeaconReader(path)` with `read() -> BeaconReport | None` and `close()`.
  `read` returns `None` when the file is shorter than `BEACON_BYTES`, the magic is
  wrong, or `seq != seq_check`. `read` never raises for a short, bad-magic, or torn
  cell.

## 2. `metal_trace` axis in `shader_variants.py`

Traces: shader-variant-key "A Metal-only trace axis carries the beacon session
define". Follow the `metal_neural` / `metal_records` precedent (lines 141–196), but
with the two documented divergences.

- [x] Add a boolean field `metal_trace: bool = False` on `ShaderVariantKey`.
- [x] In `__post_init__`, refuse `metal_trace` on a non-Metal target, matching the
  Metal-only rule for `metal_neural` / `metal_records`.
- [x] Do NOT restrict `metal_trace` to the wavefront family. Accept it on ALL Metal
  families: megakernel, wavefront, preview, and debug-raster.
- [x] Emit `SKINNY_METAL_TRACE` from `session_defines()` only, riding the Metal
  `base` define segment like `metal_neural` / `metal_records`.
- [x] Do NOT touch `cache_token()` or `spv_cache_key()`. A traced build is a Metal
  in-process compile with no `.spv` disk artifact.

## 3. Shader helper, gated declaration, and gated store calls

Traces: "The shader helper stores the kernel id at kernel entry", "The trace gate
keeps production shaders byte-identical". ALL beacon shader code sits behind `#if
defined(SKINNY_METAL_TRACE)`. New file `beacon.slang` — `git add -f`.

- [x] Add `beacon.slang` with `void beaconStore(uint kernelId)` (writes kernel id,
  phase 0, trip 0) and the reserved `void beaconStore(uint kernelId, uint phase,
  uint trip)`. Wrap the whole file body in `#if defined(SKINNY_METAL_TRACE)`.
- [x] Add the `gKernelBeacon` buffer declaration in `bindings.slang` under `#if
  defined(SKINNY_METAL_TRACE)`. Gate the declaration itself, not only its uses, so
  an unused Metal global never consumes an argument-table slot.
- [x] Insert `beaconStore(KERNEL_ID)` as the FIRST statement of every compute entry
  point, each guarded by `#if defined(SKINNY_METAL_TRACE)`. Cover `mainImage`, the
  35 wavefront entries, the preview entry, and the debug-raster entry.
- [x] Set each `KERNEL_ID` to the exact id the host `KERNEL_NAMES` table assigns to
  that entry. The shader id and the host id MUST match per entry.
- [x] Recompile the checked-in Metal/SPIR-V artifacts. Confirm the default (gate
  off) output is byte-unchanged. If any interface would need to change to make the
  ids match, STOP and report.

## 4. Host allocation, bind, and pre-dispatch stamp

Traces: "A shared beacon buffer records the running kernel identity", "The child
stamps the kernel id before each dispatch". All host beacon work is gated the same
way as the shader gate. `metal_compute.py` dispatch wrappers own the stamp.

- [x] Allocate one 256-byte UMA shared-storage beacon buffer on the Metal backend
  ONLY under the trace gate. The Vulkan backend never allocates or binds it.
- [x] Bind the beacon buffer at the fixed shader-global name `gKernelBeacon` on
  every Metal compute dispatch, under the trace gate only.
- [x] Call `BeaconWriter.stamp(kernel_id)` immediately BEFORE each `submit` +
  `wait_for_idle`, in the `metal_compute.py` dispatch wrappers. The stamp lands
  before the dispatch that may hang.
- [x] Resolve each dispatch's `kernel_id` from `kernel_id_for(entry)` using the
  entry name the pipeline compiled. Never hand-number an id at a call site.

## 5. Child/parent mmap wiring and additive hygiene report

Traces: "The parent polls the beacon and reports the hung kernel", "The parent
never reports a kernel the child never reached", metal-dispatch-hygiene "The
timeout path reports the hung kernel identity". Wire the beacon into the existing
child-process timeout pattern (`tests/metal_cleanup_child.py` +
`tests/test_metal_cleanup.py`).

- [x] Pass the mmap beacon-file path from parent to child. The child opens the cell
  and initializes it to `KERNEL_ID_NONE` before its first dispatch.
- [x] The child stamps each kernel id through `BeaconWriter.stamp` before its
  dispatch, per group 4.
- [x] In the parent SIGTERM timeout path, after SIGTERM and the grace wait, call
  `BeaconReader.read()`. Report `kernel_id=N (name)` from `kernel_name_for(N)`. A
  `None` read reports "no valid beacon".
- [x] Keep the SIGTERM-first order unchanged: chained handler runs `destroy()`,
  wait the grace period, escalate to SIGKILL ONLY after confirming no in-flight
  dispatch. The beacon report is additive.

## 6. Verification (verifier-owned, spec-derived)

Derive these from the spec scenarios, not from the code. Group them late. Add
hostless tests with `git add -f`.

- [x] Byte-layout golden: pin the 256-byte offsets (magic 0, seq 4, kernel_id 8,
  phase 12, trip 16, seq_check 20). Traces the beacon-buffer requirement.
- [x] Seqlock torn read → `None`: write a cell with `seq != seq_check` and assert
  `BeaconReader.read()` returns `None` and does not raise. Traces "a torn or
  uninitialized cell is not reported as a kernel".
- [x] Reader error semantics: a short file and a bad-magic cell each return `None`
  without raising. Traces the reader contract.
- [x] Phantom-id guard: `BeaconWriter.stamp` on an id absent from `KERNEL_NAMES`
  raises `ValueError`. Traces "no phantom report".
- [x] Append-only id golden: pin the full id → name mapping and assert id 0 is
  `"<none>"`. Fail on any renumber or reuse. Traces "ids are append-only".
- [x] Table coverage: assert every dispatchable entry (`mainImage`, the 35
  wavefront entries, preview, debug-raster) has a table id. Traces "every dispatched
  kernel has a table entry".
- [x] `kernel_name_for` fallback: an unknown id returns `"<unknown:N>"` and never
  raises. Traces the reader-report contract.
- [x] Variant-key goldens: extend `tests/test_shader_variants.py` so `metal_trace`
  emits `SKINNY_METAL_TRACE` into `session_defines()` only, is refused on a Vulkan
  key, is accepted on all four Metal families, and leaves `cache_token()` /
  `spv_cache_key()` unchanged. Traces the shader-variant-key requirement.
- [x] Gate-off byte-identical: assert every Vulkan `.spv` and the default Metal
  binary are byte-identical to the pre-change output. Traces "The trace gate keeps
  production shaders byte-identical".
- [x] SIGTERM-then-report (gpu-marked, one guarded Metal process): a child hangs in
  a known kernel; the parent times out, SIGTERMs, and reports that kernel by name.
  Respect the one-guarded-Metal-process rule and never SIGKILL before confirming no
  in-flight dispatch. Traces "a stalled kernel is reported by name" and "the beacon
  report does not weaken the kill order".
