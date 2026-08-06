# Design — metal-kernel-beacon

## Problem and constraint

macOS cannot cancel a committed Metal command buffer and cannot reset another
process's GPU. An infinite kernel wedges the GPU until reboot. skinny dispatches
through SlangPy and slang-rhi, which hide the `MTLCommandBuffer`. Python sees only
`create_command_encoder → begin_compute_pass → dispatch → submit_command_buffer →
wait_for_idle` (`metal_compute.py`, `metal_context.py`). No command-buffer status,
completion handler, `MTLFunctionLog`, or `errorOptions` is reachable.
`wait_for_idle()` is where an infinite kernel wedges the host — it never returns.

The design goal: after a wedge, report **which** kernel was stuck, by name,
without Xcode and without the native command-buffer path.

## Ownership and seam

**Owner.** The `MetalContext` dispatch wrapper in `metal_compute.py` owns the
beacon write on the host side. A new host module, `metal_beacon.py`, owns the
memory-mapped cell format, the reader, the writer, and the kernel-identity table.
`shader_variants.py` owns the `SKINNY_METAL_TRACE` define.

**Seam.** One memory-mapped file holds a single 256-byte beacon cell. The child
process writes the cell before each dispatch. The parent process reads the cell on
a wall-clock poll. A UMA shared-storage GPU buffer of the same 256-byte layout is
bound to every compute kernel; the GPU writes its kernel id there under the trace
gate. The two writers never conflict, because the GPU buffer confirms only the
kernel the child is currently dispatching.

## Data flow

1. The parent forks the child and passes the mmap file path.
2. The child opens the mmap and initializes the cell to "no kernel".
3. For each kernel dispatch, the child:
   a. stamps the kernel id and a bumped sequence number into the mmap cell, then
      flushes it (`msync`);
   b. binds the UMA beacon buffer and submits the dispatch;
   c. the GPU kernel, under `SKINNY_METAL_TRACE`, writes its kernel id into the
      UMA beacon buffer at its first line;
   d. calls `wait_for_idle()`. A healthy kernel returns and the child moves to
      the next kernel. An infinite kernel never returns — the mmap cell already
      holds this kernel id from step (a).
4. The parent polls the mmap cell. On a wall-clock timeout it sends SIGTERM to the
   child, waits the grace period, reads the last valid cell, and reports the
   kernel id and name.

The pre-dispatch host stamp is the load-bearing record. It lands before the
dispatch that may hang, so it survives the wedge. The GPU beacon write is the
byte-identical-gated shader half. It confirms the kernel actually started on the
device and it reserves the byte slots for the per-loop trip counts that come in a
follow-up.

## Why not the native path (discarded alternative)

The native Apple path attaches `errorOptions = .encoderExecutionStatus` to the
`MTLCommandBuffer`, then a completion handler reads `MTLFunctionLog` for the
function name and the shader `file:line`. This path is richer — it gives the
source location, not just the kernel name. It is rejected because slang-rhi hides
the `MTLCommandBuffer`. No status, completion handler, `MTLFunctionLog`, or
`errorOptions` crosses the slang-rhi boundary into Python. The native path is a
non-goal until slang-rhi exposes the raw command buffer. The beacon design needs
no raw command buffer: it uses only a shared buffer, an mmap file, and the
existing child-process timeout pattern, so it is portable, headless, and
CI-usable today.

## Frozen interfaces

The following interfaces are FROZEN once the design gate approves. Downstream
agents implement against them and do not change them.

### Beacon byte layout (256 bytes, little-endian)

The same layout describes the UMA GPU buffer and the mmap cell.

| Offset | Size | Field        | Meaning                                             |
|--------|------|--------------|-----------------------------------------------------|
| 0      | 4    | `magic`      | `u32` = `0xB0AC0001`; a reader rejects any other    |
| 4      | 4    | `seq`        | `u32` monotonic write counter; the writer bumps it  |
| 8      | 4    | `kernel_id`  | `u32` static kernel id; 0 = no kernel               |
| 12     | 4    | `phase`      | `u32` reserved; this change writes 0                |
| 16     | 4    | `trip`       | `u32` reserved trip count; this change writes 0     |
| 20     | 4    | `seq_check`  | `u32` mirror of `seq`; a reader accepts only `seq == seq_check` |
| 24     | 232  | `reserved`   | zero; room for later per-kernel fields              |

Constants:

- `BEACON_BYTES = 256`
- `BEACON_MAGIC = 0xB0AC0001`
- `KERNEL_ID_NONE = 0`
- `PHASE_ENTRY = 0`

The mmap write is a seqlock. The writer bumps `seq`, writes the payload, then
writes `seq_check = seq`. The reader reads `magic`, `seq`, the payload, and
`seq_check`; it accepts the snapshot only when `magic == BEACON_MAGIC` and
`seq == seq_check`. The seqlock guards the mmap cell against a torn read while the
child stamps.

### `SKINNY_METAL_TRACE` build-gate contract

- `shader_variants.py` gains a boolean `metal_trace` axis on `ShaderVariantKey`,
  default `False`, valid only on Metal-target keys. A Vulkan-target key with
  `metal_trace` set is refused in `__post_init__`, matching the existing rule for
  `SKINNY_METAL_NEURAL` and `SKINNY_METAL_RECORDS`.
- The axis rides the Metal `base` define segment and reaches `session_defines()`
  ONLY, exactly like `metal_neural` and `metal_records`. It emits
  `SKINNY_METAL_TRACE` into the Metal session defines, so a traced session
  compiles the beacon helper in. It does NOT touch `cache_token()` or
  `spv_cache_key()` — those name and hash the Vulkan `.spv` disk artifact, and a
  traced build is always a Metal in-process compile. There is no Metal `.spv` disk
  cache, so no on-disk collision exists to guard against.
- One real divergence from the precedent: `metal_neural` and `metal_records` are
  wavefront-only, but `metal_trace` is valid on ALL Metal families — megakernel,
  wavefront, preview, and debug-raster — because the beacon binds to every compute
  kernel. `__post_init__` does not refuse `metal_trace` on any Metal family.
- Every `beaconStore` call, the beacon helper module, AND the `gKernelBeacon`
  buffer declaration in `bindings.slang` compile only under
  `#if defined(SKINNY_METAL_TRACE)`. The host-side allocation and bind of the
  beacon buffer are gated the same way. With the gate off, the Metal binaries and
  the Vulkan SPIR-V are byte-identical to today's production output. An
  ungated-but-unused Metal global would still consume an argument-table slot, so
  the declaration itself — not only the store calls — must sit behind the gate.
- The beacon buffer binds at a fixed Metal shader-global name `gKernelBeacon`. It
  occupies an argument-table slot only under `SKINNY_METAL_TRACE`, so the
  production 128-slot budget is unchanged.

### Shader helper

Under `SKINNY_METAL_TRACE`, the beacon helper exposes:

```
// beacon.slang — compiled only under SKINNY_METAL_TRACE
void beaconStore(uint kernelId);                   // writes kernel id, phase=0, trip=0
void beaconStore(uint kernelId, uint phase, uint trip);  // reserved for the follow-up
```

Every compute entry point calls `beaconStore(KERNEL_ID)` as its first statement.
`KERNEL_ID` is the kernel's static id, taken from the kernel-identity table.

### Host module `metal_beacon.py`

```
BEACON_BYTES: int = 256
BEACON_MAGIC: int = 0xB0AC0001
KERNEL_ID_NONE: int = 0
PHASE_ENTRY: int = 0

@dataclass(frozen=True)
class BeaconReport:
    kernel_id: int
    kernel_name: str
    phase: int
    trip: int
    seq: int

class BeaconWriter:
    """Child-side owner of the mmap cell."""
    def __init__(self, path: str) -> None: ...
    def stamp(self, kernel_id: int) -> None:
        """Bump seq, write kernel_id (phase=0, trip=0), write seq_check, msync.
        Call right BEFORE submitting the dispatch.
        Raises ValueError if kernel_id is not in the kernel-identity table."""
    def close(self) -> None: ...

class BeaconReader:
    """Parent-side poller of the mmap cell."""
    def __init__(self, path: str) -> None: ...
    def read(self) -> BeaconReport | None:
        """Return the current cell, or None when the cell is smaller than
        BEACON_BYTES, the magic is wrong, or seq != seq_check (a torn read).
        Never raises for a partial or torn cell."""
    def close(self) -> None: ...

# Kernel-identity table — the single source of id <-> name.
KERNEL_NAMES: dict[int, str]        # id -> entry-point name; 0 -> "<none>"
def kernel_id_for(entry: str) -> int: ...       # raises KeyError for an unknown entry
def kernel_name_for(kernel_id: int) -> str: ... # unknown id -> "<unknown:N>", never raises
```

Error semantics:

- `BeaconReader.read()` returns `None` for a short file, a magic mismatch, or a
  torn read (`seq != seq_check`). It never raises for those.
- `BeaconWriter.stamp()` raises `ValueError` when `kernel_id` is absent from the
  table, so a caller cannot stamp a phantom id.
- `kernel_name_for()` returns `"<unknown:N>"` for an id absent from the table and
  never raises, so a report always resolves.

### Kernel-identity table contract

- `KERNEL_NAMES` is the single source that maps each integer id to its
  entry-point name. It covers the megakernel `mainImage`, the 35 wavefront entry
  names in `vk_wavefront.py`, the preview kernel, and the debug-raster kernel.
- Id 0 is reserved for `"<none>"`.
- Ids are append-only. A new kernel takes the next free id. An existing id is
  never renumbered and never reused. A pinned golden test enforces the mapping, so
  a renumber fails the build.

## Timeout path (in `metal-dispatch-hygiene`)

The parent's timeout path is unchanged in order: SIGTERM first, wait the grace
period, escalate to SIGKILL only after it confirms no in-flight dispatch. The
beacon report is additive. After SIGTERM the parent calls `BeaconReader.read()`
and reports `kernel_id=N (name)` from `kernel_name_for(N)`. A `None` read reports
"no valid beacon" rather than a kernel id.

## Safety argument (modeled in TLA+)

The parent's reported kernel id is always a kernel the child reached. The mmap
cell holds a kernel id only after the child stamps it before a dispatch. The
seqlock makes a torn read observable, and the reader maps a torn read or a magic
mismatch to `None`, so the parent never reports a partial or stale-composite id.
The child dispatches kernels sequentially, so when it hangs on kernel N the cell
holds exactly N. The TLA+ module `specs/tla/MetalKernelBeacon.tla` models this
protocol and states the safety invariant (no phantom report) and the liveness
property (a hung child is eventually reported). The file uses the TLA-valid name
`MetalKernelBeacon.tla`, because SANY requires the filename to match the module
identifier and TLA+ identifiers forbid the hyphen in the change id.
