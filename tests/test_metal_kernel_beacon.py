"""Spec-derived verification for change ``metal-kernel-beacon``.

Every test traces to a ``#### Scenario`` in
``openspec/changes/metal-kernel-beacon/specs/**`` or to a frozen interface in
``design.md`` — never to the implementation. Green means "matches the spec".

Hostless tests (default ``pytest``) exercise the device-free host module
``skinny.metal_beacon`` and the ``skinny.shader_variants`` trace axis. The one
gpu-marked test drives the ``beacon`` child mode of ``metal_cleanup_child.py``
under the guarded Metal runner (one process at a time, SIGTERM never SIGKILL).
"""

from __future__ import annotations

import inspect
import os
import platform
import shutil
import signal
import struct
import subprocess
import sys
import time
from pathlib import Path

import pytest

from skinny.metal_beacon import (
    BEACON_BYTES,
    BEACON_MAGIC,
    KERNEL_ID_NONE,
    KERNEL_NAMES,
    BeaconReader,
    BeaconWriter,
    kernel_id_for,
    kernel_name_for,
)
from skinny.shader_variants import Family, ShaderVariantKey, Target

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SHADERS = _REPO_ROOT / "src" / "skinny" / "shaders"
_GENSLANG = _REPO_ROOT / "src" / "skinny" / "mtlx" / "genslang"
_CHILD = Path(__file__).resolve().parent / "metal_cleanup_child.py"
_ON_APPLE_SILICON = sys.platform == "darwin" and platform.machine() == "arm64"


# ── The append-only kernel-identity golden (design.md "Kernel-identity table")
# Pinned from the frozen contract: id 0 = "<none>", then mainImage, the 35
# dispatched wavefront entries (path/sppm/mlt/bdpt/neural/indirect), preview,
# and the five debug-raster kernels — append-only ids 1..42. A renumber or a
# reuse changes this dict and fails the build.
GOLDEN_KERNEL_NAMES: dict[int, str] = {
    0: "<none>",
    1: "mainImage",
    2: "wfPathGenerate",
    3: "wfPathIntersect",
    4: "wfBuildArgs",
    5: "wfScatter",
    6: "wfPathShadeFlat",
    7: "wfPathResolve",
    8: "wfPathShade",
    9: "wfSppmEye",
    10: "wfSppmGridCount",
    11: "wfSppmGridScanBlock",
    12: "wfSppmGridScanBlockSums",
    13: "wfSppmGridScanAdd",
    14: "wfSppmGridScatter",
    15: "wfSppmPhotonTrace",
    16: "wfSppmUpdate",
    17: "wfMltBootstrap",
    18: "wfMltInit",
    19: "wfMltMutate",
    20: "wfMltResolve",
    21: "wfBdptClassify",
    22: "wfBdptBuildArgs",
    23: "wfBdptScatter",
    24: "wfBdptConnectNee",
    25: "wfBdptConnectFull",
    26: "wfBdptResolve",
    27: "wfBdptGenEye",
    28: "wfBdptWalkClassify",
    29: "wfBdptBounceEye",
    30: "wfBdptWalk",
    31: "wfBdptLightTail",
    32: "wfBdptGenLight",
    33: "wfBdptBounceLight",
    34: "wfBdptSplat",
    35: "wfNeuralProposal",
    36: "wfIndirectPaint",
    37: "previewMain",
    38: "clearImage",
    39: "clearDepth",
    40: "depthLines",
    41: "colorLines",
    42: "blendTris",
}


# ── 1. Beacon byte-layout golden ─────────────────────────────────────
# Scenario: the beacon-buffer requirement — a fixed 256-byte layout with a
# magic word, seq, kernel-id, reserved phase/trip, and a seq mirror.


def test_beacon_byte_layout_golden(tmp_path):
    assert BEACON_BYTES == 256
    assert BEACON_MAGIC == 0xB0AC0001
    path = str(tmp_path / "cell")
    writer = BeaconWriter(path)
    try:
        writer.stamp(kernel_id_for("wfPathShade"))  # id 8, first stamp → seq 1
    finally:
        writer.close()
    raw = Path(path).read_bytes()
    assert len(raw) == 256
    # Frozen field offsets (design.md byte-layout table), little-endian u32.
    assert struct.unpack_from("<I", raw, 0)[0] == BEACON_MAGIC      # magic @0
    assert struct.unpack_from("<I", raw, 4)[0] == 1                 # seq @4
    assert struct.unpack_from("<I", raw, 8)[0] == 8                 # kernel_id @8
    assert struct.unpack_from("<I", raw, 12)[0] == 0               # phase @12
    assert struct.unpack_from("<I", raw, 16)[0] == 0               # trip @16
    assert struct.unpack_from("<I", raw, 20)[0] == 1               # seq_check @20


# ── 2. Seqlock torn read → None; and reader error semantics ──────────
# Scenario "a torn or uninitialized cell is not reported as a kernel" +
# the BeaconReader contract (short file / bad magic → None, never raises).


def _write_raw(path: str, magic: int, seq: int, kernel_id: int, seq_check: int) -> None:
    buf = bytearray(BEACON_BYTES)
    struct.pack_into("<II", buf, 0, magic, seq)
    struct.pack_into("<I", buf, 8, kernel_id)
    struct.pack_into("<I", buf, 20, seq_check)
    Path(path).write_bytes(bytes(buf))


def test_torn_read_returns_none(tmp_path):
    path = str(tmp_path / "torn")
    # A valid magic but seq (7) != seq_check (8): observably torn.
    _write_raw(path, BEACON_MAGIC, 7, 8, 8)
    assert BeaconReader(path).read() is None


def test_bad_magic_returns_none(tmp_path):
    path = str(tmp_path / "badmagic")
    _write_raw(path, 0xDEADBEEF, 3, 8, 3)
    assert BeaconReader(path).read() is None


def test_short_file_returns_none(tmp_path):
    path = str(tmp_path / "short")
    Path(path).write_bytes(b"\x00" * (BEACON_BYTES - 1))
    assert BeaconReader(path).read() is None


def test_missing_file_returns_none_and_never_raises(tmp_path):
    # A cell the child has not created yet: read must not raise.
    assert BeaconReader(str(tmp_path / "nope")).read() is None


# ── 3. Reader/writer round-trip ──────────────────────────────────────
# Scenario "a running kernel writes its id to the beacon" (host mirror):
# stamp(id) then read() returns a BeaconReport carrying that id + name.


def test_writer_reader_round_trip(tmp_path):
    path = str(tmp_path / "cell")
    writer = BeaconWriter(path)
    try:
        writer.stamp(kernel_id_for("wfSppmPhotonTrace"))  # id 15
    finally:
        writer.close()
    report = BeaconReader(path).read()
    assert report is not None
    assert report.kernel_id == 15
    assert report.kernel_name == "wfSppmPhotonTrace"
    assert report.phase == 0 and report.trip == 0
    assert report.seq == 1


def test_stamp_bumps_sequence_monotonically(tmp_path):
    path = str(tmp_path / "cell")
    writer = BeaconWriter(path)
    try:
        writer.stamp(kernel_id_for("mainImage"))
        writer.stamp(kernel_id_for("wfPathShade"))
    finally:
        writer.close()
    report = BeaconReader(path).read()
    assert report is not None
    assert report.kernel_id == 8 and report.kernel_name == "wfPathShade"
    assert report.seq == 2  # two stamps → seq advanced


# ── 4. Phantom-id guard ──────────────────────────────────────────────
# Scenario "no phantom report": stamp() refuses an id absent from the table.


def test_stamp_phantom_id_raises_value_error(tmp_path):
    writer = BeaconWriter(str(tmp_path / "cell"))
    try:
        with pytest.raises(ValueError):
            writer.stamp(9999)  # not a kernel the child could ever reach
    finally:
        writer.close()


def test_stamp_accepts_the_reserved_none_id(tmp_path):
    # The child initializes the cell to KERNEL_ID_NONE — it is a valid table id.
    writer = BeaconWriter(str(tmp_path / "cell"))
    try:
        writer.stamp(KERNEL_ID_NONE)  # must not raise
    finally:
        writer.close()


# ── 5. Name/id table fallbacks ───────────────────────────────────────
# design.md error semantics: kernel_name_for never raises (unknown → sentinel);
# kernel_id_for raises KeyError for an unknown entry.


def test_kernel_name_for_unknown_id_returns_sentinel():
    assert kernel_name_for(9999) == "<unknown:9999>"
    assert kernel_name_for(-1) == "<unknown:-1>"  # never raises


def test_kernel_name_for_known_ids_resolve():
    assert kernel_name_for(0) == "<none>"
    assert kernel_name_for(1) == "mainImage"


def test_kernel_id_for_unknown_entry_raises_key_error():
    with pytest.raises(KeyError):
        kernel_id_for("wfNoSuchKernel")


# ── 6. Append-only golden + table coverage ───────────────────────────
# Scenario "ids are append-only" + "every dispatched kernel has a table entry".


def test_kernel_names_match_append_only_golden():
    assert KERNEL_NAMES == GOLDEN_KERNEL_NAMES
    assert KERNEL_NAMES[0] == "<none>"
    # Ids are dense and append-only from 0 — no gap, no reuse.
    assert sorted(KERNEL_NAMES) == list(range(len(KERNEL_NAMES)))


def test_id_to_name_is_a_bijection():
    # A reuse would collapse two ids onto one name; a rename collides too.
    names = list(KERNEL_NAMES.values())
    assert len(names) == len(set(names)), "duplicate kernel name = a reused/renamed id"


def _wavefront_entries_in_source() -> set[str]:
    """The dispatched ``wf*`` entry names, read from their single owner
    ``wavefront_driver.KERNEL_ENTRY_NAMES`` (change choice-table-wavefront-owners).
    The names used to be string literals grepped from ``vk_wavefront.py``; they are
    now ``WF_*`` constants owned by the driver and imported by both backends, so the
    beacon table's coverage is checked against the owner, not a source scrape."""
    from skinny import wavefront_driver

    return set(wavefront_driver.KERNEL_ENTRY_NAMES)


def test_table_covers_every_dispatchable_kernel():
    covered = set(KERNEL_NAMES.values())
    wf = _wavefront_entries_in_source()
    assert len(wf) == 35, f"expected 35 dispatched wavefront entries, found {len(wf)}"
    missing = wf - covered
    assert not missing, f"wavefront kernels absent from the table: {sorted(missing)}"
    # Megakernel, preview, and the five debug-raster kernels (design coverage).
    for entry in ("mainImage", "previewMain",
                  "clearImage", "clearDepth", "depthLines", "colorLines", "blendTris"):
        assert entry in covered, f"{entry} missing from KERNEL_NAMES"


# ── 7. metal_trace variant-key goldens ───────────────────────────────
# shader-variant-key spec: a Metal-only axis emitting SKINNY_METAL_TRACE into
# session_defines() only, refused on Vulkan, valid on ALL Metal families, and
# leaving cache_token()/spv_cache_key() untouched.

_METAL_FAMILIES = (Family.MEGAKERNEL, Family.WAVEFRONT, Family.PREVIEW, Family.DEBUG_RASTER)


def test_metal_trace_defaults_off():
    key = ShaderVariantKey(Target.METAL, Family.MEGAKERNEL)
    assert key.metal_trace is False
    assert "SKINNY_METAL_TRACE" not in key.session_defines()


@pytest.mark.parametrize("family", _METAL_FAMILIES)
def test_metal_trace_accepted_and_emitted_on_every_metal_family(family):
    key = ShaderVariantKey(Target.METAL, family, metal_trace=True)
    assert key.session_defines().get("SKINNY_METAL_TRACE") == "1"


def test_metal_trace_refused_on_vulkan():
    with pytest.raises(ValueError):
        ShaderVariantKey(Target.VULKAN, Family.MEGAKERNEL, metal_trace=True)


def test_metal_trace_does_not_change_cache_token():
    off = ShaderVariantKey(Target.METAL, Family.MEGAKERNEL)
    on = ShaderVariantKey(Target.METAL, Family.MEGAKERNEL, metal_trace=True)
    assert on.cache_token() == off.cache_token()  # both "" — no .spv artifact tag


def test_metal_trace_is_not_a_vulkan_define():
    # It rides the Metal-only set; no Vulkan key can carry it, so it never
    # reaches a slangc flag tuple / spv_cache_key input.
    from skinny.shader_variants import METAL_ONLY_DEFINES
    assert "SKINNY_METAL_TRACE" in METAL_ONLY_DEFINES


# ── 8. Gate-off byte-identity (SPIR-V, slangc, no GPU) ───────────────
# Scenario "The trace gate keeps production shaders byte-identical": with the
# gate off the megakernel SPIR-V carries no beacon buffer; only the gate flips
# it in. The exact 10 MB size is generated_materials-dependent, so we assert the
# portable invariant (determinism + no beacon symbol off + present on) rather
# than a scene-specific hash.

_DEFAULT_GENERATED_MATERIALS = (_REPO_ROOT / "src" / "skinny" / "shaders"
                                / "generated_materials.slang")


def _resolve_slangc() -> str | None:
    exe = shutil.which("slangc")
    if exe:
        return exe
    sdk = os.environ.get("VULKAN_SDK")
    if sdk and (Path(sdk) / "bin" / "slangc").exists():
        return str(Path(sdk) / "bin" / "slangc")
    return None


def _find_generated_materials() -> Path | None:
    """The megakernel imports ``generated_materials`` — a per-scene stub written
    at render time (untracked, ``.gitignore`` is ``*``). Locate one so the
    compile resolves the import; None if no render has produced it anywhere."""
    if _DEFAULT_GENERATED_MATERIALS.exists():
        return _DEFAULT_GENERATED_MATERIALS
    # Fall back to the primary checkout that shares this worktree.
    try:
        common = subprocess.check_output(
            ["git", "rev-parse", "--git-common-dir"], cwd=str(_REPO_ROOT), text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    primary = Path(common).resolve().parent / "src" / "skinny" / "shaders" / "generated_materials.slang"
    return primary if primary.exists() else None


def _compile_megakernel_spv(out: Path, gen_dir: Path, *, trace: bool) -> None:
    slangc = _resolve_slangc()
    assert slangc is not None
    cmd = [slangc, str(_SHADERS / "main_pass.slang"), "-target", "spirv",
           "-entry", "mainImage", "-stage", "compute", "-o", str(out),
           "-I", str(gen_dir), "-I", str(_SHADERS), "-I", str(_GENSLANG)]
    if trace:
        cmd += ["-DSKINNY_METAL_TRACE=1"]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300.0)
    assert proc.returncode == 0, f"slangc failed:\n{proc.stderr}"


def test_gate_off_megakernel_spirv_carries_no_beacon(tmp_path):
    if _resolve_slangc() is None:
        pytest.skip("slangc not on PATH / $VULKAN_SDK/bin — cannot compile SPIR-V")
    gen = _find_generated_materials()
    if gen is None:
        pytest.skip("no generated_materials.slang — run a render once to produce it")
    gen_dir = tmp_path / "gen"
    gen_dir.mkdir()
    shutil.copyfile(gen, gen_dir / "generated_materials.slang")

    off1 = tmp_path / "off1.spv"
    off2 = tmp_path / "off2.spv"
    on = tmp_path / "on.spv"
    _compile_megakernel_spv(off1, gen_dir, trace=False)
    _compile_megakernel_spv(off2, gen_dir, trace=False)
    _compile_megakernel_spv(on, gen_dir, trace=True)

    off_bytes = off1.read_bytes()
    # (a) The gate-off compile is deterministic — no beacon-induced churn.
    assert off_bytes == off2.read_bytes(), "gate-off SPIR-V is not reproducible"
    # (b) Production (gate off) carries NO beacon buffer binding.
    assert b"gKernelBeacon" not in off_bytes
    # (c) The gate is real: turning it on is the only thing that adds the beacon.
    on_bytes = on.read_bytes()
    assert b"gKernelBeacon" in on_bytes
    assert on_bytes != off_bytes


# ── 9. SIGTERM-then-report (gpu-marked, one guarded Metal process) ───
# Scenario "a stalled kernel is reported by name" + metal-dispatch-hygiene
# "the beacon report does not weaken the kill order": the child stamps a known
# kernel id before a bounded dispatch; the parent SIGTERMs, reads the cell, and
# reports that kernel by name — SIGTERM first, never SIGKILL-first.


@pytest.mark.gpu
@pytest.mark.skipif(not _ON_APPLE_SILICON, reason="beacon report needs Apple-Silicon Metal")
def test_sigterm_then_report_names_the_stamped_kernel(tmp_path):
    cell = tmp_path / "beacon.cell"
    sentinel = tmp_path / "frame1.flag"
    entry = "wfPathShade"  # a distinctive non-megakernel id (8) to prove name resolution
    env = os.environ.copy()  # inherits VULKAN_SDK / DYLD_LIBRARY_PATH
    env["PYTHONPATH"] = str(_REPO_ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    child = subprocess.Popen(
        [sys.executable, str(_CHILD), "beacon", str(cell), str(sentinel), entry],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, cwd=str(_REPO_ROOT),
    )
    try:
        deadline = time.monotonic() + 90.0
        while not sentinel.exists():
            assert child.poll() is None, (
                "beacon child died before frame 1: "
                f"{child.stderr.read().decode(errors='replace') if child.stderr else ''}"
            )
            assert time.monotonic() < deadline, "child never wrote the frame-1 sentinel"
            time.sleep(0.1)
        # Kill order (unchanged): SIGTERM first — the chained handler runs
        # destroy(); the child's bounded queue drains and it dies by SIGTERM.
        # No SIGKILL is sent (the child holds no in-flight dispatch once it
        # returns from the bounded wait), preserving the never-SIGKILL-first rule.
        child.send_signal(signal.SIGTERM)
        child.wait(timeout=60.0)
    finally:
        if child.poll() is None:  # safety net only — never the tested path
            child.send_signal(signal.SIGTERM)
            child.wait(timeout=30.0)
    assert child.returncode == -signal.SIGTERM, (
        f"expected SIGTERM death, got rc={child.returncode}\n"
        f"stderr: {child.stderr.read().decode(errors='replace') if child.stderr else ''}"
    )

    # Parent report path (frozen 2-liner from design.md "Timeout path").
    report = BeaconReader(str(cell)).read()
    assert report is not None, "no valid beacon after SIGTERM"
    assert report.kernel_id == kernel_id_for(entry)
    assert report.kernel_name == entry
    text = (f"kernel_id={report.kernel_id} ({report.kernel_name})"
            if report is not None else "no valid beacon")
    assert text == "kernel_id=8 (wfPathShade)"


# ══════════════════════════════════════════════════════════════════════
# change beacon-wavefront-attribution — verifier (task group 5)
# ══════════════════════════════════════════════════════════════════════

# ── 10. Trace-gated per-kernel submit in MetalFrameEncoder (hostless) ─
# design.md frozen contract "MetalFrameEncoder.dispatch / .dispatch_indirect —
# trace-gated submit granularity":
#   - trace False → returns after cpass.end() WITHOUT submitting (batched).
#   - trace True  → runs the flush() body (submit + drain + reopen), and SHALL
#     NOT set _submitted, so a same-frame later dispatch still encodes and the
#     frame-end submit() stays valid.
# Scenarios "trace on submits one wavefront kernel at a time" / "trace off keeps
# the batched single submit" (metal-kernel-beacon + metal-dispatch-hygiene).
# Driven with a device-free fake ctx/encoder so the FROZEN behavioral contract is
# asserted without a GPU — the crux GPU proof is test 12 below.

from types import SimpleNamespace  # noqa: E402


class _FakeCpass:
    def bind_pipeline(self, *a):
        pass

    def dispatch_compute(self, *a):
        pass

    def dispatch_compute_indirect(self, *a):
        pass

    def end(self):
        pass


class _FakeEncoder:
    def begin_compute_pass(self):
        return _FakeCpass()


def _make_fake_frame_encoder(trace: bool):
    """A ``MetalFrameEncoder`` with every device call stubbed, so only the
    trace-gated submit-granularity branch is exercised."""
    from skinny.metal_compute import MetalFrameEncoder

    enc = MetalFrameEncoder.__new__(MetalFrameEncoder)
    enc.ctx = SimpleNamespace(trace=trace, beacon_stamp=lambda entry: None)
    enc._spy = SimpleNamespace(
        math=SimpleNamespace(uint3=lambda *a: None),
        BufferOffsetPair=lambda *a: None,
    )
    enc._enc = _FakeEncoder()
    enc._submitted = False
    enc._root = lambda *a, **k: object()  # bypass create_root_shader_object
    calls: list[str] = []
    # The trace branch MUST reuse flush() (submit + drain + reopen), NOT submit()
    # (design "Reuse the flush() mechanism, not submit()"); record which fires.
    enc.flush = lambda: calls.append("flush")
    enc.submit = lambda: calls.append("submit")
    return enc, calls


def _drive_dispatch(enc, method: str) -> None:
    pipe = SimpleNamespace(entry_point="wfPathShade", entry="wfPathShade",
                           pipeline=object())
    if method == "dispatch":
        enc.dispatch(pipe, [1, 1, 1], bindings=None)
    else:
        enc.dispatch_indirect(pipe, object(), 0, bindings=None)


@pytest.mark.parametrize("method", ["dispatch", "dispatch_indirect"])
def test_trace_on_submits_one_kernel_per_command_buffer(method):
    # Scenario "trace on submits one wavefront kernel at a time": each encoded
    # kernel is submitted + drained via flush() before the next is encoded.
    enc, calls = _make_fake_frame_encoder(trace=True)
    _drive_dispatch(enc, method)
    assert calls == ["flush"], "trace-on dispatch must per-kernel submit via flush()"
    # Reuse flush(), not submit(): _submitted stays clear so the frame-end
    # submit() is not skipped and a later same-frame dispatch still encodes.
    assert enc._submitted is False


@pytest.mark.parametrize("method", ["dispatch", "dispatch_indirect"])
def test_trace_off_keeps_batched_single_submit(method):
    # Scenario "trace off keeps the batched single submit": the dispatch encodes
    # only — no submit, no drain — so the frame accumulates into one command
    # buffer that submit() drains once at frame end.
    enc, calls = _make_fake_frame_encoder(trace=False)
    _drive_dispatch(enc, method)
    assert calls == [], "trace-off dispatch must not submit or drain per kernel"
    assert enc._submitted is False


# ── 11. destroy() releases the beacon buffer, idempotently (hostless) ─
# design.md "MetalContext.destroy — beacon-buffer release contract" +
# scenario "destroy clears the beacon buffer": release _beacon_buffer beside
# _beacon_writer, set the handle None, safe when None, safe on a second call.


class _FakeBeaconBuffer:
    def __init__(self):
        self.destroyed = 0

    def destroy(self):
        self.destroyed += 1


def _make_stub_context(*, beacon_buffer):
    """A ``MetalContext`` shell with only the fields ``destroy()`` touches —
    no GPU device (``device=None`` skips the drain/close branch)."""
    from skinny.metal_context import MetalContext

    ctx = MetalContext.__new__(MetalContext)
    ctx._destroyed = False
    ctx._beacon_writer = None
    ctx._beacon_buffer = beacon_buffer
    ctx.device = None
    ctx.surface = None
    ctx.swapchain_info = None
    return ctx


def test_destroy_releases_and_clears_the_beacon_buffer():
    buf = _FakeBeaconBuffer()
    ctx = _make_stub_context(beacon_buffer=buf)
    ctx.destroy()
    assert buf.destroyed == 1, "destroy() must release the beacon buffer"
    assert ctx._beacon_buffer is None, "destroy() must clear the beacon handle"
    # A second destroy() is a safe no-op — it does not re-release the buffer.
    ctx.destroy()
    assert buf.destroyed == 1
    assert ctx._beacon_buffer is None


def test_destroy_is_none_safe_when_no_beacon_buffer():
    # Trace off, or trace on but no dispatch ran, so beacon_native was never
    # touched: the handle is None and destroy() must not raise.
    ctx = _make_stub_context(beacon_buffer=None)
    ctx.destroy()  # must not raise
    assert ctx._beacon_buffer is None
    ctx.destroy()  # idempotent no-op
    assert ctx._beacon_buffer is None


# ── 12. Seqlock quiescence comment (hostless) ────────────────────────
# design.md "BeaconWriter seqlock comment — documentation contract" + scenario
# "the seqlock comment records the straddle limit": the comment states the guard
# catches gross tearing (not a payload straddle) and rests on writer quiescence
# at read time.


def test_seqlock_comment_records_straddle_limit_and_quiescence():
    src = inspect.getsource(BeaconWriter.stamp).lower()
    # Gross-tearing-only, not a payload straddle.
    assert "straddle" in src, "seqlock comment must name the payload straddle limit"
    assert "gross" in src or "tearing" in src, (
        "seqlock comment must state it catches gross tearing only"
    )
    # Correctness rests on writer quiescence at read time.
    assert "quiescen" in src, "seqlock comment must state the quiescence assumption"
    # The quiescence mechanism: the parent reads only after it SIGTERMs the child.
    assert "sigterm" in src or "wait_for_idle" in src, (
        "seqlock comment must tie quiescence to the post-SIGTERM read"
    )


# ── 13. Seqlock primitive rejects a straddled/uninitialized cell ─────
# The documented reader invariant at the primitive level: accept only when
# magic is valid AND seq == seq_check; a straddled or uninitialized cell → None.
# (Re-uses the shipped primitive to assert the invariant the comment states.)


def test_seqlock_reader_accepts_only_committed_cell(tmp_path):
    path = str(tmp_path / "cell")
    # A committed cell (seq == seq_check, valid magic) reads back the kernel.
    _write_raw(path, BEACON_MAGIC, 5, kernel_id_for("wfPathShade"), 5)
    report = BeaconReader(path).read()
    assert report is not None and report.kernel_id == kernel_id_for("wfPathShade")
    # A straddled cell (seq advanced past seq_check, mid-write) → None.
    _write_raw(path, BEACON_MAGIC, 6, kernel_id_for("wfPathShade"), 5)
    assert BeaconReader(path).read() is None
    # An uninitialized cell (magic 0) → None.
    _write_raw(path, 0, 0, 0, 0)
    assert BeaconReader(path).read() is None


# ── 14. Traced wavefront wedge names the in-flight stage (gpu-marked) ─
# The crux (metal-kernel-beacon "a wedged wavefront kernel is named under trace";
# metal-dispatch-hygiene "a traced wavefront wedge isolates one kernel"). Drives
# the ACTUAL trace-gated MetalFrameEncoder per-kernel-submit path (NOT the sync
# harness of test 9). A LATER-encoded stage (wfPathShade) hangs bounded; under
# per-kernel submit the child blocks in its drain and never encodes the still-
# later stage (wfPathResolve). The parent SIGTERMs and reads the in-flight stage.
# This FAILS against the old batched behavior, which would stamp wfPathResolve
# (the last-encoded kernel of the in-flight batch) over the hung wfPathShade.


@pytest.mark.gpu
@pytest.mark.skipif(not _ON_APPLE_SILICON, reason="beacon report needs Apple-Silicon Metal")
def test_traced_wavefront_wedge_names_the_inflight_stage(tmp_path):
    cell = tmp_path / "beacon.cell"
    sentinel = tmp_path / "stageA.flag"
    env = os.environ.copy()  # inherits VULKAN_SDK / DYLD_LIBRARY_PATH
    env["PYTHONPATH"] = str(_REPO_ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    # Turning trace on = pointing SKINNY_METAL_TRACE at the cell. This is what
    # flips ctx.trace and makes MetalFrameEncoder per-kernel-submit.
    env["SKINNY_METAL_TRACE"] = str(cell)
    child = subprocess.Popen(
        [sys.executable, str(_CHILD), "beacon_wavefront", str(sentinel)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, cwd=str(_REPO_ROOT),
    )
    try:
        deadline = time.monotonic() + 120.0
        while not sentinel.exists():
            assert child.poll() is None, (
                "wavefront-beacon child died before stage A drained: "
                f"{child.stderr.read().decode(errors='replace') if child.stderr else ''}"
            )
            assert time.monotonic() < deadline, "child never wrote the stage-A sentinel"
            time.sleep(0.1)
        # Stage A drained; stage B (wfPathShade) is now the in-flight kernel and
        # the child is blocked in its bounded drain. SIGTERM while B is in flight —
        # SIGTERM first, never SIGKILL-first (the bounded loop self-terminates and
        # the chained handler runs destroy()).
        child.send_signal(signal.SIGTERM)
        child.wait(timeout=90.0)
    finally:
        if child.poll() is None:  # safety net only — never the tested path
            child.send_signal(signal.SIGTERM)
            child.wait(timeout=60.0)

    report = BeaconReader(str(cell)).read()
    assert report is not None, (
        "no valid beacon after SIGTERM: "
        f"{child.stderr.read().decode(errors='replace') if child.stderr else ''}"
    )
    # The in-flight, hung stage — NOT the still-later stage encoded after it.
    assert report.kernel_name == "wfPathShade", (
        f"expected the in-flight stage wfPathShade, got {report.kernel_name!r} "
        f"(id {report.kernel_id}); wfPathResolve here = the old batched "
        f"last-encoded misattribution the fix removes"
    )
    assert report.kernel_id == kernel_id_for("wfPathShade")
    assert report.kernel_name != "wfPathResolve", (
        "reported a later-encoded stage — the batched misattribution bug"
    )
