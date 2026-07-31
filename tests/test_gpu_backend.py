"""Gates for the declared GPU backend seam (change ``gpu-backend-adapter``).

Four layers, all hostless — no ``vulkan`` extension, no Metal device:

1. **Conformance** — the three adapters expose the same surface, modulo the
   declared one-sided / divergent tables. Fails on drift from the pinned
   fixture that is not declared.
2. **No attribute-presence probes** — the two broken backend tests
   (``hasattr(ctx, "compute_queue")`` and the ``descriptor_sets is None``
   sentinel) must not come back.
3. **Capacity agreement** — the bindless texture capacity in the capability
   record, in each adapter, and compiled into ``bindings.slang`` for that
   target are one number, enforced here rather than by a source comment.
4. **Recording adapter** — dispatch ordering and binding coverage are
   assertable with no device.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest

from skinny import gpu_backend as gb

SRC = Path(__file__).resolve().parent.parent / "src" / "skinny"
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "gpu_adapter_surface.json"

#: Consumers of the seam. The adapters and `backend_select` are excluded where a
#: gate is about *reaching past* the seam; the adapters may of course name
#: themselves.
ADAPTER_FILES = {"vk_compute.py", "metal_compute.py", "recording_compute.py"}


def _surfaces() -> dict[str, dict]:
    return {name: gb.adapter_surface(mod)
            for name, mod in gb.ADAPTER_MODULES.items()}


def _one_sided_for(key: str) -> str | None:
    entry = gb.ONE_SIDED_MEMBERS.get(key)
    return entry["only"] if entry else None


def _py_sources(exclude_adapters: bool = False):
    for path in sorted(SRC.rglob("*.py")):
        if exclude_adapters and path.name in ADAPTER_FILES:
            continue
        if path.name == "gpu_backend.py":
            continue  # documents the removed probes by name, on purpose
        yield path


# ── 1. conformance ──────────────────────────────────────────────────────


def test_adapters_expose_the_same_classes_modulo_the_declared_table():
    surfaces = _surfaces()
    every = set().union(*(set(s["classes"]) for s in surfaces.values()))
    for cls in sorted(every):
        present = {n for n, s in surfaces.items() if cls in s["classes"]}
        if present == set(surfaces):
            continue
        only = _one_sided_for(cls)
        assert only is not None, (
            f"class {cls} is on {sorted(present)} only and is not declared in "
            f"gpu_backend.ONE_SIDED_MEMBERS — declare it (with the reason) or "
            f"give the other adapters one")
        assert present == {only}, (
            f"class {cls} is declared one-sided on {only!r} but is actually on "
            f"{sorted(present)}")


def test_shared_methods_agree_on_name_and_parameters():
    """A method on more than one adapter takes the same parameters, in the same
    order, under the same names — the `rgba_f32` vs `data` break."""
    surfaces = _surfaces()
    drift = []
    every = set().union(*(set(s["classes"]) for s in surfaces.values()))
    for cls in sorted(every):
        holders = {n: s["classes"][cls] for n, s in surfaces.items()
                   if cls in s["classes"]}
        methods = set().union(*(set(m) for m in holders.values()))
        for meth in sorted(methods):
            key = f"{cls}.{meth}"
            present = {n for n, m in holders.items() if meth in m}
            devices = {n for n in holders if n != "recording"}
            if present != set(holders):
                # The recording adapter executes nothing, so it is free to skip
                # a one-sided member of either device adapter; what must be
                # declared is a member the two DEVICE adapters disagree on.
                if present >= devices:
                    continue
                assert _one_sided_for(key) is not None, (
                    f"{key} is on {sorted(present)} of {sorted(holders)} and is "
                    f"not declared in ONE_SIDED_MEMBERS")
                continue
            sigs = {n: tuple(holders[n][meth]) for n in present}
            if len(set(sigs.values())) > 1:
                if key in gb.DIVERGENT_SIGNATURES:
                    continue
                drift.append((key, sigs))
    assert not drift, (
        "shared members whose signatures disagree and are not declared in "
        f"gpu_backend.DIVERGENT_SIGNATURES: {drift}")


def test_module_functions_and_constants_are_declared_when_one_sided():
    surfaces = _surfaces()
    for kind in ("functions", "constants"):
        every = set().union(*(set(s[kind]) for s in surfaces.values()))
        for name in sorted(every):
            present = {n for n, s in surfaces.items() if name in s[kind]}
            if present == set(surfaces):
                continue
            only = _one_sided_for(name)
            assert only is not None, (
                f"module {kind[:-1]} {name} is on {sorted(present)} only and is "
                f"not declared in ONE_SIDED_MEMBERS")
            # `recording_compute` is free to skip a one-sided member of either
            # device adapter — it executes nothing.
            assert present <= {only, "recording"}, (
                f"{name} is declared one-sided on {only!r} but is on "
                f"{sorted(present)}")


def test_surface_matches_the_pinned_fixture():
    """The fixture is the drift this change removed. Regenerate it deliberately
    (``python -m tests.test_gpu_backend``) — never edit it to match the code."""
    pinned = json.loads(FIXTURE.read_text())
    live = _surfaces()
    for name in ("vulkan", "metal"):
        assert live[name] == pinned[name], (
            f"the {name} adapter surface drifted from "
            f"tests/fixtures/gpu_adapter_surface.json")


def test_one_sided_and_divergent_entries_are_real():
    """Every declared exception names something that actually exists — a stale
    entry silently re-admits the drift it was meant to bound."""
    surfaces = _surfaces()

    def exists(key: str) -> bool:
        cls, _, meth = key.partition(".")
        for s in surfaces.values():
            if meth:
                if meth in s["classes"].get(cls, {}):
                    return True
            elif cls in s["classes"] or cls in s["functions"] or cls in s["constants"]:
                return True
        return False

    stale = [k for k in {**gb.ONE_SIDED_MEMBERS, **gb.DIVERGENT_SIGNATURES}
             if not exists(k)]
    assert not stale, f"declared members that no adapter has: {stale}"


# ── 2. the broken probes stay gone ──────────────────────────────────────


def test_no_attribute_presence_backend_probe_remains():
    """`MetalContext.compute_queue` is `None`, not absent, so this probe was
    unconditionally true (design D3). It must not return."""
    probe = re.compile(r"hasattr\(\s*[\w.]*ctx[\w.]*\s*,\s*[\"']compute_queue[\"']")
    offenders = [f"{p.name}:{i}" for p in _py_sources()
                 for i, line in enumerate(p.read_text().splitlines(), 1)
                 if probe.search(line.split("#", 1)[0])]
    assert not offenders, (
        f"attribute-presence backend probe reintroduced at {offenders} — read "
        f"gpu_backend.capabilities(ctx) instead")


def test_descriptor_sets_is_not_used_as_a_backend_test():
    """`descriptor_sets is None` may still ask *are the sets built yet*, but
    never *which backend is this* — the give-away is compounding it with
    `is_metal` in one expression (design D2)."""
    offenders = []
    for path in _py_sources():
        lines = path.read_text().splitlines()
        for i, line in enumerate(lines):
            window = " ".join(lines[i:i + 2])
            if "descriptor_sets is" in window and "is_metal" in window:
                offenders.append(f"{path.name}:{i + 1}")
    assert not offenders, (
        f"`descriptor_sets` compounded with `is_metal` at {offenders} — the "
        f"same fact stated twice; read caps.has_descriptor_sets")


def test_renderer_is_metal_branches_are_only_two_implementation_splits():
    """Spec scenario: every remaining `is_metal` in `renderer.py` is the adapter
    selection itself or a genuine second implementation."""
    src = (SRC / "renderer.py").read_text()
    tree = ast.parse(src)
    lines = src.splitlines()
    allowed_calls = {
        "metal_wavefront.ensure_pass",     # vs vk_wavefront.ensure_pass
        "_render_material_preview_metal",  # record-and-submit vs bind-by-name
        "_render_windowed_metal",          # swapchain frame vs surface blit
        "_render_headless_metal",
    }
    # Map each `if`/ternary that tests `is_metal` to its FULL body span. A fixed
    # line window silently mis-judges a branch whose dispatching call sits
    # further down (main's `render_headless` drains and derives a plan first),
    # which reads as a violation the moment unrelated code lands above the call.
    spans: dict[int, tuple[int, int]] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.If, ast.IfExp)):
            for sub in ast.walk(node.test):
                if isinstance(sub, ast.Attribute) and sub.attr == "is_metal":
                    end = getattr(node, "end_lineno", sub.lineno + 6)
                    prev = spans.get(sub.lineno)
                    lo, hi = (sub.lineno, end)
                    spans[sub.lineno] = (lo, max(hi, prev[1])) if prev else (lo, hi)

    live = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute) or node.attr != "is_metal":
            continue
        text = lines[node.lineno - 1]
        if "self.is_metal = " in text:
            continue  # the adapter selection itself
        lo, hi = spans.get(node.lineno, (node.lineno, node.lineno + 6))
        body = "\n".join(lines[lo - 1:hi])
        if any(name in body for name in allowed_calls):
            continue
        live.append(f"renderer.py:{node.lineno}: {text.strip()}")
    assert not live, (
        "`is_metal` branches in renderer.py that are neither adapter selection "
        f"nor a two-implementation split: {live}")


def test_no_consumer_imports_an_adapter_private():
    """Spec scenario: no reach past the seam for a backend-module private."""
    bad = re.compile(r"from\s+skinny\.(vk_compute|metal_compute|recording_compute)\s+import\s+[^\n]*\b_\w")
    offenders = [f"{p.name}:{i}" for p in _py_sources(exclude_adapters=True)
                 for i, line in enumerate(p.read_text().splitlines(), 1)
                 if bad.search(line)]
    assert not offenders, f"import of an adapter private at {offenders}"


# ── 3. the capacity is one number ───────────────────────────────────────


def _shader_capacities() -> dict[str, int]:
    """`flatMaterialTextures[N]` per target, read out of `bindings.slang`."""
    text = (SRC / "shaders" / "bindings.slang").read_text()
    metal_block = re.search(
        r"#if defined\(SKINNY_METAL\)(.*?)#endif", text, re.S)
    plain_block = re.search(
        r"#if !defined\(SKINNY_METAL\)(.*?)#endif", text, re.S)
    assert metal_block and plain_block, "bindings.slang lost its target blocks"
    decl = re.compile(r"flatMaterialTextures\[(\d+)\]")
    metal = decl.search(metal_block.group(1))
    plain = decl.search(plain_block.group(1))
    assert metal and plain, "no flatMaterialTextures declaration found"
    return {"metal": int(metal.group(1)), "vulkan": int(plain.group(1))}


@pytest.mark.parametrize("backend", ["vulkan", "metal"])
def test_bindless_capacity_matches_the_shader(backend):
    """Was a hand-maintained "MUST equal ..." comment in `bindings.slang`."""
    caps = {"vulkan": gb.VULKAN_CAPABILITIES, "metal": gb.METAL_CAPABILITIES}[backend]
    assert caps.bindless_texture_capacity == _shader_capacities()[backend]


@pytest.mark.parametrize("backend", ["vulkan", "metal", "recording"])
def test_bindless_capacity_matches_the_adapter_constant(backend):
    mod = gb.ADAPTER_MODULES[backend]
    src = (SRC.parent / (mod.replace(".", "/") + ".py")).read_text()
    tree = ast.parse(src)
    value = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "BINDLESS_TEXTURE_CAPACITY"
                for t in node.targets):
            value = node.value
    assert value is not None, f"{mod} lost BINDLESS_TEXTURE_CAPACITY"
    if isinstance(value, ast.Constant):
        assert value.value == getattr(
            gb, f"{backend.upper()}_CAPABILITIES").bindless_texture_capacity
    else:
        # `recording_compute` derives it from the record, which is the point.
        assert "RECORDING_CAPABILITIES" in ast.unparse(value)


# ── 4. capability record ────────────────────────────────────────────────


def test_capabilities_resolve_from_a_context():
    class _Vk:
        backend_name, is_metal = "vulkan", False

    class _Mt:
        backend_name, is_metal = "metal", True
        supports_shared_memory = True
        supports_indirect_dispatch = False

    # A context that declares no probe keeps the static record's values.
    assert gb.capabilities(_Vk()) == gb.VULKAN_CAPABILITIES
    mt = gb.capabilities(_Mt())
    # The device probes are folded in, so a consumer reads one named fact.
    assert mt.has_shared_in_place_write is True
    assert mt.has_indirect_dispatch is False
    assert mt.has_descriptor_sets is False


def test_capabilities_fall_back_to_is_metal_for_a_stub_context():
    class _Stub:
        is_metal = False

    assert gb.capabilities(_Stub()).name == "vulkan"


def test_the_two_backends_differ_on_every_declared_capability():
    """A capability earns its place by replacing a real branch (design D1), so
    a field the backends agree on is either dead or misnamed.

    Compared **after** the probe fold, on a fully-capable device of each kind:
    the static records are deliberately pessimistic about every device-probed
    field (codex pre-merge review, HIGH 1), so comparing the static records
    would now report the probed fields as agreeing when the devices differ.
    """
    class _FullMetal:
        backend_name, is_metal = "metal", True
        supports_shared_memory = True
        supports_indirect_dispatch = True
        supports_external_memory = False    # no CUDA interop on Metal
        supports_external_semaphore = False

    class _FullVulkan:
        backend_name, is_metal = "vulkan", False
        supports_shared_memory = False      # exported memory, not in-place UMA
        supports_indirect_dispatch = True
        supports_external_memory = True
        supports_external_semaphore = True

    metal = gb.capabilities(_FullMetal())
    vulkan = gb.capabilities(_FullVulkan())
    same = [f for f in vulkan.__dataclass_fields__
            if f != "has_indirect_dispatch"
            and getattr(vulkan, f) == getattr(metal, f)]
    assert not same, f"capabilities the two backends agree on: {same}"


# ── 5. the recording adapter ────────────────────────────────────────────


def test_recording_adapter_records_allocation_and_dispatch_order():
    from skinny import recording_compute as rec

    ctx = rec.RecordingContext(32, 32)
    buf = rec.StorageBuffer(ctx, 4096)
    img = rec.StorageImage(ctx, 32, 32)
    pipe = rec.ComputePipeline(ctx, "shaders", "main_pass", "mainImage")
    pipe.dispatch(32, 32, binds={"outputBuffer": img, "sceneBuffer": buf})
    pipe.dispatch_kernel((8, 1, 1), buffers={"outputBuffer": img})

    assert ctx.recorder.allocations() == [
        ("StorageBuffer", 4096), ("StorageImage", 0), ("ComputePipeline", 0)]
    assert ctx.recorder.dispatch_entries() == ["mainImage", "mainImage"]
    assert ctx.recorder.ops("dispatch")[0].detail["groups"] == (32, 32, 1)


def test_recording_adapter_returns_zero_filled_data_and_never_pixels():
    from skinny import recording_compute as rec

    ctx = rec.RecordingContext(4, 4)
    assert rec.StorageBuffer(ctx, 64).download_sync(16) == b"\x00" * 16
    assert not rec.StorageImage(ctx, 4, 4).read_rgba().any()


def test_recording_adapter_reports_a_missing_binding():
    """Spec scenario: the gap is caught before the GPU, not as a black image."""
    from skinny import recording_compute as rec

    ctx = rec.RecordingContext()
    pipe = rec.ComputePipeline(ctx, "shaders", "main_pass", "mainImage")
    pipe.reflect_globals({"outputBuffer", "sceneBuffer", "envMap"})
    pipe.dispatch(8, 8, binds={"outputBuffer": rec.StorageBuffer(ctx, 16)})

    assert ctx.recorder.missing_bindings() == [
        ("mainImage", "envMap"), ("mainImage", "sceneBuffer")]


def test_recording_adapter_treats_a_none_resource_as_unbound():
    """A key with no resource is NOT a binding — `metal_compute.dispatch` skips
    `None` rather than binding it, so recording it as bound would mask exactly
    the gap this adapter exists to find (codex pre-merge review, MEDIUM 2)."""
    from skinny import recording_compute as rec

    ctx = rec.RecordingContext()
    pipe = rec.ComputePipeline(ctx, "shaders", "main_pass", "mainImage")
    pipe.reflect_globals({"outputBuffer"})
    pipe.dispatch(8, 8, binds={"outputBuffer": None})

    assert ctx.recorder.missing_bindings() == [("mainImage", "outputBuffer")]


def test_recording_adapter_counts_the_non_binds_channels():
    """A resource does not have to arrive through `binds`: the uniform blob
    binds `fc`, a push block `pc`, the preview output `previewOutput`, and
    `bindless` the global it names. Ignoring those reported real bindings as
    absent (codex pre-merge review, MEDIUM 2)."""
    from skinny import recording_compute as rec

    ctx = rec.RecordingContext()
    pipe = rec.ComputePipeline(ctx, "shaders", "main_pass", "mainImage")
    pipe.reflect_globals({"fc", "matTextures"})
    pipe.dispatch(8, 8, uniform_blob=b"\x00" * 64,
                  bindless=("matTextures", []))
    assert ctx.recorder.missing_bindings() == []

    prev = rec.PreviewPipeline(ctx, "shaders")
    prev.reflect_globals({"fc", "pc", "previewOutput"})
    prev.dispatch(64, push_bytes=b"\x00" * 32, uniform_blob=b"\x00" * 64,
                  output_image=rec.StorageImage(ctx, 64, 64))
    assert ctx.recorder.missing_bindings() == []


def test_recording_adapter_reports_nothing_for_an_undeclared_pipeline():
    """No reflection declared ⇒ report nothing, rather than report everything."""
    from skinny import recording_compute as rec

    ctx = rec.RecordingContext()
    pipe = rec.ComputePipeline(ctx, "shaders", "main_pass", "mainImage")
    pipe.dispatch(8, 8, binds={})
    assert ctx.recorder.missing_bindings() == []


def test_recording_context_needs_no_device():
    """It is importable and usable with neither GPU package present."""
    import subprocess
    import sys

    code = (
        "import sys\n"
        "class _Block:\n"
        "    def find_module(self, name, path=None):\n"
        "        return self if name.split('.')[0] in ('vulkan','slangpy') else None\n"
        "    def load_module(self, name):\n"
        "        raise ImportError(name)\n"
        "sys.meta_path.insert(0, _Block())\n"
        "from skinny import recording_compute as rec\n"
        "ctx = rec.RecordingContext()\n"
        "rec.ComputePipeline(ctx, 'shaders', 'm', 'e').dispatch(4, 4)\n"
        "assert ctx.recorder.dispatch_entries() == ['e']\n"
        "assert 'vulkan' not in sys.modules and 'slangpy' not in sys.modules\n"
        "print('ok')\n"
    )
    env_src = str(SRC.parent)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, env={"PYTHONPATH": env_src, "PATH": ""})
    assert out.returncode == 0, out.stderr
    assert "ok" in out.stdout


# ── 6. codex pre-merge review fixes ─────────────────────────────────────


def test_no_device_dependent_capability_is_asserted_statically():
    """HIGH 1: a capability the device probes must not be a backend-family
    constant. Every static record states the pessimistic default, and
    `capabilities()` promotes it from the context — for BOTH backends, not just
    Metal. Asserting Vulkan external memory statically handed the CUDA publisher
    buffers that `vk_compute` had silently allocated non-exportable, deferring
    the failure to first publish."""
    from skinny.gpu_backend import (
        METAL_CAPABILITIES,
        VULKAN_CAPABILITIES,
        capabilities,
    )

    for rec in (VULKAN_CAPABILITIES, METAL_CAPABILITIES):
        assert rec.has_external_memory is False
        assert rec.has_external_semaphore is False
        assert rec.has_shared_in_place_write is False

    class _Ctx:
        backend_name = "vulkan"
        supports_external_memory = True
        supports_external_semaphore = True

    caps = capabilities(_Ctx())
    assert caps.has_external_memory and caps.has_external_semaphore

    # the graded device build can enable memory WITHOUT the semaphore
    class _MemOnly(_Ctx):
        supports_external_semaphore = False

    half = capabilities(_MemOnly())
    assert half.has_external_memory and not half.has_external_semaphore


def test_interop_handoff_refuses_by_naming_the_missing_capability():
    """HIGH 1: the CUDA protocol needs external memory AND an external timeline
    semaphore. A device with one but not the other must refuse here, naming what
    is missing, not allocate silently-unexportable buffers."""
    import pytest

    from skinny.sampling.neural_handoff import make_publisher

    class _Buf:
        class ctx:
            backend_name = "vulkan"
            supports_external_memory = True
            supports_external_semaphore = False

    with pytest.raises(NotImplementedError) as exc:
        make_publisher("interop", weights_buffer=_Buf())
    msg = str(exc.value)
    assert "external timeline semaphore" in msg
    assert "vulkan" in msg


def test_bindless_capacity_never_read_from_one_adapter_by_name():
    """HIGH 2: `build_wavefront_shade_passes` imported the Vulkan 128 while the
    Metal shader compiles 119 — the descriptor/shader extent mismatch the
    capability exists to prevent. No renderer path may import the constant."""
    import ast
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent / "src" / "skinny"
    offenders = []
    for path in sorted(root.rglob("*.py")):
        # The adapters each legitimately declare their own constant; every
        # other module must go through the capability record.
        if path.name in {"vk_compute.py", "metal_compute.py",
                         "recording_compute.py", "gpu_backend.py"}:
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            # `from skinny.vk_compute import BINDLESS_TEXTURE_CAPACITY`
            if isinstance(node, ast.ImportFrom) and node.module in {
                    "skinny.vk_compute", "skinny.metal_compute",
                    "skinny.recording_compute"}:
                if any(a.name == "BINDLESS_TEXTURE_CAPACITY" for a in node.names):
                    offenders.append(f"{path.name}:{node.lineno} (import)")
            # `gpu.BINDLESS_TEXTURE_CAPACITY` / `self._gpu.BINDLESS_...` — an
            # attribute read off the adapter MODULE. This is the form the
            # import-only check missed: it yields the right number today only
            # because `gpu` happens to be the resolved adapter, which gives the
            # capacity a second owner (codex re-review).
            if isinstance(node, ast.Attribute) and \
                    node.attr == "BINDLESS_TEXTURE_CAPACITY":
                offenders.append(f"{path.name}:{node.lineno} (attribute)")
    assert not offenders, (
        f"the bindless capacity is read from an adapter instead of the "
        f"capability record at {offenders}; use "
        f"capabilities(ctx).bindless_texture_capacity")


def test_resource_module_knows_every_declared_adapter():
    """MEDIUM 1: `gpu_backend.ADAPTER_MODULES` declared three adapters while the
    selection seam knew two vendors, so a RecordingContext silently selected
    vk_compute and the recording adapter was unreachable through a Renderer."""
    from skinny.backend_select import _RESOURCE_MODULES, resource_module
    from skinny.gpu_backend import ADAPTER_MODULES
    from skinny.recording_compute import RecordingContext

    assert set(_RESOURCE_MODULES) == set(ADAPTER_MODULES)

    mod = resource_module(RecordingContext())
    assert mod.__name__ == "skinny.recording_compute"


def test_adapters_share_one_argument_domain():
    """MEDIUM 3: parameter NAMES agreeing is not the promise — the argument
    domain is. `format` took a VkFormat int on one adapter and a neutral string
    on the other under the same name, so the old name-only surface could not
    catch it."""
    from skinny.gpu_backend import ADAPTER_MODULES, adapter_surface

    domains = {n: adapter_surface(m)["domains"] for n, m in ADAPTER_MODULES.items()}
    conflicts = []
    for member in set().union(*[set(d) for d in domains.values()]):
        present = {n: d[member] for n, d in domains.items() if member in d}
        for param in set().union(*[set(v) for v in present.values()]):
            kinds = {n: v[param] for n, v in present.items() if param in v}
            if len(set(kinds.values())) > 1:
                conflicts.append((member, param, kinds))
    assert not conflicts, f"argument-domain drift across adapters: {conflicts}"


def test_vulkan_refuses_a_raw_vkenum_across_the_seam():
    """MEDIUM 3: accepting an int is what allowed the domains to diverge. A call
    site that passes one must fail loudly on Vulkan rather than working there
    and raising on Metal."""
    import pytest

    vk_compute = pytest.importorskip("skinny.vk_compute")

    with pytest.raises(TypeError, match="VkFormat int"):
        vk_compute._vk_format_token(37)
    with pytest.raises(TypeError, match="VkSamplerAddressMode int"):
        vk_compute._vk_address_token(0)
    assert vk_compute._vk_format_token(None) is not None


def test_docs_do_not_teach_the_removed_probe_or_the_removed_name():
    """LOW 1: live documentation that still teaches `hasattr(ctx,
    "compute_queue")` or `PreviewPipelineMetal` invites the exact bug this
    change removed, and the repo's no-drift rule (CLAUDE.md) forbids it.

    Naming the probe is allowed **only** where the surrounding prose says it was
    wrong — that is the cautionary passage this change deliberately added. The
    disclaimer wraps across lines, so the check reads a small window rather than
    the single matching line.
    """
    root = Path(__file__).resolve().parent.parent
    probe = re.compile(r'hasattr\(\s*[\w.]*ctx[\w.]*\s*,\s*["\']compute_queue["\']')
    disclaimer = re.compile(
        r"never|removed|must not|unconditionally true|cautionary|was wrong",
        re.I)
    offenders = []
    for doc in sorted(root.glob("docs/*.md")) + [root / "CLAUDE.md",
                                                 root / "README.md"]:
        if not doc.exists():
            continue
        lines = doc.read_text().splitlines()
        for i, line in enumerate(lines, 1):
            if "PreviewPipelineMetal" in line:
                offenders.append(f"{doc.name}:{i} PreviewPipelineMetal")
            if probe.search(line):
                window = "\n".join(lines[max(0, i - 4):i + 3])
                if not disclaimer.search(window):
                    offenders.append(f"{doc.name}:{i} compute_queue probe")
    assert not offenders, (
        f"documentation still teaches a removed probe/name: {offenders}")


def test_docs_state_the_implemented_bindless_capacity():
    """LOW 1: `Backends.md` documented 120 Metal slots while the implementation
    compiles 119 — a doc number that disagrees with the capability is how the
    'MUST equal' comment failed in the first place."""
    root = Path(__file__).resolve().parent.parent
    backends = (root / "docs" / "Backends.md").read_text()
    cap = gb.METAL_CAPABILITIES.bindless_texture_capacity
    assert f"Texture2D[{cap}]" in backends, (
        f"docs/Backends.md does not state the implemented Metal pool size "
        f"Texture2D[{cap}]")
    assert "Texture2D[120]" not in backends


def test_implementation_map_lists_the_new_owner_modules():
    """LOW 1: a new owner module missing from the per-file map is drift the
    repo's documentation-upkeep rule explicitly forbids."""
    root = Path(__file__).resolve().parent.parent
    imap = (root / "docs" / "ImplementationMap.md").read_text()
    for mod in ("gpu_backend.py", "recording_compute.py"):
        assert mod in imap, f"docs/ImplementationMap.md does not list {mod}"


def test_no_renderer_init_path_reaches_a_member_the_adapter_lacks():
    """The reachability claim, tested — not just module routing.

    `test_resource_module_knows_every_declared_adapter` proves `resource_module`
    returns the right module; it does NOT prove a `Renderer` can be built over
    it. It could not: `Renderer.__init__` picked the shared bindless sampler
    with `not has_descriptor_sets`, so the recording adapter (bind-by-name, but
    it splits no samplers) took the Metal arm and died on a missing
    `make_sampler`. `has_descriptor_sets` had been overloaded into "is
    bind-by-name", which is the same answer on two backends and the wrong one on
    a third (fallback pre-merge review, MEDIUM 1).

    So: for every capability-gated reach at a one-sided member, the capability
    must be true only where the member exists.
    """
    for name, module in gb.ADAPTER_MODULES.items():
        surface = gb.adapter_surface(module)
        caps = gb._BY_NAME[name]
        for member, entry in gb.ONE_SIDED_MEMBERS.items():
            if "." in member:
                continue  # methods are reached through an instance, not module
            gate = _ONE_SIDED_GATES.get(member)
            if gate is None:
                continue
            present = (member in surface["functions"]
                       or member in surface["classes"]
                       or member in surface["constants"])
            if getattr(caps, gate):
                assert present, (
                    f"{name}: capability {gate} is True but {member} is absent "
                    f"— a consumer gated on it would reach a member this "
                    f"adapter does not have")


#: One-sided module members a consumer reaches through a capability, and the
#: capability that gates the reach. The gate above asserts the capability is
#: true ONLY where the member exists, so the pair cannot drift apart.
_ONE_SIDED_GATES = {
    "make_sampler": "needs_shared_bindless_sampler",
}


def test_capability_names_the_reason_not_the_binding_model():
    """`has_descriptor_sets` must not stand in for the argument-table or the
    shader-build facts it was overloaded with. Both now have their own field,
    and the three consumers read those."""
    tree = ast.parse((SRC / "renderer.py").read_text())

    def _caps_fields(node) -> set[str]:
        """Capability fields named anywhere inside an expression."""
        return {n.attr for n in ast.walk(node)
                if isinstance(n, ast.Attribute)
                and isinstance(n.value, ast.Attribute)
                and n.value.attr == "caps"}

    # Find the guard around each `_gpu.make_sampler(...)` reach by AST, not by
    # matching source text: a whitespace-exact `not in` assert passes the moment
    # the expression is reindented, which is a gate that goes vacuous silently.
    reaches = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.If, ast.IfExp)):
            continue
        body = node.body if isinstance(node.body, list) else [node.body]
        for sub in ast.walk(node):
            if (isinstance(sub, ast.Attribute) and sub.attr == "make_sampler"):
                reaches.append(_caps_fields(node.test))
                break
    assert reaches, "no guarded make_sampler reach found in renderer.py"
    for fields in reaches:
        assert "needs_shared_bindless_sampler" in fields, (
            f"a make_sampler reach is guarded by {fields or '{}'} — it must be "
            f"gated on needs_shared_bindless_sampler, the fact that names why "
            f"the member exists")
        assert "has_descriptor_sets" not in fields, (
            "a make_sampler reach is gated on the binding model again")

    # both record-drain sites read the shader-build fact
    assert (SRC / "renderer.py").read_text().count(
        "self.caps.has_merged_record_header") == 2


if __name__ == "__main__":  # regenerate the pinned surface fixture
    FIXTURE.write_text(json.dumps(
        {n: gb.adapter_surface(m) for n, m in gb.ADAPTER_MODULES.items()
         if n in ("vulkan", "metal")},
        indent=2, sort_keys=True) + "\n")
    print(f"regenerated {FIXTURE}")
