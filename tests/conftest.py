from __future__ import annotations

from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SHADER_DIR = PROJECT_ROOT / "src" / "skinny" / "shaders"
MTLX_DIR = PROJECT_ROOT / "src" / "skinny" / "mtlx" / "genslang"
HARNESS_DIR = Path(__file__).resolve().parent / "harnesses"


@pytest.fixture(scope="session")
def shader_dir():
    return SHADER_DIR


@pytest.fixture(scope="session")
def mtlx_dir():
    return MTLX_DIR


@pytest.fixture(scope="session")
def harness_dir():
    return HARNESS_DIR


@pytest.fixture
def metal_probe_device():
    """A raw slangpy Metal device for isolated struct/kernel probe tests.

    These probes (cloud-noise, flat-material / std-surface layout) don't need a
    full ``MetalContext`` — no surface, no megakernel — but they still fall under
    the repo's Metal dispatch-hygiene rule: *every* piece of GPU work must
    guarantee device teardown so a killed run can't abandon a live device + its
    committed command buffer and wedge the GPU. This fixture drains
    (``wait_for_idle``) and closes the device in its finalizer on every exit path
    — normal return, assertion failure, or error — mirroring
    ``MetalContext.destroy``. Skips when native Metal isn't available.
    """
    try:
        from skinny.backend_select import metal_available
    except OSError as exc:  # pragma: no cover — Vulkan SDK missing from dylib path
        pytest.skip(f"needs the Vulkan SDK on the dylib path: {exc}")
    ok, reason = metal_available()
    if not ok:
        pytest.skip(f"native Metal unavailable: {reason}")
    spy = pytest.importorskip("slangpy")
    dev = spy.create_device(type=spy.DeviceType.metal)
    try:
        yield dev
    finally:
        try:
            dev.wait_for_idle()
        except Exception:  # noqa: BLE001 — best-effort drain before teardown
            pass
        try:
            dev.close()
        except Exception:  # noqa: BLE001 — device may already be torn down
            pass


@pytest.fixture(scope="session")
def device(shader_dir, mtlx_dir, harness_dir):
    spy = pytest.importorskip("slangpy")
    try:
        return spy.create_device(
            include_paths=[str(shader_dir), str(mtlx_dir), str(harness_dir)]
        )
    except Exception:
        pytest.skip("No Vulkan device available")


@pytest.fixture(scope="session")
def load_shader(device):
    spy = pytest.importorskip("slangpy")
    _cache: dict[str, object] = {}

    def _load(filename: str):
        if filename not in _cache:
            _cache[filename] = spy.Module.load_from_file(device, filename)
        return _cache[filename]

    return _load


@pytest.fixture(scope="session")
def load_source(device):
    spy = pytest.importorskip("slangpy")
    _cache: dict[str, object] = {}

    def _load(name: str, source: str):
        if name not in _cache:
            _cache[name] = spy.Module(device.load_module_from_source(name, source))
        return _cache[name]

    return _load


@pytest.fixture(scope="session")
def common_harness(load_shader):
    return load_shader("test_common_harness.slang")


@pytest.fixture(scope="session")
def sampler_harness(load_shader):
    return load_shader("test_sampler_harness.slang")


@pytest.fixture(scope="session")
def light_harness(load_shader):
    return load_shader("test_light_harness.slang")


@pytest.fixture(scope="session")
def skin_harness(load_shader):
    return load_shader("test_skin_harness.slang")


@pytest.fixture(scope="session")
def volume_harness(load_shader):
    return load_shader("test_volume_harness.slang")


@pytest.fixture(scope="session")
def restir_harness(load_shader):
    return load_shader("test_restir_harness.slang")


@pytest.fixture(scope="session")
def neural_flow_harness(load_shader):
    return load_shader("test_neural_flow_harness.slang")
