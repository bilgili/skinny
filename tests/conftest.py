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
