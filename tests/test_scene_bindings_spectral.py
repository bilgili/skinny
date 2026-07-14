"""Regression: `scene_bindings_only` must forward the `spectral` flag.

Wavefront mode builds the shared set-0 descriptor-set layout via
`ComputePipeline.scene_bindings_only`. When it dropped the `spectral` kwarg the
layout was built RGB (no bindings 45-51) while `--spectral` made the renderer
write those descriptors -> VUID-VkWriteDescriptorSet-dstBinding-00315 -> segfault.
Hostless: capture what the classmethod forwards to __init__.
"""
from __future__ import annotations

import pytest


@pytest.mark.parametrize("module", ["skinny.vk_compute", "skinny.metal_compute"])
def test_scene_bindings_only_forwards_spectral(module, monkeypatch):
    import importlib
    try:
        mod = importlib.import_module(module)
    except Exception as exc:  # vk_compute raises OSError w/o Vulkan SDK dylib
        pytest.skip(f"{module} unimportable here: {exc}")
    captured = {}

    def fake_init(self, *args, **kwargs):
        captured["spectral"] = kwargs.get("spectral")
        captured["compile_pipeline"] = kwargs.get("compile_pipeline")

    monkeypatch.setattr(mod.ComputePipeline, "__init__", fake_init)

    mod.ComputePipeline.scene_bindings_only(None, None, spectral=True)
    assert captured["spectral"] is True
    assert captured["compile_pipeline"] is False

    mod.ComputePipeline.scene_bindings_only(None, None)
    assert captured["spectral"] is False
