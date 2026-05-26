"""Tests for the headless render API (skinny.headless) + loader stage support."""

from __future__ import annotations

from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCENE = PROJECT_ROOT / "assets" / "cornell_box_sphere.usda"


def _have_usd() -> bool:
    try:
        import pxr  # noqa: F401
        return True
    except Exception:
        return False


needs_usd = pytest.mark.skipif(
    not _have_usd() or not SCENE.exists(),
    reason="OpenUSD (pxr) not installed or cornell_box_sphere.usda missing",
)


@needs_usd
class TestLoadSceneFromStage:
    def test_stage_matches_path(self):
        from pxr import Usd
        from skinny.usd_loader import load_scene_from_stage, load_scene_from_usd

        by_path = load_scene_from_usd(SCENE)
        stage = Usd.Stage.Open(str(SCENE))
        by_stage = load_scene_from_stage(stage)

        assert len(by_stage.instances) == len(by_path.instances)
        assert len(by_stage.materials) == len(by_path.materials)
        assert len(by_stage.lights_dir) == len(by_path.lights_dir)
