"""Hostless tests for `MaterialReloader` under render-thread ownership.

The Python Material Editor dock no longer holds a GUI-thread render lock: its
reload runs on the render worker (single owner of the renderer), so the reloader
must accept *no* lock and still perform its pipeline-rebuild region.
"""
from __future__ import annotations

import skinny.material_reloader as mr


def _fake_renderer(calls: list[str]):
    class FakeRenderer:
        _material_version = 0

        def _build_pipeline_for_current_graphs(self) -> None:
            calls.append("rebuilt")

    return FakeRenderer()


def test_material_reloader_runs_without_a_render_lock(tmp_path, monkeypatch) -> None:
    src = tmp_path / "python_materials" / "demo_material.py"
    src.parent.mkdir(parents=True)
    src.write_text("# original\n", encoding="utf-8")

    monkeypatch.setattr(
        mr, "resolve_source_path",
        lambda name: src if name == "python_materials.demo_material" else None,
    )
    # slangpile only touches on-disk .slang output + import machinery; stub it so
    # the test isolates the locked pipeline-rebuild region.
    monkeypatch.setattr(mr.MaterialReloader, "_run_slangpile", lambda self, name: None)

    calls: list[str] = []
    renderer = _fake_renderer(calls)

    reloader = mr.MaterialReloader(renderer)  # no render_lock argument

    result = reloader.reload("python_materials.demo_material", "# edited\n")

    assert result.ok
    assert result.stage == "done"
    assert calls == ["rebuilt"]
    assert renderer._material_version == 1
    assert src.read_text(encoding="utf-8") == "# edited\n"
