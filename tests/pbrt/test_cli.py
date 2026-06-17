"""Tests for the skinny-import-pbrt CLI (task 10.3)."""

from __future__ import annotations

from skinny.pbrt.cli import main

SCENE = """
Camera "perspective" "float fov" 45
WorldBegin
LightSource "distant" "rgb L" [1 1 1]
Material "diffuse" "rgb reflectance" [0.5 0.5 0.5]
Shape "sphere" "float radius" 1
"""


def test_cli_emits_loadable_usd(tmp_path, capsys):
    src = tmp_path / "scene.pbrt"
    src.write_text(SCENE)
    out = tmp_path / "scene.usda"
    rc = main([str(src), "-o", str(out)])
    assert out.exists()
    assert rc == 0  # nothing unsupported
    captured = capsys.readouterr()
    assert "pbrt import report" in captured.out


def test_cli_reports_unsupported_with_nonzero_exit(tmp_path):
    src = tmp_path / "scene.pbrt"
    src.write_text('WorldBegin\nShape "curve"\n')  # unsupported shape
    out = tmp_path / "scene.usda"
    rc = main([str(src), "-o", str(out), "-q"])
    assert rc == 1
