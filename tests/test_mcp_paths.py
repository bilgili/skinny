"""Hostless tests for the MCP filesystem allowlist (src/skinny/mcp_paths.py).

``resolve_roots``/``check_path`` are pure path logic. ``validate_added_subtree``
is exercised against small in-memory/on-disk USD fixtures built at runtime --
plain ``pxr.Usd`` composition, no renderer, no GPU context.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

pxr = pytest.importorskip("pxr")

from skinny.mcp_paths import check_path, resolve_roots, validate_added_subtree  # noqa: E402


# ── resolve_roots ────────────────────────────────────────────────────

def test_cli_value_takes_precedence_over_env(tmp_path) -> None:
    cli_dir = tmp_path / "cli"
    env_dir = tmp_path / "env"
    cli_dir.mkdir()
    env_dir.mkdir()
    roots = resolve_roots(str(cli_dir), str(env_dir))
    assert roots == [os.path.realpath(cli_dir)]


def test_env_used_when_no_cli_value(tmp_path) -> None:
    env_dir = tmp_path / "env"
    env_dir.mkdir()
    roots = resolve_roots(None, str(env_dir))
    assert roots == [os.path.realpath(env_dir)]


def test_default_covers_both_tmp_spellings_and_cwd(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    roots = resolve_roots(None, None)
    assert os.path.realpath(tempfile.gettempdir()) in roots
    assert os.path.realpath("/tmp") in roots
    assert os.path.realpath(tmp_path) in roots


def test_roots_are_deduplicated_preserving_order(tmp_path) -> None:
    roots = resolve_roots(f"{tmp_path},{tmp_path},{tmp_path}/../{tmp_path.name}", None)
    assert roots == [os.path.realpath(tmp_path)]


def test_roots_are_comma_split_and_stripped(tmp_path) -> None:
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    roots = resolve_roots(f" {a} , {b} ", None)
    assert roots == [os.path.realpath(a), os.path.realpath(b)]


# ── check_path ───────────────────────────────────────────────────────

def test_path_inside_root_is_accepted(tmp_path) -> None:
    roots = [os.path.realpath(tmp_path)]
    assert check_path(tmp_path / "file.usda", roots) is None


def test_path_outside_every_root_is_rejected(tmp_path) -> None:
    roots = [os.path.realpath(tmp_path / "allowed")]
    reason = check_path(tmp_path / "elsewhere" / "file.usda", roots)
    assert reason is not None
    assert "outside the allowed roots" in reason


def test_symlink_escape_is_rejected(tmp_path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    link = allowed / "escape"
    link.symlink_to(outside)
    roots = [os.path.realpath(allowed)]
    reason = check_path(link / "secret.usda", roots)
    assert reason is not None
    assert str(outside.resolve()) in reason


# ── validate_added_subtree fixtures ─────────────────────────────────

def _write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


@pytest.fixture()
def dirs(tmp_path):
    roots_dir = tmp_path / "roots"
    outside_dir = tmp_path / "outside"
    roots_dir.mkdir()
    outside_dir.mkdir()
    return roots_dir, outside_dir


def _new_stage_with_pre_layers():
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.CreateInMemory(load=Usd.Stage.LoadNone)
    UsdGeom.Xform.Define(stage, "/World")
    pre_layers = stage.GetUsedLayers()
    return stage, pre_layers


def test_argument_inside_roots_composes_cleanly(dirs) -> None:
    roots_dir, _outside_dir = dirs
    from pxr import UsdGeom

    plain = _write(
        roots_dir / "plain.usda",
        '#usda 1.0\n(\n    defaultPrim = "S"\n)\n\ndef Sphere "S" {}\n',
    )
    stage, pre_layers = _new_stage_with_pre_layers()
    prim = UsdGeom.Xform.Define(stage, "/World/Nested").GetPrim()
    prim.GetReferences().AddReference(str(plain))
    # Confirms the reference actually composed (a broken reference resolves
    # silently in this USD build, which would make the test pass vacuously).
    assert any(layer.identifier == str(plain) for layer in stage.GetUsedLayers())

    validate_added_subtree(stage, prim, pre_layers, roots=[os.path.realpath(roots_dir)])


def test_nested_reference_escapes_the_roots(dirs) -> None:
    roots_dir, outside_dir = dirs
    from pxr import UsdGeom

    secret = _write(
        outside_dir / "secret.usda",
        '#usda 1.0\n(\n    defaultPrim = "Secret"\n)\n\ndef Sphere "Secret" {}\n',
    )
    inroot = _write(
        roots_dir / "inroot.usda",
        (
            '#usda 1.0\n(\n    defaultPrim = "Nested"\n)\n\n'
            f'def Xform "Nested" (\n    references = @{secret}@\n)\n{{\n}}\n'
        ),
    )
    stage, pre_layers = _new_stage_with_pre_layers()
    prim = UsdGeom.Xform.Define(stage, "/World/Nested").GetPrim()
    prim.GetReferences().AddReference(str(inroot))

    with pytest.raises(ValueError, match="referenced layer outside roots"):
        validate_added_subtree(stage, prim, pre_layers, roots=[os.path.realpath(roots_dir)])


def test_operators_pre_existing_layers_are_not_policed(dirs) -> None:
    """A layer already composed before the add is not re-checked."""
    roots_dir, outside_dir = dirs
    from pxr import UsdGeom

    pre_existing = _write(
        outside_dir / "pre_existing.usda",
        '#usda 1.0\n(\n    defaultPrim = "Old"\n)\n\ndef Sphere "Old" {}\n',
    )
    unrelated = _write(
        roots_dir / "unrelated.usda",
        '#usda 1.0\n(\n    defaultPrim = "New"\n)\n\ndef Sphere "New" {}\n',
    )

    from pxr import Usd

    stage = Usd.Stage.CreateInMemory(load=Usd.Stage.LoadNone)
    UsdGeom.Xform.Define(stage, "/World")
    old_prim = UsdGeom.Xform.Define(stage, "/World/Old").GetPrim()
    old_prim.GetReferences().AddReference(str(pre_existing))

    pre_layers = stage.GetUsedLayers()  # taken AFTER the pre-existing ref composed

    new_prim = UsdGeom.Xform.Define(stage, "/World/New").GetPrim()
    new_prim.GetReferences().AddReference(str(unrelated))
    used = {layer.identifier for layer in stage.GetUsedLayers()}
    assert str(pre_existing) in used and str(unrelated) in used

    validate_added_subtree(stage, new_prim, pre_layers, roots=[os.path.realpath(roots_dir)])


def test_texture_behind_deferred_payload_is_caught(dirs) -> None:
    roots_dir, outside_dir = dirs
    texture = outside_dir / "tex.png"
    texture.write_bytes(b"\x89PNG\r\n")

    payload_content = _write(
        roots_dir / "payload_content.usda",
        (
            '#usda 1.0\n(\n    defaultPrim = "Tex"\n)\n\n'
            f'def Shader "Tex"\n{{\n    asset inputs:file = @{texture}@\n}}\n'
        ),
    )
    holder = _write(
        roots_dir / "holder.usda",
        (
            '#usda 1.0\n(\n    defaultPrim = "Nested"\n)\n\n'
            f'def Xform "Nested" (\n    payload = @{payload_content}@\n)\n{{\n}}\n'
        ),
    )
    from pxr import UsdGeom

    stage, pre_layers = _new_stage_with_pre_layers()
    prim = UsdGeom.Xform.Define(stage, "/World/Nested").GetPrim()
    prim.GetReferences().AddReference(str(holder))

    with pytest.raises(ValueError, match="asset outside roots"):
        validate_added_subtree(stage, prim, pre_layers, roots=[os.path.realpath(roots_dir)])


def test_texture_inside_instanced_prototype_is_caught(dirs) -> None:
    roots_dir, outside_dir = dirs
    texture = outside_dir / "inst_tex.png"
    texture.write_bytes(b"\x89PNG\r\n")

    instanced_content = _write(
        roots_dir / "instanced_content.usda",
        (
            '#usda 1.0\n(\n    defaultPrim = "Root"\n)\n\n'
            'def Xform "Root"\n{\n'
            f'    def Shader "InstTex"\n    {{\n        asset inputs:file = @{texture}@\n    }}\n'
            '}\n'
        ),
    )
    holder = _write(
        roots_dir / "instance_holder.usda",
        (
            '#usda 1.0\n(\n    defaultPrim = "InstanceHolder"\n)\n\n'
            'def Xform "InstanceHolder" (\n'
            '    instanceable = true\n'
            f'    references = @{instanced_content}@\n'
            ')\n{}\n'
        ),
    )
    from pxr import Usd, UsdGeom

    stage, pre_layers = _new_stage_with_pre_layers()
    prim = UsdGeom.Xform.Define(stage, "/World/Nested").GetPrim()
    prim.GetReferences().AddReference(str(holder))

    # Sanity: a plain traversal (no instance-proxy flag) does not reach the
    # instanced child at all -- this is the escape TraverseInstanceProxies closes.
    plain_children = list(Usd.PrimRange(prim))
    assert not any(c.GetName() == "InstTex" for c in plain_children)

    with pytest.raises(ValueError, match="asset outside roots"):
        validate_added_subtree(stage, prim, pre_layers, roots=[os.path.realpath(roots_dir)])


def test_relative_asset_path_resolves_inside_roots(dirs) -> None:
    roots_dir, _outside_dir = dirs
    (roots_dir / "tex.png").write_bytes(b"\x89PNG\r\n")
    holder = _write(
        roots_dir / "relative_tex.usda",
        (
            '#usda 1.0\n(\n    defaultPrim = "Tex"\n)\n\n'
            'def Shader "Tex"\n{\n    asset inputs:file = @./tex.png@\n}\n'
        ),
    )
    from pxr import UsdGeom

    stage, pre_layers = _new_stage_with_pre_layers()
    prim = UsdGeom.Xform.Define(stage, "/World/Nested").GetPrim()
    prim.GetReferences().AddReference(str(holder))
    assert any(layer.identifier == str(holder) for layer in stage.GetUsedLayers())

    validate_added_subtree(stage, prim, pre_layers, roots=[os.path.realpath(roots_dir)])


def test_unresolvable_asset_is_rejected(dirs) -> None:
    roots_dir, _outside_dir = dirs
    holder = _write(
        roots_dir / "missing_tex.usda",
        (
            '#usda 1.0\n(\n    defaultPrim = "Tex"\n)\n\n'
            'def Shader "Tex"\n{\n    asset inputs:file = @./does_not_exist.png@\n}\n'
        ),
    )
    from pxr import UsdGeom

    stage, pre_layers = _new_stage_with_pre_layers()
    prim = UsdGeom.Xform.Define(stage, "/World/Nested").GetPrim()
    prim.GetReferences().AddReference(str(holder))

    with pytest.raises(ValueError, match="unresolved asset"):
        validate_added_subtree(stage, prim, pre_layers, roots=[os.path.realpath(roots_dir)])


def test_udim_template_is_rejected(dirs) -> None:
    roots_dir, _outside_dir = dirs
    holder = _write(
        roots_dir / "udim_tex.usda",
        (
            '#usda 1.0\n(\n    defaultPrim = "Tex"\n)\n\n'
            'def Shader "Tex"\n{\n    asset inputs:file = @./tex.<UDIM>.png@\n}\n'
        ),
    )
    from pxr import UsdGeom

    stage, pre_layers = _new_stage_with_pre_layers()
    prim = UsdGeom.Xform.Define(stage, "/World/Nested").GetPrim()
    prim.GetReferences().AddReference(str(holder))

    with pytest.raises(ValueError, match="unresolved asset"):
        validate_added_subtree(stage, prim, pre_layers, roots=[os.path.realpath(roots_dir)])
