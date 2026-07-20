"""Regressions for the pre-merge review findings.

Each test pins one behaviour that was wrong or unguarded before review.
"""

from __future__ import annotations

import os
import stat
import threading
import time

import pytest

from skinny.mcp_auth import (
    check_request,
    load_or_create_token,
    registration_command,
    token_is_from_env,
)
from skinny.mcp_server import SceneToolError, SceneTools
from skinny.render_session import RenderCommandQueue
from skinny.scene_graph import RendererRef, SceneGraphNode, SceneGraphProperty

PORT = 8765
TOKEN = "t" * 40


# ── Timed-out writes must not apply later ────────────────────────────

def test_timed_out_write_is_cancelled_and_never_runs() -> None:
    """A queued write must not mutate the scene after the client saw an error."""
    queue = RenderCommandQueue()
    tools = SceneTools(queue, timeout=0.05)
    ran: list[str] = []

    with pytest.raises(SceneToolError, match="had no effect"):
        tools._read(lambda _r: ran.append("executed"))

    # The render thread catches up only now -- far too late.
    queue.run_pending(object())
    assert ran == [], "a cancelled command must not execute"


def test_cancelled_command_does_not_break_later_commands() -> None:
    queue = RenderCommandQueue()
    tools = SceneTools(queue, timeout=0.05)
    ran: list[str] = []

    with pytest.raises(SceneToolError):
        tools._read(lambda _r: ran.append("stale"))
    queue.post(lambda _r: ran.append("fresh"))
    queue.run_pending(object())

    assert ran == ["fresh"]


# ── Type validation ──────────────────────────────────────────────────

class Recorder:
    def __init__(self, prop) -> None:
        node = SceneGraphNode(
            path="/n", name="n", type_name="Mesh", children=[],
            properties=[prop], renderer_ref=RendererRef(kind="material", index=0),
        )
        self.scene_graph = SceneGraphNode(
            path="/", name="/", type_name="Stage", children=[node],
            properties=[], renderer_ref=None,
        )
        self._material_version = 0
        self._scene_graph_version = 0
        self._usd_stage = None
        self.calls: list = []

    def apply_material_override(self, index, key, value):
        self.calls.append((key, value))


def _drive(prop, value):
    renderer = Recorder(prop)
    queue = RenderCommandQueue()
    tools = SceneTools(queue, timeout=2.0)
    stop = threading.Event()

    def loop():
        while not stop.is_set():
            queue.run_pending(renderer)
            time.sleep(0.001)

    t = threading.Thread(target=loop, daemon=True)
    t.start()
    try:
        return renderer, tools.scene_set("/n", prop.name, value)
    finally:
        stop.set()
        t.join(timeout=2.0)


def _prop(name, type_name, value, **meta):
    return SceneGraphProperty(
        name=name, display_name=name, type_name=type_name, value=value,
        editable=True, metadata=meta,
    )


def test_string_is_not_coerced_into_a_bool() -> None:
    """`"false"` must not silently become True."""
    with pytest.raises(SceneToolError, match="expects a boolean"):
        _drive(_prop("flag", "bool", True), "false")


def test_float_rejects_a_string() -> None:
    with pytest.raises(SceneToolError, match="expects a number"):
        _drive(_prop("roughness", "float", 0.5), "0.5")


def test_float_rejects_non_finite() -> None:
    with pytest.raises(SceneToolError, match="must be finite"):
        _drive(_prop("roughness", "float", 0.5), float("nan"))


def test_vector_rejects_wrong_length() -> None:
    with pytest.raises(SceneToolError, match="exactly 3"):
        _drive(_prop("translate", "vec3f", (0.0, 0.0, 0.0)), [1.0, 2.0])


def test_vector_rejects_a_string() -> None:
    with pytest.raises(SceneToolError, match="expects 3 numbers"):
        _drive(_prop("translate", "vec3f", (0.0, 0.0, 0.0)), "1,2,3")


def test_valid_float_still_applies() -> None:
    renderer, _result = _drive(_prop("roughness", "float", 0.5), 0.25)
    assert renderer.calls == [("roughness", 0.25)]


# ── Dome texture / lens file routing ─────────────────────────────────

def test_failed_dome_texture_load_is_reported() -> None:
    from skinny.ui.scene_edit_actions import apply_scene_property

    class R:
        def apply_dome_light_texture(self, index, path):
            return False  # missing or unreadable HDR

    node = SceneGraphNode(
        path="/env", name="env", type_name="DomeLight", children=[], properties=[],
        renderer_ref=RendererRef(kind="light_env", index=0),
    )
    reason = apply_scene_property(
        R(), node, _prop("texture", "texture_file", ""), "/nope.hdr",
    )
    assert reason is not None and "could not load" in reason


def test_lens_file_does_not_go_to_apply_camera_param() -> None:
    """apply_camera_param would attempt float(path) on a lens file."""
    from skinny.ui.scene_edit_actions import apply_scene_property

    calls: list = []

    class R:
        def apply_camera_lens_file(self, path):
            calls.append(path)
            return True

        def apply_camera_param(self, key, value):  # pragma: no cover
            raise AssertionError("lens file must not route to apply_camera_param")

    node = SceneGraphNode(
        path="/Camera", name="Camera", type_name="Camera", children=[], properties=[],
        renderer_ref=RendererRef(kind="renderer_camera", index=0),
    )
    reason = apply_scene_property(
        R(), node, _prop("lens", "lens_file", ""), "/lenses/a.txt",
    )
    assert reason is None
    assert calls == ["/lenses/a.txt"]


# ── Host validation ──────────────────────────────────────────────────

def _headers(**kwargs):
    return {k.replace("_", "-"): v for k, v in kwargs.items()}


def test_missing_host_is_refused() -> None:
    assert check_request(_headers(authorization=f"Bearer {TOKEN}"), TOKEN, PORT)


def test_host_without_a_port_is_refused() -> None:
    headers = _headers(authorization=f"Bearer {TOKEN}", host="127.0.0.1")
    assert check_request(headers, TOKEN, PORT) is not None


def test_exact_loopback_authority_is_accepted() -> None:
    headers = _headers(authorization=f"Bearer {TOKEN}", host=f"127.0.0.1:{PORT}")
    assert check_request(headers, TOKEN, PORT) is None


# ── Token file hardening ─────────────────────────────────────────────

def test_truncated_token_reports_an_actionable_error(tmp_path, monkeypatch) -> None:
    """A corrupt token is never silently overwritten.

    Overwriting is what let a racing starter clobber a token another server was
    already serving. Refusing to repair removes that class of bug; the operator
    deletes one file instead.
    """
    monkeypatch.delenv("SKINNY_MCP_TOKEN", raising=False)
    path = tmp_path / "mcp_token"
    path.write_text("")
    path.chmod(0o600)

    with pytest.raises(SystemExit, match="Delete it and restart"):
        load_or_create_token(path)


def test_fresh_token_is_created_owner_only(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("SKINNY_MCP_TOKEN", raising=False)
    path = tmp_path / "mcp_token"

    token = load_or_create_token(path)

    assert len(token) >= 32
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert not list(tmp_path.glob("*.tmp")), "staging file leaked"


def test_symlinked_token_is_refused(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("SKINNY_MCP_TOKEN", raising=False)
    real = tmp_path / "real"
    real.write_text("x" * 40)
    real.chmod(0o600)
    link = tmp_path / "mcp_token"
    os.symlink(real, link)

    with pytest.raises(SystemExit, match="refusing to read"):
        load_or_create_token(link)


def test_registration_points_at_the_env_var_when_it_is_in_force(monkeypatch) -> None:
    """Printing `cat ~/.skinny/mcp_token` is wrong when the env var overrides."""
    monkeypatch.setenv("SKINNY_MCP_TOKEN", "from-env")
    assert token_is_from_env()
    line = registration_command(PORT)
    assert "$SKINNY_MCP_TOKEN" in line
    assert "from-env" not in line


def test_registration_points_at_the_file_otherwise(monkeypatch) -> None:
    monkeypatch.delenv("SKINNY_MCP_TOKEN", raising=False)
    assert "mcp_token" in registration_command(PORT)


# ── Depth-bounded kind filter ────────────────────────────────────────

def test_kind_filter_respects_depth() -> None:
    deep = SceneGraphNode(
        path="/a/b/c", name="c", type_name="Light", children=[], properties=[],
        renderer_ref=RendererRef(kind="light_dir", index=0),
    )
    mid = SceneGraphNode(path="/a/b", name="b", type_name="Xform", children=[deep],
                         properties=[], renderer_ref=None)
    top = SceneGraphNode(path="/a", name="a", type_name="Xform", children=[mid],
                         properties=[], renderer_ref=None)
    root = SceneGraphNode(path="/", name="/", type_name="Stage", children=[top],
                          properties=[], renderer_ref=None)

    class R:
        scene_graph = root
        _material_version = 0
        _scene_graph_version = 0

    queue = RenderCommandQueue()
    tools = SceneTools(queue, timeout=2.0)
    renderer = R()

    def call(depth):
        stop = threading.Event()

        def loop():
            while not stop.is_set():
                queue.run_pending(renderer)
                time.sleep(0.001)

        t = threading.Thread(target=loop, daemon=True)
        t.start()
        try:
            return tools.scene_list("/", depth=depth, kind="light_dir")
        finally:
            stop.set()
            t.join(timeout=2.0)

    assert call(1)["nodes"] == []          # light sits 3 levels down
    assert len(call(5)["nodes"]) == 1


# ── Second-round review fixes ────────────────────────────────────────

def test_token_helpers_do_not_hard_require_posix_only_os_attrs() -> None:
    """Windows is a supported platform; O_NOFOLLOW/getuid are POSIX-only."""
    import pathlib

    source = (
        pathlib.Path(__file__).resolve().parents[1] / "src" / "skinny" / "mcp_auth.py"
    ).read_text()
    assert "os.O_NOFOLLOW" not in source, "unguarded os.O_NOFOLLOW breaks Windows"
    assert 'getattr(os, "O_NOFOLLOW"' in source
    assert 'getattr(os, "getuid"' in source


def test_token_read_works_when_posix_attrs_are_absent(tmp_path, monkeypatch) -> None:
    """Simulate Windows: no O_NOFOLLOW, no getuid."""
    monkeypatch.delenv("SKINNY_MCP_TOKEN", raising=False)
    monkeypatch.delattr(os, "O_NOFOLLOW", raising=False)
    monkeypatch.delattr(os, "getuid", raising=False)

    path = tmp_path / "mcp_token"
    token = load_or_create_token(path)

    assert len(token) >= 32
    assert load_or_create_token(path) == token


def test_concurrent_first_creation_agrees_on_one_token(tmp_path, monkeypatch) -> None:
    """Two first-time starts must not each publish a different token."""
    monkeypatch.delenv("SKINNY_MCP_TOKEN", raising=False)
    path = tmp_path / "mcp_token"
    results: list[str] = []
    barrier = threading.Barrier(8)

    def worker() -> None:
        barrier.wait()
        results.append(load_or_create_token(path))

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5.0)

    assert len(results) == 8
    assert len(set(results)) == 1, "racing starts published different tokens"
    # Every starter must serve exactly what is on disk -- a winner whose token
    # got replaced would authenticate nobody.
    assert results[0] == path.read_text().strip()
    assert not list(tmp_path.glob("*.tmp")), "staging file leaked"


def test_lens_file_value_is_not_written_back(tmp_path) -> None:
    """The renderer republishes lens_file, so writing our value would mislead."""
    calls: list = []

    class R:
        _material_version = 0
        _scene_graph_version = 0
        _usd_stage = None

        def apply_camera_lens_file(self, path):
            calls.append(path)
            return True

    prop = _prop("lens_file", "lens_file", "(load .usda)")
    node = SceneGraphNode(
        path="/Camera", name="Camera", type_name="Camera", children=[],
        properties=[prop], renderer_ref=RendererRef(kind="renderer_camera", index=0),
    )
    renderer = R()
    renderer.scene_graph = SceneGraphNode(
        path="/", name="/", type_name="Stage", children=[node],
        properties=[], renderer_ref=None,
    )

    queue = RenderCommandQueue()
    # roots must cover the fixture's lens path -- scene_set now root-checks
    # every asset-typed write (mcp-scene-structure change), a deliberate
    # tightening of the pre-change "any string" behavior this test relied on.
    tools = SceneTools(queue, timeout=2.0, roots=[os.path.realpath("/lenses")])
    stop = threading.Event()

    def loop():
        while not stop.is_set():
            queue.run_pending(renderer)
            time.sleep(0.001)

    t = threading.Thread(target=loop, daemon=True)
    t.start()
    try:
        tools.scene_set("/Camera", "lens_file", "/lenses/a.usda")
    finally:
        stop.set()
        t.join(timeout=2.0)

    assert calls == ["/lenses/a.usda"]
    # The placeholder stays; a later scene_get must not claim otherwise.
    assert prop.value == "(load .usda)"


def test_no_replace_fallback_in_token_creation() -> None:
    """os.replace is atomic but NOT exclusive.

    A fallback that used it could let two starters both publish, destroying a
    token the other was already serving. Round 3 flagged that bug; round 4 found
    it reintroduced as a hardlink fallback. Pin it shut.
    """
    import pathlib

    source = (
        pathlib.Path(__file__).resolve().parents[1] / "src" / "skinny" / "mcp_auth.py"
    ).read_text()
    create = source[source.index("def _create_token"):source.index("def _read_token")]
    # Match the call, not the word: the comment explaining why it is absent
    # legitimately names it.
    code = "\n".join(
        line for line in create.splitlines() if not line.lstrip().startswith("#")
    )
    assert "os.replace(" not in code, "non-exclusive publish reintroduced"
    assert "os.link(" in code


def test_unlinkable_filesystem_refuses_rather_than_racing(tmp_path, monkeypatch) -> None:
    """No hardlinks -> refuse with guidance, never a non-exclusive write."""
    monkeypatch.delenv("SKINNY_MCP_TOKEN", raising=False)
    monkeypatch.setattr(
        os, "link", lambda *a, **k: (_ for _ in ()).throw(OSError("unsupported")),
    )

    with pytest.raises(SystemExit, match="SKINNY_MCP_TOKEN"):
        load_or_create_token(tmp_path / "mcp_token")

    assert not (tmp_path / "mcp_token").exists()
    assert not list(tmp_path.glob("*.tmp")), "staging file leaked"


def test_env_registration_command_trims_whitespace(monkeypatch) -> None:
    """A padded env var must still yield a command that authenticates."""
    monkeypatch.setenv("SKINNY_MCP_TOKEN", "  padded-token-value  ")
    line = registration_command(PORT)
    assert "sed" in line or "Trim()" in line
    assert "padded-token-value" not in line


@pytest.mark.parametrize(
    "raw", ["  padded  ", "\ttabbed\n", "  tok en  ", "internal space"],
)
def test_printed_command_expands_to_exactly_the_auth_token(raw, monkeypatch) -> None:
    """The emitted shell expression must equal what auth compares.

    `tr -d` would strip *internal* whitespace too, so a token containing a
    space would be sent differently than it is checked.
    """
    import subprocess

    monkeypatch.setenv("SKINNY_MCP_TOKEN", raw)
    expected = load_or_create_token()

    expr = registration_command(PORT).split("Bearer ", 1)[1].rstrip('"')
    produced = subprocess.run(
        ["sh", "-c", f'printf %s "{expr}"'],
        capture_output=True, text=True, env=dict(os.environ), check=True,
    ).stdout

    assert produced == expected
