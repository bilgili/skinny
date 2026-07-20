"""Filesystem allowlist for MCP structural tools.

The MCP client is a local AI agent with its own file tools, on the same
loopback+token trust domain as the renderer process (see the module docstring
of ``mcp_server.py``). This is a guardrail against agent mistakes -- pointing a
tool at the wrong path -- not a sandbox boundary; residual gaps (resolver
plugins, edits made outside this server) are accepted and documented, not
implied closed.

Enforcement happens at three points, all funneled through ``check_path``:
the tool argument itself, the layer stack a model add newly composes, and the
resolved asset attributes of the subtree it adds. See design D2/D3 in
``openspec/changes/mcp-scene-structure/design.md``.
"""

from __future__ import annotations

import os
import tempfile


def resolve_roots(cli_value: "str | None", env: "str | None" = None) -> list[str]:
    """Resolve the allowed filesystem roots.

    Precedence: ``cli_value`` (comma-separated) > ``env`` (comma-separated,
    typically the ``SKINNY_MCP_ROOTS`` environment variable) > a default of
    the platform temporary directories and the current working directory.

    The default spells out both the per-user temporary directory and ``/tmp``
    because they differ on macOS (``tempfile.gettempdir()`` resolves to a
    per-user ``/var/folders/...`` path while an agent writing to ``/tmp``
    resolves through the ``/private/tmp`` symlink) -- omitting either produces
    mystery rejections for a client using the other spelling.

    Every root is resolved through symlinks and deduplicated, preserving
    first-seen order.
    """
    raw = cli_value if cli_value else env
    if raw:
        candidates = [part.strip() for part in raw.split(",") if part.strip()]
    else:
        candidates = [tempfile.gettempdir(), "/tmp", os.getcwd()]

    roots: list[str] = []
    for candidate in candidates:
        real = os.path.realpath(candidate)
        if real not in roots:
            roots.append(real)
    return roots


def check_path(path, roots: list[str]) -> "str | None":
    """``None`` if ``path`` (resolved through symlinks) lies under a root.

    Otherwise a retry-grade reason string naming the resolved path and the
    configured roots, so a client can see exactly what it needs to change.
    """
    real = os.path.realpath(str(path))
    for root in roots:
        if real == root or real.startswith(root + os.sep):
            return None
    joined = ", ".join(roots) or "(none configured)"
    return (
        f"path {str(path)!r} resolves to {real!r}, which is outside the "
        f"allowed roots: {joined}"
    )


def validate_added_subtree(stage, prim, pre_layers, roots: list[str]) -> None:
    """Veto ``add_model``'s composed result unless it stays inside ``roots``.

    Called as the ``validate`` callback ``add_model`` invokes after the
    reference has recomposed but before the geometry re-sync. Raises
    ``ValueError`` (caught and reported as a tool error by the MCP layer)
    naming the first offending layer or asset path found.

    Two USD composition gaps are closed before the walk: payloads on the
    added subtree do not compose at reference time, so they are loaded
    explicitly; and a plain traversal stops at ``instanceable`` boundaries, so
    instance proxies are traversed too. Both closed, a texture reachable only
    through a payload or only inside an instanced prototype is still caught.

    ``pre_layers`` must be the stage's used-layer set captured *before*
    ``add_model`` was called, so layers the operator's own pre-existing scene
    already composes are never re-policed -- only layers this add newly
    introduces are checked.
    """
    from pxr import Sdf, Usd

    prim.Load(Usd.LoadWithDescendants)

    pre_identifiers = {layer.identifier for layer in pre_layers}
    for layer in stage.GetUsedLayers():
        if layer.anonymous or layer.identifier in pre_identifiers:
            continue
        candidate = layer.realPath or layer.identifier
        reason = check_path(candidate, roots)
        if reason is not None:
            raise ValueError(f"add_model: referenced layer outside roots -- {reason}")

    asset_types = {Sdf.ValueTypeNames.Asset, Sdf.ValueTypeNames.AssetArray}
    for visited in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
        for attr in visited.GetAttributes():
            type_name = attr.GetTypeName()
            if type_name not in asset_types:
                continue
            value = attr.Get()
            if value is None:
                continue
            asset_values = list(value) if type_name == Sdf.ValueTypeNames.AssetArray else [value]
            for asset_value in asset_values:
                resolved = asset_value.resolvedPath
                if not resolved:
                    raise ValueError(
                        f"add_model: unresolved asset on {visited.GetPath()} "
                        f"attribute {attr.GetName()!r} ({asset_value.path!r}) -- "
                        "cannot verify it lies within the allowed roots"
                    )
                reason = check_path(resolved, roots)
                if reason is not None:
                    raise ValueError(
                        f"add_model: asset outside roots on {visited.GetPath()} "
                        f"attribute {attr.GetName()!r} -- {reason}"
                    )
