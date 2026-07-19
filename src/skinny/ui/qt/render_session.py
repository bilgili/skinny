"""Backwards-compatible re-export of :mod:`skinny.render_session`.

The module moved up to the package root when the command queue and renderer
proxy gained callers outside the Qt front-end (the GLFW loop and the MCP server
thread). It never imported Qt, so nothing about it is Qt-specific; only its
location was. Existing imports from this path keep working.
"""

from __future__ import annotations

from skinny.render_session import (
    DebugFrame,
    FrameSnapshot,
    QtRendererConfig,
    QtRendererProxy,
    RenderCommand,
    RenderCommandQueue,
    RendererStateSnapshot,
    SceneStateSnapshot,
    build_scene_state,
    choice_names_from_renderer,
    renderer_online_status,
)

__all__ = [
    "DebugFrame",
    "FrameSnapshot",
    "QtRendererConfig",
    "QtRendererProxy",
    "RenderCommand",
    "RenderCommandQueue",
    "RendererStateSnapshot",
    "SceneStateSnapshot",
    "build_scene_state",
    "choice_names_from_renderer",
    "renderer_online_status",
]
