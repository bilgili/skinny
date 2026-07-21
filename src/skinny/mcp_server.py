"""In-process MCP server exposing the live scene graph.

Path-addressed inspection/property tools (``scene_list`` / ``scene_get`` /
``scene_set``) plus structural tools (``scene_add_model`` /
``scene_add_primitive`` / ``scene_add_light`` / ``scene_remove`` /
``scene_save`` / ``scene_job_status``) over the scene-graph model the Qt dock
already edits. Attaches to a renderer that is already running; it never
builds one.

**Material authoring (mcp-material-authoring)** adds a read-only discovery
tool (``material_list``, renderer-free) and two structural tools
(``scene_add_material``, ``scene_bind_material``) plus a ``material``
argument on ``scene_add_primitive``. Spec validation and MaterialX synthesis
run on the MCP thread, entirely before any stage or filesystem write
(``mtlx_synthesis``); only stage authoring and resync happen inside a posted
closure. A created material is not live (rendered, loaded, editable) until a
primitive binds it -- see each tool's docstring.

**Filesystem trust (design D2, mcp-scene-structure):** every tool argument
naming a path -- a model to reference, a save destination, an asset-typed
property write -- is checked against a configurable allowlist (``mcp_paths``).
The MCP client is a local agent on the same loopback+token trust domain as
this process, with its own file tools; this is a guardrail against a
misdirected tool call, not a sandbox boundary.

**Structural tools are jobs, not plain writes (design D7).** A model add can
outlast a flat request timeout. Each structural tool waits a short inline
grace period and returns its result directly if it finishes in time; otherwise
it returns a ``job_id`` to poll via ``scene_job_status`` rather than
cancelling -- a cancelled-but-already-running add would otherwise leave the
client unsure whether the scene changed.

**Threading invariant (design D2):** this module never touches ``Renderer``
directly. ``Renderer`` has no internal lock, and its scene graph is rebuilt and
swapped by the streaming load thread, so *reads* race too. Every read and every
write goes through the render-thread command queue and is awaited with a
timeout.

Writes await deliberately. Resolving the node, validating the value, and
dispatching all have to happen on the render thread, and a client needs to know
whether its edit was applied or rejected — a fire-and-forget write would report
success for an edit that a bounds, type, or routing check then threw away. The
cost is that writes cannot coalesce (``post_with_reply`` takes no
``coalesce_key``), so a client value-sweep is paced by the round-trip rather
than collapsed into a single edit. That is the right trade for a caller that
must be told what happened; the operator's own slider drags still coalesce
through the dock's proxy verbs.

Review rule: no ``renderer.`` outside a posted closure.
"""

from __future__ import annotations

import functools
import inspect
import logging
import math
import os
import threading
import time
import uuid
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from skinny import mtlx_synthesis
from skinny.mcp_auth import check_request, load_or_create_token, registration_command
from skinny.mcp_paths import check_path, resolve_roots, validate_added_subtree
from skinny.mtlx_synthesis import MaterialSpecError
from skinny.scene_graph import compose_trs_matrix, find_node_by_path, scene_graph_to_dict
from skinny.ui.scene_edit_actions import apply_scene_property, has_editable_stage, is_deletable

log = logging.getLogger(__name__)

# A read that cannot complete within this many seconds means the render thread
# is wedged; report it rather than blocking the client forever.
REQUEST_TIMEOUT_S = 10.0

# Structural tools wait this long inline before degrading to a pollable job.
# Kept short: FastMCP runs sync tool bodies directly on the event loop (no
# to_thread), so this is a deliberate loop-wide stall -- no other MCP request,
# including a scene_job_status poll, is serviced while it runs (design D7).
STRUCTURAL_GRACE_S = 2.0

# Pollable job retention: whichever bound is hit first prunes the job.
JOB_RETENTION_MAX = 50
JOB_RETENTION_S = 600.0

DEFAULT_LIST_DEPTH = 2

# Property types whose displayed value the renderer owns and republishes when it
# rebuilds the node (the camera's lens_file shows a "(load .usda)" placeholder,
# not the loaded path). Writing our value onto the property object would be
# silently discarded, so a later read would contradict what the write reported.
_RENDERER_OWNED_VALUES = frozenset({"lens_file"})

# Property types whose value is a filesystem path, gated through the same
# allowlist as the structural tools (design D-set / F8) -- generic on
# asset-ness, not a hardcoded name so a future asset-valued type isn't
# silently exempt. "string"/"token" are plain text, not paths.
_ASSET_PROPERTY_TYPES = frozenset({"asset", "texture_file", "lens_file"})


class SceneToolError(Exception):
    """A tool-level failure to report to the client, not a transport error."""


# ── Read helpers (all run on the render thread) ──────────────────────

def _versions(renderer) -> dict[str, int]:
    """Both counters: property edits move only ``_material_version``.

    ``_scene_graph_version`` is bumped on *structural* change only — the
    renderer deliberately leaves it alone on a property edit because the dock's
    widgets are bound to the live property objects. Reporting only that one
    would leave a client unable to observe any material or light edit.
    """
    return {
        "scene_graph_version": int(getattr(renderer, "_scene_graph_version", 0)),
        "material_version": int(getattr(renderer, "_material_version", 0)),
    }


def _summarize(node, depth: int, kind: str | None) -> dict:
    """Structure only — no properties. See design D4."""
    entry: dict[str, Any] = {
        "path": node.path,
        "name": node.name,
        "type": node.type_name,
        "child_count": len(node.children),
    }
    if node.renderer_ref is not None:
        entry["kind"] = node.renderer_ref.kind
    if depth > 0 and node.children:
        children = [_summarize(c, depth - 1, kind) for c in node.children]
        if kind is not None:
            children = [c for c in children if c]
        if children:
            entry["children"] = children
    return entry


def _matches_kind(node, kind: str | None) -> bool:
    if kind is None:
        return True
    ref = node.renderer_ref
    return ref is not None and ref.kind == kind


def _collect_by_kind(node, kind: str, out: list, depth: int) -> None:
    if _matches_kind(node, kind):
        out.append({
            "path": node.path,
            "name": node.name,
            "type": node.type_name,
            "kind": node.renderer_ref.kind,
            "child_count": len(node.children),
        })
    if depth <= 0:
        return
    for child in node.children:
        _collect_by_kind(child, kind, out, depth - 1)


class _QueueProxy:
    """Adapter giving a bare command queue the ``request`` surface.

    The Qt front-end already owns a ``QtRendererProxy``; the GLFW front-end has
    only a queue. Tools use nothing but ``request``, so either works.
    """

    def __init__(self, queue) -> None:
        self._commands = queue

    def request(self, callback):
        return self._commands.post_with_reply(callback)


def _as_proxy(proxy_or_queue):
    if hasattr(proxy_or_queue, "request"):
        return proxy_or_queue
    return _QueueProxy(proxy_or_queue)


@dataclass
class _Job:
    future: Any
    created: float


class SceneTools:
    """Tool bodies, separated from transport so they can be tested headless."""

    def __init__(
        self,
        proxy_or_queue,
        *,
        timeout: float = REQUEST_TIMEOUT_S,
        structural_grace: float = STRUCTURAL_GRACE_S,
        roots: "list[str] | None" = None,
    ) -> None:
        # Normalize here so no caller has to remember: the Qt front-end passes a
        # QtRendererProxy, the GLFW one a bare queue.
        self._proxy = _as_proxy(proxy_or_queue)
        self._timeout = timeout
        self._structural_grace = structural_grace
        # Front-ends resolve the CLI/env precedence themselves and pass the
        # final list (design D2); a caller that skips it (e.g. existing
        # hostless tests) gets the same default resolve_roots(None, ...) would.
        self._roots = list(roots) if roots is not None else resolve_roots(
            None, os.environ.get("SKINNY_MCP_ROOTS"),
        )
        self._jobs: "dict[str, _Job]" = {}
        self._jobs_lock = threading.Lock()
        # Owns the tempdir every synthesized material's .mtlx (+ mapping
        # sidecar) is flushed to, before the structural closure that
        # references it is posted (design D2/D9). One store per SceneTools
        # instance -- i.e. per server process.
        self._material_store = mtlx_synthesis.SessionMaterialStore()
        # Serializes synthesized-name picking so two concurrent material adds
        # can't choose the same salted name and clobber each other's session
        # file before the first's rollback is installed (design D6/finding #5).
        # A name stays reserved from pick until its `.mtlx` exists (the file then
        # guards reuse via write_document's refuse-overwrite).
        self._name_lock = threading.Lock()
        self._reserved_names: set[str] = set()

    def _prune_jobs_locked(self) -> None:
        now = time.monotonic()
        stale = [jid for jid, job in self._jobs.items() if now - job.created > JOB_RETENTION_S]
        for jid in stale:
            del self._jobs[jid]
        if len(self._jobs) > JOB_RETENTION_MAX:
            oldest_first = sorted(self._jobs.items(), key=lambda kv: kv[1].created)
            for jid, _job in oldest_first[: len(self._jobs) - JOB_RETENTION_MAX]:
                del self._jobs[jid]

    def _structural(self, callback):
        """Post structural work; return its result if it finishes within the
        grace period, else park it as a pollable job (design D7).
        """
        future = self._proxy.request(callback)
        try:
            result = future.result(timeout=self._structural_grace)
            return {"status": "done", **result}
        except TimeoutError:
            job_id = uuid.uuid4().hex
            with self._jobs_lock:
                self._prune_jobs_locked()
                self._jobs[job_id] = _Job(future=future, created=time.monotonic())
            return {"status": "pending", "job_id": job_id}

    def scene_job_status(self, job_id: str) -> dict:
        """Poll a structural tool's pending job. Never blocks (design D7/F6):
        checks whether the future is already done rather than waiting on it,
        so a poll can never extend the render-thread stall a slow add causes.
        """
        with self._jobs_lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise SceneToolError(f"no such job: {job_id!r}")
        if not job.future.done():
            return {"status": "pending", "job_id": job_id}
        try:
            result = job.future.result()
        except SceneToolError as exc:
            return {"status": "failed", "job_id": job_id, "error": str(exc)}
        except Exception as exc:  # noqa: BLE001 - report any renderer-thrown
            # failure (ValueError/RuntimeError/pxr.Tf.ErrorException/...) as a
            # retry-grade result rather than letting it escape as a transport
            # error; a poll must always resolve to done/failed, never raise.
            return {
                "status": "failed", "job_id": job_id,
                "error": f"{type(exc).__name__}: {exc}",
            }
        return {"status": "done", "job_id": job_id, **result}

    def _read(self, callback):
        """Run work on the render thread and wait for its reply.

        On timeout the command is cancelled so a still-queued write cannot
        apply minutes later, after the client already saw an error and possibly
        retried. If it had already started running, cancellation fails and the
        outcome is genuinely unknown — say so rather than implying nothing
        happened.
        """
        future = self._proxy.request(callback)
        try:
            return future.result(timeout=self._timeout)
        except TimeoutError:
            if future.cancel():
                raise SceneToolError(
                    f"render thread did not respond within {self._timeout:g}s; "
                    "the request was cancelled and had no effect"
                )
            raise SceneToolError(
                f"render thread did not respond within {self._timeout:g}s; "
                "the request had already started and its outcome is unknown"
            )

    # ── Tools ────────────────────────────────────────────────────────

    def scene_list(
        self,
        path: str = "/",
        depth: int = DEFAULT_LIST_DEPTH,
        kind: str | None = None,
    ) -> dict:
        """Enumerate scene structure. No property values — use scene_get."""
        def read(renderer) -> dict:
            graph = getattr(renderer, "scene_graph", None)
            if graph is None:
                raise SceneToolError("no scene is loaded")
            node = find_node_by_path(graph, path)
            if node is None:
                raise SceneToolError(f"no such path: {path!r}")
            if kind is not None:
                found: list = []
                _collect_by_kind(node, kind, found, depth)
                return {"nodes": found, **_versions(renderer)}
            return {"root": _summarize(node, depth, None), **_versions(renderer)}

        return self._read(read)

    def scene_get(self, path: str) -> dict:
        """Read one node's full properties, with editable flags and bounds."""
        def read(renderer) -> dict:
            graph = getattr(renderer, "scene_graph", None)
            if graph is None:
                raise SceneToolError("no scene is loaded")
            node = find_node_by_path(graph, path)
            if node is None:
                raise SceneToolError(f"no such path: {path!r}")
            # Serialize this node alone: scene_graph_to_dict recurses, and
            # building a whole subtree only to discard it blocks the render
            # thread between frames. Structure belongs to scene_list.
            detached = replace(node, children=[])
            # deepcopy: scene_graph_to_dict places each property's live
            # `metadata` mapping straight into the result, and `replace` is
            # shallow -- without this, a renderer-owned mutable would cross the
            # future boundary and be serialized off-thread.
            node_dict = deepcopy(scene_graph_to_dict(detached))
            return {"node": node_dict, **_versions(renderer)}

        return self._read(read)

    def scene_set(self, path: str, property: str, value: Any) -> dict:
        """Write one property.

        Resolves the node once on the render thread, validates type then
        bounds, and dispatches through the same function the Qt dock uses.
        Awaited, so the client is told whether the edit applied or why it did
        not; see the module docstring for why that beats coalescing here.
        """
        def write(renderer) -> dict:
            graph = getattr(renderer, "scene_graph", None)
            if graph is None:
                raise SceneToolError("no scene is loaded")
            node = find_node_by_path(graph, path)
            if node is None:
                raise SceneToolError(f"no such path: {path!r}")
            prop = next((p for p in node.properties if p.name == property), None)
            if prop is None:
                names = ", ".join(sorted(p.name for p in node.properties)) or "none"
                raise SceneToolError(
                    f"no property {property!r} on {path!r} (has: {names})"
                )
            if not prop.editable:
                raise SceneToolError(f"{property!r} on {path!r} is not editable")

            checked = _check_bounds(prop, _coerce(prop, value))
            if prop.type_name in _ASSET_PROPERTY_TYPES:
                path_reason = check_path(checked, self._roots)
                if path_reason is not None:
                    raise SceneToolError(path_reason)
            reason = apply_scene_property(renderer, node, prop, checked, graph=graph)
            if reason is not None:
                raise SceneToolError(f"cannot set {property!r} on {path!r}: {reason}")
            if prop.type_name not in _RENDERER_OWNED_VALUES:
                # Types the renderer republishes (it rebuilds the node and
                # restores its own placeholder) must not be written back here —
                # the write would be discarded and a later scene_get would
                # disagree with what this call reported.
                prop.value = checked
            return {"applied": {"path": path, "property": property, "value": checked},
                    **_versions(renderer)}

        return self._read(write)

    # ── Structural tools ────────────────────────────────────────────

    def scene_create(self, force: bool = False) -> dict:
        """Create a fresh, empty, editable scene so structural editing can begin
        without a scene having been loaded from disk.

        Synthesizes an in-memory USD stage holding a single ``/World`` Xform
        (Y-up, 1 m/unit) and attaches a non-destructive edit layer, so the
        ``scene_add_*`` tools and ``scene_save`` work immediately. The renderer's
        synthetic default light, dome, and camera appear in the scene graph, so
        the created scene is lit and enumerable.

        Refuses when an editable stage is already loaded -- replacing it would
        silently discard unsaved structural edits. Pass ``force = true`` to
        replace the current stage with a fresh empty one anyway.

        Returns the scene version counters only; the stage is anonymous until
        ``scene_save`` writes it to an allowed root.
        """
        def write(renderer) -> dict:
            if has_editable_stage(renderer) and not force:
                raise SceneToolError(
                    "a scene is already loaded; pass force=true to replace it "
                    "and discard unsaved structural edits"
                )
            renderer.create_empty_scene()
            return {**_versions(renderer)}

        return self._structural(write)

    def scene_add_model(
        self,
        usd_path: str,
        name: "str | None" = None,
        parent: "str | None" = None,
        translate=None,
        rotate_euler_deg=None,
        scale=None,
        matrix=None,
    ) -> dict:
        """Reference a USD file into the scene under a new prim.

        ``usd_path`` must resolve inside the configured allowed roots (see the
        module docstring). Referenced content is checked further after it
        composes -- any newly introduced layer and every asset-valued
        attribute in the added subtree must also resolve inside the roots --
        and the add is rolled back if it doesn't. This is a guardrail within
        one trust domain, not a sandbox: it catches a misdirected path, not a
        determined adversary.

        Degrades to a pollable job (``scene_job_status``) if the add outlasts
        a short inline grace period; see the module docstring.
        """
        reason = check_path(usd_path, self._roots)
        if reason is not None:
            raise SceneToolError(reason)
        transform = _resolve_transform(
            translate=translate, rotate_euler_deg=rotate_euler_deg,
            scale=scale, matrix=matrix,
        )
        parent_path = parent or "/World"
        roots = self._roots

        def write(renderer) -> dict:
            if not has_editable_stage(renderer):
                raise SceneToolError("no editable USD stage is loaded")
            stage = renderer._usd_stage
            pre_layers = stage.GetUsedLayers()

            def validate(stage_arg, added_prim) -> None:
                validate_added_subtree(stage_arg, added_prim, pre_layers, roots)

            path = renderer.add_model(
                usd_path, parent_prim_path=parent_path, name=name,
                transform=transform, validate=validate,
            )
            return {"path": path, **_versions(renderer)}

        return self._structural(write)

    def scene_add_primitive(
        self,
        type: str,
        color=None,
        roughness: "float | None" = None,
        metallic: "float | None" = None,
        material: "str | None" = None,
        name: "str | None" = None,
        parent: "str | None" = None,
        translate=None,
        rotate_euler_deg=None,
        scale=None,
        matrix=None,
    ) -> dict:
        """Add an analytic primitive with its own bound, editable material.

        ``type`` is one of Sphere, Cube, Cylinder, Cone, Capsule, Plane -- the
        gprim types the loader tessellates. The primitive is never authored
        bare: an unbound prim would resolve to the protected fallback
        material slot and its appearance could never be edited afterwards.

        With no ``material``, behavior is unchanged: a dedicated inline
        preview-surface material is authored and bound alongside the
        primitive, seeded by ``color``/``roughness``/``metallic`` (omitted
        ones get sensible defaults).

        ``material`` (mcp-material-authoring, design D6) replaces that inline
        material with a curated preset, a procedural template, or an
        existing bound material -- distinguished by a leading ``/``: a
        ``/Materials/...`` path binds that exact prim (erroring if absent);
        any other string must name a known preset or template and is created
        (with preset dedup) as if by ``scene_add_material``, then bound.
        Supplying ``material`` together with ``color``/``roughness``/
        ``metallic`` is rejected as ambiguous -- an inline seed and a bound
        material are two different ways to skin the primitive.

        A synthesized (template) material's first bind changes the scene's
        graph-set signature and rebuilds the render pipeline (design D9), so
        a ``material=<template>`` call is more likely than a plain add to
        degrade to a pollable job (``scene_job_status``).
        """
        if material is not None and (
            color is not None or roughness is not None or metallic is not None
        ):
            raise SceneToolError(
                "scene_add_primitive: 'material' cannot be combined with "
                "'color', 'roughness', or 'metallic' -- bind a material or "
                "seed an inline preview material, not both"
            )
        transform = _resolve_transform(
            translate=translate, rotate_euler_deg=rotate_euler_deg,
            scale=scale, matrix=matrix,
        )
        color_value = _as_vec3("color", color) if color is not None else None
        parent_path = parent or "/World"

        # Material creation (if a name, not a path) is deferred into the write
        # closure so the whole create + add-primitive + bind is one atomic
        # render-thread transaction (finding #8): if the bind (or the resync
        # after it) fails, the just-created material is torn down with the
        # primitive rather than left orphaned. The MCP-thread synthesis (dry-run
        # + file flush) for a template still runs here, before the closure is
        # posted (design D9: the bind is the step expected to degrade to a job).
        material_plan = (
            self._resolve_material_plan(material) if material is not None else None
        )

        def write(renderer) -> dict:
            if not has_editable_stage(renderer):
                raise SceneToolError("no editable USD stage is loaded")
            if material_plan is not None:
                mat_path, cleanup = material_plan(renderer)
                try:
                    path = renderer.add_primitive(
                        type, parent_prim_path=parent_path, name=name,
                        transform=transform, bind_material_path=mat_path,
                    )
                except Exception:
                    if cleanup is not None:
                        try:
                            cleanup(renderer)
                        except Exception:  # rollback is best-effort
                            pass
                    raise
                return {
                    "path": path, "material_path": mat_path,
                    **_versions(renderer),
                }
            path = renderer.add_primitive(
                type, parent_prim_path=parent_path, name=name, transform=transform,
                color=color_value, roughness=roughness, metallic=metallic,
            )
            return {"path": path, **_versions(renderer)}

        return self._structural(write)

    def _resolve_material_plan(self, material: str):
        """Return a render-thread callable resolving ``scene_add_primitive``'s
        ``material`` to ``(material_path, cleanup)``.

        A leading ``/`` means an existing ``/Materials/...`` path -- bound as-is
        with no cleanup. Otherwise ``material`` must be a known preset (created
        with dedup) or template name (synthesized fresh); anything else is an
        explicit error listing both catalogs. The returned callable runs inside
        the caller's write closure so material creation and the primitive
        add+bind share one rollback scope (finding #8); ``cleanup`` (or ``None``)
        removes a holder this call freshly created -- never a deduped reuse.
        """
        if material.startswith("/"):
            return lambda renderer: (material, None)

        presets = mtlx_synthesis.list_presets()
        if material in presets:
            holder_name = mtlx_synthesis.preset_holder_name(material)
            preset_path = presets[material]

            def plan(renderer):
                if not has_editable_stage(renderer):
                    raise SceneToolError("no editable USD stage is loaded")
                path, created = _add_or_dedup_preset(renderer, holder_name, preset_path)
                # Hard-remove a freshly-created holder on rollback (finding #8/E)
                # so a same-name retry is not blocked by an active=false tombstone
                # (a deduped reuse is never torn down).
                cleanup = (lambda r, p=path: r.discard_created_prim(p)) if created else None
                return path, cleanup

            return plan

        if material in mtlx_synthesis.TEMPLATES:
            normalized = mtlx_synthesis.validate_spec({"template": material})
            candidate, written_path = self._synthesize_and_stage_material(
                normalized, None, material,
            )

            def plan(renderer):
                path = self._add_synth_material(renderer, candidate, written_path)

                def cleanup(r, p=path, name=candidate):
                    # Hard-remove the spec (finding #8/E): remove_node deactivates
                    # (a tombstone the same-name collision check would then block
                    # forever); this deletes the edit-layer spec so the name is
                    # reusable, matching add_material's own rollback.
                    r.discard_created_prim(p)
                    self._material_store.delete(name)

                return path, cleanup

            return plan

        available_presets = ", ".join(sorted(presets)) or "(none)"
        available_templates = ", ".join(sorted(mtlx_synthesis.TEMPLATES)) or "(none)"
        raise SceneToolError(
            f"scene_add_primitive: material {material!r} is not a "
            f"/Materials/... path, a known preset, or a known template; "
            f"presets: {available_presets}; templates: {available_templates}"
        )

    def material_list(self) -> dict:
        """Discovery: everything needed to build a valid ``scene_add_material``
        spec, in one call (mcp-material-authoring).

        Renderer-free -- it never touches the render thread, only
        ``mtlx_synthesis``'s catalogs (derived from disk / the whitelist /
        the template registry at call time, so this can never hand-drift
        from what a spec actually accepts). Per-preset editable inputs come
        from a generator reflection run per preset file, mtime-cached, so
        the first call after a preset file changes may be slow -- acceptable
        for a discovery call.

        Returns ``{"presets": [{"name", "editable_inputs"}], "models":
        {"preview": {...}, "standard_surface": {...}}, "graph_nodes": [...],
        "templates": {name: {...}}}``.
        """
        presets = [
            {"name": preset_name, "editable_inputs": mtlx_synthesis.list_preset_inputs(preset_name)}
            for preset_name in sorted(mtlx_synthesis.list_presets())
        ]
        models = {
            model: mtlx_synthesis.model_param_schema(model)
            for model in ("preview", "standard_surface")
        }
        templates = {
            template_name: mtlx_synthesis.template_param_schema(template_name)
            for template_name in sorted(mtlx_synthesis.TEMPLATES)
        }
        return {
            "presets": presets,
            "models": models,
            "graph_nodes": list(mtlx_synthesis.NODE_WHITELIST),
            "templates": templates,
        }

    def _synthesize_and_stage_material(
        self, normalized_spec: dict, requested_name: "str | None", default_base: str,
    ) -> "tuple[str, str]":
        """MCP-thread synth (design D9): reserve a unique salted name, run the
        generator dry-run, flush the session ``.mtlx`` (+ descriptor sidecar) --
        all *before* any stage closure is posted, so a rejected spec never
        touches the stage and never leaves a file behind.

        Name picking is reserved under ``_name_lock`` (finding #5): the candidate
        must miss the current ``/Materials`` children, any in-flight reservation,
        AND any existing session file, so two concurrent adds cannot pick the same
        name and clobber. The reservation is released once the ``.mtlx`` exists
        (the file then guards reuse via ``write_document``'s refuse-overwrite) or
        on any failure, which also deletes whatever was written.
        """
        from pxr import Tf

        base = Tf.MakeValidIdentifier(requested_name or default_base)

        def read(renderer) -> "set[str]":
            if not has_editable_stage(renderer):
                raise SceneToolError("no editable USD stage is loaded")
            scope = renderer._usd_stage.GetPrimAtPath("/Materials")
            if scope and scope.IsValid():
                return {child.GetName() for child in scope.GetChildren()}
            return set()

        existing = self._read(read)
        with self._name_lock:
            candidate = base
            i = 1
            while (
                candidate in existing
                or candidate in self._reserved_names
                or self._material_store.path_for(candidate).exists()
            ):
                candidate = f"{base}_{i}"
                i += 1
            self._reserved_names.add(candidate)

        try:
            result = mtlx_synthesis.synthesize(normalized_spec, candidate)
            written_path = self._material_store.write_document(
                candidate, result.document_xml, result.descriptors,
            )
        except MaterialSpecError as exc:
            self._material_store.delete(candidate)
            raise SceneToolError(str(exc)) from exc
        except Exception:
            self._material_store.delete(candidate)
            raise
        finally:
            with self._name_lock:
                self._reserved_names.discard(candidate)
        return candidate, written_path

    def _add_synth_material(self, renderer, candidate: str, written_path: str) -> str:
        """``add_material`` for a synthesized doc, deleting its session file on
        ANY failure after the file was written (finding #5).

        The stage-readiness check and ``add_material``'s holder-collision guard
        both raise *before* ``add_material`` wires its own rollback, so without
        this wrapper either one orphans the just-written ``.mtlx`` + sidecar.
        Deleting the file also releases the name (``write_document``'s
        refuse-overwrite keyed off the file). ``delete`` is idempotent.
        """
        if not has_editable_stage(renderer):
            self._material_store.delete(candidate)
            raise SceneToolError("no editable USD stage is loaded")
        try:
            return renderer.add_material(
                candidate, mtlx_path=written_path,
                session_dir=str(self._material_store.dir),
            )
        except Exception:
            self._material_store.delete(candidate)
            raise

    def scene_add_material(self, spec: dict, name: "str | None" = None) -> dict:
        """Create a material holder under ``/Materials`` (mcp-material-authoring,
        design D2/D4/D8).

        ``spec`` is exactly one of: ``{"preset": name}`` (curated corpus,
        server-resolved -- never a client path); ``{"model": "preview"|
        "standard_surface", "params": {...}, "graph": {...}?}`` (flat params,
        optionally an explicit nodegraph on standard_surface); or
        ``{"template": name, "params": {...}}`` (a server-owned procedural
        recipe that expands to a standard_surface graph). Call ``material_list``
        for the exact catalogs and schemas. Validation -- and, for a
        synthesized document, the Slang generator dry-run -- runs entirely on
        this call before any stage or filesystem write: a rejected spec never
        creates a prim, a file, or bumps a version counter.

        ``name`` seeds the holder prim name for the ``preview`` and
        ``standard_surface``/``template`` forms (uniquified if taken; default
        ``"Material"`` or the template name). It is rejected for a preset
        spec -- design D6's naming contract fixes a preset holder's name to
        the curated document's surfacematerial element name, which ``name``
        cannot override.

        Adding a preset that already has a ``/Materials`` holder returns the
        existing holder's path instead of creating a duplicate (design D6:
        a curated document's element name is fixed, so a second holder could
        never resolve anyway). Synthesized/template materials are never
        deduplicated -- each call builds a fresh, name-salted document.

        The result's ``live`` is always ``False``: participation is
        binding-driven (design D8) -- a material is loaded, rendered, and
        exposes its editable properties only once a primitive binds it
        (``scene_bind_material`` or ``scene_add_primitive(material=...)``).
        A synthesized material's *first* bind, not this call, is what
        changes the scene's graph-set signature and is expected to degrade
        to a pollable job (design D9).
        """
        try:
            normalized = mtlx_synthesis.validate_spec(spec)
        except MaterialSpecError as exc:
            raise SceneToolError(str(exc)) from exc

        form = normalized["form"]

        if form == "preset":
            if name is not None:
                raise SceneToolError(
                    "scene_add_material: 'name' is not accepted for a preset "
                    "spec -- the holder prim name is fixed to the curated "
                    "document's surfacematerial element name"
                )
            holder_name = mtlx_synthesis.preset_holder_name(normalized["preset"])
            preset_path = normalized["path"]

            def write(renderer) -> dict:
                if not has_editable_stage(renderer):
                    raise SceneToolError("no editable USD stage is loaded")
                path, _created = _add_or_dedup_preset(renderer, holder_name, preset_path)
                return {"path": path, "live": False, **_versions(renderer)}

            return self._structural(write)

        if form == "preview":
            material_name = name or "Material"
            params = normalized["params"]

            def write(renderer) -> dict:
                if not has_editable_stage(renderer):
                    raise SceneToolError("no editable USD stage is loaded")
                path = renderer.add_material(material_name, preview_params=params)
                return {"path": path, "live": False, **_versions(renderer)}

            return self._structural(write)

        # form == "standard_surface": a raw graph/flat spec, or an expanded
        # template (validate_spec lost the original template name, so read
        # it off the *raw* client spec for the default base name).
        default_base = spec.get("template") if isinstance(spec, dict) else None
        candidate, written_path = self._synthesize_and_stage_material(
            normalized, name, default_base or "Material",
        )

        def write(renderer) -> dict:
            path = self._add_synth_material(renderer, candidate, written_path)
            return {"path": path, "live": False, **_versions(renderer)}

        return self._structural(write)

    def scene_bind_material(self, prim_path: str, material_path: str) -> dict:
        """Bind ``material_path`` to ``prim_path`` (mcp-material-authoring,
        design D6): the moment of participation for a material created by
        ``scene_add_material`` (design D8) -- it becomes live (loaded,
        rendered, editable) only now.

        Authors explicit binding-relationship targets (set, not prepended),
        so this *replaces* rather than merges with any file-authored binding
        under LIVRPS; rebinding an already-bound prim is last-write-wins.
        Errors when ``prim_path`` doesn't exist or isn't bindable geometry,
        when ``material_path`` doesn't exist, or when it is neither
        ``Material``-typed nor carries a ``.mtlx`` reference (renderer
        checks, surfaced as-is).

        A material's *first* bind changes the scene's graph-set signature
        and triggers a full render-pipeline rebuild (design D9) -- expect
        this call to degrade to a pollable job (``scene_job_status``) past
        the inline grace period more often than a plain structural add.
        """
        def write(renderer) -> dict:
            if not has_editable_stage(renderer):
                raise SceneToolError("no editable USD stage is loaded")
            renderer.bind_material(prim_path, material_path)
            return {
                "prim_path": prim_path, "material_path": material_path,
                **_versions(renderer),
            }

        return self._structural(write)

    def scene_add_light(
        self,
        light_type: str,
        intensity: "float | None" = None,
        color=None,
        name: "str | None" = None,
        parent: "str | None" = None,
        translate=None,
        rotate_euler_deg=None,
        scale=None,
        matrix=None,
    ) -> dict:
        """Add a light. ``light_type`` is one of DistantLight, SphereLight,
        DomeLight, RectLight, DiskLight. ``intensity``/``color`` are authored
        at creation when given; a dome's texture is set afterwards via
        ``scene_set`` (it applies to only one of the five types)."""
        transform = _resolve_transform(
            translate=translate, rotate_euler_deg=rotate_euler_deg,
            scale=scale, matrix=matrix,
        )
        color_value = _as_vec3("color", color) if color is not None else None
        parent_path = parent or "/World"

        def write(renderer) -> dict:
            if not has_editable_stage(renderer):
                raise SceneToolError("no editable USD stage is loaded")
            path = renderer.add_light(
                light_type, parent_prim_path=parent_path, name=name,
                transform=transform, intensity=intensity, color=color_value,
            )
            return {"path": path, **_versions(renderer)}

        return self._structural(write)

    def scene_remove(self, path: str) -> dict:
        """Remove a node by deactivation (non-destructive -- the prim stays on
        the stage with ``active = false``). The pseudo-root and synthesized
        ``/Skinny/*`` nodes are refused."""
        def write(renderer) -> dict:
            if not has_editable_stage(renderer):
                raise SceneToolError("no editable USD stage is loaded")
            graph = getattr(renderer, "scene_graph", None)
            node = find_node_by_path(graph, path) if graph is not None else None
            if node is None:
                raise SceneToolError(f"no such node: {path!r}")
            if not is_deletable(node):
                raise SceneToolError(f"not deletable: {path!r}")
            renderer.remove_node(path)
            return {"path": path, **_versions(renderer)}

        return self._structural(write)

    def scene_save(self, path: str) -> dict:
        """Write the USD edit layer to ``path`` (required, and must resolve
        inside the allowed roots -- the renderer's own unrequested default
        lands beside the loaded scene, typically outside them).

        Only structural edits (adds, removals, transforms) are captured.
        Property edits made via ``scene_set`` mutate in-memory render state
        without authoring to the USD edit layer, so they are NOT included in
        a save -- the same partial-save behavior the graphical editor's save
        action has.
        """
        if not path:
            raise SceneToolError(
                "scene_save requires an explicit path inside the allowed "
                "roots; omitting it would fall back to a location beside the "
                "loaded scene, which is typically outside them"
            )
        reason = check_path(path, self._roots)
        if reason is not None:
            raise SceneToolError(reason)

        def write(renderer) -> dict:
            if not has_editable_stage(renderer):
                raise SceneToolError("no editable USD stage is loaded")
            written = renderer.save_edits(path)
            return {"path": written, **_versions(renderer)}

        return self._structural(write)


def _add_or_dedup_preset(renderer, holder_name: str, preset_path: str) -> "tuple[str, bool]":
    """Return ``(holder_path, created)`` for a preset (design D6).

    Reuses the existing ``/Materials`` holder if one already references
    ``preset_path`` (dedup, ``created=False``); else authors a new one
    (``created=True``). The flag lets a transactional caller (finding #8) know
    whether it may remove the holder on a later failure -- a deduped reuse must
    never be torn down. Must run on the render thread -- it reads the live
    composed stage.
    """
    from skinny import usd_material_edit as ume

    resolved_abs = str(Path(preset_path).resolve())
    for holder_path, asset in ume.collect_material_holders(renderer._usd_stage).items():
        if str(Path(asset).resolve()) == resolved_abs:
            return holder_path, False
    return renderer.add_material(holder_name, mtlx_path=preset_path), True


def _as_vec3(label: str, value) -> tuple:
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise SceneToolError(f"{label} expects 3 numbers, got {value!r}")
    if len(value) != 3:
        raise SceneToolError(f"{label} expects exactly 3 numbers, got {len(value)}")
    out = []
    for component in value:
        if isinstance(component, bool) or not isinstance(component, (int, float)):
            raise SceneToolError(f"{label} expects numbers, got {component!r}")
        fv = float(component)
        if not math.isfinite(fv):
            raise SceneToolError(f"{label} must be finite, got {component!r}")
        out.append(fv)
    return tuple(out)


def _as_scale(value) -> tuple:
    if isinstance(value, bool) or not isinstance(value, (int, float, list, tuple)):
        raise SceneToolError(f"scale expects a number or 3 numbers, got {value!r}")
    if isinstance(value, (int, float)):
        fv = float(value)
        if not math.isfinite(fv):
            raise SceneToolError(f"scale must be finite, got {value!r}")
        return (fv, fv, fv)
    return _as_vec3("scale", value)


def _as_matrix16(value) -> list:
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise SceneToolError(f"matrix expects 16 numbers, got {value!r}")
    if len(value) != 16:
        raise SceneToolError(f"matrix expects exactly 16 numbers, got {len(value)}")
    out = []
    for component in value:
        if isinstance(component, bool) or not isinstance(component, (int, float)):
            raise SceneToolError(f"matrix expects numbers, got {component!r}")
        fv = float(component)
        if not math.isfinite(fv):
            raise SceneToolError(f"matrix must be finite, got {component!r}")
        out.append(fv)
    return out


def _resolve_transform(
    *, translate=None, rotate_euler_deg=None, scale=None, matrix=None,
):
    """TRS (compose_trs_matrix) or a raw matrix, never both (design D6/X3).

    ``None`` when nothing was given, matching the add verbs' own
    ``transform=None`` identity default. A sheared ``matrix`` does not
    round-trip through the TRS properties a node read reports -- documented
    on the tools that accept it.
    """
    trs_given = translate is not None or rotate_euler_deg is not None or scale is not None
    if trs_given and matrix is not None:
        raise SceneToolError(
            "provide either translate/rotate_euler_deg/scale or matrix, not both"
        )
    if matrix is not None:
        flat = _as_matrix16(matrix)
        return np.asarray(flat, dtype=float).reshape(4, 4)
    if not trs_given:
        return None
    t = _as_vec3("translate", translate) if translate is not None else (0.0, 0.0, 0.0)
    r = (
        _as_vec3("rotate_euler_deg", rotate_euler_deg)
        if rotate_euler_deg is not None else (0.0, 0.0, 0.0)
    )
    s = _as_scale(scale) if scale is not None else (1.0, 1.0, 1.0)
    return compose_trs_matrix(t, r, s)


def _coerce(prop, value):
    """Validate and normalize a written value against the property's type.

    A client can send anything JSON can carry, so type is checked before the
    value reaches renderer code — otherwise ``"false"`` becomes ``True``, a
    two-element list reaches transform recomposition, and a NaN silently
    corrupts a material override that only fails later at upload.
    """
    type_name = getattr(prop, "type_name", "")

    if type_name == "bool":
        if not isinstance(value, bool):
            raise SceneToolError(f"{prop.name} expects a boolean, got {value!r}")
        return value

    if type_name in ("float", "color3f"):
        return _finite(prop, value) if type_name == "float" else _vector(prop, value, 3)

    if type_name == "int":
        if isinstance(value, bool) or not isinstance(value, int):
            raise SceneToolError(f"{prop.name} expects an integer, got {value!r}")
        return value

    if type_name == "vec3f":
        return _vector(prop, value, 3)

    if type_name == "vec2f":
        return _vector(prop, value, 2)

    if type_name in _ASSET_PROPERTY_TYPES or type_name in ("string", "token"):
        if not isinstance(value, str):
            raise SceneToolError(f"{prop.name} expects a string, got {value!r}")
        return value

    return value


def _finite(prop, value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SceneToolError(f"{prop.name} expects a number, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise SceneToolError(f"{prop.name} must be finite, got {value!r}")
    return number


def _vector(prop, value, length: int) -> tuple:
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise SceneToolError(
            f"{prop.name} expects {length} numbers, got {value!r}"
        )
    if len(value) != length:
        raise SceneToolError(
            f"{prop.name} expects exactly {length} numbers, got {len(value)}"
        )
    return tuple(_finite(prop, component) for component in value)


def _check_bounds(prop, value):
    """Reject out-of-bounds writes; never clamp. See design D12.

    The published ranges are editor affordances, not legal bounds — the dock
    itself raises a growable property's spin-box maximum far past ``max``.
    Clamping would make a client less capable than the operator and would
    silently alter a render (``roughness=0.0`` quietly becoming ``0.04``).
    """
    meta = prop.metadata or {}
    if meta.get("growable"):
        return value
    lo, hi = meta.get("min"), meta.get("max")
    if lo is None and hi is None:
        return value
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return value
    if lo is not None and value < lo:
        raise SceneToolError(
            f"{prop.name}={value} is below its minimum {lo} (range {lo}..{hi})"
        )
    if hi is not None and value > hi:
        raise SceneToolError(
            f"{prop.name}={value} is above its maximum {hi} (range {lo}..{hi})"
        )
    return value


# ── Transport ────────────────────────────────────────────────────────

def build_app(tools: SceneTools, token: str, port: int):
    """Build the MCP streamable-HTTP ASGI app with the request guards applied."""
    from mcp.server.fastmcp import FastMCP

    from mcp.server.fastmcp.exceptions import ToolError

    server = FastMCP("skinny")

    def _wrap(fn):
        """Preserve the signature FastMCP reflects on, and signal real errors.

        ``functools.wraps`` alone is not enough: it copies ``__wrapped__``, and
        FastMCP builds the input schema from the *signature*. Without an explicit
        ``__signature__`` the tool is advertised as ``(*args, **kwargs)`` and no
        client can call it.

        Failures raise ``ToolError`` rather than returning an ``{"error": ...}``
        payload, which FastMCP would report as a *successful* call — leaving a
        client unable to tell a failed edit from an applied one.
        """
        @functools.wraps(fn)
        def call(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except SceneToolError as exc:
                raise ToolError(str(exc)) from exc
            except Exception as exc:  # noqa: BLE001 - renderer exceptions
                # (ValueError/RuntimeError/pxr.Tf.ErrorException/...) arrive
                # here via Future.result(); report every one as a tool
                # failure rather than letting it escape as a transport error.
                raise ToolError(f"{type(exc).__name__}: {exc}") from exc

        call.__signature__ = inspect.signature(fn)
        return call

    for tool in (
        tools.scene_list, tools.scene_get, tools.scene_set,
        tools.scene_create,
        tools.scene_add_model, tools.scene_add_primitive, tools.scene_add_light,
        tools.scene_remove, tools.scene_save, tools.scene_job_status,
        tools.material_list, tools.scene_add_material, tools.scene_bind_material,
    ):
        server.tool()(_wrap(tool))

    app = server.streamable_http_app()

    class Guard:
        """Refuse browser and unauthenticated requests before they reach a tool."""

        def __init__(self, inner):
            self.inner = inner

        async def __call__(self, scope, receive, send):
            if scope["type"] == "http":
                reason = check_request(dict(scope.get("headers") or []), token, port)
                if reason is not None:
                    await send({
                        "type": "http.response.start",
                        "status": 403,
                        "headers": [(b"content-type", b"text/plain")],
                    })
                    await send({"type": "http.response.body", "body": reason.encode()})
                    return
            await self.inner(scope, receive, send)

    return Guard(app)


def serve(
    proxy_or_queue, port: int, sock, roots: "list[str] | None" = None,
) -> threading.Thread:
    """Run the MCP server on a daemon thread over an already-bound socket.

    Signal handlers are explicitly not installed. ``uvicorn`` installs SIGINT and
    SIGTERM handlers by default; from a non-main thread that raises, and were it
    to succeed it would overwrite ``MetalContext``'s chained teardown handlers —
    the backstop that stops an abandoned kernel from wedging the GPU until
    reboot.

    ``roots`` is the resolved filesystem allowlist for structural tools (see
    ``mcp_paths.resolve_roots``); ``None`` falls back to its own default.
    """
    import uvicorn

    token = load_or_create_token()
    app = build_app(SceneTools(proxy_or_queue, roots=roots), token, port)
    config = uvicorn.Config(app, log_level="warning")
    server = uvicorn.Server(config)
    server.install_signal_handlers = lambda: None  # never touch process signals

    def run() -> None:
        try:
            server.run(sockets=[sock])
        except Exception:  # noqa: BLE001 - a dead server must not kill the render loop
            log.exception("MCP server stopped")

    thread = threading.Thread(target=run, name="skinny-mcp", daemon=True)
    thread.start()
    return thread


def start(
    proxy_or_queue, port: int, roots: "list[str] | None" = None,
) -> threading.Thread | None:
    """Bind, start, and print the registration line. ``None`` if the port is taken.

    A collision leaves the renderer running with MCP disabled — it does not
    exit, and does not silently pick another port.

    ``roots`` is the resolved filesystem allowlist for structural tools (see
    ``mcp_paths.resolve_roots``); ``None`` falls back to its own default.
    """
    from skinny.mcp_auth import bind_loopback_socket

    try:
        sock = bind_loopback_socket(port)
    except OSError as exc:
        log.warning(
            "MCP server unavailable: port %d is already in use (%s). "
            "The renderer is running normally without it.", port, exc,
        )
        return None

    thread = serve(proxy_or_queue, port, sock, roots=roots)
    print(f"\nMCP server on http://127.0.0.1:{port}/mcp — register it with:\n")
    print(registration_command(port))
    print()
    return thread
