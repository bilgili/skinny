"""One owner for the persisted session snapshot (change session-settings-owner).

`settings.py` owns the *file* — load, merge, atomic write. This module owns the
*schema*: which keys the snapshot has, who contributes each one, and the capture
and restore of every renderer-owned key.

Two sections:

- **Shared** (`SHARED_KEYS`) — renderer-owned state. `capture_shared` reads it
  off a live `Renderer`, `restore_shared` writes it back. Both interactive
  front-ends use these verbatim, so their settings files are interchangeable.
- **Contributed** (`GLFW_KEYS`, `QT_KEYS`) — front-end-owned state that no
  renderer can produce: the GLFW window position, the Qt dock layout. The module
  preserves these opaquely; it never interprets a Qt geometry blob. A front-end
  hands them to `contribute`, which REFUSES any key not declared here — an
  undeclared key fails at the first exit rather than being written and then
  silently erased by the other front-end.

The front-ends MUST NOT re-author the schema: they call `capture_shared` +
`contribute`, never build the dict themselves. Before the change each authored
its own 11-key dict and `save_settings` wrote wholesale, so exiting one erased
five of the other's keys.
"""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import Mapping
from typing import Any

import numpy as np

from skinny.params import _apply_saved_params, _snapshot_params, build_all_params

log = logging.getLogger(__name__)

# Renderer-owned; captured and restored by this module on both front-ends.
SHARED_KEYS = frozenset({
    "params",
    "camera",
    "gizmo_mode",
    "backend",
    "encoding",
    "sppm_glossy_roughness",
    "neural_handoff",
    "neural_trainer",
    "train_precision",
    "online_training",
})

# Front-end-owned, contributed and preserved opaquely. `last_dirs` sits under
# QT_KEYS because `skinny-gui` is the only front-end that snapshots it on exit;
# `settings.record_last_dir` also writes it through directly, from any front-end
# with a file dialog, and that path merges into the same key.
GLFW_KEYS = frozenset({"vulkan_window"})
QT_KEYS = frozenset({
    "open_docks",
    "last_dirs",
    "section_states",
    "qt_geometry",
    "qt_dock_state",
})

CONTRIBUTED_KEYS = GLFW_KEYS | QT_KEYS
DECLARED_KEYS = SHARED_KEYS | CONTRIBUTED_KEYS


# ── camera ──────────────────────────────────────────────────────────

def capture_camera(renderer) -> dict[str, Any]:
    """Snapshot both cameras plus which one is active."""
    orbit = renderer.orbit_camera
    free = renderer.free_camera
    return {
        "mode": renderer.camera_mode,
        "orbit": {
            "yaw": float(orbit.yaw),
            "pitch": float(orbit.pitch),
            "distance": float(orbit.distance),
            "fov": float(orbit.fov),
            "target": [float(orbit.target[0]), float(orbit.target[1]), float(orbit.target[2])],
        },
        "free": {
            "position": [float(free.position[0]), float(free.position[1]), float(free.position[2])],
            "yaw": float(free.yaw),
            "pitch": float(free.pitch),
            "fov": float(free.fov),
            "move_speed": float(free.move_speed),
        },
    }


def _vec3(raw, fallback):
    if isinstance(raw, (list, tuple)) and len(raw) == 3:
        try:
            return np.array([float(raw[0]), float(raw[1]), float(raw[2])], dtype=np.float32)
        except (TypeError, ValueError):
            pass
    return fallback


def _flt(raw, fallback):
    try:
        return float(raw)
    except (TypeError, ValueError):
        return fallback


def restore_camera(renderer, saved_cam) -> None:
    """Restore both cameras. Missing / out-of-range values keep the default.

    One rule for both front-ends: a persisted orbit distance beyond the current
    cap RAISES the cap to fit. `skinny-gui` used to clamp the distance to 50 and
    ignore `max_distance`, which silently destroyed a legitimately persisted
    wide view; `max_distance` is the cap authority, so the cap moves, not the
    view.
    """
    if not isinstance(saved_cam, dict):
        return

    orbit_raw = saved_cam.get("orbit")
    if isinstance(orbit_raw, dict):
        o = renderer.orbit_camera
        o.yaw = _flt(orbit_raw.get("yaw"), o.yaw)
        o.pitch = float(np.clip(
            _flt(orbit_raw.get("pitch"), o.pitch), -np.pi / 2 + 0.01, np.pi / 2 - 0.01
        ))
        o.distance = max(0.5, _flt(orbit_raw.get("distance"), o.distance))
        o.max_distance = max(o.max_distance, o.distance)
        o.fov = float(np.clip(_flt(orbit_raw.get("fov"), o.fov), 1.0, 170.0))
        o.target = _vec3(orbit_raw.get("target"), o.target)

    free_raw = saved_cam.get("free")
    if isinstance(free_raw, dict):
        f = renderer.free_camera
        f.position = _vec3(free_raw.get("position"), f.position)
        f.yaw = _flt(free_raw.get("yaw"), f.yaw)
        f.pitch = float(np.clip(
            _flt(free_raw.get("pitch"), f.pitch), -np.pi / 2 + 0.01, np.pi / 2 - 0.01
        ))
        f.fov = float(np.clip(_flt(free_raw.get("fov"), f.fov), 1.0, 170.0))
        f.move_speed = float(np.clip(_flt(free_raw.get("move_speed"), f.move_speed), 0.05, 50.0))

    mode = saved_cam.get("mode")
    if mode in ("orbit", "free"):
        renderer.camera_mode = mode


def restore_gizmo_mode(renderer, saved_mode) -> None:
    """Restore the persisted transform-gizmo mode (an int 0..3)."""
    from skinny.gizmo import GizmoMode
    try:
        renderer.gizmo.mode = GizmoMode(int(saved_mode))
    except (TypeError, ValueError):
        return


# ── shared section ──────────────────────────────────────────────────

def capture_shared(renderer, *, backend: str) -> dict[str, Any]:
    """Capture every renderer-owned key. Call with a LIVE renderer — on the
    thread that owns it, which for `skinny-gui` means inside a render-thread
    request.

    Parameters are captured with `build_all_params`, never the
    visibility-filtered set: filtering on capture permanently loses the
    fallback-light values whenever a scene with authored lighting is loaded.
    Visibility governs what is displayed, not what is stored.

    `online_training` records the user's INTENT, not whether the session
    managed to arm the loop — a per-session prerequisite refusal is not a
    preference change.
    """
    return {
        "params": _snapshot_params(renderer, build_all_params(renderer)),
        "camera": capture_camera(renderer),
        "gizmo_mode": int(renderer.gizmo.mode),
        "backend": backend,
        "encoding": renderer._neural_config.encoding.value,
        "sppm_glossy_roughness": getattr(renderer, "_sppm_glossy_roughness_override", None),
        "neural_handoff": renderer._neural_handoff_kind,
        "neural_trainer": renderer._neural_trainer_kind,
        "train_precision": renderer._train_precision,
        "online_training": bool(getattr(renderer, "_online_training_requested", False)),
    }


def restore_shared(renderer, data: Mapping[str, Any]) -> None:
    """Apply the renderer-owned keys of a loaded snapshot.

    Restores the full parameter set (`build_all_params`), so a value captured
    under one light authority is not dropped under another. The caller owns any
    follow-up refresh (`_update_light`) and any CLI-wins precedence.

    The three steps are **independently fault-isolated**: a settings file that
    breaks the parameter restore must still give the user their camera back. The
    isolation lives here, in the owner, rather than in a caller's try/except —
    `skinny` had none at all, and `skinny-gui` had three.

    `backend` and `encoding` are in `SHARED_KEYS` but are NOT restored here: they
    are session-fixed bring-up inputs, consumed by `plan_bringup(persisted=…)`
    before a renderer exists. This module owns the KEY; `bringup.py` owns what a
    startup value does.
    """
    for step, apply in (
        ("params", lambda: _apply_saved_params(
            renderer, data.get("params", {}), build_all_params(renderer))),
        ("camera", lambda: restore_camera(renderer, data.get("camera"))),
        # A missing gizmo_mode needs no guard here: restore_gizmo_mode returns on
        # the TypeError from int(None).
        ("gizmo mode", lambda: restore_gizmo_mode(renderer, data.get("gizmo_mode"))),
    ):
        try:
            apply()
        except Exception as exc:  # noqa: BLE001 — one bad key must not cost the rest
            log.warning("Failed to restore persisted %s: %s", step, exc)


def restore_params(renderer, data: Mapping[str, Any]) -> None:
    """Parameter-only restore, for a front-end that mirrors params outside the
    renderer thread (`skinny-gui` applies them to its proxy as well)."""
    _apply_saved_params(renderer, data.get("params", {}), build_all_params(renderer))


# ── persisted CLI flags ─────────────────────────────────────────────

def _one_of(*allowed: Any):
    def check(value):
        return value if value in allowed else None
    return check


def _as_bool(value):
    return bool(value)


def _as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# Flags `cli_common.add_render_flags` documents as "persisted on the interactive
# front-ends": key → (flag spelling, env var, validator). The validator rejects a
# stale/garbage persisted value by returning None, which falls back to the CLI
# value. `skinny-gui` used to restore only `sppm_glossy_roughness`, so the
# documented persistence held on one front-end of two.
PERSISTED_FLAGS: dict[str, tuple[str, str, Any]] = {
    "neural_handoff": (
        "--neural-handoff", "SKINNY_NEURAL_HANDOFF", _one_of("file", "interop", "shared"),
    ),
    "neural_trainer": (
        "--neural-trainer", "SKINNY_NEURAL_TRAINER", _one_of("cpu", "cuda", "mlx", "auto"),
    ),
    "train_precision": (
        "--train-precision", "SKINNY_TRAIN_PRECISION", _one_of("fp32", "fp16"),
    ),
    "online_training": (
        "--online-training", "SKINNY_ONLINE_TRAINING", _as_bool,
    ),
    "sppm_glossy_roughness": (
        "--sppm-glossy-roughness", "SKINNY_SPPM_GLOSSY_ROUGHNESS", _as_float,
    ),
}


def resolve_persisted_flag(
    key: str,
    cli_value: Any,
    saved: Mapping[str, Any],
    *,
    argv: list[str] | None = None,
    environ: Mapping[str, str] | None = None,
) -> Any:
    """Resolve one persisted flag: an explicit CLI flag or env var wins, else the
    persisted value, else ``cli_value`` (the argparse default).

    The explicit-CLI test is the flag's presence in ``argv``, not its value —
    argparse cannot distinguish a default from a value typed by hand.
    """
    flag, env, coerce = PERSISTED_FLAGS[key]
    argv = sys.argv if argv is None else argv
    environ = os.environ if environ is None else environ
    if flag in argv or environ.get(env):
        return cli_value
    raw = saved.get(key)
    if raw is None:
        return cli_value
    value = coerce(raw)
    return cli_value if value is None else value


# ── contributed sections ────────────────────────────────────────────

def contribute(
    shared: Mapping[str, Any],
    contributed: Mapping[str, Any],
    *,
    owned: frozenset[str],
) -> dict[str, Any]:
    """Merge a front-end's own keys onto the shared section.

    `owned` is the caller's declared section (`GLFW_KEYS` or `QT_KEYS`), stated at
    the call site so the check is per-front-end rather than against the union: a
    key one front-end owns is refused to the other, which would otherwise write it
    once and have it erased on the owner's next exit. An undeclared key is refused
    outright — declare it in `GLFW_KEYS` / `QT_KEYS` first.
    """
    if not owned <= CONTRIBUTED_KEYS:
        raise ValueError(
            f"{sorted(owned - CONTRIBUTED_KEYS)} is not a declared contributed section"
        )
    unknown = sorted(set(contributed) - owned)
    if unknown:
        raise ValueError(
            "undeclared session-settings key(s) "
            f"{unknown}: declare them in skinny.session_snapshot "
            "(GLFW_KEYS / QT_KEYS) before contributing them"
        )
    out = dict(shared)
    out.update(contributed)
    return out
