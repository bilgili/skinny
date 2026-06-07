"""Hot-reload pipeline for Python-authored slangpile materials.

The renderer transpiles every `.py` under `python_materials/` into Slang at
compute-pipeline startup via `vk_compute.ComputePipeline._run_codegen`. That
path is silent on failure and only runs once per pipeline build, which is
fine at boot but useless for interactive editing.

`MaterialReloader` lets the GUI (or any caller) re-run codegen for a single
module after editing its source, then trigger a full pipeline rebuild so the
viewport picks up the new Slang. Errors at each stage are captured and
returned as a structured `ReloadResult` so the caller can surface them
inline instead of swallowing them.
"""

from __future__ import annotations

import importlib
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional


ReloadStage = Literal["resolve", "write", "slangpile", "pipeline", "done"]


@dataclass
class ReloadResult:
    """Outcome of a `MaterialReloader.reload` call.

    `stage` reports the last stage that ran (on success this is `"done"`;
    on failure it is the stage that raised). `message` carries the error
    traceback when `ok=False`, empty otherwise.
    """

    ok: bool
    stage: ReloadStage
    message: str = ""
    duration_ms: float = 0.0


def _python_materials_root() -> Path:
    """Repo-root `python_materials/` directory."""
    # src/skinny/material_reloader.py → repo root is parents[2].
    return Path(__file__).resolve().parents[2] / "python_materials"


def _genslang_out_dir(shader_dir: Path) -> Path:
    """Mirror of `ComputePipeline._run_codegen`'s output layout."""
    return shader_dir.parent / "mtlx" / "genslang"


def resolve_source_path(module_name: str) -> Optional[Path]:
    """Convert ``python_materials.<stem>`` → ``<repo>/python_materials/<stem>.py``.

    Returns None if the module name doesn't live under the
    ``python_materials`` package or the file doesn't exist on disk.
    """
    if not module_name.startswith("python_materials."):
        return None
    stem = module_name.split(".", 1)[1]
    if not stem or "." in stem:
        return None
    path = _python_materials_root() / f"{stem}.py"
    return path if path.is_file() else None


class MaterialReloader:
    """Thread-safe orchestrator: write source → run slangpile → rebuild pipeline.

    Caller owns the render-thread lock (typically `RenderViewport._render_lock`)
    and passes it in so this class can serialise its pipeline rebuild against
    the render worker.
    """

    def __init__(self, renderer, render_lock: threading.Lock) -> None:
        self.renderer = renderer
        self._lock = render_lock

    def reload(self, module_name: str, source_text: str) -> ReloadResult:
        t0 = time.perf_counter()

        source_path = resolve_source_path(module_name)
        if source_path is None:
            return ReloadResult(
                ok=False,
                stage="resolve",
                message=(
                    f"Unknown python material module: {module_name!r}. "
                    f"Expected a file under {_python_materials_root()}."
                ),
                duration_ms=(time.perf_counter() - t0) * 1000.0,
            )

        try:
            source_path.write_text(source_text, encoding="utf-8")
        except OSError as exc:
            return ReloadResult(
                ok=False,
                stage="write",
                message=f"Failed to write {source_path}: {exc}",
                duration_ms=(time.perf_counter() - t0) * 1000.0,
            )

        try:
            self._run_slangpile(module_name)
        except Exception:  # noqa: BLE001
            return ReloadResult(
                ok=False,
                stage="slangpile",
                message=traceback.format_exc(),
                duration_ms=(time.perf_counter() - t0) * 1000.0,
            )

        try:
            with self._lock:
                # Forces a fresh ComputePipeline build against the now-
                # regenerated .slang files. Reuses the renderer's existing
                # slangc-failure fallback (renders affected materials
                # magenta and re-raises with stderr attached).
                self.renderer._build_pipeline_for_current_graphs()
                # Bumps the state-hash so progressive accumulation resets
                # on the next frame.
                self.renderer._material_version += 1
        except Exception:  # noqa: BLE001
            return ReloadResult(
                ok=False,
                stage="pipeline",
                message=traceback.format_exc(),
                duration_ms=(time.perf_counter() - t0) * 1000.0,
            )

        return ReloadResult(
            ok=True,
            stage="done",
            message="",
            duration_ms=(time.perf_counter() - t0) * 1000.0,
        )

    def _run_slangpile(self, module_name: str) -> None:
        """Re-import the module from disk and transpile it via slangpile.

        Must be called outside the render lock — slangpile only touches the
        on-disk `.slang` output, not the GPU. Caller wraps the subsequent
        pipeline rebuild in the lock.
        """
        import linecache

        from skinny.slangpile import build_module

        materials_root = _python_materials_root()
        out_dir = _genslang_out_dir(self.renderer.shader_dir)

        # Make the package importable even if the entry point never
        # touched it (e.g. tests that skip ComputePipeline startup).
        repo_root = str(materials_root.parent)
        added_path = False
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
            added_path = True
        try:
            # Drop the cached module + its line-cache entry so slangpile's
            # `inspect.getsource` picks up the new file contents instead of
            # the version Python first imported at app start.
            sys.modules.pop(module_name, None)
            source_path = resolve_source_path(module_name)
            if source_path is not None:
                linecache.checkcache(str(source_path))
            importlib.import_module(module_name)
            build_module(module_name, out_dir)
        finally:
            if added_path:
                try:
                    sys.path.remove(repo_root)
                except ValueError:
                    pass
