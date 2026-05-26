# Clever Camera Placement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When a model loads, rotate Z-up USD geometry to the scene's Y-up and frame the camera at a 3/4 hero angle so the whole model is visible and upright.

**Architecture:** Bake an up-axis correction (Z→Y rotation) into all stage-authored transforms inside `usd_loader._read_open_stage` — the single function every USD load path funnels through — so the TLAS, bounds, lights, camera, and scene graph all see one consistent Y-up scene with no renderer/shader changes. Separately, change the two auto-frame functions in `renderer.py` to set a 3/4 hero orientation (yaw 30°, pitch 15°) instead of the current flat yaw=0/pitch=0, and make the F-key reset re-run that framing for the loaded model.

**Tech Stack:** Python 3.12+, numpy, OpenUSD (`pxr`), pytest.

**Spec:** `docs/superpowers/specs/2026-05-26-clever-camera-placement-design.md`

---

## Background the engineer needs

**Matrix convention (read carefully).** This codebase stores 4×4 transforms in
*math-transpose* form: a point is a **row vector** and transforms multiply on
the right — `p_world_row = p_local_row @ M_stored`. USD's
`ComputeLocalToWorldTransform` already returns matrices in exactly this stored
form (`usd_loader._world_transform`), and `MeshInstance.world_bounds`
(`scene.py:163`) applies them as `corners @ self.transform`.

To add a **world-space rotation `R` about the origin** *after* an instance's
own transform, in column convention you'd write `p' = R · M · p`. Converted to
this codebase's row/stored convention that becomes:

```
M_stored_new = M_stored @ Rᵀ
```

and for a bare world-space vector (light position, direction, camera
position/forward):

```
v_new_row = v_row @ Rᵀ
```

So everything is a right-multiply by `Rᵀ`. We therefore build `Rᵀ` directly and
never need the un-transposed `R`.

**Z-up → Y-up rotation.** The standard conversion is −90° about X. In column
convention `R = [[1,0,0],[0,0,1],[0,-1,0]]` (maps +Z→+Y, +Y→−Z). Its transpose,
which is what we actually use, is:

```
Rᵀ = [[1, 0,  0],
      [0, 0,  1],
      [0, -1, 0]]
```

Wait — verify by hand (this is the matrix the code uses, so get it right):
`R = [[1,0,0],[0,0,1],[0,-1,0]]`, so `Rᵀ = [[1,0,0],[0,0,-1],[0,1,0]]`.
Check `(0,0,1) @ Rᵀ` = row picked by the `1` in slot 2 = `Rᵀ[2]` = `(0,1,0)` ✓
(+Z maps to +Y). Check `(0,1,0) @ Rᵀ` = `Rᵀ[1]` = `(0,0,-1)` ✓ (+Y maps to −Z).
**The correct matrix to write in code is `Rᵀ = [[1,0,0],[0,0,-1],[0,1,0]]`.**

**Where USD loads funnel.** `load_scene_from_usd` (disk) and
`load_scene_from_stage` (open stage) both call `_read_open_stage`;
`_read_usd_stage` opens the file then calls `_read_open_stage`. So a single
correction inside `_read_open_stage` covers every entry point including the
live app's streaming loader. Mesh instances are baked *after* `_read_open_stage`
returns, from the `prim_data` list of `(MeshSource, transform, material_id)`
tuples it produces — so correcting the transforms inside `prim_data` propagates
into the baked `MeshInstance`s. Emissive-light instances are already populated
in `scene.instances` at correction time and must be corrected directly.

**Setup / running tests.** From repo root:
```bash
.venv/bin/pytest tests/test_camera_placement.py -v
```
The up-axis tests need only `pxr` + numpy (CPU). They are gated with a
`needs_usd` skip identical to `tests/test_usd_gprims.py`.

---

## File Structure

- **Create** `tests/test_camera_placement.py` — all new tests for this feature:
  up-axis correction (CPU, needs `pxr`), the hero-angle orientation helper (CPU,
  pure numpy), and the F-key reframe decision logic (CPU, via a lightweight stub
  + the unbound `Renderer.reset_camera` method). One focused test file.
- **Modify** `src/skinny/usd_loader.py` — add `_apply_up_axis_correction(...)`
  helper near `_world_transform`; call it at the end of `_read_open_stage`.
- **Modify** `src/skinny/renderer.py` — add hero-angle orientation to
  `_frame_camera_to_scene` and `_frame_camera_to_mesh` (via a shared private
  helper `_apply_hero_orientation`); rewrite `reset_camera` to re-frame the
  loaded model.

---

## Task 1: Up-axis rotation matrix helper

**Files:**
- Modify: `src/skinny/usd_loader.py` (add helper after `_world_transform`, ~line 1356)
- Test: `tests/test_camera_placement.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_camera_placement.py`:

```python
"""Up-axis correction on USD load + camera hero-angle framing."""

from __future__ import annotations

import numpy as np
import pytest


def _have_usd() -> bool:
    try:
        import pxr  # noqa: F401
        return True
    except Exception:
        return False


needs_usd = pytest.mark.skipif(not _have_usd(), reason="OpenUSD (pxr) not installed")


class TestUpAxisRotation:
    def test_y_up_returns_none(self):
        from skinny.usd_loader import _up_axis_rt
        assert _up_axis_rt("Y") is None

    def test_z_up_matrix_maps_z_to_y(self):
        from skinny.usd_loader import _up_axis_rt
        rt = _up_axis_rt("Z")
        assert rt is not None
        assert rt.shape == (3, 3)
        # Row-vector right-multiply: +Z world axis maps to +Y.
        np.testing.assert_allclose(np.array([0, 0, 1], np.float32) @ rt,
                                   np.array([0, 1, 0], np.float32), atol=1e-6)
        # +Y maps to -Z.
        np.testing.assert_allclose(np.array([0, 1, 0], np.float32) @ rt,
                                   np.array([0, 0, -1], np.float32), atol=1e-6)
        # +X unchanged.
        np.testing.assert_allclose(np.array([1, 0, 0], np.float32) @ rt,
                                   np.array([1, 0, 0], np.float32), atol=1e-6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_camera_placement.py::TestUpAxisRotation -v`
Expected: FAIL — `ImportError: cannot import name '_up_axis_rt'`.

- [ ] **Step 3: Write minimal implementation**

In `src/skinny/usd_loader.py`, add after `_world_transform` (after line ~1356,
before `# ─── Public entry point ───`):

```python
def _up_axis_rt(up_axis: str) -> "Optional[np.ndarray]":
    """Stored-form (transpose) rotation that maps a stage's up axis to +Y.

    Returns ``None`` for a Y-up stage (no correction needed). For a Z-up
    stage returns ``Rᵀ`` for the −90°-about-X rotation, ready to
    right-multiply this codebase's row-vector/stored transforms:
    ``M_new = M_stored @ rt`` and ``v_new = v_row @ rt``.

    Rᵀ maps +Z→+Y and +Y→−Z.
    """
    if up_axis != "Z":
        return None
    return np.array(
        [[1.0, 0.0, 0.0],
         [0.0, 0.0, -1.0],
         [0.0, 1.0, 0.0]],
        dtype=np.float32,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_camera_placement.py::TestUpAxisRotation -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add -f src/skinny/usd_loader.py tests/test_camera_placement.py
git commit -m "feat(usd): add up-axis rotation matrix helper"
```

---

## Task 2: Apply correction to a loaded Scene

**Files:**
- Modify: `src/skinny/usd_loader.py` (add `_apply_up_axis_correction` after `_up_axis_rt`; call it in `_read_open_stage` before the `return`, ~line 1424)
- Test: `tests/test_camera_placement.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_camera_placement.py`:

```python
@needs_usd
class TestUpAxisCorrectionOnLoad:
    def _z_up_stage_with_tall_mesh(self):
        """A stage where geometry is tall along +Z (Z-up), 1 unit thin in X/Y."""
        from pxr import Usd, UsdGeom
        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        mesh = UsdGeom.Mesh.Define(stage, "/m")
        # A thin column standing along +Z: spans z in [0, 4], x/y in [-0.5, 0.5].
        pts = [(-0.5, -0.5, 0.0), (0.5, -0.5, 0.0), (0.5, 0.5, 0.0),
               (-0.5, 0.5, 0.0), (-0.5, -0.5, 4.0), (0.5, -0.5, 4.0),
               (0.5, 0.5, 4.0), (-0.5, 0.5, 4.0)]
        mesh.GetPointsAttr().Set([(float(x), float(y), float(z)) for x, y, z in pts])
        mesh.GetFaceVertexCountsAttr().Set([4, 4])
        mesh.GetFaceVertexIndicesAttr().Set([0, 1, 2, 3, 4, 5, 6, 7])
        return stage

    def test_z_up_mesh_stands_along_y_after_load(self):
        from skinny.usd_loader import load_scene_from_stage
        stage = self._z_up_stage_with_tall_mesh()
        scene = load_scene_from_stage(stage)
        amin, amax = scene.world_bounds()
        ext = amax - amin
        # Before correction the long axis was Z (~4). After correction the
        # tall extent must be on Y, and Z must be the short (~1) axis.
        assert ext[1] > 3.0, f"expected tall Y extent, got {ext}"
        assert ext[2] < 1.5, f"expected short Z extent, got {ext}"

    def test_y_up_mesh_unchanged(self):
        from pxr import Usd, UsdGeom
        from skinny.usd_loader import load_scene_from_stage
        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
        mesh = UsdGeom.Mesh.Define(stage, "/m")
        pts = [(-0.5, 0.0, -0.5), (0.5, 0.0, -0.5), (0.5, 0.0, 0.5),
               (-0.5, 0.0, 0.5), (-0.5, 4.0, -0.5), (0.5, 4.0, -0.5),
               (0.5, 4.0, 0.5), (-0.5, 4.0, 0.5)]
        mesh.GetPointsAttr().Set([(float(x), float(y), float(z)) for x, y, z in pts])
        mesh.GetFaceVertexCountsAttr().Set([4, 4])
        mesh.GetFaceVertexIndicesAttr().Set([0, 1, 2, 3, 4, 5, 6, 7])
        scene = load_scene_from_stage(stage)
        amin, amax = scene.world_bounds()
        ext = amax - amin
        assert ext[1] > 3.0 and ext[2] < 1.5, f"Y-up scene should be untouched, got {ext}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_camera_placement.py::TestUpAxisCorrectionOnLoad -v`
Expected: FAIL — `test_z_up_mesh_stands_along_y_after_load` fails because the
tall extent is still on Z (`ext[2] ≈ 4`, `ext[1] ≈ 1`). `test_y_up_mesh_unchanged`
should PASS already (sanity).

- [ ] **Step 3: Write minimal implementation**

In `src/skinny/usd_loader.py`, add immediately after `_up_axis_rt`:

```python
def _apply_up_axis_correction(
    prim_data: "list[tuple[MeshSource, np.ndarray, int]]",
    scene: Scene,
    up_axis: str,
) -> "list[tuple[MeshSource, np.ndarray, int]]":
    """Rotate a Z-up stage's geometry, lights, and camera to scene +Y.

    Returns prim_data with corrected transforms (mesh instances are baked
    from it after this returns). Lights, emissive instances, and the camera
    override already live on ``scene`` and are mutated in place. Y-up stages
    are returned untouched.
    """
    rt = _up_axis_rt(up_axis)
    if rt is None:
        return prim_data

    rt4 = np.eye(4, dtype=np.float32)
    rt4[:3, :3] = rt

    prim_data = [
        (src, (xf @ rt4).astype(np.float32), mat) for (src, xf, mat) in prim_data
    ]
    for inst in scene.instances:  # emissive-light instances
        inst.transform = (inst.transform @ rt4).astype(np.float32)
    for ls in scene.lights_sphere:
        ls.position = (ls.position @ rt).astype(np.float32)
    for ld in scene.lights_dir:
        ld.direction = (ld.direction @ rt).astype(np.float32)
    ov = scene.camera_override
    if ov is not None:
        ov.position = (ov.position @ rt).astype(np.float32)
        ov.forward = (ov.forward @ rt).astype(np.float32)
    return prim_data
```

Then in `_read_open_stage`, replace the final lines (currently):

```python
    partial_scene = Scene(
        instances=list(emissive_instances),
        materials=materials,
        lights_dir=lights_dir,
        lights_sphere=lights_sphere,
        environment=environment,
        camera_override=camera_override,
        mm_per_unit=mm_per_unit,
    )
    return partial_scene, prim_data, (stage if keep_stage else None)
```

with:

```python
    partial_scene = Scene(
        instances=list(emissive_instances),
        materials=materials,
        lights_dir=lights_dir,
        lights_sphere=lights_sphere,
        environment=environment,
        camera_override=camera_override,
        mm_per_unit=mm_per_unit,
    )
    up_axis = str(UsdGeom.GetStageUpAxis(stage))
    prim_data = _apply_up_axis_correction(prim_data, partial_scene, up_axis)
    return partial_scene, prim_data, (stage if keep_stage else None)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_camera_placement.py::TestUpAxisCorrectionOnLoad -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add -f src/skinny/usd_loader.py tests/test_camera_placement.py
git commit -m "feat(usd): rotate Z-up stages to scene Y-up on load"
```

---

## Task 3: Camera override is corrected too

**Files:**
- Test only: `tests/test_camera_placement.py` (implementation already done in Task 2; this locks the behavior)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_camera_placement.py`:

```python
@needs_usd
class TestCameraOverrideCorrection:
    def test_z_up_camera_forward_corrected(self):
        from pxr import Usd, UsdGeom, Gf
        from skinny.usd_loader import _read_open_stage
        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        # Mesh so the stage has geometry (loader requires usable geometry).
        mesh = UsdGeom.Mesh.Define(stage, "/m")
        mesh.GetPointsAttr().Set([(0, 0, 0), (1, 0, 0), (0, 0, 1)])
        mesh.GetFaceVertexCountsAttr().Set([3])
        mesh.GetFaceVertexIndicesAttr().Set([0, 1, 2])
        # Camera with identity xform: USD forward is local -Z = world (0,0,-1)
        # in a Z-up stage. After correction it must rotate to (0,1,0).
        UsdGeom.Camera.Define(stage, "/cam")
        partial_scene, _prim_data, _ = _read_open_stage(stage)
        ov = partial_scene.camera_override
        assert ov is not None
        # A Z-up camera with identity xform looks down local -Z = world (0,0,-1).
        # Correction maps it by Rᵀ: (0,0,-1) @ Rᵀ = -Rᵀ[2] = (0,-1,0).
        np.testing.assert_allclose(ov.forward,
                                   np.array([0, -1, 0], np.float32), atol=1e-5)
```

- [ ] **Step 2: Run test to verify it passes (already implemented)**

Run: `.venv/bin/pytest tests/test_camera_placement.py::TestCameraOverrideCorrection -v`
Expected: PASS — Task 2's `_apply_up_axis_correction` already rotates
`camera_override.forward`. (If it FAILS, the override correction in Task 2 is
wrong — fix Task 2, do not patch here.)

- [ ] **Step 3: (no implementation — behavior from Task 2)**

- [ ] **Step 4: Commit**

```bash
git add -f tests/test_camera_placement.py
git commit -m "test(usd): lock camera-override up-axis correction"
```

---

## Task 4: Hero-angle orientation in the frame helpers

**Files:**
- Modify: `src/skinny/renderer.py` — `_frame_camera_to_scene` (~1253-1256) and `_frame_camera_to_mesh` (~1327-1330); add private helper `_apply_hero_orientation` near them.
- Test: `tests/test_camera_placement.py`

- [ ] **Step 1: Write the failing test**

The frame helpers are `Renderer` methods that read `self.orbit_camera`. Test the
shared orientation helper directly on a minimal object to avoid constructing a
full GPU `Renderer`. Append to `tests/test_camera_placement.py`:

```python
class TestHeroOrientation:
    def test_hero_angles_applied(self):
        from skinny.renderer import OrbitCamera, _hero_yaw_pitch
        yaw, pitch = _hero_yaw_pitch()
        np.testing.assert_allclose(yaw, np.radians(30.0), atol=1e-6)
        np.testing.assert_allclose(pitch, np.radians(15.0), atol=1e-6)
        # Camera sits above and to the side of the target, looking down.
        cam = OrbitCamera()
        cam.target = np.array([0.0, 0.0, 0.0], np.float32)
        cam.distance = 5.0
        cam.yaw, cam.pitch = yaw, pitch
        pos = cam.position
        assert pos[1] > 0.0, "camera should be elevated (pitch>0)"
        assert pos[0] > 0.0, "camera should be turned to +X side (yaw>0)"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_camera_placement.py::TestHeroOrientation -v`
Expected: FAIL — `ImportError: cannot import name '_hero_yaw_pitch'`.

- [ ] **Step 3: Write minimal implementation**

In `src/skinny/renderer.py`, add a module-level helper near `_look_at`
(after line ~612):

```python
def _hero_yaw_pitch() -> tuple[float, float]:
    """Default 3/4 hero-view orbit angles (radians): yaw 30°, pitch 15°."""
    return float(np.radians(30.0)), float(np.radians(15.0))
```

Then in `_frame_camera_to_scene`, replace:

```python
        self.orbit_camera.target = center
        self.orbit_camera.distance = distance
        self.orbit_camera.yaw = 0.0
        self.orbit_camera.pitch = 0.0
```

with:

```python
        yaw, pitch = _hero_yaw_pitch()
        self.orbit_camera.target = center
        self.orbit_camera.distance = distance
        self.orbit_camera.yaw = yaw
        self.orbit_camera.pitch = pitch
```

And in `_frame_camera_to_mesh`, replace the identical four lines:

```python
        self.orbit_camera.target = center
        self.orbit_camera.distance = distance
        self.orbit_camera.yaw = 0.0
        self.orbit_camera.pitch = 0.0
```

with:

```python
        yaw, pitch = _hero_yaw_pitch()
        self.orbit_camera.target = center
        self.orbit_camera.distance = distance
        self.orbit_camera.yaw = yaw
        self.orbit_camera.pitch = pitch
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_camera_placement.py::TestHeroOrientation -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -f src/skinny/renderer.py tests/test_camera_placement.py
git commit -m "feat(camera): 3/4 hero angle for auto-frame on model load"
```

---

## Task 5: F-key reset re-frames the loaded model

**Files:**
- Modify: `src/skinny/renderer.py` — `reset_camera` (~1111-1123)
- Test: `tests/test_camera_placement.py`

- [ ] **Step 1: Write the failing test**

`reset_camera` currently always builds a fresh `OrbitCamera()` (yaw=0/pitch=0).
After the change it must re-run framing for a loaded model. We test the
decision logic without a GPU `Renderer` by constructing a stand-in object that
has the same attributes `reset_camera` touches and calling the *unbound*
method. Append to `tests/test_camera_placement.py`:

```python
class TestResetReframes:
    class _Stub:
        """Minimal stand-in exposing only what reset_camera reads/writes."""
        def __init__(self):
            from skinny.renderer import OrbitCamera, FreeCamera
            self.orbit_camera = OrbitCamera()
            self.free_camera = FreeCamera()
            self.camera_mode = "free"
            self._usd_scene = None
            self._mesh_sources = []
            self._framed = None
            self._refreshed = False

        def _frame_camera_to_scene(self, scene):
            self._framed = ("scene", scene)

        def _frame_camera_to_mesh(self, src):
            self._framed = ("mesh", src)

        def _apply_camera_override(self, scene):
            pass

        def _refresh_camera_node(self):
            self._refreshed = True

    def test_reset_frames_usd_scene(self):
        from skinny.renderer import Renderer
        stub = self._Stub()
        stub._usd_scene = object()
        Renderer.reset_camera(stub)
        assert stub._framed == ("scene", stub._usd_scene)
        assert stub.camera_mode == "orbit"
        assert stub._refreshed

    def test_reset_frames_obj_mesh(self):
        from skinny.renderer import Renderer
        stub = self._Stub()
        src = object()
        stub._mesh_sources = [src]
        Renderer.reset_camera(stub)
        assert stub._framed == ("mesh", src)
        assert stub.camera_mode == "orbit"

    def test_reset_default_when_nothing_loaded(self):
        from skinny.renderer import Renderer
        stub = self._Stub()
        Renderer.reset_camera(stub)
        assert stub._framed is None
        assert stub.camera_mode == "orbit"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_camera_placement.py::TestResetReframes -v`
Expected: FAIL — current `reset_camera` never calls `_frame_camera_to_scene`/
`_frame_camera_to_mesh`, so `_framed` stays `None`.

- [ ] **Step 3: Write minimal implementation**

In `src/skinny/renderer.py`, replace the body of `reset_camera` (lines
~1118-1123):

```python
        self.orbit_camera = OrbitCamera()
        self.free_camera = FreeCamera()
        self.camera_mode = "orbit"
        if self._usd_scene is not None and self._usd_scene.camera_override is not None:
            self._apply_camera_override(self._usd_scene)
        self._refresh_camera_node()
```

with:

```python
        self.orbit_camera = OrbitCamera()
        self.free_camera = FreeCamera()
        self.camera_mode = "orbit"
        if self._usd_scene is not None:
            # Re-frame the loaded scene; honors an authored camera override
            # internally and otherwise applies the hero-angle auto-frame.
            self._frame_camera_to_scene(self._usd_scene)
        elif self._mesh_sources:
            self._frame_camera_to_mesh(self._mesh_sources[0])
        self._refresh_camera_node()
```

> Note: `_frame_camera_to_scene` already calls `_apply_camera_override` when
> `scene.camera_override is not None`, so the explicit override re-apply in the
> old code is preserved through that path — no separate call needed.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_camera_placement.py::TestResetReframes -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add -f src/skinny/renderer.py tests/test_camera_placement.py
git commit -m "feat(camera): F-key reset re-frames the loaded model"
```

---

## Task 6: Full suite + lint

**Files:** none (verification only)

- [ ] **Step 1: Run the new test file**

Run: `.venv/bin/pytest tests/test_camera_placement.py -v`
Expected: all tests PASS (USD tests skip only if `pxr` is unavailable).

- [ ] **Step 2: Run lint on touched files**

Run: `.venv/bin/ruff check src/skinny/usd_loader.py src/skinny/renderer.py tests/test_camera_placement.py`
Expected: no errors.

- [ ] **Step 3: Syntax check**

Run: `.venv/bin/python -m py_compile src/skinny/usd_loader.py src/skinny/renderer.py`
Expected: no output (success).

- [ ] **Step 4: Regression — existing USD + headless suites still pass**

Run: `.venv/bin/pytest tests/test_usd_gprims.py -v`
Expected: unchanged PASS/skip results (correction must not break Y-up gprim loads).

- [ ] **Step 5: Manual smoke (if a GPU env is available)**

Load a known Z-up USD and a Y-up OBJ head in the app; confirm both appear
upright and fully framed at the 3/4 angle, and that pressing **F** re-frames.
If no GPU/display is available, state that explicitly rather than claiming the
UI was verified.

- [ ] **Step 6: Commit any lint fixes**

```bash
git add -f src/skinny/usd_loader.py src/skinny/renderer.py tests/test_camera_placement.py
git commit -m "chore(camera): lint + suite green for camera placement"
```

---

## Self-Review Notes (for the implementer)

- **Spec coverage:** up-axis rotation (Tasks 1-3), hero-angle framing (Task 4),
  F-key reframe (Task 5), edge cases via existing early-returns (untouched),
  testing (every task is TDD). All spec sections map to a task.
- **Watch the camera-override sign:** Task 3 explicitly corrects the expected
  value to `(0,-1,0)`. Use that, not `(0,1,0)`.
- **`git add -f` is required** because the repo's `.gitignore` is `*` and tracks
  files explicitly (existing specs/plans/src are already tracked this way).
