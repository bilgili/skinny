# Settable Camera Distance with 10× Initial Cap — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the orbit-camera distance field accept any value while the scene-size cap (now `10×` longest edge, keeping the 50-unit floor) becomes the initial seed/slider bound that auto-grows when the user types or zooms past it.

**Architecture:** `OrbitCamera.max_distance` stays the single live ceiling. A new `OrbitCamera.set_distance(v)` lower-clamps to 0.5, raises `max_distance` when `v` exceeds it (never shrinks), and is the one write path used by wheel-zoom and the renderer's `apply_camera_param`. The Qt and Panel scene-graph distance widgets allow free numeric entry and rescale their slider range in place to follow `max_distance`. A `growable` metadata flag on the distance property gates the widget behaviour so other float properties are unaffected.

**Tech Stack:** Python 3.12/3.13, NumPy, PySide/Qt (`skinny-gui`), Panel (`skinny-web`), pytest, ruff.

**Spec:** `docs/superpowers/specs/2026-05-27-camera-distance-any-value-design.md`

---

## File Structure

- `src/skinny/renderer.py` — cap formula `_orbit_distance_cap`; `OrbitCamera.set_distance` + `zoom`; `Renderer.apply_camera_param` distance branch. (Shared core.)
- `src/skinny/app.py` — settings-restore distance clamp.
- `src/skinny/scene_graph.py` — `growable` flag on the distance property metadata.
- `src/skinny/ui/qt/windows/scene_graph.py` — Qt `_add_float`: free entry + live slider rescale.
- `src/skinny/ui/panel/windows.py` — Panel float widget: `EditableFloatSlider` + grow.
- `tests/test_camera_placement.py` — extend `TestDistanceCap` (renderer + scene-graph + settings-restore logic).

## Notes on running the renderer-gated tests

`tests/test_camera_placement.py` gates renderer-importing tests behind `@needs_renderer`, which skips when `import skinny.renderer` fails (no Vulkan SDK on the dylib path). In the plain `.venv` (3.12) these tests **skip**. To actually execute them, use the repo-root 3.13 venv with the Vulkan SDK exported (per `CLAUDE.md`):

```bash
export VULKAN_SDK=/Users/ahmetbilgili/VulkanSDK/1.4.341.1/macOS
export DYLD_LIBRARY_PATH=$VULKAN_SDK/lib
./bin/python3.13 -m pytest tests/test_camera_placement.py -v
```

Each task's "Run test" step assumes this environment so the gated tests run rather than skip.

---

### Task 1: Cap multiplier 4× → 10×

**Files:**
- Modify: `src/skinny/renderer.py:620-627` (`_orbit_distance_cap`)
- Test: `tests/test_camera_placement.py:198-215` (`test_cap_floor_and_scaling`, `test_frame_mesh_sets_cap`)

- [ ] **Step 1: Update the failing tests**

In `tests/test_camera_placement.py`, replace `test_cap_floor_and_scaling` and `test_frame_mesh_sets_cap` inside `class TestDistanceCap`:

```python
    def test_cap_floor_and_scaling(self):
        from skinny.renderer import _orbit_distance_cap
        assert _orbit_distance_cap(2.0) == 50.0       # small scene → 50 floor
        assert _orbit_distance_cap(5.0) == 50.0       # boundary: 10×5 == 50
        assert _orbit_distance_cap(20.0) == 200.0     # large scene → 10×longest

    def test_frame_mesh_sets_cap(self):
        import types
        from skinny.renderer import Renderer, OrbitCamera
        stub = types.SimpleNamespace(orbit_camera=OrbitCamera())
        # A 40-unit-tall mesh: longest dim 40 → cap = max(50, 400) = 400.
        positions = np.array(
            [[-1, 0, -1], [1, 0, 1], [-1, 40, -1], [1, 40, 1]], dtype=np.float32
        )
        source = types.SimpleNamespace(positions=positions)
        Renderer._frame_camera_to_mesh(stub, source)
        assert stub.orbit_camera.max_distance == 400.0
        assert stub.orbit_camera.distance <= 400.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./bin/python3.13 -m pytest tests/test_camera_placement.py::TestDistanceCap::test_cap_floor_and_scaling tests/test_camera_placement.py::TestDistanceCap::test_frame_mesh_sets_cap -v`
Expected: FAIL — `_orbit_distance_cap(20.0)` returns `80.0` (not `200.0`); `max_distance` is `160.0` (not `400.0`).

- [ ] **Step 3: Change the multiplier**

In `src/skinny/renderer.py`, replace `_orbit_distance_cap`:

```python
def _orbit_distance_cap(longest_dim: float) -> float:
    """Initial max orbit distance for a scene whose longest AABB edge is
    ``longest_dim``.

    At least 10× the longest dimension so large scenes can be framed and
    zoomed out, never below the legacy 50-unit floor for small scenes. This is
    the *initial* ceiling only — ``OrbitCamera.set_distance`` raises
    ``max_distance`` past this when the user types or zooms further out.
    """
    return float(max(50.0, 10.0 * longest_dim))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./bin/python3.13 -m pytest tests/test_camera_placement.py::TestDistanceCap -v`
Expected: PASS for both (other `TestDistanceCap` tests may still fail until later tasks — that is expected).

- [ ] **Step 5: Commit**

```bash
git add src/skinny/renderer.py tests/test_camera_placement.py
git commit -m "feat(camera): scale orbit-distance cap to 10x longest edge"
```

---

### Task 2: `OrbitCamera.set_distance` + grow-on-zoom

**Files:**
- Modify: `src/skinny/renderer.py:734-737` (`OrbitCamera.zoom`; add `set_distance` above it)
- Test: `tests/test_camera_placement.py` (`TestDistanceCap`: add two tests, rewrite `test_zoom_respects_dynamic_cap`)

- [ ] **Step 1: Write/rewrite the failing tests**

In `class TestDistanceCap`, **delete** `test_zoom_respects_dynamic_cap` and add:

```python
    def test_set_distance_grows_max(self):
        from skinny.renderer import OrbitCamera
        cam = OrbitCamera()                  # max_distance defaults to 50
        cam.set_distance(120.0)
        assert cam.distance == 120.0
        assert cam.max_distance == 120.0     # grew to fit the larger value
        cam.set_distance(10.0)               # within ceiling → does not shrink
        assert cam.distance == 10.0
        assert cam.max_distance == 120.0

    def test_set_distance_clamps_floor_and_guard(self):
        from skinny.renderer import OrbitCamera
        cam = OrbitCamera()
        cam.set_distance(-5.0)
        assert cam.distance == 0.5           # lower floor
        cam.set_distance(1e12)
        assert cam.distance == 1e9           # upper degeneracy guard
        assert cam.max_distance == 1e9

    def test_zoom_grows_past_ceiling(self):
        from skinny.renderer import OrbitCamera
        cam = OrbitCamera()
        cam.max_distance = 200.0
        cam.distance = 199.0
        for _ in range(50):                  # zoom-out: distance *= 1.1 each step
            cam.zoom(-1.0)
        assert cam.distance > 200.0          # grew past the old ceiling
        assert cam.max_distance == cam.distance
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./bin/python3.13 -m pytest tests/test_camera_placement.py::TestDistanceCap -v`
Expected: FAIL — `OrbitCamera` has no `set_distance` (AttributeError); `zoom` still clamps so `test_zoom_grows_past_ceiling` would see `distance == 200.0`.

- [ ] **Step 3: Add `set_distance` and route `zoom` through it**

In `src/skinny/renderer.py`, replace the `zoom` method (currently lines 734-737) with:

```python
    def set_distance(self, value: float) -> None:
        """Set the orbit distance to any value ≥ 0.5, growing ``max_distance``
        to fit.

        ``max_distance`` is the current ceiling (slider range + wheel-zoom
        limit). Writing a larger distance raises it so the UI stays consistent;
        it never shrinks here — only a re-frame/model-load resets it. The 1e9
        cap is a degeneracy guard (avoids inf/NaN and int-slider precision
        loss), effectively unbounded for real scenes.
        """
        v = float(np.clip(value, 0.5, 1e9))
        if v > self.max_distance:
            self.max_distance = v
        self.distance = v

    def zoom(self, delta: float) -> None:
        self.set_distance(self.distance * (1.0 - delta * 0.1))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./bin/python3.13 -m pytest tests/test_camera_placement.py::TestDistanceCap -v`
Expected: PASS for `test_set_distance_grows_max`, `test_set_distance_clamps_floor_and_guard`, `test_zoom_grows_past_ceiling`.

- [ ] **Step 5: Commit**

```bash
git add src/skinny/renderer.py tests/test_camera_placement.py
git commit -m "feat(camera): add OrbitCamera.set_distance that grows max_distance"
```

---

### Task 3: Route `apply_camera_param` distance through `set_distance`

**Files:**
- Modify: `src/skinny/renderer.py:4720-4722` (`Renderer.apply_camera_param`, orbit `distance` branch)
- Test: `tests/test_camera_placement.py` (`TestDistanceCap`: add one test)

- [ ] **Step 1: Write the failing test**

In `class TestDistanceCap`, add:

```python
    def test_apply_camera_param_distance_grows(self):
        import types
        from skinny.renderer import Renderer, OrbitCamera
        cam = OrbitCamera()                  # max_distance defaults to 50
        stub = types.SimpleNamespace(
            camera=cam, camera_mode="orbit", _material_version=0,
        )
        Renderer.apply_camera_param(stub, "distance", 333.0)
        assert cam.distance == 333.0
        assert cam.max_distance == 333.0     # grew via set_distance
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./bin/python3.13 -m pytest tests/test_camera_placement.py::TestDistanceCap::test_apply_camera_param_distance_grows -v`
Expected: FAIL — the branch clamps to `max_distance` (50), so `distance == 50.0` and `max_distance == 50.0`.

- [ ] **Step 3: Use `set_distance` in the distance branch**

In `src/skinny/renderer.py`, in `apply_camera_param`, change the orbit `distance` branch (line 4722):

```python
        elif self.camera_mode == "orbit":
            if key == "distance":
                cam.set_distance(v)
            elif key in ("target_x", "target_y", "target_z"):
                axis = "xyz".index(key[-1])
                cam.target[axis] = v
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./bin/python3.13 -m pytest tests/test_camera_placement.py::TestDistanceCap::test_apply_camera_param_distance_grows -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/skinny/renderer.py tests/test_camera_placement.py
git commit -m "feat(camera): route apply_camera_param distance through set_distance"
```

---

### Task 4: Settings-restore honours large distances

**Files:**
- Modify: `src/skinny/app.py:78` (`_apply_saved_camera`, orbit distance clamp)
- Test: `tests/test_camera_placement.py` (`TestDistanceCap`: add one test)

- [ ] **Step 1: Write the failing test**

In `class TestDistanceCap`, add:

```python
    def test_restore_honors_large_distance(self):
        import types
        from skinny.renderer import OrbitCamera
        from skinny.app import _apply_saved_camera
        stub = types.SimpleNamespace(orbit_camera=OrbitCamera(), free_camera=None)
        _apply_saved_camera(stub, {"orbit": {"distance": 250.0}})
        assert stub.orbit_camera.distance == 250.0
        assert stub.orbit_camera.max_distance >= 250.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./bin/python3.13 -m pytest tests/test_camera_placement.py::TestDistanceCap::test_restore_honors_large_distance -v`
Expected: FAIL — current code clamps to `[0.5, 50.0]`, so `distance == 50.0`.

- [ ] **Step 3: Replace the stale 50-unit clamp**

In `src/skinny/app.py`, in `_apply_saved_camera`, replace line 78:

```python
        o.distance = max(0.5, _flt(orbit_raw.get("distance"), o.distance))
        o.max_distance = max(o.max_distance, o.distance)
```

(`_flt` is the local helper defined earlier in the same function.)

- [ ] **Step 4: Run test to verify it passes**

Run: `./bin/python3.13 -m pytest tests/test_camera_placement.py::TestDistanceCap::test_restore_honors_large_distance -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/skinny/app.py tests/test_camera_placement.py
git commit -m "fix(camera): keep restored orbit distance above the 50-unit cap"
```

---

### Task 5: `growable` flag on the distance property

**Files:**
- Modify: `src/skinny/scene_graph.py:921-926` (`inject_renderer_camera`, distance property)
- Test: `tests/test_camera_placement.py` (`TestDistanceCap`: extend `test_scene_graph_slider_uses_max_distance`)

- [ ] **Step 1: Extend the failing test**

In `class TestDistanceCap`, replace `test_scene_graph_slider_uses_max_distance` with:

```python
    def test_scene_graph_slider_uses_max_distance(self):
        from skinny.renderer import OrbitCamera
        from skinny.scene_graph import SceneGraphNode, inject_renderer_camera
        cam = OrbitCamera()
        cam.max_distance = 160.0
        root = SceneGraphNode(path="/", name="root", type_name="Scope")
        inject_renderer_camera(root, cam, "orbit")
        synth = next(c for c in root.children if c.path == "/Skinny/MainCamera")
        dist = next(p for p in synth.properties if p.name == "distance")
        assert dist.metadata["max"] == 160.0
        assert dist.metadata.get("growable") is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./bin/python3.13 -m pytest tests/test_camera_placement.py::TestDistanceCap::test_scene_graph_slider_uses_max_distance -v`
Expected: FAIL — `dist.metadata.get("growable")` is `None`.

- [ ] **Step 3: Add the flag**

In `src/skinny/scene_graph.py`, in `inject_renderer_camera`, update the distance property's metadata (line 925):

```python
        node.properties.append(SceneGraphProperty(
            name="distance", display_name="distance",
            type_name="float", value=float(camera.distance),
            editable=True,
            metadata={
                "min": 0.5,
                "max": float(getattr(camera, "max_distance", 50.0)),
                "growable": True,
            },
        ))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./bin/python3.13 -m pytest tests/test_camera_placement.py::TestDistanceCap::test_scene_graph_slider_uses_max_distance -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/skinny/scene_graph.py tests/test_camera_placement.py
git commit -m "feat(camera): mark distance scene-graph property growable"
```

---

### Task 6: Qt field — free entry + live slider rescale

**Files:**
- Modify: `src/skinny/ui/qt/windows/scene_graph.py:310-356` (`_add_float`)

No unit test — this repo has no Qt widget tests. Verify manually in `skinny-gui`.

- [ ] **Step 1: Replace `_add_float`**

In `src/skinny/ui/qt/windows/scene_graph.py`, replace the whole `_add_float` method with:

```python
    def _add_float(
        self, layout: QHBoxLayout, node: SceneGraphNode, prop: SceneGraphProperty,
    ) -> None:
        lo = float(prop.metadata.get("min", 0.0))
        hi = float(prop.metadata.get("max", 1.0))
        cur = float(prop.value)
        growable = bool(prop.metadata.get("growable"))

        slider = QSlider(Qt.Horizontal)
        slider.setRange(0, 1000)
        spin = QDoubleSpinBox()
        spin.setRange(lo, 1e9 if growable else hi)
        spin.setDecimals(3)
        # Mutable mapping bounds so a growable range can be rescaled in place
        # without rebuilding the widget (preserves the slider grab mid-drag).
        rng = {"hi": hi, "span": max(hi - lo, 1e-9)}

        def to_int(v: float) -> int:
            return int(round((v - lo) / rng["span"] * 1000.0))

        def from_int(i: int) -> float:
            return lo + (i / 1000.0) * rng["span"]

        with QSignalBlocker(slider), QSignalBlocker(spin):
            slider.setValue(to_int(cur))
            spin.setValue(cur)

        def apply(v: float) -> None:
            prop.value = float(v)
            self._apply_property(node, prop, float(v))

        slider.valueChanged.connect(lambda i: (spin.setValue(from_int(i))))
        spin.valueChanged.connect(lambda v: (
            self._update_slider_from_spin(slider, to_int, v),
            apply(v),
        ))

        layout.addWidget(slider, stretch=1)
        layout.addWidget(spin)

        if node.renderer_ref is not None and node.renderer_ref.kind == "renderer_camera":
            def pull() -> None:
                cam = self.renderer.camera
                live = _read_camera_param(cam, prop.name)
                if live is None:
                    return
                if growable:
                    live_max = float(getattr(cam, "max_distance", rng["hi"]))
                    if abs(live_max - rng["hi"]) > 1e-4:
                        rng["hi"] = live_max
                        rng["span"] = max(live_max - lo, 1e-9)
                        with QSignalBlocker(slider):
                            slider.setValue(to_int(float(live)))
                if abs(spin.value() - float(live)) > 1e-4:
                    with QSignalBlocker(slider), QSignalBlocker(spin):
                        spin.setValue(float(live))
                        slider.setValue(to_int(float(live)))
                    prop.value = float(live)
            self._pulls.append(pull)
```

Key changes vs. the original: `growable` read from metadata; spinbox upper range becomes `1e9` when growable (free entry); `hi`/`span` moved into a mutable `rng` dict that `to_int`/`from_int` read; the `pull()` rescales `rng` when the live `max_distance` grows.

- [ ] **Step 2: Lint**

Run: `.venv/bin/ruff check src/skinny/ui/qt/windows/scene_graph.py`
Expected: no new errors.

- [ ] **Step 3: Manual verification**

Run (with the Vulkan env from the "Notes" section exported): `./bin/python3.13 -c "from skinny.ui.qt.app import main; main()"`

1. Open **Scene Graph**, select `/Skinny/MainCamera`, find the **distance** row.
2. In the numeric spinbox, type a value **larger than the slider's right end** (e.g. if the slider tops out near 50, type `400`). Expected: the camera dollies out, the value applies, and the slider thumb is **not** pinned at the far right — its range has grown so the thumb sits proportionally.
3. Scroll the mouse wheel to zoom **out** in the viewport past the initial cap. Expected: the distance slider's range grows to follow; thumb tracks rather than sticking at the end.
4. Drag the slider — it still maps smoothly over `[0.5, current max]`.

- [ ] **Step 4: Commit**

```bash
git add src/skinny/ui/qt/windows/scene_graph.py
git commit -m "feat(camera): Qt distance field accepts any value, slider auto-grows"
```

---

### Task 7: Panel field — `EditableFloatSlider` + grow

**Files:**
- Modify: `src/skinny/ui/panel/windows.py:147-161` (float-property branch in the property-widget builder)

No unit test — verify manually in `skinny-web`.

- [ ] **Step 1: Replace the float branch**

In `src/skinny/ui/panel/windows.py`, replace the `if prop.type_name == "float" and prop.editable:` block (lines 147-161) with:

```python
    if prop.type_name == "float" and prop.editable:
        lo = float(prop.metadata.get("min", 0.0))
        hi = float(prop.metadata.get("max", 1.0))
        step = (hi - lo) / 100.0 if hi > lo else 0.01

        if prop.metadata.get("growable"):
            w = pn.widgets.EditableFloatSlider(
                name=prop.display_name, start=lo, end=hi,
                fixed_start=lo, fixed_end=1e9, step=step,
                value=float(prop.value),
            )

            def on_change(event, p=prop, r=ref, sl=w):
                with session._lock:
                    _apply_prop_value(renderer, r, p, float(event.new))
                if float(event.new) > sl.end:
                    sl.end = float(event.new)

            w.param.watch(on_change, "value")
            return w

        w = pn.widgets.FloatSlider(
            name=prop.display_name, start=lo, end=hi,
            step=step, value=float(prop.value),
        )

        def on_change(event, p=prop, r=ref):
            with session._lock:
                _apply_prop_value(renderer, r, p, float(event.new))

        w.param.watch(on_change, "value")
        return w
```

`EditableFloatSlider` exposes a numeric text field alongside the slider; `fixed_end=1e9` lets the user type past the soft `end` (the slider track), and the watcher grows `end` so the track follows.

- [ ] **Step 2: Lint**

Run: `.venv/bin/ruff check src/skinny/ui/panel/windows.py`
Expected: no new errors.

- [ ] **Step 3: Manual verification**

Run (Vulkan env exported): `./bin/python3.13 -c "from skinny.web_app import main; main()"` and open the served URL.

1. Open the scene-graph panel, select the camera, find **distance** — it renders as an editable slider (slider + number box).
2. Type a value past the slider's right end (e.g. `400`). Expected: applies, camera dollies out, and the slider track extends so the handle is not pinned.
3. Plain bounded float properties (e.g. `fov`) still render as a normal `FloatSlider`.

- [ ] **Step 4: Commit**

```bash
git add src/skinny/ui/panel/windows.py
git commit -m "feat(camera): Panel distance field accepts any value, slider auto-grows"
```

---

### Task 8: Full verification

**Files:** none (verification only)

- [ ] **Step 1: Run the full camera test module (gated tests execute)**

Run:
```bash
export VULKAN_SDK=/Users/ahmetbilgili/VulkanSDK/1.4.341.1/macOS
export DYLD_LIBRARY_PATH=$VULKAN_SDK/lib
./bin/python3.13 -m pytest tests/test_camera_placement.py -v
```
Expected: all `TestDistanceCap` tests PASS (none skipped); other classes pass/skip as before.

- [ ] **Step 2: Run the full suite + lint**

Run:
```bash
./bin/python3.13 -m pytest
.venv/bin/ruff check src/
```
Expected: no failures, no new ruff errors.

- [ ] **Step 3: Confirm both GUIs manually (if not already done in Tasks 6–7)**

`skinny-gui` and `skinny-web`: distance field accepts a value beyond the initial cap and the slider range grows consistently in each.
