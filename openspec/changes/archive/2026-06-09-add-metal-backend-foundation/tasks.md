## 1. Backend-selection seam

- [x] 1.1 Add `--backend {auto,metal,vulkan}` to `add_render_flags` in `cli_common.py` (+ `SKINNY_BACKEND` env; `default=None` sentinel = use env/persisted/auto, mirroring `--integrator`), so all four front-ends share one definition.
- [x] 1.2 Add `src/skinny/backend_select.py`: `select_backend(prefer, *, persisted=None)` applying precedence explicit > env > persisted > `auto`. Per D7, in this foundation phase `auto` resolves to **vulkan** (the renderer is not yet Metal-ready); explicit `metal` returns `metal` only when the Metal device constructs (else a clear `RuntimeError`). `make_context(backend, window, width, height, **kw)` returns a `VulkanContext` or `MetalContext`.
- [x] 1.3 Route all four context-construction sites through `make_context` and thread `args.backend`: `app.py`, `headless.py`, `ui/qt/app.py`, `web_app.py`. Persist/restore the selected backend on the interactive front-ends (`app.py`, `skinny-gui`) like the other flags; a front-end that resolves to `metal` exits with `METAL_FOUNDATION_NOTICE` (full render is P2).
- [x] 1.4 Add `backend_name` ("vulkan"/"metal") + `is_metal` attributes to `VulkanContext` (and `MetalContext`) for downstream Metal-vs-Vulkan guards. Vulkan behavior unchanged.

## 2. Metal device foundation

- [x] 2.1 Spike (resolves O1): confirm SlangPy compiles a Slang compute entry to Metal in-process (slang-rhi loads + dispatches); if not, fall back to shelling `slangc -target metal` → `.metallib` and loading that. Record the outcome in the change before building §2.3.
  - **Outcome (resolved in-process — no `slangc` shell-out needed):** `spy.create_device(type=DeviceType.metal)` →
    `load_module_from_source` → `link_program` → `create_compute_kernel` → `kernel.dispatch(thread_count, vars={...})`
    compiles + dispatches a Slang compute entry on Metal entirely in-process; reading the buffer back via
    `Buffer.to_numpy()` after `wait_for_idle()` returned the exact `arange(N)` the kernel wrote (bit-identical).
    **Gotcha:** the entry point must NOT be named `main` — Slang's Metal target renames `main`→`main_0`
    (`main` is reserved) and the resulting compute-pipeline creation fails with `SLANG_FAIL`. Foundation
    kernels use `computeMain`. **O2 also resolved here:** `device.create_surface(WindowHandle(nswindow=<int>))`
    + `Surface.configure/acquire_next_image/present` drives the swapchain directly (no manual `CAMetalLayer`);
    the GLFW NSWindow pointer comes from `glfw.get_cocoa_window(window)`.
- [x] 2.2 `src/skinny/metal_context.py` — `MetalContext` wrapping a SlangPy `Device(type=metal)`: surface from the GLFW Cocoa `NSWindow` (O2: `device.create_surface(WindowHandle(nswindow))`, slang-rhi `Surface` — no manual `CAMetalLayer`), swapchain-equivalent + `present_clear`. Mirrors the `VulkanContext` consumed surface (`width`/`height`, `compute_queue`/`present_queue` placeholders, `swapchain_info`, `allocate_command_buffers`, `recreate_swapchain`, `destroy`). All four capability flags = `False`. Platform/import-guarded (Apple-Silicon macOS only).
- [x] 2.3 `src/skinny/metal_compute.py` (minimal) — `StorageBuffer`, `StorageImage`, `ComputePipeline` wrapping SlangPy resources, matching the `vk_compute.py` constructor signatures. Resource binding via `dispatch(vars=…)`; uniform blocks (none in P1) via `set_data` byte blobs only — never per-field cursor writes (D4 / the known fence hang).
- [x] 2.4 Trivial compute path: `foundation_trivial.slang` (`computeMain`) dispatched on Metal via `MetalContext` + `metal_compute`, read back = `arange(N)` (verified). Windowed `present_clear` clears + presents frames with the fence signalling each frame (verified by the present smoke).

## 3. Tests

- [x] 3.1 Backend-selection unit tests (`tests/test_backend_select.py`, no GPU / mocked probe): P1 `auto`→vulkan even on a Metal-capable host; explicit `--backend metal` raises clearly when unavailable; `--backend vulkan`→vulkan; the precedence chain (explicit > env > persisted > auto) holds; `--backend` choices+default identical from the one shared definition.
- [x] 3.2 Headless trivial-dispatch parity (`tests/test_metal_foundation.py`): the same kernel on metal vs vulkan yields a byte-identical buffer (both == `arange`); skips cleanly when either backend is unavailable. Verified passing with the Vulkan SDK on the dylib path.
- [x] 3.3 Windowed Metal present smoke (`tests/test_metal_foundation.py`): open + clear + present several frames, every frame's fence signals (`present_clear` returns True); skips without a display. Verified passing on a Mac display.

## 4. Docs

- [x] 4.1 `README.md`: added the `--backend {auto,metal,vulkan}` flag description (new "GPU backend" section).
- [x] 4.2 `docs/Architecture.md`: documented the backend-selection seam (`backend_select.make_context`) + the `MetalContext` foundation (new "Backend selection" section), noted full render parity is staged in later changes, and flagged the `gfx/` ABC as distinct/unused.
- [x] 4.3 `CHANGELOG.md`: noted the native Metal backend foundation + `--backend` selection. Aligned `CLAUDE.md` (Metal plumbing → `metal_context.py`/`metal_compute.py`; backend-selection paragraph) and `openspec/config.yaml` (context + backend lines) with the real modules, scoped to what P1 ships.

## 5. Validate

- [x] 5.1 Ruff: the three new modules + all edited files are clean; the repo's other pre-existing src/ lint findings are untouched (out of scope). Pytest: full suite **collects with zero import errors** (717 tests — the four edited front-ends import fine); `tests/test_backend_select.py` + `tests/test_metal_foundation.py` + `tests/test_cli_common.py` pass (Metal trivial dispatch + Metal↔Vulkan parity + windowed present verified on this Apple-Silicon Mac; parity skips cleanly without the Vulkan SDK). `tests/test_web.py`'s 13 failures are **pre-existing** (raw-`vulkan` device enumeration in this headless session) — reproduced identically on unmodified `main`, so the Vulkan path is byte-unchanged.
- [x] 5.2 `openspec validate add-metal-backend-foundation --strict` → "Change 'add-metal-backend-foundation' is valid".
