# Skinny — Installation

This document covers what skinny needs and how to install it: the platform and
dependency requirements, the virtual environment, and — only for a platform
outside the prebuilt-wheel matrix — the MaterialX from-source build. On a
supported platform `pip install -e ".[dev]"` installs every Python package.

Two prerequisites are external to pip, and this document is the one place that
states them:

- **A Vulkan loader library**, on every platform. `skinny.renderer` imports the
  `vulkan` binding at module scope, and that binding `dlopen`s
  `libvulkan.so.1` / `vulkan-1.dll` / `libvulkan.dylib`, so `import skinny.app`
  fails without one — including when you intend to render on Metal. Its error
  says `Cannot find Vulkan SDK version`, but what it could not find is the
  loader. On Linux and Windows a current Vulkan driver package provides it
  system-wide. macOS has no system Vulkan, so install the LunarG SDK
  (MoltenVK) and put `$VULKAN_SDK/lib` on `DYLD_LIBRARY_PATH`.
- **The Slang compiler `slangc` on `PATH`**, for Vulkan rendering only. No
  SPIR-V is checked in, so the shaders compile on first run. The native Metal
  backend compiles in-process via SlangPy and does not need it.

For the shortest path to a rendered frame see the quick start in
[README.md](../README.md). For how to run the renderer once installed see
[Usage.md](Usage.md).

---

## Requirements

- Python 3.12, 3.13, or 3.14 (`pyproject.toml` sets `requires-python = ">=3.12"`, and the prebuilt wheels start at `cp312`)
- A GPU supported by one of the compute backends: Vulkan 1.2 with a current
  graphics driver, or native Metal on Apple Silicon
- Slang compiler (`slangc`) on `PATH`
- MaterialX **built from source** with the Slang code generator enabled — the
  PyPI wheel does not ship `PyMaterialXGenSlang`. See
  [MaterialX from source (required for the Slang backend)](#materialx-from-source-required-for-the-slang-backend).
- GLFW-compatible desktop environment (only required for the `skinny`
  shader-debug entry; `skinny-gui` runs on Qt and `skinny-web` is headless)

Python dependencies (`pyproject.toml`):

| Package | Purpose |
|---------|---------|
| `numpy` | Linear algebra, mesh processing |
| `slangpy` | Slang shader compilation and reflection |
| `vulkan` | Vulkan API bindings |
| `glfw` | Window creation and input (debug entry) |
| `PySide6` | Qt desktop UI |
| `Pillow` | Image I/O (HDR, textures, tattoos) |
| `imageio[freeimage]` | HDR / EXR screenshot output |
| `MaterialX` | Material definitions and Slang code generation |

Optional:

| Package | Purpose |
|---------|---------|
| `usd-core` | OpenUSD scene loading (`pip install -e ".[usd]"`) |
| `panel` | Web UI framework (`pip install -e ".[web]"`) |
| `bokeh` | Panel dependency (Tornado server) |
| `av` (PyAV) | H264 video encoding via FFmpeg bindings |

## Setup

```powershell
python -m venv .
.\Scripts\python -m pip install --upgrade pip
.\Scripts\python -m pip install -e .
```

For USD scene support:

```powershell
.\Scripts\python -m pip install -e ".[usd]"
```

For web mode (Panel + H264 streaming):

```powershell
.\Scripts\python -m pip install -e ".[web]"
```

For development tools:

```powershell
.\Scripts\python -m pip install -e ".[dev]"
```

### Pre-commit hooks

`.pre-commit-config.yaml` runs `ruff-check` (lint, scoped to `src/`) plus
basic hygiene checks (trailing whitespace, EOF newline, YAML/TOML syntax,
merge conflicts) over the repo minus vendored build output, data/asset dirs,
generated Slang, and the openspec corpus — see the comment atop the config
for the exact exclude list. Install the `[dev]` extra (above), then enable
the git hook:

```bash
.venv/bin/pre-commit install
```

Run it manually against staged changes at any time:

```bash
.venv/bin/pre-commit run
```

If `core.hooksPath` is already customized in this repo (e.g. by another tool's
hook installer), `pre-commit install` refuses rather than clobbering it — run
`pre-commit run` manually in that case, or reconcile the hooks path first.

Verify the Slang compiler:

```powershell
slangc -version
```

### MaterialX from source (required for the Slang backend)

The MaterialX wheel published on PyPI (1.39.x) ships the GLSL, MDL, MSL, and
OSL code generators, but **not** the Slang code generator. Skinny's MaterialX
runtime (`materialx_runtime.py`) imports `PyMaterialXGenSlang` to compile both
the `ND_skinny_layered_skin_stack` skin shader and arbitrary nodegraphs
(`standard_surface`, marble, wood, brass, etc.) into Slang modules at runtime.
Without the Slang generator the renderer fails at import time with
`ImportError: cannot import name 'PyMaterialXGenSlang'`.

**On a supported platform you don't need to do anything below** — `pyproject.toml`
already pulls prebuilt wheels for both packages as base (non-extra) dependencies:

- `materialx-python-standalone` — MaterialX built with `MATERIALX_BUILD_GEN_SLANG=ON`,
  providing `import MaterialX` + `PyMaterialXGenSlang`.
- `openusd-materialx` — OpenUSD (v26.05) built with the `usdMtlx` plugin, providing
  `import pxr`.

Both are published as direct-URL GitHub Release wheels from
[`bilgili/openusd-materialx`](https://github.com/bilgili/openusd-materialx) (not
PyPI — the PyPI `MaterialX`/`usd-core` wheels lack GenSlang/usdMtlx), one entry per
`(python_version, sys_platform, platform_machine)` combination the release ships —
Python 3.12/3.13/3.14 × (`darwin`/`arm64` [Apple Silicon], `linux`/`x86_64`,
`win32`/`AMD64`), matching wheel filename tags `cp312`-`cp314` ×
`macosx_26_0_arm64`/`linux_x86_64`/`win_amd64`. `pip` resolves the matching
entry automatically from the environment markers, so a plain `pip install -e .`
(or `-e ".[dev]"`) installs the Slang- and usdMtlx-capable builds directly —
no compiler, no CMake.

The manual from-source build below is only needed if your platform isn't in that
matrix (e.g. Linux aarch64, Intel macOS) or you need a newer MaterialX than the
pinned `v1.0.11` release provides.

> **The recipe below covers MaterialX only.** Off the wheel matrix the
> environment markers skip *both* direct dependencies, so `openusd-materialx`
> is not installed either — and `skinny.usd_loader` imports `pxr`
> unconditionally, so the renderer still fails with `ModuleNotFoundError: pxr`
> after a MaterialX-only build. Such a platform also needs OpenUSD built with
> the `usdMtlx` plugin (`PXR_BUILD_USD_IMAGING`/`usdMtlx` enabled); see
> [`bilgili/openusd-materialx`](https://github.com/bilgili/openusd-materialx)
> for the build this project's wheels are produced from. That build is not
> scripted here.

Build and install MaterialX with Python bindings + Slang generator enabled:

```bash
# 1. Clone upstream MaterialX (>= 1.39)
git clone --depth 1 https://github.com/AcademySoftwareFoundation/MaterialX.git
cd MaterialX

# 2. Configure with Python bindings and the Slang generator enabled.
#    Point MATERIALX_PYTHON_EXECUTABLE at the same interpreter you will use
#    to run skinny (your venv's python), so the bindings match its ABI.
cmake -S . -B build \
  -DMATERIALX_BUILD_PYTHON=ON \
  -DMATERIALX_BUILD_GEN_SLANG=ON \
  -DMATERIALX_PYTHON_EXECUTABLE="$(pwd)/../.venv/bin/python" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$(pwd)/install"

# 3. Build and install
cmake --build build --parallel
cmake --install build

# 4. Install the Python package into skinny's venv. The build emits a
#    standard setup.py / pyproject under build/python (or install/python
#    depending on the version) — install it in place, do NOT `pip install
#    MaterialX` afterwards or the wheel will overwrite the source build.
../.venv/bin/pip install ./install/python
```

Verify the Slang generator is available:

```bash
.venv/bin/python -c "from MaterialX import PyMaterialXGenSlang; print(PyMaterialXGenSlang.__file__)"
```

Notes:

- On Windows use the same CMake invocation with the Visual Studio generator
  (`cmake -S . -B build -G "Visual Studio 17 2022" -A x64 ...`) and install
  with `cmake --build build --config Release --target install`.
- If you previously installed the PyPI wheel into the venv, uninstall it first
  (`pip uninstall MaterialX`) before installing the from-source build.
- Keep the MaterialX checkout around — re-installing the venv requires
  re-running step 4 against the same `install/python` tree.
