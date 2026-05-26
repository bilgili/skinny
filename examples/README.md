# Examples — offscreen rendering

The supported offscreen-render path is `skinny.headless` (Python API) and the
`skinny-render` CLI. Both accept a file path or a live `Usd.Stage` mutated
between calls; the GPU context is held across calls by `HeadlessRenderer` so
you pay for pipeline compile only once.

## Quick start

### CLI — single image

```bash
skinny-render assets/cornell_box_sphere.usda -o out/cornell.png \
    --width 1024 --height 1024 --samples 128
```

### CLI — animation

```bash
skinny-render assets/cornell_box_sphere.usda --animate \
    --frames 1:48:1 --outdir out/frames --samples 64
```

### API — mutate a stage and re-render per frame

```python
from pxr import Usd, UsdGeom
from skinny.headless import HeadlessRenderer

stage = Usd.Stage.Open("assets/cornell_box_sphere.usda")
xf = UsdGeom.Xformable(stage.GetPrimAtPath("/Cornell/Sphere"))
with HeadlessRenderer(800, 800) as r:
    for i in range(24):
        xf.ClearXformOpOrder()
        xf.AddTranslateOp().Set((i * 0.1, 0.0, 0.0))
        r.render_scene(stage, f"out/frame_{i:04d}.png", samples=64)
```

## macOS — Vulkan SDK setup

The Python that has a working Vulkan runtime and `PyMaterialXGenSlang` is
`./bin/python3.13` (not `.venv`). Export these before any headless run:

```bash
export VULKAN_SDK=/Users/ahmetbilgili/VulkanSDK/1.4.341.1/macOS
export DYLD_LIBRARY_PATH=$VULKAN_SDK/lib
./bin/python3.13 examples/render_image.py -o out/cornell.png
```

## Example scripts

| Script | What it does |
|--------|--------------|
| `render_image.py` | Minimal wrapper: calls `render_scene()` and exits |
| `render_turntable.py` | Orbits the existing scene camera 360° by mutating its stage transform per frame |

Both scripts are thin demos of the public API — see `skinny.headless` for the
full interface and `Architecture.md` for renderer internals.

## Output formats

Extension determines the format (`--format` overrides it):

- `png` / `jpeg` / `bmp` — tonemapped LDR (exposure + tonemap applied)
- `exr` / `hdr` — linear HDR straight from the accumulation buffer
