# Prism dispersion asset

A BK7 (dispersive dielectric) triangular prism lit by a bright thin backlight,
viewed through the prism so its refracted image splits into a spectrum under
`--spectral`. Demonstrates hero-wavelength Cauchy glass dispersion (Group 6.4).

- `prism_dispersion.pbrt` — pbrt v4 source (`Material "dielectric" "spectrum eta" "glass-BK7"`).
- `prism_dispersion.usda` — imported USD (preserves `skinnyOverrides.glass_dispersion = "bk7"`).

Render (native Metal or Vulkan):

    export VULKAN_SDK=/path/to/VulkanSDK/macOS DYLD_LIBRARY_PATH=$VULKAN_SDK/lib
    skinny-render assets/prism_dispersion/prism_dispersion.usda \
        -o prism.png --width 700 --height 700 --samples 768 --spectral --backend metal

Under `--spectral` the refracted edges show red→blue chromatic separation; without
`--spectral` (RGB) the glass is achromatic (white edges), because BK7's IOR is
constant per channel. Chroma-spread of the brightest refracted pixels: RGB 0.00
vs spectral ~0.32.
