"""Layered-skin parameter record and its std140 pack — device-free.

Split out of ``renderer.py`` (change renderer-pure-core-extraction, task 2.7).
Its own module rather than a lodger in ``material_pack``: this is the skin
path's record with its own documented std140 layout, and the flat-material
packers know nothing about it.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

import numpy as np

@dataclass
class SkinParameters:
    """Physically-based skin parameters.

    Layered skin model: epidermis -> dermis -> subcutaneous fat.
    Absorption and scattering coefficients are spectral (RGB approximation).
    """

    # Epidermis
    melanin_fraction: float = 0.15
    epidermis_thickness_mm: float = 0.1

    # Dermis
    hemoglobin_fraction: float = 0.05
    blood_oxygenation: float = 0.75
    dermis_thickness_mm: float = 1.0

    # Subcutaneous
    subcut_thickness_mm: float = 3.0

    # Scattering
    scattering_coefficient: np.ndarray = field(
        default_factory=lambda: np.array([3.7, 4.4, 5.05], dtype=np.float32)
    )
    anisotropy_g: float = 0.8

    # Surface
    roughness: float = 0.35
    ior: float = 1.4

    # Sub-millimeter surface detail (pores + vellus hair). Defaults to 0 so
    # loading a pre-detail preset renders identically to pre-change output.
    pore_density: float = 0.0
    pore_depth: float = 0.0
    hair_density: float = 0.0
    hair_tilt: float = 0.0

    def pack(self) -> bytes:
        """Pack into std140-compatible bytes matching the Slang SkinParams struct.

        std140 layout (offsets in bytes):
          0: melaninFraction      (float)
          4: hemoglobinFraction   (float)
          8: bloodOxygenation     (float)
         12: epidermisThickness   (float)
         16: dermisThickness      (float)
         20: subcutThickness      (float)
         24: <8 bytes padding>    (align float3 to 16)
         32: scatteringCoeff      (float3, 12 bytes)
         44: anisotropy           (float, fills vec3 trailing slot)
         48: roughness            (float)
         52: ior                  (float)
         56: poreDensity          (float)
         60: poreDepth            (float)
         64: hairDensity          (float)
         68: hairTilt             (float)
         72: <8 bytes padding>    (struct rounds to 16)
        Total: 80 bytes
        """
        return struct.pack(
            "6f 2I 3f f 2f 4f 2I",
            self.melanin_fraction,
            self.hemoglobin_fraction,
            self.blood_oxygenation,
            self.epidermis_thickness_mm,
            self.dermis_thickness_mm,
            self.subcut_thickness_mm,
            0, 0,                                # 8 bytes padding
            *self.scattering_coefficient,
            self.anisotropy_g,
            self.roughness,
            self.ior,
            self.pore_density,
            self.pore_depth,
            self.hair_density,
            self.hair_tilt,
            0, 0,                                # 8 bytes struct tail padding
        )
