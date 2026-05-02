"""Biological skin-parameter presets.

A preset is a named bundle of ``mtlx.*`` values keyed by the same dotted paths
used by ``app._set_nested``.  Applying a preset writes directly into
``Renderer.mtlx_overrides``, triggering the accumulation reset via
``Renderer._current_state_hash``.

The Fitzpatrick scale (I-VI, palest to darkest) is the dominant clinical axis
for skin colour; here we cross it with a coarse male/female split that nudges
dermis thickness, vellus hair density/tilt, surface roughness, and subcutaneous
thickness.  The numbers are starting points -- the sliders let the user tune
from there.

References:
    [1] Donner, Jensen, "A Spectral BSSRDF for Shading Human Skin", EGSR 2006.
        Melanin volume fractions (_FITZ_MELANIN) are derived from the light /
        medium / dark skin classification in Table 1.
    [2] Fitzpatrick, "A New Approach to the Classification of Photomorphotypes",
        Archives of Dermatology, 1988.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Preset:
    name: str
    values: dict[str, float] = field(default_factory=dict)
    # False for user-saved presets loaded from ~/.skinny/presets/. The Tk
    # Delete button only operates on non-builtin entries.
    is_builtin: bool = True


_FITZ_ROMAN = ["I", "II", "III", "IV", "V", "VI"]

# Melanin volume fraction per Fitzpatrick type [1, 2].
# Rough fit to Donner & Jensen Table 1 light / medium / dark bands,
# stretched across six Fitzpatrick steps:
#   c_mel ∈ [0.03, 0.65]   (eumelanin volume fraction driving σ_a in skin_bssrdf.slang)
_FITZ_MELANIN = {1: 0.03, 2: 0.07, 3: 0.15, 4: 0.25, 5: 0.40, 6: 0.65}


def _make(fitz: int, sex: str) -> Preset:
    m = _FITZ_MELANIN[fitz]
    female = sex == "F"
    return Preset(
        name=f"Fitzpatrick {_FITZ_ROMAN[fitz - 1]} ({sex})",
        values={
            "mtlx.layer_top_melanin":              m,
            "mtlx.layer_middle_hemoglobin":         0.055 if female else 0.05,
            "mtlx.layer_middle_blood_oxygenation":  0.75,
            "mtlx.layer_top_thickness":             0.08 + 0.01 * fitz,
            "mtlx.layer_middle_thickness":          0.95 if female else 1.10,
            "mtlx.layer_bottom_thickness":          3.5 if female else 3.0,
            "mtlx.skin_bsdf_roughness":             0.34 if female else 0.42,
            "mtlx.skin_bsdf_ior":                   1.4,
            "mtlx.skin_bsdf_pore_density":          0.30 if female else 0.40,
            "mtlx.skin_bsdf_pore_depth":            0.35 if female else 0.45,
            "mtlx.skin_bsdf_hair_density":          0.08 if female else 0.35,
            "mtlx.skin_bsdf_hair_tilt":             0.2 if female else 0.5,
        },
    )


PRESETS: list[Preset] = [
    _make(fitz, sex) for fitz in range(1, 7) for sex in ("F", "M")
]


def apply_preset(renderer, preset: Preset) -> None:
    from skinny.app import _SKIN_TO_MTLX, _set_nested

    for path, value in preset.values.items():
        path = _SKIN_TO_MTLX.get(path, path)
        _set_nested(renderer, path, value)
