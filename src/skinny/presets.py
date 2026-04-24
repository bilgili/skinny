"""Biological skin-parameter presets.

A preset is a named bundle of values for fields on Renderer.skin (and nearby
state), keyed by the same dotted paths used by `app._set_nested`. This means
applying a preset goes through the same path that keyboard and Tk sliders use,
so the accumulation reset in `Renderer._current_state_hash` fires automatically.

The Fitzpatrick scale (I–VI, palest to darkest) is the dominant clinical axis
for skin colour; here we cross it with a coarse male/female split that nudges
dermis thickness, vellus hair density/tilt, surface roughness, and subcutaneous
thickness. The numbers are starting points — the sliders let the user tune
from there.
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

# Melanin volume fraction per Fitzpatrick type. Rough fit matching Donner &
# Jensen (2006) light / medium / dark bands, stretched to six steps.
_FITZ_MELANIN = {1: 0.03, 2: 0.07, 3: 0.15, 4: 0.25, 5: 0.40, 6: 0.65}


def _make(fitz: int, sex: str) -> Preset:
    m = _FITZ_MELANIN[fitz]
    female = sex == "F"
    return Preset(
        name=f"Fitzpatrick {_FITZ_ROMAN[fitz - 1]} ({sex})",
        values={
            "skin.melanin_fraction":       m,
            "skin.hemoglobin_fraction":    0.055 if female else 0.05,
            "skin.blood_oxygenation":      0.75,
            "skin.epidermis_thickness_mm": 0.08 + 0.01 * fitz,
            "skin.dermis_thickness_mm":    0.95 if female else 1.10,
            "skin.subcut_thickness_mm":    3.5 if female else 3.0,
            "skin.roughness":              0.34 if female else 0.42,
            "skin.ior":                    1.4,
            "skin.pore_density":           0.30 if female else 0.40,
            "skin.pore_depth":             0.35 if female else 0.45,
            "skin.hair_density":           0.08 if female else 0.35,
            "skin.hair_tilt":              0.2 if female else 0.5,
        },
    )


PRESETS: list[Preset] = [
    _make(fitz, sex) for fitz in range(1, 7) for sex in ("F", "M")
]


def apply_preset(renderer, preset: Preset) -> None:
    from skinny.app import _set_nested

    for path, value in preset.values.items():
        _set_nested(renderer, path, value)
