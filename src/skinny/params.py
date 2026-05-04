"""Shared parameter definitions and accessors.

Extracted from app.py so both the desktop (GLFW + Tkinter) and web (Panel)
entry points can share ParamSpec, get/set helpers, and persistence logic
without pulling in GLFW.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# Common render resolutions surfaced in the desktop and web "Resolution"
# sections. First entry "Custom" is the sentinel selected when the live
# width/height don't match any preset; its (0, 0) tuple is never applied.
RESOLUTION_PRESETS: list[tuple[str, int, int]] = [
    ("Custom",      0,    0),
    ("640x360",     640,  360),
    ("960x540",     960,  540),
    ("1280x720",    1280, 720),
    ("1600x900",    1600, 900),
    ("1920x1080",   1920, 1080),
    ("2560x1440",   2560, 1440),
    ("3840x2160",   3840, 2160),
    ("1024x1024",   1024, 1024),
    ("2048x2048",   2048, 2048),
]


def find_resolution_preset_index(width: int, height: int) -> int:
    """Return RESOLUTION_PRESETS index whose (W, H) matches; 0 ("Custom") otherwise."""
    for i, (_name, w, h) in enumerate(RESOLUTION_PRESETS):
        if w == width and h == height:
            return i
    return 0


@dataclass
class ParamSpec:
    name: str
    path: str
    kind: str  # "continuous" or "discrete"
    step: float = 0.0
    lo: float = 0.0
    hi: float = 0.0
    choice_source: str | None = None  # for discrete: attribute on renderer


def _cont(name: str, path: str, step: float, lo: float, hi: float) -> ParamSpec:
    return ParamSpec(name, path, "continuous", step, lo, hi)


def _disc(name: str, path: str, choice_source: str) -> ParamSpec:
    return ParamSpec(name, path, "discrete", choice_source=choice_source)


# Discrete first so Preset / Environment show up at the top.
STATIC_PARAMS: list[ParamSpec] = [
    _disc("Preset",            "preset_index",                "presets"),
    _disc("Environment",       "env_index",                   "environments"),
    _cont("IBL intensity",     "env_intensity",               0.05, 0.0,  3.0),
    _cont("mm per unit",       "mm_per_unit",                 5.0,  1.0,  500.0),
    _disc("Direct light",      "direct_light_index",          "direct_light_modes"),
    _disc("Scattering",        "scatter_index",               "scatter_modes"),
    _disc("Sampling",          "integrator_index",            "integrator_modes"),
    _disc("Furnace mode",      "furnace_index",               "furnace_modes"),
    _disc("Head model",        "head_index",                  "head_models"),
    _disc("Detail maps",       "detail_maps_index",           "detail_maps_modes"),
    _cont("Normal map strength", "normal_map_strength",       0.05, 0.0,  2.0),
    _cont("Displacement (mm)", "displacement_scale_mm",       0.05, 0.0,  2.0),
    _disc("Tattoo",            "tattoo_index",                "tattoos"),
    _cont("Tattoo density",    "tattoo_density",              0.05, 0.0,  1.0),

    _cont("Melanin",            "mtlx.layer_top_melanin",            0.01, 0.0,  1.0),
    _cont("Hemoglobin",         "mtlx.layer_middle_hemoglobin",      0.01, 0.0,  1.0),
    _cont("Blood oxygenation",  "mtlx.layer_middle_blood_oxygenation", 0.05, 0.0, 1.0),
    _cont("Epidermis thickness", "mtlx.layer_top_thickness",         0.02, 0.01, 1.0),
    _cont("Dermis thickness",   "mtlx.layer_middle_thickness",       0.1,  0.1,  5.0),
    _cont("Subcut thickness",   "mtlx.layer_bottom_thickness",       0.2,  0.5,  10.0),
    _cont("Anisotropy (g)",     "mtlx.layer_top_anisotropy",         0.02, 0.0,  0.99),
    _cont("Roughness",          "mtlx.skin_bsdf_roughness",          0.02, 0.01, 1.0),
    _cont("IOR",                "mtlx.skin_bsdf_ior",                0.02, 1.0,  2.0),

    _cont("Pore density",       "mtlx.skin_bsdf_pore_density",       0.05, 0.0,  1.0),
    _cont("Pore depth",         "mtlx.skin_bsdf_pore_depth",         0.05, 0.0,  1.0),
    _cont("Vellus hair density", "mtlx.skin_bsdf_hair_density",      0.05, 0.0,  1.0),
    _cont("Vellus hair tilt",   "mtlx.skin_bsdf_hair_tilt",          0.05, 0.0,  1.0),

    _cont("Light elevation",    "light_elevation",             5.0, -90.0, 90.0),
    _cont("Light azimuth",      "light_azimuth",               5.0, -180.0, 180.0),
    _cont("Light intensity",    "light_intensity",             0.2,  0.0,  20.0),
    _cont("Light color R",      "light_color_r",               0.05, 0.0,  1.0),
    _cont("Light color G",      "light_color_g",               0.05, 0.0,  1.0),
    _cont("Light color B",      "light_color_b",               0.05, 0.0,  1.0),
]

# Backward-compat alias. New code should call build_all_params(renderer)
# to get the live list including dynamic material params; older code that
# imports ALL_PARAMS directly still gets the static base.
ALL_PARAMS = STATIC_PARAMS


_GANGED_MTLX_FIELDS: dict[str, list[str]] = {
    "layer_top_anisotropy": ["layer_middle_anisotropy", "layer_bottom_anisotropy"],
    "skin_bsdf_ior": ["layer_top_ior", "layer_middle_ior", "layer_bottom_ior"],
    "layer_top_scattering_coeff": ["layer_middle_scattering_coeff"],
}

_SKIN_TO_MTLX: dict[str, str] = {
    "skin.melanin_fraction":       "mtlx.layer_top_melanin",
    "skin.hemoglobin_fraction":    "mtlx.layer_middle_hemoglobin",
    "skin.blood_oxygenation":      "mtlx.layer_middle_blood_oxygenation",
    "skin.epidermis_thickness_mm": "mtlx.layer_top_thickness",
    "skin.dermis_thickness_mm":    "mtlx.layer_middle_thickness",
    "skin.subcut_thickness_mm":    "mtlx.layer_bottom_thickness",
    "skin.anisotropy_g":           "mtlx.layer_top_anisotropy",
    "skin.roughness":              "mtlx.skin_bsdf_roughness",
    "skin.ior":                    "mtlx.skin_bsdf_ior",
    "skin.pore_density":           "mtlx.skin_bsdf_pore_density",
    "skin.pore_depth":             "mtlx.skin_bsdf_pore_depth",
    "skin.hair_density":           "mtlx.skin_bsdf_hair_density",
    "skin.hair_tilt":              "mtlx.skin_bsdf_hair_tilt",
}


def build_dynamic_params(renderer) -> list[ParamSpec]:
    """Return one ParamSpec per editable MaterialX uniform on the active
    skin material, excluding fields already covered by STATIC_PARAMS.
    Empty when the runtime hasn't loaded a material.
    """
    cm = getattr(renderer, "_mtlx_skin_material", None)
    if cm is None or not getattr(cm, "uniform_block", None):
        return []
    static_paths = {p.path for p in STATIC_PARAMS}
    ganged_targets = set()
    for targets in _GANGED_MTLX_FIELDS.values():
        for t in targets:
            ganged_targets.add(t)
    from skinny.materialx_runtime import ui_specs_from_uniform_block
    out: list[ParamSpec] = []
    for spec in ui_specs_from_uniform_block(cm.uniform_block):
        if spec["path"] in static_paths:
            continue
        field_name = spec["path"].split(".")[1] if "." in spec["path"] else ""
        if field_name in ganged_targets:
            continue
        out.append(ParamSpec(
            name=spec["name"], path=spec["path"], kind=spec["kind"],
            step=spec["step"], lo=spec["lo"], hi=spec["hi"],
        ))
    return out


def build_all_params(renderer) -> list[ParamSpec]:
    """Concatenate the static base + material-driven dynamic params."""
    return STATIC_PARAMS + build_dynamic_params(renderer)


def _get_nested(obj, path):
    """Resolve `path` on `obj`. Routes:
    - "mtlx.<field>"          -> obj.mtlx_overrides[field]   (scalar)
    - "mtlx.<field>.<idx>"    -> obj.mtlx_overrides[field][idx]  (vector comp)
    - "<a>.<b>"               -> getattr chain (legacy)
    Falls back to the active material's uniform_block default when an
    `mtlx.*` path hasn't been explicitly overridden yet, so the slider
    starts at the authored MaterialX value.
    """
    parts = path.split(".")
    if parts[0] == "mtlx" and len(parts) >= 2:
        field_name = parts[1]
        overrides = getattr(obj, "mtlx_overrides", {})
        cm = getattr(obj, "_mtlx_skin_material", None)
        if field_name in overrides:
            value = overrides[field_name]
        else:
            try:
                base = obj._mtlx_skin_overrides()
            except (AttributeError, TypeError):
                base = {}
            if field_name in base:
                value = base[field_name]
            elif cm is not None:
                value = next(
                    (uf.default for uf in cm.uniform_block if uf.name == field_name),
                    0.0,
                )
            else:
                value = 0.0
        if len(parts) == 3:
            idx = int(parts[2])
            if isinstance(value, (tuple, list)):
                return float(value[idx]) if idx < len(value) else 0.0
            return 0.0
        return float(value) if isinstance(value, (int, float)) else value
    for p in parts:
        obj = getattr(obj, p)
    return obj


def _set_nested(obj, path, value):
    parts = path.split(".")
    if parts[0] == "mtlx" and len(parts) >= 2:
        field_name = parts[1]
        overrides = obj.mtlx_overrides
        if len(parts) == 3:
            idx = int(parts[2])
            cur = overrides.get(field_name)
            if cur is None:
                cm = getattr(obj, "_mtlx_skin_material", None)
                cur = next(
                    (uf.default for uf in cm.uniform_block if uf.name == field_name),
                    (0.0, 0.0, 0.0),
                ) if cm is not None else (0.0, 0.0, 0.0)
            cur_list = list(cur) if isinstance(cur, (tuple, list)) else [0.0, 0.0, 0.0]
            while len(cur_list) <= idx:
                cur_list.append(0.0)
            cur_list[idx] = float(value)
            val_tuple = tuple(cur_list)
            overrides[field_name] = val_tuple
            for linked in _GANGED_MTLX_FIELDS.get(field_name, ()):
                overrides[linked] = val_tuple
        else:
            fval = float(value)
            overrides[field_name] = fval
            for linked in _GANGED_MTLX_FIELDS.get(field_name, ()):
                overrides[linked] = fval
        return
    for p in parts[:-1]:
        obj = getattr(obj, p)
    setattr(obj, parts[-1], value)


# preset_index intentionally stays out of the saved snapshot: the user's
# custom preset list can change between sessions, so a stored index loses
# meaning. The underlying param values restore themselves directly.
_NON_PERSISTED_PARAMS = {"preset_index"}


def _snapshot_params(renderer, params: list[ParamSpec] | None = None) -> dict[str, float | int]:
    out: dict[str, float | int] = {}
    for p in (params if params is not None else build_all_params(renderer)):
        if p.path in _NON_PERSISTED_PARAMS:
            continue
        val = _get_nested(renderer, p.path)
        if p.kind == "continuous":
            out[p.path] = float(val)
        else:
            out[p.path] = int(val)
    return out


def _apply_saved_params(renderer, saved_params, params: list[ParamSpec] | None = None) -> None:
    if not isinstance(saved_params, dict):
        return
    for old_key, new_key in _SKIN_TO_MTLX.items():
        if old_key in saved_params and new_key not in saved_params:
            saved_params[new_key] = saved_params[old_key]
    for p in (params if params is not None else build_all_params(renderer)):
        if p.path in _NON_PERSISTED_PARAMS or p.path not in saved_params:
            continue
        raw = saved_params[p.path]
        try:
            if p.kind == "continuous":
                val = float(np.clip(float(raw), p.lo, p.hi))
            else:
                choices = getattr(renderer, p.choice_source, None) or []
                if not choices:
                    continue
                idx = int(raw)
                if not (0 <= idx < len(choices)):
                    continue
                val = idx
        except (TypeError, ValueError):
            continue
        _set_nested(renderer, p.path, val)
