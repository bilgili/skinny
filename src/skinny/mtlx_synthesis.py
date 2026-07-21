"""GPU-free MaterialX material synthesis for the MCP authoring tools.

This module is the *validation + document-build + reflection* half of the
`mcp-material-authoring` change (design D2–D6). It never touches the GPU, the
renderer, or Vulkan — it only builds MaterialX documents with the MaterialX
Python API and runs the Slang generator as a **dry-run** (`MaterialLibrary`
generate + compute-fragment extraction) to (a) reject anything the generator
cannot compile *before* a prim or file exists, and (b) reflect the generated
uniform names so the editability contract (logical input → gen uniform names)
is derived, never guessed.

Why a whole module instead of inlining into `mcp_server.py`: the renderer
(group 3) and the MCP tool surface (group 4) both consume this, and keeping it
renderer-import-free means the hostless test suite can exercise the real gen
without pulling in `vulkan`. Do **not** `import skinny.renderer` here.

Public surface
--------------
``validate_spec``      normalize one of the four spec forms, or raise.
``list_presets`` / ``resolve_preset``   server-side curated catalog (dict
                                        lookup, never a client path join).
``NODE_WHITELIST``     the gen-proven nodegraph node types.
``TEMPLATES`` / ``expand_template``     server-owned procedural recipes.
``build_document``     spec + material name → MaterialX ``Document``.
``synthesize``         validate → build → gen dry-run → ``SynthesisResult``
                       (document XML, logical→uniform mapping, editable keys).
``SessionMaterialStore``   tempdir-backed ``.mtlx`` lifecycle with sidecar.
``list_preset_inputs``     mtime-cached gen reflection of a curated preset.
``preset_holder_name``     the curated document's surfacematerial element name
                           (the D6 naming contract `add_material`'s ``name``
                           must match for a preset).
``model_param_schema`` / ``template_param_schema``   ``{param: {type, range}}``
                           schemas for the `material_list` discovery tool.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import MaterialX as mx

from skinny.materialx_runtime import MaterialLibrary

# ─── Asset / catalog locations ────────────────────────────────────────
#
# Anchored the same way settings.REPO_ROOT is (this file lives at
# src/skinny/mtlx_synthesis.py, so parents[2] is the repo root). Kept local
# rather than importing settings so this module has no incidental deps.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_PRESET_DIR = _REPO_ROOT / "assets" / "Usd-Mtlx-Example" / "materials"
_PRESET_PREFIX = "standard_surface_"


class MaterialSpecError(ValueError):
    """A material-spec rejection to report to the client.

    A ``ValueError`` subclass so a caller that wants the least coupling can
    catch it as ``ValueError``; ``mcp_server`` wraps it into ``SceneToolError``
    at the tool boundary (it must not import renderer/UI types the other way).
    """


# ─── Material parameter bounds ────────────────────────────────────────
#
# Source of truth is scene_graph._MATERIAL_FLOAT_RANGES; duplicated here (not
# imported) because scene_graph is the UI tree model and a synthesis backend
# should not depend on the UI layer. Keep the two in sync when either changes.
_MATERIAL_FLOAT_RANGES: dict[str, tuple[float, float]] = {
    "roughness":      (0.04, 1.0),
    "metallic":       (0.0,  1.0),
    "specular":       (0.0,  1.0),
    "opacity":        (0.0,  1.0),
    "ior":            (1.0,  3.0),
    "coat":           (0.0,  1.0),
    "coat_roughness": (0.0,  1.0),
    "clearcoat":      (0.0,  1.0),
    "clearcoatRoughness": (0.0, 1.0),
    "specular_roughness": (0.04, 1.0),
    "metalness":      (0.0,  1.0),
    "transmission":   (0.0,  1.0),
    "base":           (0.0,  1.0),
}


# ─── Node whitelist ───────────────────────────────────────────────────
#
# `checker` from the design tuple is intentionally absent: this MaterialX
# build (1.39.x) ships the node as `checkerboard`, not `checker`, so the
# literal `checker` fails its per-node gen dry-run. Per design D4/task 1.7 the
# node is dropped from the whitelist and the `checker` template is dropped
# rather than silently substituting a differently-named node. The per-node
# hostless gen test (`test_mtlx_synthesis`) is the standing gate for re-adding
# it (or any node) once it compiles under this literal name.
NODE_WHITELIST: tuple[str, ...] = (
    "fractal3d", "noise2d", "noise3d", "position", "texcoord", "mix",
    "multiply", "add", "subtract", "sin", "power", "dotproduct",
    "ramplr", "ramptb",
)

# Node output data type when a node dict does not pin `output`. Everything not
# listed defaults to color3 (the type a base_color-driving graph wants);
# these three cannot produce color3.
_NODE_DEFAULT_OUTPUT: dict[str, str] = {
    "position": "vector3",
    "texcoord": "vector2",
    "dotproduct": "float",
    "sin": "float",
}

# Node inputs the gen expects as integer (so a JSON int is authored as
# `integer`, not `float`, which the generator's type check rejects).
_INTEGER_NODE_INPUTS: frozenset[str] = frozenset({"octaves", "index"})


# ─── Shader input schemas (reflected once, lazily) ────────────────────

_SHADER_INPUTS_CACHE: dict[str, dict[str, str]] = {}

# Spec model name -> MaterialX shader node string (the `ND_<x>_surfaceshader`
# nodedef stem). `preview` is UsdPreviewSurface's authored spelling.
_MODEL_NODE_STRING: dict[str, str] = {
    "preview": "UsdPreviewSurface",
    "standard_surface": "standard_surface",
}


# UsdPreviewSurface inputs the inline author path (`usd_material_edit.
# author_preview_material` / `_PREVIEW_INPUTS`) actually writes. The reflected
# nodedef exposes the FULL surface — `normal`/`displacement`/`occlusion` are
# texture/connection-typed and `useSpecularWorkflow`/`opacityThreshold`/
# `opacityMode` are mode inputs the inline author silently drops. Discovery AND
# validation restrict to exactly this author-able subset (design D4/finding #9)
# so a spec can never accept a `preview` param that authoring would ignore. Keep
# in sync with `usd_material_edit._PREVIEW_INPUTS`.
_PREVIEW_AUTHORABLE: frozenset[str] = frozenset({
    "diffuseColor", "emissiveColor", "specularColor", "roughness", "metallic",
    "clearcoat", "clearcoatRoughness", "opacity", "ior",
})


def _shader_inputs(model: str) -> dict[str, str]:
    """`{input_name: mtlx_type}` for a shader model, reflected + cached.

    Reflected from the loaded MaterialX library rather than hardcoded so the
    schema tracks whatever standard_surface / UsdPreviewSurface version ships.
    """
    if model not in _SHADER_INPUTS_CACHE:
        node_string = _MODEL_NODE_STRING.get(model, model)
        lib = _shared_library()
        nd = lib.document.getNodeDef(f"ND_{node_string}_surfaceshader")
        if nd is None:
            raise MaterialSpecError(f"unknown shader nodedef for model {model!r}")
        _SHADER_INPUTS_CACHE[model] = {
            i.getName(): i.getType() for i in nd.getActiveInputs()
        }
    return _SHADER_INPUTS_CACHE[model]


def _model_input_schema(model: str) -> dict[str, str]:
    """Author-able `{input_name: mtlx_type}` for a shader model (finding #9).

    Restricts `preview` to `_PREVIEW_AUTHORABLE`; `standard_surface` passes the
    full reflected schema (its flat inputs are all authored on the shader prim).
    The single source both `_validate_shader_params` and `model_param_schema`
    read, so advertised == authored.
    """
    schema = _shader_inputs(model)
    if model == "preview":
        return {k: v for k, v in schema.items() if k in _PREVIEW_AUTHORABLE}
    return schema


_SHARED_LIBRARY: Optional[MaterialLibrary] = None


def _shared_library() -> MaterialLibrary:
    """A loaded stdlib+skinny library for *reflection only* (schemas).

    Dry-runs use their own fresh libraries so imported synthesized docs never
    accumulate here. Loading is ~40ms and idempotent.
    """
    global _SHARED_LIBRARY
    if _SHARED_LIBRARY is None:
        lib = MaterialLibrary.from_install()
        lib.load()
        _SHARED_LIBRARY = lib
    return _SHARED_LIBRARY


# ─── Preset catalog (design D3) ───────────────────────────────────────

def list_presets() -> dict[str, str]:
    """`{preset_name: absolute .mtlx path}` for the curated corpus.

    Enumerated from the directory at call time (design D3) so the catalog can
    never drift from disk. Names strip the `standard_surface_` prefix.
    """
    catalog: dict[str, str] = {}
    if not _PRESET_DIR.is_dir():
        return catalog
    for path in sorted(_PRESET_DIR.glob("*.mtlx")):
        stem = path.stem
        name = stem[len(_PRESET_PREFIX):] if stem.startswith(_PRESET_PREFIX) else stem
        catalog[name] = str(path)
    return catalog


def resolve_preset(name: str) -> str:
    """Resolve a preset name to its curated file via **dict lookup only**.

    The client string is never joined onto a filesystem path (design D3): a
    `../../etc/foo` name simply misses the catalog and reports as unknown,
    listing the available names.
    """
    catalog = list_presets()
    if name not in catalog:
        available = ", ".join(sorted(catalog)) or "(none)"
        raise MaterialSpecError(
            f"unknown preset {name!r}; available presets: {available}"
        )
    return catalog[name]


def preset_holder_name(name: str) -> str:
    """Surfacematerial element name for a curated preset (design D6).

    This -- not the catalog key -- is what a `/Materials` holder referencing
    the preset must be named as (the naming contract binding resolution
    relies on): a curated document's element name is whatever its author
    gave it (e.g. ``marble_solid`` -> ``Marble_3D``), not the file stem.
    """
    doc = mx.createDocument()
    mx.readFromXmlFile(doc, resolve_preset(name))
    return _find_surfacematerial(doc)


# ─── Spec validation + form dispatch (design D4) ──────────────────────

_MODELS = ("preview", "standard_surface")


def _finite_number(label: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MaterialSpecError(f"{label} expects a number, got {value!r}")
    fv = float(value)
    if not math.isfinite(fv):
        raise MaterialSpecError(f"{label} must be finite, got {value!r}")
    return fv


def _finite_check_node_value(label: str, value: Any) -> None:
    """Reject NaN/Inf in a graph node's constant value (finding #10).

    Recurses into list/tuple (colour/vector constants). Strings (node names,
    filenames) and booleans are left alone — only numeric leaves are checked.
    """
    if isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            _finite_check_node_value(f"{label}[{i}]", v)
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return
    if not math.isfinite(float(value)):
        raise MaterialSpecError(f"{label} must be finite, got {value!r}")


def _as_color3(label: str, value: Any) -> list[float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise MaterialSpecError(f"{label} expects 3 numbers, got {value!r}")
    if len(value) != 3:
        raise MaterialSpecError(f"{label} expects exactly 3 numbers, got {len(value)}")
    return [_finite_number(f"{label}[{i}]", c) for i, c in enumerate(value)]


def _check_float_bounds(name: str, value: float) -> None:
    rng = _MATERIAL_FLOAT_RANGES.get(name)
    if rng is None:
        return
    lo, hi = rng
    if value < lo or value > hi:
        raise MaterialSpecError(
            f"{name}={value} is outside its range {lo}..{hi}"
        )


def _validate_shader_params(model: str, params: dict) -> dict:
    """Validate + normalize flat params against a shader's reflected schema.

    Bounds come from `_MATERIAL_FLOAT_RANGES`; color3 inputs are coerced to a
    3-vector; every other numeric is finite-checked. Unknown param names are
    rejected so a typo does not silently vanish into a no-op input.
    """
    if not isinstance(params, dict):
        raise MaterialSpecError(f"params must be an object, got {params!r}")
    schema = _model_input_schema(model)
    out: dict[str, Any] = {}
    for key, value in params.items():
        mtype = schema.get(key)
        if mtype is None:
            raise MaterialSpecError(
                f"unknown {model} input {key!r}; not in the shader schema"
            )
        if mtype == "color3":
            out[key] = _as_color3(key, value)
        elif mtype in ("vector3", "vector2"):
            n = 3 if mtype == "vector3" else 2
            if not isinstance(value, (list, tuple)) or len(value) != n:
                raise MaterialSpecError(f"{key} expects {n} numbers, got {value!r}")
            out[key] = [_finite_number(f"{key}[{i}]", c) for i, c in enumerate(value)]
        elif mtype == "boolean":
            if not isinstance(value, bool):
                raise MaterialSpecError(f"{key} expects a boolean, got {value!r}")
            out[key] = value
        elif mtype == "integer":
            if isinstance(value, bool) or not isinstance(value, int):
                raise MaterialSpecError(f"{key} expects an integer, got {value!r}")
            out[key] = value
        else:  # float and anything else numeric
            fv = _finite_number(key, value)
            _check_float_bounds(key, fv)
            out[key] = fv
    return out


def model_param_schema(model: str) -> dict[str, dict]:
    """``{param_name: {"type": mtlx_type, "range": [lo, hi] | None}}`` for a
    shader model -- the same reflected schema `_validate_shader_params`
    validates against, so `material_list` (design D5) can never drift from
    what a spec actually accepts.
    """
    return {
        pname: {
            "type": mtype,
            "range": list(_MATERIAL_FLOAT_RANGES[pname]) if pname in _MATERIAL_FLOAT_RANGES else None,
        }
        for pname, mtype in _model_input_schema(model).items()
    }


def _validate_graph(graph: dict) -> dict:
    """Validate a nodegraph spec: node whitelist + dangling connections.

    Shape: ``{"nodes": {name: {"type": cat, ...}}, "connections": [[src, tgt]]}``
    where ``src`` is ``"node.out"`` and ``tgt`` is ``"node.input"`` (another
    node) or a bare standard_surface input name.
    """
    if not isinstance(graph, dict):
        raise MaterialSpecError(f"graph must be an object, got {graph!r}")
    nodes = graph.get("nodes")
    if not isinstance(nodes, dict) or not nodes:
        raise MaterialSpecError("graph.nodes must be a non-empty object")
    connections = graph.get("connections", [])
    if not isinstance(connections, list):
        raise MaterialSpecError("graph.connections must be a list")

    for nname, ndef in nodes.items():
        if not isinstance(ndef, dict) or "type" not in ndef:
            raise MaterialSpecError(f"node {nname!r} must be an object with a 'type'")
        cat = ndef["type"]
        if cat not in NODE_WHITELIST:
            supported = ", ".join(NODE_WHITELIST)
            raise MaterialSpecError(
                f"node {nname!r} uses unsupported type {cat!r}; "
                f"supported node types: {supported}"
            )
        # Finite-check every constant numeric/color the node authors so a NaN/Inf
        # cannot reach the generated MaterialX (finding #10). `_connect` inputs
        # carry no value; an exposed input's value rides under `value`.
        for key, raw in ndef.items():
            if key in ("type", "output"):
                continue
            if isinstance(raw, dict):
                if raw.get("_connect"):
                    continue
                value = raw.get("value")
            else:
                value = raw
            _finite_check_node_value(f"{nname}.{key}", value)

    ss_inputs = _shader_inputs("standard_surface")
    for conn in connections:
        if not isinstance(conn, (list, tuple)) or len(conn) != 2:
            raise MaterialSpecError(f"connection must be [src, target], got {conn!r}")
        src, tgt = conn
        src_str = str(src)
        src_node = src_str.split(".", 1)[0]
        if src_node not in nodes:
            raise MaterialSpecError(
                f"connection source references missing node {src_node!r}"
            )
        # Validate the `.out`-style output suffix: whitelisted nodes are single-
        # output, so only the canonical `out` is valid — a bogus `pos.NOPE` token
        # must not silently pass (finding #10).
        if "." in src_str:
            suffix = src_str.split(".", 1)[1]
            if suffix != "out":
                raise MaterialSpecError(
                    f"connection source {src_str!r}: unknown output {suffix!r}; "
                    f"only 'out' is a valid output name"
                )
        tgt_str = str(tgt)
        head = tgt_str.split(".", 1)[0]
        if "." in tgt_str and head in nodes:
            continue  # node-input target, node exists
        if head in nodes:
            continue
        # bare target: must be a standard_surface input
        if tgt_str not in ss_inputs:
            raise MaterialSpecError(
                f"connection target {tgt_str!r} is neither a graph node nor a "
                f"standard_surface input"
            )
    return graph


def validate_spec(spec: dict) -> dict:
    """Normalize a material spec into exactly one of four forms, or raise.

    Returns a dict with a ``form`` discriminator:
      - ``{"form": "preset", "preset", "path"}``
      - ``{"form": "preview", "params"}``
      - ``{"form": "standard_surface", "params", "graph"|None, "_promote_all"}``
    Templates are expanded here into the standard_surface form (design D4);
    ``_promote_all`` marks template-origin specs whose params are all promoted.
    Rejection happens before any USD/filesystem mutation (design D4).
    """
    if not isinstance(spec, dict):
        raise MaterialSpecError(f"material spec must be an object, got {spec!r}")

    keys = {"preset", "template", "model"}
    present = [k for k in keys if k in spec]
    if len(present) == 0:
        raise MaterialSpecError(
            "material spec must supply exactly one of 'preset', 'template', or 'model'"
        )
    if len(present) > 1:
        raise MaterialSpecError(
            f"material spec mixes forms {sorted(present)}; supply exactly one of "
            f"'preset', 'template', or 'model'"
        )

    form = present[0]

    if form == "preset":
        if "graph" in spec:
            raise MaterialSpecError("'graph' is only valid with model: 'standard_surface'")
        name = spec["preset"]
        if not isinstance(name, str):
            raise MaterialSpecError(f"preset must be a string, got {name!r}")
        return {"form": "preset", "preset": name, "path": resolve_preset(name)}

    if form == "template":
        # Templates expand into a standard_surface graph spec, then flow the
        # same validation path (design D4). expand_template validates bounds.
        expanded = expand_template(spec["template"], spec.get("params", {}))
        normalized = validate_spec(expanded)
        normalized["_promote_all"] = True
        return normalized

    # form == "model"
    model = spec["model"]
    if model not in _MODELS:
        raise MaterialSpecError(
            f"unknown model {model!r}; supported models: {', '.join(_MODELS)}"
        )
    graph = spec.get("graph")
    if graph is not None and model != "standard_surface":
        raise MaterialSpecError(
            f"nodegraphs require model: 'standard_surface', not {model!r}"
        )
    params = _validate_shader_params(model, spec.get("params", {}))
    if model == "preview":
        return {"form": "preview", "params": params}
    if graph is not None:
        _validate_graph(graph)
    return {
        "form": "standard_surface",
        "params": params,
        "graph": graph,
        "_promote_all": False,
    }


# ─── Templates (design D4/D5) ─────────────────────────────────────────
#
# Each template declares a param schema with bounds and expands to a
# standard_surface graph spec expressed entirely in whitelisted nodes. All
# declared template params are promoted (exposed as nodegraph interface
# inputs), so every one becomes an editable key.


@dataclass
class TemplateParam:
    name: str
    kind: str            # "float" | "integer" | "color3"
    default: Any
    bounds: Optional[tuple[float, float]] = None


def _tp(name, kind, default, lo=None, hi=None) -> TemplateParam:
    return TemplateParam(name, kind, default, (lo, hi) if lo is not None else None)


# schema is a list of TemplateParam; builder(params) -> graph dict.
TEMPLATES: dict[str, dict] = {}


def _register_template(name: str, schema: list[TemplateParam], builder) -> None:
    TEMPLATES[name] = {"schema": schema, "builder": builder}


def _exposed(name: str, value: Any) -> dict:
    """Node-param value form that promotes it to an interface input `name`."""
    return {"expose": True, "value": value, "name": name}


def _noise_graph(p: dict) -> dict:
    """fractal3d blend of colorA/colorB with scale/octaves/lacunarity/diminish."""
    return {
        "nodes": {
            "pos": {"type": "position", "output": "vector3"},
            "scaled": {
                "type": "multiply", "output": "vector3",
                "in1": {"_connect": "pos"},
                "in2": _exposed("scale", p["scale"]),
            },
            "noise": {
                "type": "fractal3d", "output": "float",
                "position": {"_connect": "scaled"},
                "octaves": _exposed("octaves", p["octaves"]),
                "lacunarity": _exposed("lacunarity", p["lacunarity"]),
                "diminish": _exposed("diminish", p["diminish"]),
            },
            "blend": {
                "type": "mix", "output": "color3",
                "bg": _exposed("colorA", p["colorA"]),
                "fg": _exposed("colorB", p["colorB"]),
                "mix": {"_connect": "noise"},
            },
        },
        "connections": [["blend.out", "base_color"]],
    }


def _marble_graph(p: dict) -> dict:
    """The curated marble recipe: fractal3d driven veins between two colors."""
    return {
        "nodes": {
            "pos": {"type": "position", "output": "vector3"},
            "scaled": {
                "type": "multiply", "output": "vector3",
                "in1": {"_connect": "pos"},
                "in2": _exposed("vein_scale", p["vein_scale"]),
            },
            "noise": {
                "type": "fractal3d", "output": "float",
                "position": {"_connect": "scaled"},
                "octaves": _exposed("octaves", p["octaves"]),
            },
            "veins": {
                "type": "sin", "output": "float",
                "in": {"_connect": "noise"},
            },
            "blend": {
                "type": "mix", "output": "color3",
                "bg": _exposed("base_color", p["base_color"]),
                "fg": _exposed("vein_color", p["vein_color"]),
                "mix": {"_connect": "veins"},
            },
        },
        "connections": [["blend.out", "base_color"]],
    }


_register_template(
    "noise",
    [
        _tp("colorA", "color3", [0.9, 0.9, 0.85]),
        _tp("colorB", "color3", [0.2, 0.2, 0.25]),
        _tp("scale", "float", 4.0, 0.01, 64.0),
        _tp("octaves", "integer", 4, 1, 8),
        _tp("lacunarity", "float", 2.0, 1.0, 8.0),
        _tp("diminish", "float", 0.5, 0.0, 1.0),
    ],
    _noise_graph,
)

_register_template(
    "marble_veins",
    [
        _tp("base_color", "color3", [0.8, 0.8, 0.8]),
        _tp("vein_color", "color3", [0.1, 0.1, 0.3]),
        _tp("vein_scale", "float", 6.0, 0.01, 64.0),
        _tp("octaves", "integer", 3, 1, 8),
    ],
    _marble_graph,
)


def expand_template(name: str, params: dict) -> dict:
    """Validate template params against bounds, expand to a std_surface spec.

    Runs *before* any document is built (task 1.7): a param outside its
    declared bounds is rejected here, not at gen time.
    """
    if name not in TEMPLATES:
        raise MaterialSpecError(
            f"unknown template {name!r}; available: {', '.join(sorted(TEMPLATES))}"
        )
    if not isinstance(params, dict):
        raise MaterialSpecError(f"template params must be an object, got {params!r}")
    schema: list[TemplateParam] = TEMPLATES[name]["schema"]
    declared = {tp.name for tp in schema}
    unknown = set(params) - declared
    if unknown:
        raise MaterialSpecError(
            f"template {name!r} got unknown params {sorted(unknown)}; "
            f"declared: {sorted(declared)}"
        )
    resolved: dict[str, Any] = {}
    for tp in schema:
        value = params.get(tp.name, tp.default)
        if tp.kind == "color3":
            resolved[tp.name] = _as_color3(tp.name, value)
        elif tp.kind == "integer":
            if isinstance(value, bool) or not isinstance(value, int):
                raise MaterialSpecError(f"{tp.name} expects an integer, got {value!r}")
            if tp.bounds and (value < tp.bounds[0] or value > tp.bounds[1]):
                raise MaterialSpecError(
                    f"{tp.name}={value} is outside its range "
                    f"{tp.bounds[0]}..{tp.bounds[1]}"
                )
            resolved[tp.name] = value
        else:  # float
            fv = _finite_number(tp.name, value)
            if tp.bounds and (fv < tp.bounds[0] or fv > tp.bounds[1]):
                raise MaterialSpecError(
                    f"{tp.name}={fv} is outside its range "
                    f"{tp.bounds[0]}..{tp.bounds[1]}"
                )
            resolved[tp.name] = fv
    graph = TEMPLATES[name]["builder"](resolved)
    _annotate_template_descriptors(graph, schema)
    return {"model": "standard_surface", "params": {}, "graph": graph}


def _annotate_template_descriptors(graph: dict, schema: list[TemplateParam]) -> None:
    """Stamp each exposed graph input with its `TemplateParam` kind + bounds so
    descriptor derivation (design D5/finding #2) advertises the declared type and
    range instead of inferring a bare 0..1 float. Templates own these; a raw-graph
    `expose: true` input carries neither and falls back to value-inference +
    `_MATERIAL_FLOAT_RANGES`.
    """
    by_name = {tp.name: tp for tp in schema}
    for ndef in graph.get("nodes", {}).values():
        if not isinstance(ndef, dict):
            continue
        for raw in ndef.values():
            if not (isinstance(raw, dict) and raw.get("expose")):
                continue
            tp = by_name.get(raw.get("name"))
            if tp is not None:
                raw["kind"] = tp.kind
                raw["range"] = list(tp.bounds) if tp.bounds else None


def template_param_schema(name: str) -> dict[str, dict]:
    """``{param_name: {"type", "default", "range"}}`` for a template, read
    straight off its declared `TemplateParam` schema (design D5,
    `material_list`) -- never hand-duplicated.
    """
    tpl = TEMPLATES.get(name)
    if tpl is None:
        raise MaterialSpecError(f"unknown template {name!r}")
    return {
        tp.name: {
            "type": tp.kind,
            "default": tp.default,
            "range": list(tp.bounds) if tp.bounds else None,
        }
        for tp in tpl["schema"]
    }


# ─── Document builder (design D6) ─────────────────────────────────────


def _set_typed_value(inp, mtype: str, value: Any) -> None:
    """Author a MaterialX input value with the right typed wrapper."""
    if mtype == "color3":
        inp.setValue(mx.Color3(*[float(c) for c in value]))
    elif mtype == "vector3":
        inp.setValue(mx.Vector3(*[float(c) for c in value]))
    elif mtype == "vector2":
        inp.setValue(mx.Vector2(*[float(c) for c in value]))
    elif mtype == "integer":
        inp.setValue(int(value))
    elif mtype == "boolean":
        inp.setValue(bool(value))
    else:
        inp.setValue(float(value))


def _infer_const_type(param: str, value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int) and param in _INTEGER_NODE_INPUTS:
        return "integer"
    if isinstance(value, (int, float)):
        return "float"
    if isinstance(value, (list, tuple)):
        return {2: "vector2", 3: "color3", 4: "color4"}.get(len(value), "float")
    return "float"


def build_document(spec: dict, material_name: str) -> mx.Document:
    """Build a MaterialX ``Document`` for a standard_surface spec.

    Element names are salted with ``material_name`` (design D6): the
    surfacematerial element is named exactly ``material_name`` (the naming
    contract the loader's binding resolution relies on), the shader is
    ``SR_<name>``, the nodegraph ``NG_<name>``. Flat params land on the shader;
    graph nodes/connections build the nodegraph; a node param in ``expose``
    form becomes a nodegraph interface input (the promoted, editable key).
    """
    if spec.get("form") != "standard_surface":
        raise MaterialSpecError(
            f"build_document requires a standard_surface spec, got {spec.get('form')!r}"
        )
    doc = mx.createDocument()
    shader_name = f"SR_{material_name}"
    graph_name = f"NG_{material_name}"

    shader = doc.addNode("standard_surface", shader_name, "surfaceshader")
    material = doc.addNode("surfacematerial", material_name, "material")
    material.addInput("surfaceshader", "surfaceshader").setNodeName(shader_name)

    ss_schema = _shader_inputs("standard_surface")
    graph = spec.get("graph")
    graph_driven: set[str] = set()

    if graph is not None:
        ng = doc.addNodeGraph(graph_name)
        promote_all = bool(spec.get("_promote_all"))
        nodes = graph["nodes"]

        # Pass 1: create every node with its output type so later connections
        # can reference them regardless of dict ordering.
        node_out: dict[str, str] = {}
        for nname, ndef in nodes.items():
            out_t = ndef.get("output", _NODE_DEFAULT_OUTPUT.get(ndef["type"], "color3"))
            ng.addNode(ndef["type"], nname, out_t)
            node_out[nname] = out_t

        # Pass 2: author node inputs — constants, exposed interface inputs, and
        # intra-graph connections (`{"_connect": other_node}`).
        for nname, ndef in nodes.items():
            node = ng.getNode(nname)
            for key, raw in ndef.items():
                if key in ("type", "output"):
                    continue
                if isinstance(raw, dict) and raw.get("_connect"):
                    src = raw["_connect"]
                    src_t = node_out.get(src, "color3")
                    node.addInput(key, src_t).setNodeName(src)
                elif isinstance(raw, dict) and (raw.get("expose") or promote_all):
                    value = raw.get("value") if isinstance(raw, dict) else raw
                    iface = raw.get("name") or f"{nname}_{key}"
                    itype = _infer_const_type(key, value)
                    if ng.getInput(iface) is None:
                        gi = ng.addInput(iface, itype)
                        _set_typed_value(gi, itype, value)
                    node.addInput(key, itype).setInterfaceName(iface)
                else:
                    itype = _infer_const_type(key, raw)
                    _set_typed_value(node.addInput(key, itype), itype, raw)

        # Pass 3: connections into standard_surface inputs create nodegraph
        # outputs; a non-matching source type is bridged with a `convert` node
        # (the generator needs the output type to match the shader input).
        for src, tgt in graph.get("connections", []):
            src_node = str(src).split(".", 1)[0]
            tgt_str = str(tgt)
            head = tgt_str.split(".", 1)[0]
            if "." in tgt_str and head in nodes:
                inp_name = tgt_str.split(".", 1)[1]
                src_t = node_out.get(src_node, "color3")
                node = ng.getNode(head)
                (node.getInput(inp_name) or node.addInput(inp_name, src_t)).setNodeName(
                    src_node
                )
                continue
            # bare standard_surface input target
            ss_type = ss_schema.get(tgt_str, "color3")
            src_t = node_out.get(src_node, ss_type)
            feed = src_node
            if src_t != ss_type:
                conv = ng.addNode("convert", f"cv_{tgt_str}_{src_node}", ss_type)
                conv.addInput("in", src_t).setNodeName(src_node)
                feed = conv.getName()
            out_name = f"out_{tgt_str}"
            out = ng.addOutput(out_name, ss_type)
            out.setNodeName(feed)
            bind = shader.getInput(tgt_str) or shader.addInput(tgt_str, ss_type)
            bind.setAttribute("nodegraph", graph_name)
            bind.setAttribute("output", out_name)
            graph_driven.add(tgt_str)

    # Flat params on the shader (skip any input the graph drives).
    for key, value in spec.get("params", {}).items():
        if key in graph_driven:
            continue
        mtype = ss_schema.get(key, "float")
        _set_typed_value(shader.addInput(key, mtype), mtype, value)

    return doc


# ─── Gen dry-run gate + mapping (design D4/D5) ────────────────────────


@dataclass
class SynthesisResult:
    """The output of a passing synthesis: what to write and how to edit it."""

    document_xml: str
    # logical (promoted) input name -> the gen uniform field names it drives.
    mapping: dict[str, list[str]] = field(default_factory=dict)
    # The promoted logical keys a client can `scene_set` (mapping keys).
    editable_inputs: list[str] = field(default_factory=list)
    # Full editable-input DESCRIPTORS persisted to the sidecar + Material
    # (design D5/finding #2): `{name: {"uniforms": [...], "type": "float"|
    # "color3"|"int", "default": <authored value>, "range": [lo, hi] | None}}`.
    # The scene graph builds correctly-typed, correctly-bounded, authored-default
    # properties from these — the old `{name: [uniforms]}` shape lost type/
    # default/range and surfaced every input as a 0..1 float.
    descriptors: dict[str, dict] = field(default_factory=dict)


def _descriptor_kind(mtlx_type: "str | None") -> str:
    """Collapse an mtlx/param type to the descriptor's coarse kind."""
    if mtlx_type in ("color3", "color4"):
        return "color3"
    if mtlx_type in ("integer", "int"):
        return "int"
    return "float"


def _collect_exposed(spec: dict) -> dict[str, dict]:
    """`{iface_name: {"value", "kind", "range"}}` for every promoted graph input.

    Walks the normalized spec's graph nodes (not the built doc) so the authored
    value — and, for template inputs, the `TemplateParam`-stamped kind/range —
    are available for descriptor derivation. Deduped by interface name (a
    shattered input feeding N node inputs shares one authored value).
    """
    graph = spec.get("graph") or {}
    promote_all = bool(spec.get("_promote_all"))
    out: dict[str, dict] = {}
    for nname, ndef in graph.get("nodes", {}).items():
        if not isinstance(ndef, dict):
            continue
        for key, raw in ndef.items():
            if key in ("type", "output") or not isinstance(raw, dict):
                continue
            if raw.get("_connect"):
                continue
            if raw.get("expose") or promote_all:
                iface = raw.get("name") or f"{nname}_{key}"
                out.setdefault(iface, {
                    "value": raw.get("value"),
                    "kind": raw.get("kind"),   # template-stamped, else None
                    "range": raw.get("range"),
                })
    return out


def _build_descriptors(spec: dict, mapping: dict[str, list[str]]) -> dict[str, dict]:
    """Editable-input descriptors from the spec + gen mapping (design D5).

    Only inputs the generator actually emitted uniforms for (mapping keys) become
    descriptors. Type comes from the template-stamped kind, else value inference;
    default is the authored value; range is the template bound, else
    `_MATERIAL_FLOAT_RANGES` for a name match on a float input, else None
    (unbounded finite).
    """
    exposed = _collect_exposed(spec)
    descriptors: dict[str, dict] = {}
    for iface, uniforms in mapping.items():
        meta = exposed.get(iface, {})
        value = meta.get("value")
        kind = meta.get("kind") or _infer_const_type(iface, value)
        dtype = _descriptor_kind(kind)
        rng = meta.get("range")
        if rng is None and dtype == "float" and iface in _MATERIAL_FLOAT_RANGES:
            rng = list(_MATERIAL_FLOAT_RANGES[iface])
        descriptors[iface] = {
            "uniforms": list(uniforms),
            "type": dtype,
            "default": value,
            "range": rng,
        }
    return descriptors


def _promoted_interface_inputs(doc: mx.Document) -> dict[str, list[tuple[str, str]]]:
    """`{interface_input: [(node, input), ...]}` for each nodegraph interface.

    An interface input feeding N node inputs lists all N — the generator names
    each resulting uniform `<node>_<input>`, so this is what the shattered-input
    mapping is derived from (design D5, M1).
    """
    result: dict[str, list[tuple[str, str]]] = {}
    for ng in doc.getNodeGraphs():
        ifaces = {i.getName() for i in ng.getInputs()}
        for iname in ifaces:
            result.setdefault(iname, [])
        for node in ng.getNodes():
            for inp in node.getInputs():
                iface = inp.getInterfaceName()
                if iface:
                    result.setdefault(iface, []).append((node.getName(), inp.getName()))
    return result


def _derive_mapping(doc: mx.Document, gen_uniform_names: set[str]) -> dict[str, list[str]]:
    """logical input -> [gen uniform names], intersected with what gen emitted.

    The generator names a public uniform `<node>_<input>` for each node input a
    promoted interface input feeds. We compute those candidate names and keep
    the ones the reflection actually produced, so an interface feeding several
    node inputs maps to all its uniforms and dropped (unused) ones vanish.
    """
    mapping: dict[str, list[str]] = {}
    for iface, consumers in _promoted_interface_inputs(doc).items():
        names = [
            f"{node}_{inp}" for node, inp in consumers
            if f"{node}_{inp}" in gen_uniform_names
        ]
        if names:
            mapping[iface] = names
    return mapping


def _find_surfacematerial(doc: mx.Document) -> str:
    """Name of the (single) surfacematerial element in a document."""
    for child in doc.getChildren():
        if child.getCategory() == "surfacematerial":
            return child.getName()
    raise MaterialSpecError("document has no surfacematerial element")


def _gen_dry_run(doc: mx.Document, target_name: str):
    """Run the GPU-free generator dry-run; return (compiled, fragment_or_None).

    Any generator exception is re-raised as a MaterialSpecError so the caller
    rejects the spec before authoring a prim or file (design D4).
    """
    lib = MaterialLibrary.from_install()
    lib.load()
    lib.import_document(doc)
    try:
        cm = lib.generate(target_name, write_to_disk=False, compile_check=False)
    except Exception as e:  # generator bailout
        raise MaterialSpecError(
            f"material {target_name!r} failed the Slang generator dry-run: {e}"
        ) from e
    if not cm.pixel_source:
        raise MaterialSpecError(
            f"material {target_name!r} produced no generated source"
        )
    try:
        frag = lib.generate_for_compute(target_name, write_to_disk=False, compiled=cm)
    except Exception as e:
        raise MaterialSpecError(
            f"material {target_name!r} failed compute-fragment extraction: {e}"
        ) from e
    return cm, frag


def synthesize(spec: dict, material_name: str) -> SynthesisResult:
    """Validate (if needed), build, gen-dry-run, and reflect a material.

    Accepts either a raw client spec or an already-normalized one. Returns a
    SynthesisResult carrying the document XML, the logical→uniform mapping, and
    the editable keys. Raises MaterialSpecError on any rejection — nothing is
    written here (the session file is the caller's step, design D2).
    """
    if spec.get("form") not in ("standard_surface",):
        spec = validate_spec(spec)
    if spec.get("form") != "standard_surface":
        raise MaterialSpecError(
            f"synthesize only builds standard_surface materials, got {spec.get('form')!r}"
        )
    doc = build_document(spec, material_name)
    _cm, frag = _gen_dry_run(doc, material_name)

    mapping: dict[str, list[str]] = {}
    if frag is not None:
        gen_names = {u.name for u in frag.uniform_block}
        mapping = _derive_mapping(doc, gen_names)

    return SynthesisResult(
        document_xml=mx.writeToXmlString(doc),
        mapping=mapping,
        editable_inputs=sorted(mapping),
        descriptors=_build_descriptors(spec, mapping),
    )


# ─── Session file lifecycle (design D2) ───────────────────────────────


class SessionMaterialStore:
    """Owns a tempdir of synthesized ``.mtlx`` files + editability sidecars.

    One file per material, named after the material prim, so the loader
    (group 2) and the renderer (group 3) can reference / re-read it by prim
    name. The directory is server configuration in the same trust domain as
    the preset catalog (design D2) — clients never address it, so it is not
    constrained to the allowed roots.

    Sidecar schema (``<name>.json``) — the editability contract read straight
    off disk by ``usd_loader._read_mtlx_mapping_sidecar`` (design D5)::

        {"logical_inputs": {
            "<logical input name>": {
                "uniforms": ["<gen uniform>", ...],   # fan-out write targets
                "type":     "float" | "color3" | "int",
                "default":  <authored value> | null,  # unedited control value
                "range":    [lo, hi] | null            # null = unbounded finite
            }, ...
        }}

    The pre-descriptor shape ``{"<name>": ["<uniform>", ...]}`` is still read
    (upgraded on load with null type/default/range → then value inference), so
    older sidecars keep working.
    """

    def __init__(self, base_dir: Optional[str] = None) -> None:
        self.dir = Path(base_dir) if base_dir else Path(tempfile.mkdtemp(prefix="skinny_mtlx_"))
        self.dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, material_name: str) -> Path:
        return self.dir / f"{material_name}.mtlx"

    def sidecar_for(self, material_name: str) -> Path:
        return self.dir / f"{material_name}.json"

    def write_document(
        self,
        material_name: str,
        xml: str,
        mapping: Optional[dict] = None,
        *,
        overwrite: bool = False,
    ) -> str:
        """Write ``<name>.mtlx`` (+ optional sidecar); return the absolute path.

        Flushed to disk before returning (design D2: a resync re-reads the
        document from disk, so it must be present first). Refuses to clobber an
        existing document unless ``overwrite=True`` (design D6/finding #5): the
        name reservation upstream should already keep two concurrent requests
        from colliding, and this makes a lost-update structurally impossible
        rather than merely unlikely — a second request that reached the same
        name fails loudly instead of overwriting the first's file.
        """
        path = self.path_for(material_name)
        if not overwrite and path.exists():
            raise MaterialSpecError(
                f"session material {material_name!r} already exists at {path}; "
                f"refusing to overwrite (name collision)"
            )
        path.write_text(xml, encoding="utf-8")
        if mapping is not None:
            self.sidecar_for(material_name).write_text(
                json.dumps({"logical_inputs": mapping}, indent=2),
                encoding="utf-8",
            )
        return str(path.resolve())

    def read_mapping(self, material_name: str) -> dict:
        """Read the persisted editability sidecar (descriptors or legacy map)."""
        sidecar = self.sidecar_for(material_name)
        if not sidecar.exists():
            return {}
        return json.loads(sidecar.read_text(encoding="utf-8")).get("logical_inputs", {})

    def delete(self, material_name: str) -> None:
        """Remove a material's document + sidecar (the rollback hook)."""
        for p in (self.path_for(material_name), self.sidecar_for(material_name)):
            try:
                p.unlink()
            except FileNotFoundError:
                pass


# ─── Preset editable-input reflection (design D5, mtime-cached) ───────

# {mtlx_path: (mtime, {logical name: descriptor})}
_PRESET_INPUT_CACHE: dict[str, tuple[float, dict[str, dict]]] = {}


def _reflect_identity_descriptors(doc: mx.Document, target: str) -> dict[str, dict]:
    """Identity descriptor map from a doc's gen dry-run (design D3/finding #3).

    Each generated uniform is its own logical input (identity mapping) so a
    curated preset's advertised keys are exactly its writable scene-graph
    properties. Type comes from the reflected ``UniformField``, default from its
    authored value, range from ``_MATERIAL_FLOAT_RANGES`` on a name match.
    filename/string uniforms (textures, framerange tokens) are skipped — not
    ``scene_set``-able scalars.
    """
    _cm, frag = _gen_dry_run(doc, target)
    # Graph preset (marble): frag uniforms are the writable keys. Constant-shader
    # preset (chrome/glass/jade): the std_surface param uniforms are.
    fields = list(frag.uniform_block) if frag is not None else list(_cm.uniform_block)
    descriptors: dict[str, dict] = {}
    for u in fields:
        if u.type_name in ("filename", "string"):
            continue
        dtype = _descriptor_kind(u.type_name)
        rng = (
            list(_MATERIAL_FLOAT_RANGES[u.name])
            if dtype == "float" and u.name in _MATERIAL_FLOAT_RANGES else None
        )
        descriptors[u.name] = {
            "uniforms": [u.name],
            "type": dtype,
            "default": u.default,
            "range": rng,
        }
    return descriptors


def identity_descriptors_for_file(path: str) -> dict[str, dict]:
    """mtime-cached identity descriptors for a curated/plain ``.mtlx`` file.

    The loader attaches these to a preset material that ships no sidecar (design
    D3/finding #3), so its advertised keys (``material_list``) are exactly the
    editable scene-graph properties. Runs the GPU-free gen dry-run once per file
    (mtime-cached); callers bound *which* files this runs on (e.g. only the
    curated corpus / session dir) so it is not run over arbitrary user materials.
    """
    mtime = os.path.getmtime(path)
    cached = _PRESET_INPUT_CACHE.get(path)
    if cached is not None and cached[0] == mtime:
        return {k: dict(v) for k, v in cached[1].items()}
    doc = mx.createDocument()
    mx.readFromXmlFile(doc, path)
    target = _find_surfacematerial(doc)
    descriptors = _reflect_identity_descriptors(doc, target)
    _PRESET_INPUT_CACHE[path] = (mtime, descriptors)
    return {k: dict(v) for k, v in descriptors.items()}


def list_preset_inputs(name: str) -> list[str]:
    """Editable input names for a curated preset (the *writable keys*, design D5).

    The keys of the identity descriptor map — gen uniform names, never parsed
    ``.mtlx`` interface names (which are not writable keys). mtime-cached.
    """
    return sorted(identity_descriptors_for_file(resolve_preset(name)))
