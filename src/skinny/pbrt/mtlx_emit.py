"""Sidecar MaterialX (``.mtlx``) writer + stage→.mtlx reference authoring.

This is the exporter half of ``-mtlx`` (tasks group 2). It turns the
``standard_surface`` input dicts produced by
:func:`skinny.pbrt.materials.map_material_mtlx` into a standalone, portable
MaterialX document, and authors the USD stage so the document round-trips back
through skinny's loader.

The authored shape is dictated by the loader's intake contract
(``skinny.usd_loader._load_mtlx_materials`` / ``_collect_mtlx_asset_paths`` /
``_find_image_file_in_nodegraph``), verified empirically:

* The ``standard_surface`` shader is a **document-root node** referenced from
  the ``surfacematerial`` by ``nodename`` (the loader reads
  ``ss_input.getNodeName()``). It is **not** nested under a ``<nodegraph>`` and
  the ``surfacematerial`` does **not** use ``nodegraph``/``output`` to reach it.
* The ``surfacematerial`` element name equals the bound USD material's **leaf
  name** (``_load_mtlx_materials`` matches by ``node.getName()`` against the
  binding target leaf).
* Constant scalar/color inputs are authored as plain ``value`` inputs on the
  shader (the loader reads them via ``inp.getValueString()``).
* Texture-bound inputs are wired through an ``<image>`` node inside a
  ``<nodegraph>``; the shader input carries ``nodegraph``/``output`` attributes
  so ``_find_image_file_in_nodegraph`` can walk back to the image ``file``.
* The exported stage authors the ``.mtlx`` reference on the **Material prim**
  and does **not** author a shadowing UsdPreviewSurface shader for that prim
  (so ``ComputeBoundMaterial`` fails and the ``.mtlx`` fallback fires, yielding
  the rich overrides whether or not the host has the usdMtlx plugin).
"""

from __future__ import annotations

from typing import Mapping

try:  # pragma: no cover - exercised indirectly via importorskip in tests
    import MaterialX as mx

    _HAS_MATERIALX = True
except ImportError:  # pragma: no cover
    mx = None  # type: ignore[assignment]
    _HAS_MATERIALX = False

from pxr import Sdf, Usd  # noqa: F401  (Usd used in type-hint strings)

__all__ = ["write_mtlx_document", "author_mtlx_reference"]


# standard_surface 3-component inputs authored as MaterialX ``vector3`` rather
# than ``color3``. Any other 3-component constant (``base_color``,
# ``transmission_color``, ``coat_color``, ``specular_color``, ``emission_color``,
# ``subsurface_color``, …) is authored as ``color3``.
_VECTOR3_INPUTS = frozenset({"subsurface_radius"})

# tex_inputs value_type (USD-flavoured, as produced by map_material_mtlx) ->
# MaterialX node/output/connection type.
_TEXVT_TO_MTLX = {
    "color3f": "color3",
    "color3": "color3",
    "float": "float",
}


def _split_inputs(material: Mapping) -> tuple[dict, dict]:
    """Normalise one ``materials_by_name`` value into ``(constants, tex_inputs)``.

    Accepts either the wrapped form ``{"inputs": {...}, "tex_inputs": {...}}``
    (what the CLI builds from :func:`map_material_mtlx`'s return) or a plain
    flat ``{input_name: value}`` dict (no texture connections).
    """
    if "inputs" in material or "tex_inputs" in material:
        constants = dict(material.get("inputs", {}) or {})
        tex_inputs = dict(material.get("tex_inputs", {}) or {})
    else:
        constants = dict(material)
        tex_inputs = {}
    return constants, tex_inputs


def _mtlx_type_for_constant(name: str, value: object) -> str:
    """Infer the MaterialX input type for a constant standard_surface value."""
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (list, tuple)):
        if name in _VECTOR3_INPUTS:
            return "vector3"
        return "color3"
    return "float"


def _set_constant_input(node: "mx.Node", name: str, value: object) -> None:
    """Author one constant input on a ``standard_surface`` node."""
    vtype = _mtlx_type_for_constant(name, value)
    if vtype == "boolean":
        node.setInputValue(name, bool(value), "boolean")
    elif vtype == "color3":
        seq = list(value)
        node.setInputValue(
            name, mx.Color3(float(seq[0]), float(seq[1]), float(seq[2])), "color3"
        )
    elif vtype == "vector3":
        seq = list(value)
        node.setInputValue(
            name, mx.Vector3(float(seq[0]), float(seq[1]), float(seq[2])), "vector3"
        )
    else:
        node.setInputValue(name, float(value), "float")


def write_mtlx_document(
    materials_by_name: Mapping[str, Mapping], out_mtlx_path: str
) -> None:
    """Author a standalone ``.mtlx`` document and write it to *out_mtlx_path*.

    *materials_by_name* maps the surfacematerial element name (which MUST equal
    the bound USD material's leaf name) to a standard_surface-input description.
    Each value is either the wrapped ``{"inputs": ..., "tex_inputs": ...}`` form
    produced from :func:`skinny.pbrt.materials.map_material_mtlx`, or a plain
    ``{input_name: value}`` dict.

    For each entry this authors a document-root ``standard_surface`` shader
    (constants set as ``value`` inputs, textures wired through ``<image>`` nodes
    in a per-material nodegraph) plus a ``surfacematerial`` referencing the
    shader by ``nodename``. The document is validated with ``doc.validate()``
    before writing; an invalid document raises ``ValueError``.
    """
    if not _HAS_MATERIALX:  # pragma: no cover - environment guard
        raise RuntimeError(
            "MaterialX Python package not installed — cannot author .mtlx documents"
        )

    doc = mx.createDocument()

    for mat_name, material in materials_by_name.items():
        constants, tex_inputs = _split_inputs(material)

        shader_name = f"{mat_name}_surface"
        ss = doc.addNode("standard_surface", shader_name, "surfaceshader")

        # Constants first; a texture-bound input overrides the constant
        # connection below (we still author the constant as a sensible
        # fallback value on the input element).
        for in_name, value in constants.items():
            try:
                _set_constant_input(ss, in_name, value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"material {mat_name!r}: cannot author input {in_name!r} "
                    f"from value {value!r}: {exc}"
                ) from exc

        # Texture-bound inputs -> <image> nodes inside a per-material nodegraph.
        if tex_inputs:
            ng = doc.addNodeGraph(f"NG_{mat_name}")
            for in_name, conn in tex_inputs.items():
                image_path, _color_space, value_type = conn
                mtlx_type = _TEXVT_TO_MTLX.get(value_type, "color3")
                img = ng.addNode("image", f"img_{in_name}", mtlx_type)
                file_input = img.addInput("file", "filename")
                file_input.setValueString(str(image_path))
                out = ng.addOutput(f"out_{in_name}", mtlx_type)
                out.setConnectedNode(img)
                # Wire the shader input to the nodegraph output. Replace any
                # constant input element with a connecting one of the right type.
                shader_input = ss.getInput(in_name)
                if shader_input is not None:
                    ss.removeInput(in_name)
                shader_input = ss.addInput(in_name, mtlx_type)
                shader_input.setNodeGraphString(f"NG_{mat_name}")
                shader_input.setOutputString(f"out_{in_name}")

        sm = doc.addNode("surfacematerial", mat_name, "material")
        sm_input = sm.addInput("surfaceshader", "surfaceshader")
        sm_input.setNodeName(shader_name)

    valid, msg = doc.validate()
    if not valid:
        raise ValueError(f"authored MaterialX document is invalid: {msg}")

    mx.writeToXmlFile(doc, out_mtlx_path)


def author_mtlx_reference(
    stage: "Usd.Stage",
    material_prim_path: str,
    mtlx_asset_path: str,
    surfacematerial_name: str,
) -> None:
    """Author the stage→``.mtlx`` reference for one Material prim.

    The Material prim at *material_prim_path* is authored as a **typeless
    ``over``** carrying a layer-level reference to *mtlx_asset_path*, so that
    :func:`skinny.usd_loader._collect_mtlx_asset_paths` reports
    *mtlx_asset_path* and the bound mesh resolves *surfacematerial_name* via
    :func:`skinny.usd_loader._load_mtlx_materials`.

    Why ``over`` and not ``def``: the exported stage must yield the **same**
    ``Material`` whether or not the host has the usdMtlx file-format plugin.

    * **Plugin present** — the ``.mtlx`` reference resolves, the ``over`` picks
      up the ``Material`` type and surface output from the document, and
      ``ComputeBoundMaterial`` succeeds → ``_extract_material`` reads it.
    * **Plugin absent** — the reference fails to compose, so the ``over`` stays
      typeless, ``ComputeBoundMaterial`` returns *no* bound material, and
      ``_resolve_material_binding`` falls through to the
      ``_load_mtlx_materials`` table (matched by binding-target leaf name).

    A ``def``-typed ``Material`` prim would make ``ComputeBoundMaterial``
    *succeed* even with the plugin absent — returning an empty material and
    **bypassing** the rich ``.mtlx`` fallback. So this routine downgrades any
    pre-existing ``def`` at *material_prim_path* (e.g. from a prior
    ``UsdShade.Material.Define`` + ``Bind``) to an ``over``. No UsdPreviewSurface
    shader is authored for the prim either, for the same reason.

    The binding ``material:binding`` relationship is a plain path target, so it
    resolves to this ``over`` regardless of specifier.

    *surfacematerial_name* must equal the leaf of *material_prim_path* — that is
    the key ``_load_mtlx_materials`` matches against.
    """
    sdf_path = Sdf.Path(material_prim_path)
    leaf = sdf_path.name
    if leaf != surfacematerial_name:
        raise ValueError(
            f"material prim leaf {leaf!r} must equal surfacematerial name "
            f"{surfacematerial_name!r} for the .mtlx loader to match the binding"
        )

    root_layer = stage.GetRootLayer()
    # Author the host prim spec on the root layer as a typeless `over`. This
    # both creates it if absent and downgrades any pre-existing `def` so the
    # plugin-absent fallback fires (see docstring).
    prim_spec = Sdf.CreatePrimInLayer(root_layer, material_prim_path)
    prim_spec.specifier = Sdf.SpecifierOver
    prim_spec.typeName = ""

    ref = Sdf.Reference(mtlx_asset_path)
    existing = list(prim_spec.referenceList.prependedItems)
    if ref not in existing:
        prim_spec.referenceList.prependedItems.append(ref)
