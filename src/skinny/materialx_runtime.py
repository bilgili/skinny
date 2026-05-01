"""MaterialX runtime — load skinny's mtlx libraries and generate Slang for them.

Phase C-1 (this module): wraps MaterialXGenSlang behind a small Python API
that the renderer and Phase D's USD loader will share. Generates per-material
Slang source on demand, captures the function names emitted, and writes the
output next to a `build/` directory for inspection.

Phase C-2+ extend `CompiledMaterial` with:
- `uniform_block`: parsed scalar layout of the per-material parameter UBO so
  the Python side can pack values matching what the gen reads on the GPU.
- `texture_slots`: list of (input_name, default_path) pairs that the bindless
  texture array binds.

Until those land we expose stubs (empty lists) so the data shape is stable.

Usage:
    lib = MaterialLibrary.from_install()
    lib.load()
    cm = lib.generate("M_skinny_skin_default")
    print(cm.pixel_source[:200])
    print(cm.functions_emitted)
"""

from __future__ import annotations

import shutil
import struct
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

import MaterialX as mx
from MaterialX import PyMaterialXGenShader as mxgenshader
from MaterialX import PyMaterialXGenSlang as mxslang


# ─── Library locations ────────────────────────────────────────────────


def _stdlib_path() -> Path:
    """Path to MaterialX's bundled libraries directory."""
    return Path(mx.__file__).resolve().parent / "libraries"


def _skinny_mtlx_path() -> Path:
    """Path to skinny's mtlx library directory (ships inside the package)."""
    return Path(__file__).resolve().parent / "mtlx"


def _build_dir() -> Path:
    """Where generated Slang artefacts get dumped for inspection."""
    return Path(__file__).resolve().parents[2] / "build"


# ─── Data classes ─────────────────────────────────────────────────────


@dataclass
class UniformField:
    """One scalar/vector entry in a generated material's UBO.

    Populated from the gen's reflection (Shader.getStage("pixel").getUniformBlocks()).
    `offset` and `size` are bytes within the scalar-layout buffer; `default`
    is the authored MaterialX value (or None when the input is a connection).
    """

    name: str
    type_name: str  # "float", "color3", "vector3", "color4", etc.
    offset: int = 0
    size: int = 0
    default: Any = None


# ─── Scalar layout tables ─────────────────────────────────────────────


# (alignment, consumed-bytes) per MaterialX type.  The renderer compiles
# with -fvk-use-scalar-layout so ALL Vulkan buffer types (UBO, SSBO /
# StructuredBuffer) use VK_EXT_scalar_block_layout: every component is
# 4-byte aligned, no vec3→16-byte promotion.
_SCALAR_LAYOUT: dict[str, tuple[int, int]] = {
    "float":     (4, 4),
    "integer":   (4, 4),
    "boolean":   (4, 4),       # bool → uint
    "vector2":   (4, 8),
    "vector3":   (4, 12),
    "vector4":   (4, 16),
    "color3":    (4, 12),
    "color4":    (4, 16),
    "matrix33":  (4, 36),      # 9 floats tightly packed
    "matrix44":  (4, 64),
}

# Types we never put in the parameter UBO. surfaceshader / displacement /
# volume / EDF / VDF / BSDF are MaterialX shader-graph handles the gen
# manages internally; filename inputs become bindless texture indices in
# Phase C-4.
_SKIP_PARAM_TYPES: set[str] = {
    "surfaceshader",
    "displacementshader",
    "volumeshader",
    "lightshader",
    "EDF",
    "VDF",
    "BSDF",
    "filename",
}


@dataclass
class TextureSlot:
    """One bindless-array slot referenced by a generated material.

    Phase C-4 populates this from <image> nodes in the active material graph.
    The renderer maps each TextureSlot to an index in the global bindless
    Sampler2D array.
    """

    input_name: str
    default_path: Optional[Path] = None
    slot_index: int = 0


@dataclass
class CompiledMaterial:
    """One MaterialX target run through the Slang shader generator."""

    target_name: str
    pixel_source: str
    vertex_source: str = ""
    functions_emitted: list[str] = field(default_factory=list)
    uniform_block: list[UniformField] = field(default_factory=list)
    texture_slots: list[TextureSlot] = field(default_factory=list)
    # Path to the on-disk dump (when one was written) for debug inspection.
    pixel_path: Optional[Path] = None
    vertex_path: Optional[Path] = None
    # SPIR-V output paths produced by the slangc sanity check; None when the
    # check was skipped or the entry point couldn't be determined.
    pixel_spv_path: Optional[Path] = None
    vertex_spv_path: Optional[Path] = None


# ─── MaterialLibrary ──────────────────────────────────────────────────


class MaterialLibrary:
    """Loads stdlib + skinny mtlx files and runs MaterialXGenSlang on demand.

    A library is constructed once at startup and kept alive for the renderer's
    lifetime. `generate(target_name)` compiles a single target through the
    shader gen; `compile_for_scene(scene)` (Phase C-2) will compile the
    materials referenced by a Scene into one combined Slang module.
    """

    # Function names we expect to see in any skin material's emitted Slang.
    # Used as a quick spot-check that the gen actually inlined our impl files.
    _SKINNY_FUNCTIONS = (
        "skinny_skin_epidermis",
        "skinny_skin_dermis",
        "skinny_skin_subcut",
        "skinny_scattering_layer",
        "skinny_skin_layered_bsdf",
        "skinny_skin_layered_vdf",
    )

    def __init__(
        self,
        *,
        stdlib_path: Path,
        skinny_mtlx_path: Path,
        build_dir: Optional[Path] = None,
    ) -> None:
        self.stdlib_path = stdlib_path
        self.skinny_mtlx_path = skinny_mtlx_path
        self.build_dir = build_dir if build_dir is not None else _build_dir()
        self._doc: Optional[mx.Document] = None

    @classmethod
    def from_install(cls) -> "MaterialLibrary":
        """Construct a MaterialLibrary using paths discovered from imports."""
        return cls(
            stdlib_path=_stdlib_path(),
            skinny_mtlx_path=_skinny_mtlx_path(),
        )

    # ─── Loading ──────────────────────────────────────────────────────

    def load(self) -> mx.Document:
        """Read stdlib + skinny libraries into a merged Document.

        Idempotent — repeated calls return the cached Document. Call this
        once at scene construction.
        """
        if self._doc is not None:
            return self._doc

        doc = mx.createDocument()

        # Standard library.
        stdlib_search = mx.FileSearchPath(str(self.stdlib_path))
        mx.loadLibraries(["stdlib", "pbrlib", "bxdf", "targets"], stdlib_search, doc)

        # Skinny library — defs first, then impls, then any default wirings.
        skinny_search = mx.FileSearchPath(str(self.skinny_mtlx_path))
        for relpath in (
            "skinny_defs.mtlx",
            "genslang/skinny_genslang_impl.mtlx",
            "skinny_skin_default.mtlx",
        ):
            target = self.skinny_mtlx_path / relpath
            if not target.exists():
                raise FileNotFoundError(f"skinny mtlx file missing: {target}")
            sub = mx.createDocument()
            mx.readFromXmlFile(sub, str(target), skinny_search)
            doc.importLibrary(sub)

        valid, message = doc.validate()
        if not valid:
            raise ValueError(
                f"MaterialX document failed validation:\n{message}"
            )

        self._doc = doc
        return doc

    @property
    def document(self) -> mx.Document:
        """The merged Document. `load()` must have been called."""
        if self._doc is None:
            raise RuntimeError("MaterialLibrary.load() not called yet")
        return self._doc

    def import_document(self, doc: "mx.Document") -> None:
        """Import an external MaterialX document into the library.

        Makes surfacematerial elements (and their supporting nodegraphs /
        shader nodes) available to `generate()`. Safe to call multiple
        times with the same document — `importLibrary` skips elements
        whose names already exist.
        """
        if self._doc is None:
            raise RuntimeError("MaterialLibrary.load() not called yet")
        self._doc.importLibrary(doc)

    # ─── Element discovery ────────────────────────────────────────────

    def list_skinny_nodedefs(self) -> list[mx.NodeDef]:
        """All ND_skinny_* nodedefs in the loaded library."""
        return [
            nd for nd in self.document.getNodeDefs()
            if nd.getName().startswith("ND_skinny")
        ]

    def find_target(self, name: str) -> mx.Element:
        """Resolve a generation target by name. Looks at top level first,
        then descends into nodegraphs.
        """
        doc = self.document
        elem = doc.getChild(name)
        if elem is not None:
            return elem
        for ng in doc.getNodeGraphs():
            child = ng.getChild(name)
            if child is not None:
                return child
        children = [c.getName() for c in doc.getChildren()][:30]
        raise KeyError(
            f"target element '{name}' not found in document; "
            f"first few children: {children}"
        )

    # ─── Generation ───────────────────────────────────────────────────

    def _make_generator_and_context(
        self,
    ) -> tuple[mxslang.SlangShaderGenerator, mxgenshader.GenContext]:
        """Create a freshly-configured (generator, context) pair.

        Search paths cover (1) the parent of the stdlib so internal includes
        like "libraries/stdlib/genslang/lib/mx_math.slang" resolve, (2) the
        pbrlib genglsl directory so `lib/mx_closure_type.glsl` resolves
        from skinny impl files, (3) skinny's own genslang impl directory.
        """
        gen = mxslang.SlangShaderGenerator.create()
        ctx = mxgenshader.GenContext(gen)
        for p in (
            self.stdlib_path.parent,
            self.stdlib_path / "pbrlib" / "genglsl",
            self.skinny_mtlx_path / "genslang",
            self.skinny_mtlx_path,
        ):
            ctx.registerSourceCodeSearchPath(mx.FilePath(str(p)))
        return gen, ctx

    @staticmethod
    def _extract_uniform_fields(block) -> list[UniformField]:
        """Walk a VariableBlock and produce scalar-layout UniformFields.

        Skips MaterialX shader-graph handles and filename inputs (the
        latter become bindless texture indices in Phase C-4).
        """
        fields: list[UniformField] = []
        offset = 0
        for i in range(block.size()):
            port = block[i]
            type_name = port.getType().getName()
            if type_name in _SKIP_PARAM_TYPES:
                continue
            layout = _SCALAR_LAYOUT.get(type_name)
            if layout is None:
                # Unknown type — leave a placeholder so the diagnostic
                # output surfaces it without crashing.
                continue
            align, size = layout
            # Pad to alignment boundary.
            if offset % align != 0:
                offset += align - (offset % align)
            default: Any = None
            val = port.getValue()
            if val is not None:
                try:
                    default = val.getData()
                except Exception:
                    default = None
            fields.append(UniformField(
                name=port.getName(),
                type_name=type_name,
                offset=offset,
                size=size,
                default=default,
            ))
            offset += size
        return fields

    @staticmethod
    def _slangc_compile(
        source_path: Path,
        *,
        stage: str,
        entry: str,
        out_path: Path,
    ) -> Optional[Path]:
        """Run slangc on a generated Slang file. Returns the SPIR-V path on
        success, None if slangc is missing or compilation fails.

        Stays best-effort: this is a syntax check, not a hard dependency.
        """
        if shutil.which("slangc") is None:
            return None
        try:
            result = subprocess.run(
                [
                    "slangc", str(source_path),
                    "-target", "spirv",
                    "-stage", stage,
                    "-entry", entry,
                    "-o", str(out_path),
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=60,
            )
        except (subprocess.TimeoutExpired, OSError):
            return None
        if result.returncode != 0:
            # Surface the first few lines of diagnostics so the caller can
            # tell why we couldn't compile.
            print(
                f"slangc check failed for {source_path.name} "
                f"(stage={stage}, entry={entry}):\n"
                + (result.stderr or result.stdout)[:1024]
            )
            return None
        return out_path if out_path.exists() else None

    def compile_for_scene_material(
        self,
        material: Any,
    ) -> CompiledMaterial:
        """Run the gen on a skinny.scene.Material by authoring a
        standard_surface MaterialX network for its captured overrides.

        Each call authors a uniquely-named surfacematerial inside the
        loaded library doc and immediately runs the gen. The element
        stays in the doc afterwards so subsequent calls keep adding to
        the same library — that's fine because element names include the
        material's path-derived id.
        """
        # Sanitize the name into a MaterialX-safe identifier. Path
        # separators in Material.name (e.g. "/World/CubeAMat") aren't
        # legal in MaterialX element names; replace with underscores.
        safe = "".join(c if (c.isalnum() or c == "_") else "_"
                       for c in material.name)
        if not safe:
            safe = "unnamed"
        # Disambiguate when the same name appears twice (shouldn't happen
        # for unique USD prim paths but defensive).
        unique = safe
        i = 0
        doc = self.document
        while doc.getChild(f"M_{unique}") is not None:
            i += 1
            unique = f"{safe}_{i}"

        target_name = add_standard_surface_material(
            doc, unique, material.parameter_overrides
        )
        return self.generate(target_name, compile_check=False)

    def generate(
        self,
        target_name: str,
        *,
        write_to_disk: bool = True,
        compile_check: bool = True,
    ) -> CompiledMaterial:
        """Run MaterialXGenSlang on a target element and return source + metadata.

        Args:
            target_name: name of a Document child (e.g. "M_skinny_skin_default")
                         or a node inside any nodegraph.
            write_to_disk: when True (default), writes pixel_source/vertex_source
                           to `<build>/<target_name>_<stage>.slang` for inspection.

        Raises:
            KeyError: target not found.
            RuntimeError: MaterialX gen failed.
        """
        target = self.find_target(target_name)

        gen, ctx = self._make_generator_and_context()
        try:
            shader = gen.generate(target_name, target, ctx)
        except Exception as e:  # PyMaterialX raises a generic Exception
            raise RuntimeError(
                f"MaterialXGenSlang failed for target '{target_name}': {e}"
            ) from e

        pixel_source = ""
        vertex_source = ""
        try:
            pixel_source = shader.getStage("pixel").getSourceCode()
        except Exception:
            pass
        try:
            vertex_source = shader.getStage("vertex").getSourceCode()
        except Exception:
            pass

        compiled = CompiledMaterial(
            target_name=target_name,
            pixel_source=pixel_source,
            vertex_source=vertex_source,
            functions_emitted=[
                fn for fn in self._SKINNY_FUNCTIONS if fn in pixel_source
            ],
        )

        if write_to_disk:
            self.build_dir.mkdir(parents=True, exist_ok=True)
            if pixel_source:
                p = self.build_dir / f"{target_name}_pixel.slang"
                p.write_text(pixel_source, encoding="utf-8")
                compiled.pixel_path = p
            if vertex_source:
                p = self.build_dir / f"{target_name}_vertex.slang"
                p.write_text(vertex_source, encoding="utf-8")
                compiled.vertex_path = p

        # Reflect the public uniform block from the pixel stage so callers
        # know what to pack. PrivateUniforms holds gen-internal slots
        # (envMatrix, viewPosition, etc.) that we don't expose; we capture
        # only PublicUniforms which carries the material's authored inputs.
        try:
            pixel_stage = shader.getStage("pixel")
            blocks = pixel_stage.getUniformBlocks()
            public = blocks.get("PublicUniforms") if hasattr(blocks, "get") else (
                blocks["PublicUniforms"] if "PublicUniforms" in blocks else None
            )
            if public is not None and not public.empty():
                compiled.uniform_block = self._extract_uniform_fields(public)
        except Exception:
            # Reflection isn't load-bearing today; fall through with an
            # empty list so the rest of the pipeline still works.
            pass

        if compile_check and compiled.pixel_path is not None:
            # GenSlang emits the pixel stage as a fragment shader with a
            # `[shader("fragment")] float4 fragmentMain(...)` entry point.
            # Compile it through slangc to confirm it's syntactically valid
            # — Phase C-2 will reuse the compiled SPIR-V as part of the
            # main_pass build.
            compiled.pixel_spv_path = self._slangc_compile(
                compiled.pixel_path,
                stage="fragment",
                entry="fragmentMain",
                out_path=self.build_dir / f"{target_name}_pixel.spv",
            )

        return compiled


# ─── UsdPreviewSurface → standard_surface conversion ────────────────


# UsdPreviewSurface input → MaterialX standard_surface input. Names match
# the canonical UsdPreviewSurface schema fields the loader captures into
# Material.parameter_overrides. Inputs not in this table pass through.
_USD_PREVIEW_TO_STD_SURFACE: dict[str, str] = {
    "diffuseColor":  "base_color",
    "roughness":     "specular_roughness",
    "metallic":      "metalness",
    "specular":      "specular",
    # opacity in UsdPreviewSurface is float; standard_surface expects
    # color3, handled with explicit replication in the builder below.
    "ior":           "specular_IOR",
}


def _override_to_color3(value: Any) -> Optional["mx.Color3"]:
    if value is None:
        return None
    if hasattr(value, "asTuple"):
        seq = value.asTuple()
    elif hasattr(value, "__getitem__") and not isinstance(value, str):
        seq = [value[i] for i in range(3)]
    else:
        return None
    return mx.Color3(float(seq[0]), float(seq[1]), float(seq[2]))


def add_standard_surface_material(
    doc: "mx.Document",
    name: str,
    overrides: dict[str, Any],
) -> str:
    """Author a NodeGraph + standard_surface + surfacematerial driven by
    UsdPreviewSurface-style overrides. Returns the surfacematerial element
    name that callers pass to MaterialLibrary.generate.

    Each material gets its own unique suffix (`name`) so multiple scene
    materials can coexist in the same Document without colliding.
    Texture-bound inputs in `overrides` are skipped; bindless texture
    integration for gen-emitted materials is a future milestone.
    """
    ng_name = f"NG_{name}"
    ng = doc.addNodeGraph(ng_name)
    ss = ng.addNode("standard_surface", "ss", "surfaceshader")

    diffuse = _override_to_color3(overrides.get("diffuseColor"))
    if diffuse is not None:
        ss.setInputValue("base_color", diffuse, "color3")

    for usd_name, ss_name in _USD_PREVIEW_TO_STD_SURFACE.items():
        if usd_name == "diffuseColor":
            continue  # handled above
        v = overrides.get(usd_name)
        if v is None:
            continue
        try:
            ss.setInputValue(ss_name, float(v))
        except (TypeError, ValueError):
            pass

    op = overrides.get("opacity")
    if op is not None:
        try:
            f = float(op)
            ss.setInputValue("opacity", mx.Color3(f, f, f), "color3")
        except (TypeError, ValueError):
            pass

    out = ng.addOutput("surface_out", "surfaceshader")
    out.setConnectedNode(ss)

    sm_name = f"M_{name}"
    sm = doc.addNode("surfacematerial", sm_name, "material")
    sm_input = sm.addInput("surfaceshader", "surfaceshader")
    sm_input.setNodeGraphString(ng_name)
    sm_input.setOutputString("surface_out")

    return sm_name


# ─── Convenience helpers ──────────────────────────────────────────────


def generate_default_skin() -> CompiledMaterial:
    """One-shot helper that builds a MaterialLibrary and emits Slang for the
    canonical skin material. Used by validate_mtlx.py and the renderer's
    Phase C-2 startup hook.
    """
    lib = MaterialLibrary.from_install()
    lib.load()
    return lib.generate("M_skinny_skin_default")


# ─── Parameter packing ────────────────────────────────────────────────


def _to_seq(value: Any, n: int) -> list[float]:
    """Coerce a Python value or MaterialX Color/Vector into n floats."""
    if value is None:
        return [0.0] * n
    if hasattr(value, "asTuple"):
        seq = list(value.asTuple())
    elif hasattr(value, "__getitem__") and not isinstance(value, str):
        seq = [value[i] for i in range(n)]
    elif isinstance(value, (tuple, list)):
        seq = list(value)
    else:
        return [float(value)] * n
    out = [float(x) for x in seq[:n]]
    while len(out) < n:
        out.append(0.0)
    return out


def _pack_into(buf: bytearray, offset: int, type_name: str, value: Any) -> None:
    """Write `value` at `offset` in `buf` according to `type_name`."""
    if type_name == "float":
        struct.pack_into("f", buf, offset, 0.0 if value is None else float(value))
    elif type_name == "integer":
        struct.pack_into("i", buf, offset, 0 if value is None else int(value))
    elif type_name == "boolean":
        struct.pack_into("I", buf, offset, 1 if bool(value) else 0)
    elif type_name == "vector2":
        struct.pack_into("ff", buf, offset, *_to_seq(value, 2))
    elif type_name in ("vector3", "color3"):
        struct.pack_into("fff", buf, offset, *_to_seq(value, 3))
    elif type_name in ("vector4", "color4"):
        struct.pack_into("ffff", buf, offset, *_to_seq(value, 4))
    elif type_name == "matrix44":
        struct.pack_into("16f", buf, offset, *_to_seq(value, 16))
    elif type_name == "matrix33":
        # Scalar layout: 9 floats tightly packed (no vec4 column padding).
        cols = _to_seq(value, 9)
        struct.pack_into("9f", buf, offset, *cols)
    # Unknown types: leave zeros in `buf`.


# ─── UI param-spec generation (Phase E-4) ─────────────────────────────


# Heuristic UI ranges per field-name suffix or whole-name match. Rough but
# good enough for hand-tuning the gen-reflected layered-skin material from
# the control panel. Float-typed fields that don't match any rule get a
# safe (0..1, step 0.02) default.
_UI_RANGE_BY_SUFFIX: list[tuple[str, tuple[float, float, float]]] = [
    ("_thickness",       (0.01, 10.0, 0.05)),
    ("_anisotropy",      (-0.99, 0.99, 0.02)),
    ("_ior",             (1.0, 2.5, 0.02)),
    ("_roughness",       (0.01, 1.0, 0.02)),
    ("_density",         (0.0, 1.0, 0.05)),
    ("_depth",           (0.0, 1.0, 0.05)),
    ("_tilt",            (0.0, 1.0, 0.05)),
    ("_melanin",         (0.0, 1.0, 0.01)),
    ("_hemoglobin",      (0.0, 1.0, 0.01)),
    ("_oxygenation",     (0.0, 1.0, 0.05)),
    ("_scattering_coeff", (0.0, 5.0, 0.05)),
]


def _ui_range_for_field(name: str, type_name: str) -> tuple[float, float, float]:
    """Return (lo, hi, step) for a continuous slider derived from field name."""
    for suffix, rng in _UI_RANGE_BY_SUFFIX:
        if name.endswith(suffix):
            return rng
    if type_name in ("color3", "color4", "vector3", "vector4"):
        return (0.0, 1.0, 0.02)
    return (0.0, 1.0, 0.02)


def _humanize_field_name(name: str) -> str:
    """Convert a snake_case MaterialX input into a slider label."""
    return name.replace("_", " ").strip().capitalize()


def ui_specs_from_uniform_block(fields: list[UniformField]) -> list[dict]:
    """Build a list of ParamSpec-compatible dicts from a UniformField list.

    Each scalar float field becomes one continuous slider; vec3/color3 each
    expand to 3 sliders (".x/.y/.z" suffix in the path). Returned dicts have
    keys (name, path, kind, step, lo, hi). Caller wraps them into ParamSpec.

    The path uses the `mtlx.<field_name>` (or `mtlx.<field_name>.<i>` for
    vector components) convention; routing happens in app._set_nested.
    """
    specs: list[dict] = []
    for f in fields:
        lo, hi, step = _ui_range_for_field(f.name, f.type_name)
        label_base = _humanize_field_name(f.name)
        if f.type_name == "float":
            specs.append({
                "name": label_base,
                "path": f"mtlx.{f.name}",
                "kind": "continuous",
                "step": step, "lo": lo, "hi": hi,
            })
        elif f.type_name in ("color3", "vector3"):
            for i, axis in enumerate(("x", "y", "z")):
                specs.append({
                    "name": f"{label_base} {axis.upper()}",
                    "path": f"mtlx.{f.name}.{i}",
                    "kind": "continuous",
                    "step": step, "lo": lo, "hi": hi,
                })
        elif f.type_name in ("color4", "vector4"):
            for i, axis in enumerate(("x", "y", "z", "w")):
                specs.append({
                    "name": f"{label_base} {axis.upper()}",
                    "path": f"mtlx.{f.name}.{i}",
                    "kind": "continuous",
                    "step": step, "lo": lo, "hi": hi,
                })
        # integers, booleans, matrices: skip for now (no good UI default)
    return specs


def pack_material_values(
    fields: list[UniformField],
    overrides: Optional[dict[str, Any]] = None,
) -> bytes:
    """Pack values for `fields` into a scalar-layout byte string.

    Per field, picks `overrides[field.name]` when present, otherwise the
    field's authored default. Missing values pack as zeros. The total size
    is rounded up to the struct's scalar alignment (4 bytes) to match the
    StructuredBuffer element stride under -fvk-use-scalar-layout.

    Returned bytes are ready to upload to a `StructuredBuffer<MaterialParams>`-
    style binding.
    """
    overrides = overrides or {}
    if not fields:
        return b""
    last = fields[-1]
    total = last.offset + last.size
    total = ((total + 3) // 4) * 4
    buf = bytearray(total)
    for f in fields:
        value = overrides.get(f.name, f.default)
        _pack_into(buf, f.offset, f.type_name, value)
    return bytes(buf)
