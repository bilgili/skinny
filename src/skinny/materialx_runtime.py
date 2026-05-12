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

import re
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
    # MaterialX `filename` inputs become bindless texture slot indices
    # (uint) in our compute pipeline; mtlx_gen_shim.SamplerTexture2D wraps
    # the slot for sample/dimension queries.
    "filename":  (4, 4),
    # `string` is used by MaterialXGenSlang for animated-image framerange
    # tokens ("first,last"). The gen lowers it to a single int in cbuffer
    # output — we pack identically and the body uses the int directly.
    "string":    (4, 4),
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

    # ─── Compute-pipeline graph extraction ────────────────────────────

    # Slang type for each MaterialX scalar/vector type. Booleans become
    # uint (Slang `bool` has no defined buffer layout).
    _SLANG_TYPES: dict[str, str] = {
        "float":    "float",
        "integer":  "int",
        "boolean":  "uint",
        "vector2":  "float2",
        "vector3":  "float3",
        "vector4":  "float4",
        "color3":   "float3",
        "color4":   "float4",
        "filename": "SamplerTexture2D",  # mtlx_gen_shim wrapper
        "string":   "int",               # framerange — gen lowers to int
    }

    def generate_for_compute(
        self,
        target_name: str,
        *,
        write_to_disk: bool = True,
    ) -> "Optional[GraphFragment]":
        """Generate a header-form Slang fragment for the compute pipeline.

        Runs MaterialXGenSlang on `target_name` (a surfacematerial), then
        extracts the nodegraph evaluation that drives the wrapping
        standard_surface's `base_color` input. The extracted math is
        wrapped as:

            float3 evalGraph_<sanitized>(float3 P_in, float3 N_in,
                                         float3 T_in,
                                         in GraphParams_<sanitized> p);

        plus a `struct GraphParams_<sanitized> { … }` whose fields match
        the gen-reflected uniforms that the graph body references. The
        result is the chunk the compute build's `generated_materials.slang`
        will `#include`.

        Relies on stable markers in MaterialXGenSlang output. Today's gen
        emits, inside `fragmentMain`:

            // Pixel shader outputs
            float4 out1;

            … graph evaluation …

            surfaceshader <wrapper>_out = surfaceshader(float3(0.0),float3(0.0));
            NG_standard_surface_surfaceshader_100(arg0, base_color_var, …);

        Extraction takes the lines between `// Pixel shader outputs` and
        the `surfaceshader ` declaration; the wrapping call's 2nd
        positional argument names the return value.

        Raises RuntimeError when markers don't appear (e.g. the target is
        not a surfacematerial wrapping standard_surface).
        """
        cm = self.generate(target_name, write_to_disk=write_to_disk,
                           compile_check=False)
        sanitized = _sanitize_ident(target_name)
        struct_name = f"GraphParams_{sanitized}"

        body, return_var = _extract_graph_body(cm.pixel_source)

        # Pure constant-input materials (e.g. plain Glass without a base_color
        # nodegraph) emit no per-pixel math — the wrapping standard_surface
        # receives base_color directly from a cbuffer uniform. In that case
        # the return_var is a uniform from the full uniform_block and the
        # body holds only geomprop normalisations. There's nothing for the
        # compute pipeline to evaluate; the existing flat-material /
        # std_surface SSBO path covers these. Return None so callers can
        # skip them.
        full_uniform_names = {u.name for u in cm.uniform_block}
        if return_var in full_uniform_names:
            return None

        # Reject geomprop inputs we don't pipe yet. UVMap is allowed
        # (becomes `UV_in` from h.uv); anything else (Color sets, secondary
        # UVs, vertex colors, etc.) falls back to flat path.
        if re.search(r"\bvd\.i_(?!geomprop_UVMap\b)[A-Za-z0-9_]+", body):
            return None

        # Identifier rewrites:
        #   vd.* — fragment-shader vertex inputs supplied by the caller.
        body = body.replace("vd.positionObject", "P_in")
        body = body.replace("vd.normalWorld",    "N_in")
        body = body.replace("vd.tangentWorld",   "T_in")
        body = body.replace("vd.positionWorld",  "P_in")  # fallback
        body = body.replace("vd.i_geomprop_UVMap", "UV_in")

        # Substitute each uniform name with `p.<name>` when it appears as
        # a whole-word identifier. Walk uniforms longest-first so prefix
        # collisions (`base` vs `base_color`) don't misfire.
        used_uniforms: list[UniformField] = []
        uniforms_sorted = sorted(cm.uniform_block, key=lambda u: -len(u.name))
        for u in uniforms_sorted:
            pat = re.compile(rf"\b{re.escape(u.name)}\b")
            if pat.search(body):
                body = pat.sub(f"p.{u.name}", body)
                used_uniforms.append(u)
        # Preserve original (offset-sorted) order for struct emission.
        used_uniforms.sort(key=lambda u: u.offset)

        struct_src = _emit_param_struct(struct_name, used_uniforms,
                                        type_map=self._SLANG_TYPES,
                                        public=True)
        func_src = (
            f"public float3 evalGraph_{sanitized}(float3 P_in, float3 N_in,\n"
            f"                              float3 T_in, float2 UV_in,\n"
            f"                              in {struct_name} p)\n"
            f"{{\n"
            f"{_indent(body, '    ')}\n"
            f"    return {return_var};\n"
            f"}}\n"
        )
        # Per-graph fragments are Slang modules: every gen-emitted top-level
        # decl (mx_* helpers, NG_* helpers, BSDF/FresnelData/etc. structs)
        # gets the `internal` modifier so two graphs in the same scene can
        # carry identically-named helpers without ambiguous-call errors.
        # Only struct_name + evalGraph_* wear `public`.
        helper_block = _extract_helper_block(cm.pixel_source)
        helper_block = _filter_helpers_for_body(helper_block, body)
        header = (
            f"// Auto-generated graph module for target '{target_name}'.\n"
            f"// Source: MaterialXGenSlang via MaterialLibrary.generate_for_compute().\n"
            f"// Edits will be overwritten on next scene load.\n\n"
            # SamplerTexture2D + texture* free functions come from the
            # shared shim, which resolves against skinny's bindless
            # flatMaterialTextures[] array (binding 14).
            f"import mtlx_gen_shim;\n\n"
        )
        full_src = (
            header
            + helper_block.rstrip()
            + "\n\n"
            + struct_src
            + "\n"
            + func_src
        )

        path: Optional[Path] = None
        if write_to_disk:
            self.build_dir.mkdir(parents=True, exist_ok=True)
            path = self.build_dir / f"{target_name}_graph.slang"
            path.write_text(full_src, encoding="utf-8")

        return GraphFragment(
            target_name=target_name,
            sanitized_name=sanitized,
            struct_name=struct_name,
            func_name=f"evalGraph_{sanitized}",
            slang_source=full_src,
            uniform_block=used_uniforms,
            source_path=path,
        )


# ─── GraphFragment + helpers ───────────────────────────────────────────


@dataclass
class GraphFragment:
    """A self-contained Slang chunk for one MaterialX nodegraph.

    Built by `MaterialLibrary.generate_for_compute()`. The compute build
    concatenates each scene material's `slang_source` into
    `shaders/generated_materials.slang`, then `main_pass.slang` calls
    `func_name` to compute a hit's base_color before evalStdSurfaceBSDF.
    """

    target_name: str
    sanitized_name: str
    struct_name: str
    func_name: str
    slang_source: str
    uniform_block: list[UniformField]
    source_path: Optional[Path] = None


# `struct` format chars per MaterialX type for scalar-layout packing.
# All sub-elements are 4-byte (float/int/uint), so per-field formats
# concatenate sizes from `_SCALAR_LAYOUT` above.
_PACK_FORMAT: dict[str, str] = {
    "float":    "<f",
    "integer":  "<i",
    "boolean":  "<I",       # 0/1
    "vector2":  "<2f",
    "vector3":  "<3f",
    "vector4":  "<4f",
    "color3":   "<3f",
    "color4":   "<4f",
    "filename": "<I",       # uint bindless slot index
    "string":   "<i",       # gen framerange → int
}


def _coerce_scalar(value: Any, type_name: str) -> tuple:
    """Coerce a Python / MaterialX value into a flat tuple of floats/ints
    matching the type's component count. None or unparseable values
    collapse to zeros."""
    # Integer-shaped types (incl. filename → bindless slot, string →
    # gen-lowered int framerange) need to return an int zero, not 0.0 —
    # struct.pack_into("<I", ...) and "<i" reject floats.
    _INT_TYPES = ("integer", "boolean", "filename", "string")
    if value is None:
        if type_name in _INT_TYPES:
            return (0,)
        if type_name == "float":
            return (0.0,)
        comps = {"vector2": 2, "vector3": 3, "vector4": 4,
                 "color3": 3, "color4": 4}.get(type_name, 1)
        return tuple([0.0] * comps)

    if type_name == "boolean":
        return (1 if bool(value) else 0,)
    if type_name in ("integer", "filename", "string"):
        try:
            return (int(value),)
        except (TypeError, ValueError):
            return (0,)
    if type_name == "float":
        try:
            return (float(value),)
        except (TypeError, ValueError):
            return (0.0,)

    # Composite types — accept Vector*/Color*/sequence/object-with-asTuple.
    if hasattr(value, "asTuple"):
        seq = value.asTuple()
    elif hasattr(value, "__getitem__") and not isinstance(value, (str, bytes)):
        try:
            seq = [value[i] for i in range(_SCALAR_LAYOUT[type_name][1] // 4)]
        except (TypeError, IndexError):
            seq = []
    else:
        seq = []
    comps = _SCALAR_LAYOUT.get(type_name, (4, 4))[1] // 4
    out = []
    for i in range(comps):
        try:
            out.append(float(seq[i]))
        except (TypeError, ValueError, IndexError):
            out.append(0.0)
    return tuple(out)


def pack_uniform_block(
    fields: Iterable[UniformField],
    overrides: Optional[dict[str, Any]] = None,
) -> bytes:
    """Pack a uniform_block into a scalar-layout byte buffer.

    Each field is written at its `offset` using `_PACK_FORMAT[type_name]`.
    Buffer size is the smallest 4-byte-rounded length covering every
    field. `overrides` (when present) replaces the authored MaterialX
    default for fields whose names match.

    Output matches the Slang struct emitted by
    `MaterialLibrary.generate_for_compute` under `-fvk-use-scalar-layout`,
    so callers can vkMapMemory-blit the bytes directly into the
    per-material graph SSBO slot.
    """
    overrides = overrides or {}
    fields = list(fields)
    total = max((f.offset + f.size for f in fields), default=0)
    total = (total + 3) & ~3
    buf = bytearray(total)
    for f in fields:
        fmt = _PACK_FORMAT.get(f.type_name)
        if fmt is None:
            raise RuntimeError(
                f"pack_uniform_block: no pack format for type '{f.type_name}' "
                f"(uniform '{f.name}')"
            )
        value = overrides.get(f.name, f.default)
        components = _coerce_scalar(value, f.type_name)
        struct.pack_into(fmt, buf, f.offset, *components)
    return bytes(buf)


def _sanitize_ident(name: str) -> str:
    """Make a MaterialX element name safe for use as a Slang identifier."""
    safe = re.sub(r"[^0-9A-Za-z_]", "_", name)
    if safe and safe[0].isdigit():
        safe = "_" + safe
    return safe or "_anon"


def _indent(text: str, prefix: str) -> str:
    return "\n".join(prefix + ln if ln else ln for ln in text.splitlines())


# Captures (var, value) from `surfaceshader VAR = surfaceshader(...);`.
_SURFACESHADER_DECL = re.compile(r"\bsurfaceshader\s+(\w+)\s*=\s*surfaceshader\s*\(")

# Captures the wrapping NG_standard_surface_surfaceshader_<ver>(arg0, arg1, …) call.
# arg1 (the second positional argument) names the base_color variable.
_NG_STD_SURFACE_CALL = re.compile(
    r"NG_standard_surface_surfaceshader_\d+\s*\(\s*"
    r"[^,]+,\s*"          # arg0 = base weight
    r"([^,]+)\s*,",       # arg1 = base_color (capture)
    re.DOTALL,
)


# Top-level declarations we mark `internal` so they stay private to a
# per-graph Slang module. Anchored at start-of-line because nested
# declarations inside function bodies don't need the modifier (and would
# break if Slang doesn't allow `internal` on locals).
_INTERNAL_DECL_PATTERN = re.compile(
    r"^("
    r"struct\s+[A-Za-z_]"
    r"|void\s+[A-Za-z_]"
    r"|float\s+[A-Za-z_]"
    r"|float[234]\s+[A-Za-z_]"
    r"|float[234]x[234]\s+[A-Za-z_]"
    r"|int\s+[A-Za-z_]"
    r"|int2\s+[A-Za-z_]"
    r"|bool\s+[A-Za-z_]"
    r"|uint\s+[A-Za-z_]"
    r"|FresnelData\s+[A-Za-z_]"
    r"|ClosureData\s+[A-Za-z_]"
    r")",
    re.MULTILINE,
)

# Function/struct headers we strip wholesale from the gen prelude before
# emitting a per-graph module. mtlx_gen_shim.slang replaces them with a
# uint-slot wrapper that resolves against skinny's bindless texture array.
_GEN_PRELUDE_DROP_HEADERS = (
    "struct SamplerTexture2D",
    "float4 textureLod(SamplerTexture2D",
    "float4 texture(SamplerTexture2D",
    "float4 textureGrad(SamplerTexture2D",
    "int2 textureSize(SamplerTexture2D",
)


def _skip_brace_block(lines: list[str], i: int) -> int:
    """Advance past a brace-balanced block starting at or after `lines[i]`.

    Walks forward until the matching closing brace of the first opening
    brace encountered. The closing line itself is consumed. Used to drop
    `cbuffer { ... }`, `struct Foo { ... };`, and individual function
    bodies whose header is on `lines[i]`.
    """
    depth = 0
    seen_open = False
    j = i
    while j < len(lines):
        opens = lines[j].count("{")
        closes = lines[j].count("}")
        if opens > 0:
            seen_open = True
        depth += opens
        depth -= closes
        j += 1
        if seen_open and depth <= 0:
            break
    return j


# Slang/HLSL keywords + skinny-provided / gen-prelude names we never want to
# treat as "decl identifiers" when walking the helper-block call graph.
# Their presence inside a kept helper body must not pull in extra decls.
_NON_DECL_TOKENS = frozenset({
    "if", "else", "for", "while", "do", "switch", "case", "default",
    "return", "break", "continue", "true", "false",
    "float", "float2", "float3", "float4",
    "float2x2", "float3x3", "float4x4",
    "int", "int2", "int3", "int4", "uint", "uint2",
    "bool", "bool2", "bool3",
    "void", "struct", "const", "static",
    "in", "out", "inout",
    "internal", "public", "private",
    "uniform",
    "NonUniformResourceIndex", "GetDimensions",
    "Texture2D", "SamplerState", "Sampler2D",
    "min", "max", "abs", "clamp", "lerp", "saturate", "pow", "exp", "log",
    "sqrt", "rsqrt", "sin", "cos", "tan", "asin", "acos", "atan", "atan2",
    "floor", "ceil", "round", "frac", "fmod", "step", "smoothstep",
    "dot", "cross", "normalize", "length", "reflect", "refract",
    "select", "isnan", "isinf", "all", "any", "mul",
    "float3x3", "asfloat", "asint", "asuint", "asuint",
    "radians", "degrees", "transpose",
    # Comes from the shared mtlx_gen_shim module — gen output's own
    # SamplerTexture2D + texture* are stripped from the helper block, so
    # filter must not try to look them up in `by_name`.
    "texture", "textureLod", "textureGrad", "textureSize",
    "SamplerTexture2D",
})


# Top-level decl header: <return-or-struct> <name>(args) at column 0.
# `internal ` is optional because callers may invoke this regex either
# before or after `_INTERNAL_DECL_PATTERN` annotation.
_HELPER_HEADER = re.compile(
    r"^(?:internal\s+)?"
    r"(?:struct\s+([A-Za-z_]\w*)"
    r"|(?:void|float|float[234]|float[234]x[234]|int|int2|bool|uint|FresnelData|ClosureData)"
    r"\s+([A-Za-z_]\w*)\s*\()",
    re.MULTILINE,
)


def _scan_idents(text: str) -> set[str]:
    """Return the set of identifier tokens in `text`, excluding keywords."""
    return {
        tok for tok in re.findall(r"\b([A-Za-z_]\w*)\b", text)
        if tok not in _NON_DECL_TOKENS
    }


def _split_helper_decls(block: str) -> list[tuple[str, str, str]]:
    """Split a helper block into (name, header_kind, chunk) tuples.

    `chunk` is the full decl text (header + brace-balanced body + optional
    trailing `;`). `header_kind` is 'struct' for struct decls, 'func' for
    everything else. We track brace depth across the whole block; a new
    decl begins whenever depth is 0 and a header pattern matches at start
    of a line.
    """
    lines = block.splitlines(keepends=True)
    decls: list[tuple[str, str, str]] = []
    i = 0
    depth = 0
    while i < len(lines):
        line = lines[i]
        if depth == 0:
            m = _HELPER_HEADER.match(line)
            if m:
                struct_name, func_name = m.group(1), m.group(2)
                name = struct_name or func_name
                kind = "struct" if struct_name else "func"
                # Collect from here until matching close brace.
                start = i
                # Walk forward to find balanced brace block. Treat the
                # first `{` we see as opening (it may be on a later line).
                seen_open = False
                while i < len(lines):
                    opens = lines[i].count("{")
                    closes = lines[i].count("}")
                    if opens > 0:
                        seen_open = True
                    depth += opens
                    depth -= closes
                    i += 1
                    if seen_open and depth <= 0:
                        break
                if depth < 0:
                    depth = 0
                # Capture trailing `;` for `struct X { ... };` patterns.
                while i < len(lines) and lines[i].strip() in (";", ""):
                    if lines[i].strip() == ";":
                        i += 1
                        break
                    i += 1
                chunk = "".join(lines[start:i])
                decls.append((name, kind, chunk))
                continue
        # Lines outside a top-level decl (macros, blanks). Track brace
        # depth defensively but don't capture.
        depth += line.count("{")
        depth -= line.count("}")
        if depth < 0:
            depth = 0
        i += 1
    return decls


def _filter_helpers_for_body(block: str, body: str) -> str:
    """Keep only decls transitively reachable from `body` identifiers.

    Discards env-/light-related helpers (which reference `u_env*` private
    uniforms and `vd.*` we don't bind) along with the rest of the
    closure-tree machinery our nodegraph extraction doesn't call. Macros
    (`#define …`) and any text outside a top-level decl pass through
    unchanged.
    """
    decls = _split_helper_decls(block)
    # Same name can repeat (function overloads on argument types). Walk
    # all chunks for each name when expanding the call graph so an
    # overload's transitive deps aren't lost.
    by_name: dict[str, list[str]] = {}
    for name, _kind, chunk in decls:
        by_name.setdefault(name, []).append(chunk)

    # Seed reachability from identifiers referenced in fragmentMain body.
    needed: set[str] = set()
    frontier = _scan_idents(body) & by_name.keys()
    while frontier:
        nxt: set[str] = set()
        for name in frontier:
            if name in needed:
                continue
            needed.add(name)
            for chunk in by_name[name]:
                nxt |= _scan_idents(chunk) & by_name.keys()
        frontier = nxt - needed

    # Preserve original ordering of kept decls + intersperse any text that
    # lived between decls (macros, blank lines, comments).
    kept: list[str] = []
    # Re-walk the block, emitting either a kept decl chunk or pass-through
    # text. We do this by recomputing positions: find each decl chunk in
    # the original block and substitute.
    cursor = 0
    for name, kind, chunk in decls:
        idx = block.find(chunk, cursor)
        if idx < 0:
            # Shouldn't happen — chunk came from `block` — but fall back
            # to skipping rather than producing garbage.
            continue
        # Pass-through interstitial text (macros, comments).
        kept.append(block[cursor:idx])
        if name in needed:
            kept.append(chunk)
        cursor = idx + len(chunk)
    kept.append(block[cursor:])
    return "".join(kept)


def _extract_helper_block(pixel_source: str) -> str:
    """Return MaterialXGenSlang's helper-function block as a single string.

    Strips the gen prelude pieces that collide with skinny / our shim:
        - `struct SamplerTexture2D` and its accompanying texture* funcs
          (replaced by mtlx_gen_shim.slang's uint-slot wrapper).
        - `cbuffer pixelCB { ... }` (each uniform becomes a struct field
          in the per-graph GraphParams).
        - `struct VertexData` + `static VertexData vd;` (caller passes
          P_in / N_in / T_in / UV_in by argument).
        - `[shader("fragment")] float4 fragmentMain(...)` (extracted
          separately by `_extract_graph_body`).

    Everything else — BSDF/EDF/VDF struct decls, mx_* helper funcs, NG_*
    nodegraph helpers, M_FLOAT_EPS and other macro #defines, etc. — is
    kept verbatim and emitted as the per-graph module's body.
    """
    lines = pixel_source.splitlines(keepends=True)
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.lstrip()

        # Stop at the fragment entry point; its body is `_extract_graph_body`'s job.
        if "[shader(\"fragment\")]" in line:
            break

        # Drop the gen `cbuffer` block — uniforms move into GraphParams.
        if stripped.startswith("cbuffer "):
            i = _skip_brace_block(lines, i)
            continue
        # Drop `struct VertexData { ... };` + the matching `static VertexData vd;`.
        if stripped.startswith("struct VertexData"):
            i = _skip_brace_block(lines, i)
            # The `;` line after `}` may live on the next line; consume it.
            while i < len(lines) and lines[i].strip() in ("", ";"):
                i += 1
            continue
        if stripped.startswith("static VertexData"):
            i += 1
            continue

        # Drop gen-prelude pieces replaced by mtlx_gen_shim.slang.
        if any(stripped.startswith(h) for h in _GEN_PRELUDE_DROP_HEADERS):
            i = _skip_brace_block(lines, i)
            continue

        out.append(line)
        i += 1

    block = "".join(out)
    # Mark every surviving top-level decl `internal` so it stays private
    # to the per-graph module (prevents ambiguous-call errors when two
    # graphs both emit stdlib helpers with identical names).
    return _INTERNAL_DECL_PATTERN.sub(r"internal \1", block)


def _extract_graph_body(pixel_source: str) -> tuple[str, str]:
    """Return (body, return_var) extracted from MaterialXGenSlang's pixel emit.

    Raises RuntimeError when expected markers aren't found.
    """
    # Locate fragmentMain's body.
    frag_idx = pixel_source.find('[shader("fragment")]')
    if frag_idx < 0:
        raise RuntimeError("generate_for_compute: '[shader(\"fragment\")]' not found in pixel emit")
    brace_idx = pixel_source.find("{", frag_idx)
    if brace_idx < 0:
        raise RuntimeError("generate_for_compute: fragmentMain opening brace not found")
    # Find matching close brace by simple counting (no nested string literals
    # exist in gen output).
    depth = 0
    end_idx = -1
    for i in range(brace_idx, len(pixel_source)):
        c = pixel_source[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                end_idx = i
                break
    if end_idx < 0:
        raise RuntimeError("generate_for_compute: fragmentMain closing brace not found")
    body_full = pixel_source[brace_idx + 1 : end_idx]

    marker = "// Pixel shader outputs"
    m_idx = body_full.find(marker)
    if m_idx < 0:
        raise RuntimeError("generate_for_compute: '// Pixel shader outputs' marker not found")
    after_marker = body_full[m_idx + len(marker):]
    # Skip the next line ('float4 out1;') and any blank lines.
    skip_eol = after_marker.find("\n")
    after_marker = after_marker[skip_eol + 1:]
    # Drop the `float4 out1;` line if present.
    after_marker = re.sub(r"^\s*float4\s+out1\s*;\s*\n", "", after_marker, count=1)

    # End of graph math = the `surfaceshader X = surfaceshader(…);` line.
    decl_match = _SURFACESHADER_DECL.search(after_marker)
    if not decl_match:
        raise RuntimeError(
            "generate_for_compute: 'surfaceshader <var> = surfaceshader(…)' marker not found — "
            "graph extraction supports surfacematerial→standard_surface wrappers only"
        )
    graph_body = after_marker[: decl_match.start()].rstrip() + "\n"

    # The wrapping call names base_color as its 2nd positional arg.
    call_match = _NG_STD_SURFACE_CALL.search(after_marker, decl_match.end())
    if not call_match:
        raise RuntimeError(
            "generate_for_compute: NG_standard_surface_surfaceshader call not found — "
            "cannot determine which graph variable feeds base_color"
        )
    return_var = call_match.group(1).strip()

    return graph_body, return_var


def _emit_param_struct(
    struct_name: str,
    uniforms: Iterable[UniformField],
    *,
    type_map: dict[str, str],
    public: bool = False,
) -> str:
    """Emit a Slang struct with one field per UniformField.

    Caller pre-filters `uniforms` to only those referenced in the graph
    body. Fields are emitted in offset order so the struct layout under
    scalar-layout matches the Python packer. `public=True` marks the
    struct visible to module importers (required when the per-graph
    fragment is compiled as a Slang module).
    """
    visibility = "public " if public else ""
    lines = [f"{visibility}struct {struct_name}", "{"]
    for u in uniforms:
        slang_t = type_map.get(u.type_name)
        if slang_t is None:
            raise RuntimeError(
                f"generate_for_compute: no Slang type mapping for MaterialX type "
                f"'{u.type_name}' (uniform '{u.name}')"
            )
        lines.append(f"    {slang_t} {u.name};  // offset {u.offset}")
    lines.append("};")
    lines.append("")
    return "\n".join(lines)


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


# Reserved graphId values in the compute pipeline's `materialTypes` SSBO.
# Real MaterialX graphs occupy ids 2.. (one per distinct GraphFragment).
GRAPH_ID_SKIN = 0
GRAPH_ID_FLAT = 1
GRAPH_ID_FIRST = 2


def assign_graph_ids(
    fragments: Iterable[GraphFragment],
) -> dict[str, int]:
    """Return `{target_name → graphId}` for the given fragment list.

    The compute pipeline's evalSceneGraph() switches on graphId; this map
    is what the renderer writes into materialTypes[matId] for materials
    whose MaterialX target produced a GraphFragment.
    """
    return {
        gf.target_name: GRAPH_ID_FIRST + idx
        for idx, gf in enumerate(fragments)
    }


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
