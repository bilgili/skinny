"""Regenerate the recording-pass declared-globals golden from real slangc
reflection (change ``recording-adapter-live-bindings``).

The declared side of the binding-coverage gate used to be a hand-written Slang
parser. Codex pre-merge review kept finding valid declaration spellings it
under-reported (qualified globals, split-across-lines, two-per-line,
block-comment-prefixed) — the exact fail-open the gate exists to prevent, because
a line/regex parser cannot tell a file-scope resource global from a function
parameter of resource type without full scope tracking.

So the declared globals now come from the **compiler's own reflection**, not a
heuristic: for each registered pass this compiles the entry module with that
pass's variant defines and reads ``-reflection-json``'s top-level ``parameters``
— the identical global set the renderer binds against. The result is checked in
as ``recording_pass_globals.json`` — a checked-in generated artifact the hostless
gate trusts the way the parity harness trusts its checked-in reference EXRs: a
``gpu``-marked freshness test regenerates it and diffs, so a stale golden fails.
Regenerate here, never hand-edit.

Run (needs ``slangc`` on PATH; the Vulkan SDK supplies it)::

    PYTHONPATH=src .venv/bin/python -m tests.fixtures.gen_recording_pass_globals
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

from skinny import recording_compute as rec

GOLDEN = Path(__file__).resolve().parent / "recording_pass_globals.json"


def _include_dirs(shader_dir: Path) -> list[Path]:
    # The two -I paths the megakernel compile uses (the generated MaterialX
    # Slang tree carries the skin BSDF the megakernel imports).
    return [shader_dir, shader_dir.parent / "mtlx" / "genslang"]


def reflect_globals(pass_) -> list[str]:
    """The top-level shader globals ``slangc`` reflects for one pass, sorted.

    These are exactly the resources the compiled kernel declares — uniform block
    included — so the coverage gate compares the compiler's truth against the
    host's bind map, with no parser in between.

    Two deliberate choices, both from codex review:

    * The target is **Metal**, not SPIR-V-with-Metal-defines: the coverage gate
      is about the Metal argument table, so reflect the variant that actually
      ships (the reflected global-scope names are the same either way, but a
      Metal reflection cannot drift from what Metal binds).
    * A bindable **entry-point** parameter (a `uniform` that lowers to a push
      constant) lives in Slang's entry-point scope, NOT the top-level
      `parameters`. Reading only `parameters` would silently miss it. The
      registered passes take only system-value thread IDs (binding ``None``), so
      this refuses on any entry-point parameter that carries a real binding
      rather than dropping it — a future pass that adds one fails here loudly.
    """
    shader_dir = rec.shader_dir()
    defines = pass_.key.session_defines()
    with tempfile.TemporaryDirectory() as tmp:
        refl = Path(tmp) / "refl.json"
        cmd = [
            "slangc",
            str(shader_dir / f"{pass_.entry_module}.slang"),
            "-target", "metal",
            "-entry", pass_.entry_point,
            "-stage", "compute",
        ]
        for name, value in defines.items():
            cmd += [f"-D{name}={value}"]
        for inc in _include_dirs(shader_dir):
            cmd += ["-I", str(inc)]
        cmd += ["-reflection-json", str(refl), "-o", str(Path(tmp) / "out.metal")]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0 or not refl.exists():
            raise RuntimeError(
                f"slangc reflection failed for {pass_.name}:\n{proc.stderr}")
        data = json.loads(refl.read_text())
    for ep in data.get("entryPoints", []):
        bound = [p["name"] for p in ep.get("parameters", [])
                 if p.get("binding") is not None]
        if bound:
            raise RuntimeError(
                f"{pass_.name}: entry point {ep.get('name')!r} declares bindable "
                f"parameter(s) {bound} — these lower to push constants in the "
                "entry-point scope and are not in top-level `parameters`. Teach "
                "the generator to include them (and the host bind map to supply "
                "them) before registering a pass that uses one.")
    return sorted(p["name"] for p in data.get("parameters", []))


def _emit_generated_sources() -> None:
    """Emit the runtime-generated MaterialX Slang the megakernel imports
    (`generated_materials.slang`), the same way the renderer does at build.

    `main_pass.slang` `import generated_materials;`, a file `emit_megakernel_sources`
    writes into the shader tree — it is not checked in (a build artifact). With
    no MaterialX graphs (`[]`) it emits the minimal default-variant form, which
    is the variant this golden reflects. Pure source emission, no device.
    """
    from skinny.megakernel_sources import emit_megakernel_sources

    emit_megakernel_sources(rec.shader_dir(), [])


def build() -> dict:
    _emit_generated_sources()
    return {
        p.name: {
            "entry_module": p.entry_module,
            "entry_point": p.entry_point,
            "globals": reflect_globals(p),
        }
        for p in rec.RECORDABLE_PASSES
    }


def main() -> None:
    GOLDEN.write_text(json.dumps(build(), indent=2, sort_keys=True) + "\n")
    print(f"wrote {GOLDEN}")


if __name__ == "__main__":
    main()
