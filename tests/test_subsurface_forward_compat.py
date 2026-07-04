"""Forward-compatibility guard (task 5.5) for the volumetric subsurface walk.

The change implements only the homogeneous, dielectric-bounded interior, but the
spec REQUIRES the transport be majorant/null-collision based and read the medium
ONLY through a density seam (`densityAt` / `mediumMajorant`) dispatched by a
`kind` tag, with the medium a handle-referenced registry entry — so a future
heterogeneous NanoVDB grid is a new `kind` + two `case` bodies, no transport
rework. These are source-level structural assertions (no GPU): they fail loudly
if a later edit collapses the seam (e.g. a closed-form homogeneous-only
transmittance path that a grid could not reuse).
"""

from __future__ import annotations

from pathlib import Path

SHADERS = Path(__file__).resolve().parent.parent / "src" / "skinny" / "shaders"
MEDIUM = (SHADERS / "materials" / "subsurface" / "medium.slang").read_text()
WALK = (SHADERS / "materials" / "subsurface" / "subsurface_walk.slang").read_text()
BINDINGS = (SHADERS / "bindings.slang").read_text()
COMMON = (SHADERS / "common.slang").read_text()


def test_density_seam_dispatches_by_kind():
    # The seam is two functions, each a switch over the medium `kind` tag.
    assert "float densityAt(Medium m" in MEDIUM
    assert "float3 mediumMajorant(Medium m" in MEDIUM
    for fn in ("densityAt", "mediumMajorant"):
        body = MEDIUM.split(fn, 1)[1]
        assert "switch (m.kind)" in body, f"{fn} must dispatch on m.kind"
        assert "MEDIUM_HOMOGENEOUS" in body


def test_medium_kind_enum_reserves_nanovdb():
    assert "MEDIUM_HOMOGENEOUS" in BINDINGS
    # Formerly the reserved additive extension point; nanovdb-volume-rendering
    # discharged it — the grid kind is now implemented (see densityAt's case).
    assert "MEDIUM_NANOVDB" in BINDINGS


def test_medium_is_handle_referenced_with_grid_transform():
    # Medium carries a kind tag + the grid resource reference (not hardwired to
    # a material interior). nanovdb-volume-rendering DISCHARGED the reserved
    # `gridHandle`: the concrete reference is now the folded world→uvw rows
    # (the density texture itself is the single always-bound binding 26), and
    # a free-standing MediumInterface registers through the same resolveMedium.
    assert "struct Medium" in COMMON
    struct = COMMON.split("struct Medium", 1)[1].split("};", 1)[0]
    assert "kind" in struct and "worldToUvw0" in struct
    assert "Medium resolveMedium(uint materialId)" in MEDIUM
    # The grid kind is a real case body in the seam, no longer a comment.
    assert "case MEDIUM_NANOVDB:" in MEDIUM


def test_walk_reads_medium_only_through_the_seam():
    # The null-collision segment traversal must look up local sigma_t via the
    # seam, and must NOT short-circuit homogeneous media with a closed-form
    # transmittance a grid could not reuse.
    assert "densityAt(medium" in WALK
    assert "mediumMajorant(medium" in WALK
    # Null-collision (Woodcock) acceptance against the majorant — the
    # heterogeneous algorithm, with constant density as its degenerate case.
    assert "majorant" in WALK and "scatterAlbedo" in WALK


def test_homogeneous_density_is_the_constant_degenerate_case():
    # densityAt(MEDIUM_HOMOGENEOUS) ≡ 1 and majorant = sigma_a+sigma_s, so a
    # uniform density field equal to the constant sigma_t yields the identical
    # walk — proving a grid lookup slots in behind the same seam (and now DOES:
    # the MEDIUM_NANOVDB case bodies live beside these). Function bodies are
    # sliced at the next declaration, not the first '}' — the grid case has a
    # braced body of its own.
    density_body = (MEDIUM.split("float densityAt(Medium m", 1)[1]
                    .split("float3 mediumMajorant", 1)[0])
    homo_case = density_body.split("MEDIUM_HOMOGENEOUS:", 1)[1].split("default", 1)[0]
    assert "return 1.0" in homo_case
    # `mediumMajorant` is now a single GLOBAL majorant σ̄_t = σ_a + σ_s for every
    # kind (nanovdb/cloud both normalise density ≤ 1, so the packed σ_t bounds
    # σ_t·density) — no per-case switch. For MEDIUM_HOMOGENEOUS that majorant is
    # exactly the constant σ_t, so density≡1 · majorant reproduces the closed
    # form and a uniform grid slots in behind the same seam.
    majorant_body = (MEDIUM.split("float3 mediumMajorant(Medium m", 1)[1]
                     .split("Medium resolveMedium", 1)[0])
    assert "sigmaA + m.sigmaS" in majorant_body  # majorant = constant σ_t
