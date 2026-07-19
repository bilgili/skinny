"""Hostless source contracts for spectral directional-proposal support."""

from pathlib import Path


_ROOT = Path(__file__).resolve().parents[1]
_SHADERS = _ROOT / "src" / "skinny" / "shaders"


def _source(relative: str) -> str:
    return (_SHADERS / relative).read_text()


def test_megakernel_spectral_path_uses_shared_proposal_mixture():
    src = _source("integrators/path_spectral.slang")

    assert "BSDFSample bs = sampleBounceDirection(mat, pctx, rng);" in src
    assert "BSDFSample bs = mat.sample(wo, rng);" not in src
    assert "weightS = flatProposalWeightS(mat, cols, wo, bs.wi, bs.pdf);" in src

    # The generating mixture density, not a recomputed bare BSDF pdf, must feed
    # environment-miss and emissive-hit MIS after the bounce.
    assert "misProposalPdf = " in src
    assert "? bs.pdf : -1.0;" in src
    assert "prevProposalPdf = spawnedBySpecular ? -1.0 : bs.pdf;" in src


def test_wavefront_spectral_path_uses_shared_proposal_mixture():
    bounce = _source("wavefront/flat_bounce.slang")
    finish = _source("wavefront/wf_shade_common.slang")

    # The flat shade stage obtains its world-space bounce direction and full
    # mixture pdf from the same proposal seam as the RGB path.
    assert "BSDFSample bs = sampleBounceDirection(mat, pctx, rng);" in bounce
    assert "bs.wi = tangentToWorld(bs.wi, T, B, N);" in bounce

    # The spectral finisher ignores the RGB weight and recolors the selected
    # direction per wavelength over that returned mixture pdf.
    assert "wiT = worldToTangent(br.bsdfSample.wi, Tsp, Bsp, Nsp);" in finish
    assert (
        "weightS = flatProposalWeightS(\n"
        "            mat, cols, woSp, wiT, br.bsdfSample.pdf);"
        in finish
    )
    assert "? br.bsdfSample.pdf : -1.0;" in finish


def test_mixed_spectral_proposal_weight_includes_opacity():
    common = _source("integrators/spectral_flat_common.slang")

    assert "Spectrum flatProposalWeightS(" in common
    assert "fc.proposalMask == PROPOSAL_BSDF" in common
    assert "? flatResponseS(mat, c, wo, wi)" in common
    assert ": flatResponseNEE(mat, c, wo, wi);" in common
