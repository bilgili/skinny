"""The Metal argument-table budget census (change ``spectral-table-fold``).

Two layers. The **baseline** layer is hostless and always runs: it reads the
checked-in ``argument_budget_baseline.json`` and asserts no enforced variant
passes a limit and no capacity literal escaped the owner. The **compiler** layer
runs only where ``slangc`` is present (the Vulkan SDK ships it): it re-derives
the census from the compiler and asserts it still matches the baseline — a count
that moves without a baseline update fails here (task 1.6).

Neither layer needs a GPU device. The exact per-kernel ``≤31`` for the
multi-kernel wavefront modules — whose union over-counts — is the gpu-marked
cross-check in ``test_argument_budget_gpu`` (task 1.7).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from skinny import argument_budget as ab
from skinny.shader_variants import Target

BASELINE = ab.load_baseline()
_HAVE_SLANGC = ab.slangc_path() is not None
_slangc = pytest.mark.skipif(not _HAVE_SLANGC,
                             reason="slangc not on PATH / VULKAN_SDK")


def test_every_registered_variant_has_a_baseline():
    recorded = set(BASELINE["variants"])
    registered = {v.name for v in ab.REGISTERED_VARIANTS}
    assert recorded == registered, (
        f"baseline drifted from REGISTERED_VARIANTS: "
        f"missing={registered - recorded}, extra={recorded - registered} — "
        f"regenerate with argument_budget.write_baseline()")


@pytest.mark.parametrize("variant", ab.REGISTERED_VARIANTS,
                         ids=[v.name for v in ab.REGISTERED_VARIANTS])
def test_no_enforced_variant_passes_a_limit(variant):
    """A variant whose census the union counts exactly (megakernel, preview)
    must stay within every Metal table limit. The message names the variant, the
    table, the count and the limit — before any GPU device is built."""
    if not ab.hard_limit_enforced(variant.key):
        pytest.skip("union over-counts this multi-kernel module; per-kernel "
                    "<=31 is the gpu cross-check")
    counts = BASELINE["variants"][variant.name]
    limits = ab.table_limits(variant.key.target)
    for table in ab.TABLES:
        limit = limits[table]
        if limit is not None:
            assert counts[table] <= limit, (
                f"{variant.name}: {table} table has {counts[table]} entries, "
                f"over Metal's limit of {limit}. Declared globals must be "
                f"compiled out, not merely unused.")


@_slangc
@pytest.mark.parametrize("variant", ab.REGISTERED_VARIANTS,
                         ids=[v.name for v in ab.REGISTERED_VARIANTS])
def test_census_matches_baseline(variant):
    """The compiler census still equals the recorded baseline. A global added,
    removed, or freed moves a count and fails here even when the new count is
    still under the limit (spec: an unnoticed growth fails)."""
    c = ab.census(variant)
    recorded = BASELINE["variants"][variant.name]
    got = {ab.BUFFER: c.buffers, ab.TEXTURE: c.textures, ab.SAMPLER: c.samplers}
    assert got == recorded, (
        f"{variant.name}: census {got} != baseline {recorded} — if this change "
        f"is intended, regenerate argument_budget.write_baseline()")


# ── The bindless capacity has one home, tied to the compiler ─────────────────

def test_bindless_capacity_matches_baseline():
    assert (ab.BINDLESS_TEXTURE_CAPACITY_METAL
            == BASELINE["bindless_texture_capacity_metal"] == 119)


@_slangc
def test_bindless_capacity_derives_from_the_compiler():
    """The owner's constant equals the slangc-derived value — 128 minus the
    discrete texture globals the megakernel declares. The constant cannot drift
    from the sources without this failing."""
    assert (ab.bindless_texture_capacity(Target.METAL)
            == ab.BINDLESS_TEXTURE_CAPACITY_METAL)
    assert (ab.bindless_texture_capacity(Target.VULKAN)
            == ab.BINDLESS_TEXTURE_CAPACITY_VULKAN)


def test_consumers_read_the_capacity_from_the_owner():
    from skinny import gpu_backend, metal_compute
    assert (metal_compute.BINDLESS_TEXTURE_CAPACITY
            == ab.BINDLESS_TEXTURE_CAPACITY_METAL)
    assert (gpu_backend.METAL_CAPABILITIES.bindless_texture_capacity
            == ab.BINDLESS_TEXTURE_CAPACITY_METAL)
    assert (gpu_backend.VULKAN_CAPABILITIES.bindless_texture_capacity
            == ab.BINDLESS_TEXTURE_CAPACITY_VULKAN)


def test_capacity_literal_has_no_second_home():
    """The 119 / 128 pool capacity is declared once, in the owner. No consumer
    restates it as a literal (spec: the cap has one home)."""
    src = Path(__file__).resolve().parent.parent / "src" / "skinny"
    literal = re.compile(r"bindless_texture_capacity\s*=\s*\d+")
    for name in ("metal_compute.py", "gpu_backend.py"):
        text = (src / name).read_text()
        assert not literal.search(text) and "= 119" not in text.replace(
            "BINDLESS_TEXTURE_CAPACITY_METAL", ""), (
            f"{name} restates the bindless capacity as a literal; import it "
            f"from argument_budget instead")
