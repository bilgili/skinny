"""Size-gated bin-packing of wavefront path-tracer shade kernels.

The wavefront shade stage compiles one compute pipeline per *group* of
materials. A group's members are compiled into a single kernel, so the kernel's
size grows with the group. MoltenVK flakes the Metal compile of a shade kernel
near ~2.83 MB, so this module packs material members into groups whose combined
size stays under a safety threshold below that line.

Pure CPU, no GPU / Vulkan imports — `pack_shade_groups` takes measured member
sizes and returns the grouping. The flat / MaterialX-graph kernel is always
slot 0 and is **not** a member here; the caller reserves slot 0 for it and
assigns these groups to slots 1.. in order.

Key safety property: a group compiled from several members shares its common
imports, so its real compiled size is <= the sum of the members' isolated
sizes. Packing by that sum therefore never *under*-splits past the threshold —
in the worst case it over-splits (one extra kernel), never produces an
over-limit kernel from members that individually fit.
"""

from __future__ import annotations

# Padded slot count for the wavefront shade counting sort. The classify /
# build-args / clear-counts shaders loop 0..MAX_SLOTS; the CPU dispatches only
# the groups actually built, so a large cap costs only shader loop iterations +
# a few buffer bytes, not dispatch count. 32 makes overflow effectively
# impossible for a real scene (slot 0 = flat/graph; 31 non-flat groups left).
MAX_SLOTS = 32

# MoltenVK's observed large-kernel Metal-compile danger line (~2.83 MB) and the
# safety threshold packing targets below it. The gap absorbs the union-vs-sum
# slack (real group size < sum) in the safe direction.
MOLTENVK_DANGER_BYTES = 2_830_000
SHADE_SIZE_THRESHOLD = 2_400_000


def pack_shade_groups(
    members,
    sizes,
    threshold: int = SHADE_SIZE_THRESHOLD,
    pin_alone=frozenset(),
):
    """First-fit-decreasing bin-pack `members` into shade groups.

    Args:
        members: iterable of hashable member keys (e.g. ("skin",),
            ("python", py_id), ("debug",)). The flat/graph kernel is slot 0 and
            is NOT a member here.
        sizes: mapping member -> measured isolated SPIR-V byte size.
        threshold: max summed member size per group (bytes).
        pin_alone: members that must each occupy their own group (e.g. skin,
            whose monolithic estimator never shares a kernel).

    Returns:
        Ordered list of groups, each a list of members. Group i maps to shade
        slot i+1 (slot 0 is the flat/graph kernel). Order is deterministic.

    Raises:
        ValueError if the grouping needs more than MAX_SLOTS-1 non-flat slots,
        or if `sizes` is missing a member.
    """
    members = list(members)
    missing = [m for m in members if m not in sizes]
    if missing:
        raise ValueError(f"pack_shade_groups: sizes missing for {missing!r}")

    pin_alone = set(pin_alone)
    # Deterministic ordering: largest first, ties broken by the member's repr.
    def _sort_key(m):
        return (-int(sizes[m]), str(m))

    pinned = sorted((m for m in members if m in pin_alone), key=_sort_key)
    rest = sorted((m for m in members if m not in pin_alone), key=_sort_key)

    # Pinned members each get their own (closed) group, emitted first in size
    # order so the layout is stable regardless of input order.
    groups: list[list] = [[m] for m in pinned]
    # Open groups are the FFD bins for the non-pinned members.
    open_groups: list[list] = []
    open_sizes: list[int] = []

    for m in rest:
        s = int(sizes[m])
        placed = False
        for gi, gsize in enumerate(open_sizes):
            if gsize + s <= threshold:
                open_groups[gi].append(m)
                open_sizes[gi] = gsize + s
                placed = True
                break
        if not placed:
            open_groups.append([m])
            open_sizes.append(s)

    groups.extend(open_groups)

    if len(groups) > MAX_SLOTS - 1:
        raise ValueError(
            f"pack_shade_groups: {len(groups)} non-flat shade groups exceeds "
            f"MAX_SLOTS-1 ({MAX_SLOTS - 1}); raise MAX_SLOTS or coarsen packing"
        )
    return groups
