"""Pure USD authoring helpers shared by the renderer and hostless tests.

Kept import-light (no `vulkan`, no GPU) so the transform-authoring logic can be
unit-tested without constructing a `Renderer`.
"""

from __future__ import annotations

import numpy as np


def author_local_transform(xformable, matrix) -> None:
    """Author ``matrix`` (numpy 4x4, USD row-major convention) as the prim's
    single ``xformOp:transform`` in the active edit target.

    When the prim already carries exactly one non-inverse ``xformOp:transform``,
    set that op's value in the edit target (a value-over that wins from the
    session edit layer) instead of clear+add: ``ClearXformOpOrder`` clears only
    the current edit target, so with a transform op authored in a stronger layer
    the subsequent ``AddTransformOp`` would see a duplicate in the composed order
    and raise. ``op.Set`` on an inverse op is illegal, so a sole inverse op falls
    through. The clear+add path is the fallback for the fresh (no ops), inverse,
    and unusual multi-op cases.
    """
    from pxr import Gf, UsdGeom
    arr = np.asarray(matrix, dtype=float).reshape(16)
    gm = Gf.Matrix4d(*[float(x) for x in arr])
    ops = xformable.GetOrderedXformOps()
    if (
        len(ops) == 1
        and ops[0].GetOpType() == UsdGeom.XformOp.TypeTransform
        and not ops[0].IsInverseOp()
    ):
        ops[0].Set(gm)
        return
    xformable.ClearXformOpOrder()
    xformable.AddTransformOp().Set(gm)
