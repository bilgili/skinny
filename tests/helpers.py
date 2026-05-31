"""Shared assertion helpers for GPU shader tests."""

from __future__ import annotations

import math


def assert_near(actual, expected, *, rel=1e-4, abs_tol=1e-6):
    if isinstance(expected, (list, tuple)):
        for a, e in zip(actual, expected):
            assert math.isclose(a, e, rel_tol=rel, abs_tol=abs_tol), (
                f"expected ~{e}, got {a}"
            )
    else:
        assert math.isclose(actual, expected, rel_tol=rel, abs_tol=abs_tol), (
            f"expected ~{expected}, got {actual}"
        )


def assert_unit_vector(v, tol=1e-4):
    length = math.sqrt(sum(c * c for c in v))
    assert math.isclose(length, 1.0, abs_tol=tol), f"|v| = {length}, expected 1.0"


def assert_on_hemisphere(direction, normal, tol=-1e-5):
    dot = sum(d * n for d, n in zip(direction, normal))
    assert dot >= tol, f"dot(dir, N) = {dot}, expected >= 0"


def vec3_dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def vec3_length(v):
    return math.sqrt(vec3_dot(v, v))


def dispatch_uint_kernel(device, func, thread_count, *, inputs, outputs, scalars=None):
    """Dispatch a slangpy compute entry with uint StructuredBuffer I/O and read
    back the outputs.

    ``func`` is a ``slangpy`` Function (e.g. ``module.buildArgs``). ``inputs``
    maps param name → 1-D uint array (uploaded as a StructuredBuffer<uint>);
    ``outputs`` maps param name → element count (allocated zeroed); ``scalars``
    maps param name → int uniform. ``uniform`` resource params want the raw
    buffer, so we pass ``NDBuffer.storage``. Returns {out_name: np.ndarray}.
    """
    import numpy as np
    import slangpy as spy

    kwargs: dict = {}
    out_bufs: dict = {}
    for name, arr in inputs.items():
        buf = spy.NDBuffer.from_numpy(device, np.asarray(arr, dtype=np.uint32))
        kwargs[name] = buf.storage
    for name, count in outputs.items():
        buf = spy.NDBuffer.from_numpy(device, np.zeros(int(count), dtype=np.uint32))
        out_bufs[name] = buf
        kwargs[name] = buf.storage
    for name, value in (scalars or {}).items():
        kwargs[name] = value

    func.dispatch(thread_count, **kwargs)
    device.wait()
    return {name: buf.to_numpy() for name, buf in out_bufs.items()}
