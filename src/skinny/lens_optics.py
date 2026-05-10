"""PBRT-v3 RealisticCamera helpers.

CPU-side ports of the geometric helpers in
``pbrt-v3/src/cameras/realistic.cpp``:

* :func:`trace_lenses_from_film` — line-by-line port of
  ``RealisticCamera::TraceLensesFromFilm``.
* :func:`bound_exit_pupil` — port of
  ``RealisticCamera::BoundExitPupil``: builds an axis-aligned rectangle
  on the rear-element plane that a film point at radius ``r ∈
  [pFilmX0, pFilmX1]`` can hit through the lens stack. The shader uses
  these per-radius bounds (rotated by the pixel's azimuth) to sample
  *only* directions that the lens won't vignette, so closing the iris
  at small fstops doesn't shrink the rendered area to a central
  pinhole — every pixel keeps a productive sample disk.

Lens convention (matches PBRT and the shader port in
``shaders/cameras/thick_lens.slang``):
  * Elements stored front (i=0) → rear (i=N-1).
  * Each interface's ``thickness`` is the gap to the *next surface
    toward the film* (the rearmost thickness is the film distance).
  * Lens-local frame: film at z=0, scene direction is -z.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


@dataclass
class LensInterface:
    """Mirror of PBRT's ``LensElementInterface`` plus the renderer's
    aperture-stop flag. All values in lens-local distance units (the
    caller decides whether mm or world; the math is invariant)."""

    radius: float          # signed curvature radius
    thickness: float       # gap to next surface toward the film
    ior: float             # medium between this surface and the next-toward-film
    half_aperture: float   # clear half-aperture (radius) of this interface
    is_stop: bool


def _intersect_spherical(
    o: np.ndarray, d: np.ndarray, radius: float, z_center: float,
) -> Optional[tuple[float, np.ndarray]]:
    """Ray-sphere intersection (PBRT IntersectSphericalElement)."""
    oc = o - np.array([0.0, 0.0, z_center])
    A = float(np.dot(d, d))
    B = 2.0 * float(np.dot(d, oc))
    C = float(np.dot(oc, oc) - radius * radius)
    disc = B * B - 4.0 * A * C
    if disc < 0.0:
        return None
    sq = math.sqrt(disc)
    inv2A = 0.5 / A
    t0 = (-B - sq) * inv2A
    t1 = (-B + sq) * inv2A
    use_closer_t = (d[2] > 0) ^ (radius < 0)
    t = min(t0, t1) if use_closer_t else max(t0, t1)
    if t < 0.0:
        return None
    hit = oc + t * d
    n = hit / np.linalg.norm(hit)
    if np.dot(n, -d) < 0.0:
        n = -n
    return float(t), n


def _refract(
    wi: np.ndarray, n: np.ndarray, eta: float,
) -> Optional[np.ndarray]:
    """PBRT Refract. ``wi`` is the incident direction pointing AWAY
    from the surface (i.e. ``-ray.d`` after normalisation)."""
    cos_i = float(np.dot(n, wi))
    sin2_i = max(0.0, 1.0 - cos_i * cos_i)
    sin2_t = eta * eta * sin2_i
    if sin2_t >= 1.0:
        return None
    cos_t = math.sqrt(1.0 - sin2_t)
    return eta * (-wi) + (eta * cos_i - cos_t) * n


def trace_lenses_from_scene(
    elements: Sequence[LensInterface],
    o0: np.ndarray, d0: np.ndarray,
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Trace ``(origin, direction)`` from outside the lens (scene side)
    through the stack front→rear. Ray must be heading TOWARD the film
    (PBRT lens-local +z direction). Used by
    ``ComputeThickLensApproximation`` to find the lens's principal
    planes and focal points. PBRT's
    ``RealisticCamera::TraceLensesFromScene``.
    """
    o = np.array(o0, dtype=np.float64).copy()
    d = np.array(d0, dtype=np.float64).copy()
    # PBRT walks front → rear; element_z starts at the front-most
    # surface (at most-negative z) and increments by each thickness.
    front_z = -sum(e.thickness for e in elements)
    element_z = front_z
    for i, e in enumerate(elements):
        if e.is_stop or e.radius == 0.0:
            if d[2] <= 0.0:
                return None  # ray must be heading toward film
            t = (element_z - o[2]) / d[2]
            n = None
        else:
            z_center = element_z + e.radius
            hit = _intersect_spherical(o, d, e.radius, z_center)
            if hit is None:
                return None
            t, n = hit
        p_hit = o + t * d
        if p_hit[0] * p_hit[0] + p_hit[1] * p_hit[1] > e.half_aperture * e.half_aperture:
            return None
        o = p_hit
        if not (e.is_stop or e.radius == 0.0):
            # IOR convention: element[i].ior is the medium between
            # element i and the *next-toward-film* surface i+1. Going
            # scene→film, leaving medium just behind element[i-1] (or
            # 1.0 air for i=0) and entering medium element[i].ior.
            eta_i = elements[i - 1].ior if i > 0 else 1.0
            eta_t = e.ior
            wi = -d / np.linalg.norm(d)
            wt = _refract(wi, n, eta_i / eta_t)
            if wt is None:
                return None
            d = wt
        element_z += e.thickness
    return o, d


def compute_thick_lens_approximation(
    elements: Sequence[LensInterface],
    film_diagonal: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Port of ``RealisticCamera::ComputeThickLensApproximation``.

    Returns ``(pz, fz)`` where ``pz[0], pz[1]`` are the world-z of the
    front and rear principal planes (in PBRT lens-local convention,
    measured as positive distances from the film) and ``fz[0], fz[1]``
    are the front and rear focal points.
    """
    if not elements:
        return ((0.0, 0.0), (0.0, 0.0))
    front_z = -sum(e.thickness for e in elements)
    rear_z = -elements[-1].thickness
    x = 0.001 * float(film_diagonal)

    # Trace a paraxial ray from scene at +infinity (well in front of
    # the lens) toward the film. Origin sits 1 unit *outside* the
    # front element (on the scene side, z < front_z); direction +z.
    r_scene_o = np.array([x, 0.0, front_z - 1.0])
    r_scene_d = np.array([0.0, 0.0, 1.0])
    r_film = trace_lenses_from_scene(elements, r_scene_o, r_scene_d)
    if r_film is None:
        raise RuntimeError("ComputeThickLensApproximation: scene→film paraxial trace failed")
    o, d = r_film
    # Image-side focal point: t such that the traced ray crosses axis (x=0).
    tf = -o[0] / d[0]
    fz1 = -(o[2] + tf * d[2])
    # Rear principal plane: t where traced ray height == x.
    tp = (x - o[0]) / d[0]
    pz1 = -(o[2] + tp * d[2])

    # Trace from film side outward (origin AT film, dir toward scene).
    r_film2_o = np.array([x, 0.0, 0.0])
    r_film2_d = np.array([0.0, 0.0, -1.0])
    r_scene2 = trace_lenses_from_film(elements, r_film2_o, r_film2_d)
    if r_scene2 is None:
        raise RuntimeError("ComputeThickLensApproximation: film→scene paraxial trace failed")
    o, d = r_scene2
    tf2 = -o[0] / d[0]
    fz0 = -(o[2] + tf2 * d[2])
    tp2 = (x - o[0]) / d[0]
    pz0 = -(o[2] + tp2 * d[2])

    return ((pz0, pz1), (fz0, fz1))


def effective_focal_length(
    elements: Sequence[LensInterface], film_diagonal: float = 43.27,
) -> float:
    """Effective focal length of the lens system (paraxial). Computes
    the rear principal plane and rear focal point, returns the
    distance between them. Independent of any user-authored
    `focalLength` field — driven entirely by the element geometry.
    """
    (pz0, pz1), (fz0, fz1) = compute_thick_lens_approximation(elements, film_diagonal)
    return float(fz0 - pz0)


def focus_thick_lens(
    elements: Sequence[LensInterface],
    film_diagonal: float,
    focus_distance: float,
) -> float:
    """Returns the rearmost-element thickness needed to image a subject
    at ``focus_distance`` (in lens-local units, e.g. mm) onto the
    film. Uses the lens's paraxial-derived effective focal length F
    and solves the thin-lens equation
        1/s + 1/s' = 1/F        ⇒  s' = F·s / (s − F)
    then *replaces* the authored rear thickness with ``s'`` (rather
    than treating the authored value as an infinity-focus baseline).
    This makes the result independent of the lens-design file's
    authored back focal length — handy when the design's nominal
    focal-length label disagrees with the geometry's true F.
    ``focus_distance`` must exceed F.
    """
    if not elements:
        return 0.0
    F = effective_focal_length(elements, film_diagonal)
    if F <= 0.0 or focus_distance <= F * 1.001:
        return float(elements[-1].thickness)
    image_distance = F * focus_distance / (focus_distance - F)
    return float(image_distance)


def trace_lenses_from_film(
    elements: Sequence[LensInterface],
    o0: np.ndarray, d0: np.ndarray,
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Trace ``(origin, direction)`` from the film through the stack
    rear→front. Returns ``(origin, direction)`` after the front
    refraction, or ``None`` if the ray was blocked.

    Direct port of ``RealisticCamera::TraceLensesFromFilm``.
    """
    o = np.array(o0, dtype=np.float64).copy()
    d = np.array(d0, dtype=np.float64).copy()
    element_z = 0.0
    N = len(elements)
    for i in range(N - 1, -1, -1):
        e = elements[i]
        element_z -= e.thickness
        if e.is_stop or e.radius == 0.0:
            if d[2] >= 0.0:
                return None
            t = (element_z - o[2]) / d[2]
            n = None
        else:
            z_center = element_z + e.radius
            hit = _intersect_spherical(o, d, e.radius, z_center)
            if hit is None:
                return None
            t, n = hit
        p_hit = o + t * d
        if p_hit[0] * p_hit[0] + p_hit[1] * p_hit[1] > e.half_aperture * e.half_aperture:
            return None
        o = p_hit
        if not (e.is_stop or e.radius == 0.0):
            eta_i = e.ior
            eta_t = elements[i - 1].ior if i > 0 else 1.0
            wi = -d / np.linalg.norm(d)
            wt = _refract(wi, n, eta_i / eta_t)
            if wt is None:
                return None
            d = wt
    return o, d


def bound_exit_pupil(
    elements: Sequence[LensInterface],
    p_film_x0: float, p_film_x1: float,
    n_samples: int = 1024,
) -> tuple[float, float, float, float]:
    """Port of ``RealisticCamera::BoundExitPupil``.

    For film points whose radius lies in ``[p_film_x0, p_film_x1]``,
    return the smallest axis-aligned rectangle on the rear-element
    plane such that a uniform-disk sample inside the rectangle still
    produces a non-vignetted ray for at least one of the swept film
    radii. The resulting bounds collapse the search space the shader
    has to sample over.
    """
    if not elements:
        return (0.0, 0.0, 0.0, 0.0)
    rear = elements[-1]
    rear_z = -rear.thickness
    rear_radius = rear.half_aperture

    n_per_axis = max(1, int(math.sqrt(n_samples)))
    pupil_min_x = pupil_min_y = float("inf")
    pupil_max_x = pupil_max_y = float("-inf")
    found = False
    for i in range(n_per_axis):
        for j in range(n_per_axis):
            # Sample rear-element disk uniformly across a 2× bounding box
            # (PBRT uses a square; valid samples lie inside its inscribed
            # disk).
            ux = (i + 0.5) / n_per_axis
            uy = (j + 0.5) / n_per_axis
            p_rear = np.array([
                (2.0 * ux - 1.0) * rear_radius,
                (2.0 * uy - 1.0) * rear_radius,
                rear_z,
            ])
            if (p_rear[0] * p_rear[0] + p_rear[1] * p_rear[1]
                    > rear_radius * rear_radius):
                continue
            # Two film positions per cell: the radius range's endpoints.
            for fr in (p_film_x0, p_film_x1):
                p_film = np.array([fr, 0.0, 0.0])
                d = p_rear - p_film
                if trace_lenses_from_film(elements, p_film, d) is not None:
                    found = True
                    pupil_min_x = min(pupil_min_x, float(p_rear[0]))
                    pupil_min_y = min(pupil_min_y, float(p_rear[1]))
                    pupil_max_x = max(pupil_max_x, float(p_rear[0]))
                    pupil_max_y = max(pupil_max_y, float(p_rear[1]))
                    break
    if not found:
        # Fall back to the full rear-element disk so the shader has
        # *something* to sample (the pixel will simply vignette in
        # the trace itself).
        return (-rear_radius, +rear_radius, -rear_radius, +rear_radius)
    # Expand by half a sample step so the boundary cells are covered
    # (PBRT also pads by `2 * pupilBoundsArea / nSamples`-ish; the
    # half-cell expansion below is the simpler stable choice).
    pad = 2.0 * rear_radius / n_per_axis
    return (
        pupil_min_x - pad, pupil_max_x + pad,
        pupil_min_y - pad, pupil_max_y + pad,
    )


def compute_exit_pupil_bounds(
    elements: Sequence[LensInterface],
    film_diagonal: float,
    num_bounds: int = 64,
    samples_per_bound: int = 1024,
) -> np.ndarray:
    """Build a per-radius table of exit-pupil bounds.

    Returns an ``(num_bounds, 4)`` float32 array; row ``i`` is the
    bound for film radii in ``[i / num_bounds, (i+1) / num_bounds] ·
    film_diagonal/2``. Layout per row is ``(xMin, xMax, yMin, yMax)``.
    """
    out = np.zeros((num_bounds, 4), dtype=np.float32)
    half_diag = 0.5 * float(film_diagonal)
    for i in range(num_bounds):
        r0 = i / num_bounds * half_diag
        r1 = (i + 1) / num_bounds * half_diag
        out[i] = bound_exit_pupil(elements, r0, r1, samples_per_bound)
    return out
