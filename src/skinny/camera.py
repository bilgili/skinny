"""Camera models and the projection/view math they share — device-free.

Split out of ``renderer.py`` (change renderer-pure-core-extraction).
``debug_viewport`` already wanted exactly this and had to reach through
``renderer`` — and so drag in the GPU package — to get it.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from skinny.scene import LensSystem

def _perspective(
    fov_deg: float, aspect: float, near: float = 0.1, far: float = 100.0,
) -> np.ndarray:
    """Reverse-depth perspective projection matrix (stored transposed for GPU).

    Math (OpenGL/Vulkan infinite-far convention, z_near = 0.1, z_far = 100):
        f   = 1 / tan(fov/2)
        P   = [[f/a,  0,        0,          0],
               [0,    f,        0,          0],
               [0,    0,  far/(n-f), n·far/(n-f)],
               [0,    0,       -1,          0]]

    numpy stores row-major → GPU reads column-major → receives Pᵀ → correct.
    """
    fov_rad = np.radians(fov_deg)
    f = 1.0 / np.tan(fov_rad / 2.0)
    proj = np.zeros((4, 4), dtype=np.float32)
    proj[0, 0] = f / aspect
    proj[1, 1] = f
    proj[2, 2] = far / (near - far)
    proj[2, 3] = -1.0
    proj[3, 2] = (near * far) / (near - far)
    return proj


def _look_at(pos: np.ndarray, forward: np.ndarray,
             world_up: Optional[np.ndarray] = None) -> np.ndarray:
    """View matrix from camera position and forward direction (stored transposed).

    Math (camera basis via cross-product):
        r = normalize(forward × up)      (right axis)
        u = r × forward                  (up axis, re-orthogonalised)
        V = [[r.x,  r.y,  r.z, −r·pos],
             [u.x,  u.y,  u.z, −u·pos],
             [−d.x, −d.y, −d.z, d·pos],
             [0,    0,    0,    1     ]]

    where d = forward. Stored transposed for the same numpy/GPU convention as
    _perspective — the GPU reads column-major and recovers V.
    """
    # Returns V^T — numpy row-major, GPU reads column-major → cancels back to V.
    # `world_up` is the reference up; an authored camera (CameraOverride.up) feeds
    # its own up here so non-Y-up pbrt cameras keep their roll. Default +Y ⇒ the
    # prior basis, byte-identical for Y-up / interactive cameras.
    if world_up is None:
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    else:
        world_up = np.asarray(world_up, dtype=np.float32).reshape(3)
    right = np.cross(forward, world_up)
    rn = np.linalg.norm(right)
    if rn < 1e-6:
        # Degenerate: up ∥ forward. Fall back to a secondary world axis so the
        # basis stays finite and orthonormal (no zero `right`).
        alt = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        if abs(float(np.dot(forward, alt))) > 0.9:
            alt = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        right = np.cross(forward, alt)
        rn = np.linalg.norm(right)
    right = right / max(rn, 1e-6)
    up = np.cross(right, forward)
    view = np.eye(4, dtype=np.float32)
    view[:3, 0] = right
    view[:3, 1] = up
    view[:3, 2] = -forward
    view[3, 0] = -np.dot(right, pos)
    view[3, 1] = -np.dot(up, pos)
    view[3, 2] = np.dot(forward, pos)
    return view


def _hero_yaw_pitch() -> tuple[float, float]:
    """Default 3/4 hero-view orbit angles (radians): yaw 30°, pitch 15°."""
    return float(np.radians(30.0)), float(np.radians(15.0))


def _orbit_distance_cap(longest_dim: float) -> float:
    """Initial max orbit distance for a scene whose longest AABB edge is
    ``longest_dim``.

    At least 10× the longest dimension so large scenes can be framed and
    zoomed out, never below the legacy 50-unit floor for small scenes. This is
    the *initial* ceiling only — ``OrbitCamera.set_distance`` raises
    ``max_distance`` past this when the user types or zooms further out.
    """
    return float(max(50.0, 10.0 * longest_dim))


class CameraBase(abc.ABC):
    """PBRT CameraBase analogue — abstract camera-model surface.

    Concrete subclasses (OrbitCamera, FreeCamera) are `@dataclass`es that
    own their controller state. The base contributes the methods that
    every camera shares: the projection matrix and the common slice of
    the change-detection signature (the fields the lens-buffer sync /
    accumulation-reset paths read off the camera).

    Subclasses must expose attribute `position: np.ndarray` and implement
    `forward`, `view_matrix`, and `state_signature`. `position` is not
    declared abstract here because dataclass subclasses use either a
    field (FreeCamera) or a computed `@property` (OrbitCamera), and an
    abstract `@property` in the base would collide with the field form.
    """

    # Attributes every subclass is expected to provide. Listed for typing
    # / readers; concrete declarations live in the @dataclass subclasses.
    fov: float
    near: float
    far: float
    fstop: float
    focus_distance: float
    focal_length_mm: float
    vertical_aperture_mm: float
    lens: Optional["LensSystem"]

    @abc.abstractmethod
    def forward(self) -> np.ndarray: ...

    @abc.abstractmethod
    def view_matrix(self) -> np.ndarray: ...

    @abc.abstractmethod
    def state_signature(self) -> tuple: ...

    def projection_matrix(self, aspect: float) -> np.ndarray:
        return _perspective(self.fov, aspect, self.near, self.far)

    def _common_signature(self) -> tuple:
        """Camera-model slice of state_signature (lens + intrinsics).

        Subclasses concatenate their controller state with this tuple so
        the accumulation-reset path notices changes to either side.
        """
        return (
            float(self.fov), float(self.near), float(self.far),
            float(self.fstop), float(self.focus_distance),
            float(self.focal_length_mm), float(self.vertical_aperture_mm),
            self.lens.signature() if self.lens is not None else ("lens", "none"),
        )


@dataclass
class OrbitCamera(CameraBase):
    """Camera that rotates around a target point (default: centre of the SDF head).

    The head's y-extent is roughly [-0.94, +1.15] and its z-extent [-0.80, +0.97],
    so target=(0, 0.1, 0.05) pins the pivot to its visual centroid. If you orbit
    around the world origin instead, the head drifts noticeably around the frame
    because that origin sits near the jaw/throat rather than the head's middle.
    """

    target: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.1, 0.05], dtype=np.float32)
    )
    # World-space up used to build the view basis. Defaults to +Y (the prior
    # behavior); an authored camera (_override_to_orbit) sets this to its up so a
    # non-Y-up pbrt camera keeps its roll.
    up: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 1.0, 0.0], dtype=np.float32)
    )
    distance: float = 3.0
    yaw: float = 0.0
    pitch: float = 0.0
    fov: float = 45.0
    near: float = 0.1
    far: float = 100.0
    fstop: float = 0.0          # 0 ⇒ wide open; >0 closes the iris to f/N
    focus_distance: float = 0.0  # 0 ⇒ track orbit distance
    focal_length_mm: float = 50.0  # used with fstop to drive iris diameter
    vertical_aperture_mm: float = 24.0  # sensor height in mm; used by the lens path
    lens: Optional["LensSystem"] = None  # PBRT-style thick lens; None ⇒ pinhole
    # Upper clamp for `distance` (wheel zoom, UI slider, auto-frame). Scales
    # with scene size — the renderer raises it to ≥4× the longest scene
    # dimension when a model loads so large scenes can be framed/zoomed out.
    max_distance: float = 50.0

    @property
    def position(self) -> np.ndarray:
        """Camera world position from spherical orbit coordinates.

        Math (spherical → Cartesian):
            x = d · cos(pitch) · sin(yaw)
            y = d · sin(pitch)
            z = d · cos(pitch) · cos(yaw)

        where  d     = orbit distance
               yaw   = azimuth angle (radians)
               pitch = elevation angle (radians, clamped ±89°)
        """
        x = self.distance * np.cos(self.pitch) * np.sin(self.yaw)
        y = self.distance * np.sin(self.pitch)
        z = self.distance * np.cos(self.pitch) * np.cos(self.yaw)
        return self.target + np.array([x, y, z], dtype=np.float32)

    def orbit(self, dx: float, dy: float) -> None:
        self.yaw -= dx * 0.005
        self.pitch += dy * 0.005
        self.pitch = float(np.clip(self.pitch, -np.pi / 2 + 0.01, np.pi / 2 - 0.01))

    def set_distance(self, value: float) -> None:
        """Set the orbit distance to any value ≥ 0.5, growing ``max_distance``
        to fit.

        ``max_distance`` is the current ceiling (slider range + wheel-zoom
        limit). Writing a larger distance raises it so the UI stays consistent;
        it never shrinks here — only a re-frame/model-load resets it. The 1e9
        cap is a degeneracy guard (bounds inf and int-slider precision loss),
        effectively unbounded for real scenes.
        """
        v = float(np.clip(value, 0.5, 1e9))
        if v > self.max_distance:
            self.max_distance = v
        self.distance = v

    def zoom(self, delta: float) -> None:
        self.set_distance(self.distance * (1.0 - delta * 0.1))

    def pan(self, dx: float, dy: float) -> None:
        f = self.target - self.position
        f = f / np.linalg.norm(f)
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        right = np.cross(f, world_up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, f)
        scale = self.distance * 0.002
        self.target = self.target + (-right * dx + up * dy) * scale

    def forward(self) -> np.ndarray:
        f = self.target - self.position
        return f / max(np.linalg.norm(f), 1e-6)

    def view_matrix(self) -> np.ndarray:
        return _look_at(self.position, self.forward(), self.up)

    def state_signature(self) -> tuple:
        return (
            "orbit",
            float(self.yaw), float(self.pitch), float(self.distance),
            float(self.target[0]), float(self.target[1]), float(self.target[2]),
            float(self.up[0]), float(self.up[1]), float(self.up[2]),
        ) + self._common_signature()


@dataclass
class FreeCamera(CameraBase):
    """FPS-style camera: WASD translates, mouse look rotates yaw/pitch."""

    position: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 3.0], dtype=np.float32)
    )
    yaw: float = 0.0
    pitch: float = 0.0
    fov: float = 45.0
    move_speed: float = 1.5   # world units / second
    near: float = 0.1
    far: float = 100.0
    fstop: float = 0.0
    focus_distance: float = 0.0
    focal_length_mm: float = 50.0
    vertical_aperture_mm: float = 24.0
    lens: Optional["LensSystem"] = None

    def forward(self) -> np.ndarray:
        cp = np.cos(self.pitch)
        return np.array([
            np.sin(self.yaw) * cp,
            np.sin(self.pitch),
            -np.cos(self.yaw) * cp,
        ], dtype=np.float32)

    def _right_vec(self) -> np.ndarray:
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        r = np.cross(self.forward(), world_up)
        return r / max(np.linalg.norm(r), 1e-6)

    def look(self, dx: float, dy: float) -> None:
        self.yaw += dx * 0.005
        self.pitch -= dy * 0.005
        self.pitch = float(np.clip(self.pitch, -np.pi / 2 + 0.01, np.pi / 2 - 0.01))

    def move(self, forward: float, right: float, up: float, dt: float) -> None:
        step = self.move_speed * dt
        self.position = (
            self.position
            + self.forward() * (forward * step)
            + self._right_vec() * (right * step)
            + np.array([0.0, 1.0, 0.0], dtype=np.float32) * (up * step)
        )

    def zoom(self, delta: float) -> None:
        # Scroll changes movement speed in free mode.
        self.move_speed = float(np.clip(self.move_speed * (1.0 + delta * 0.1), 0.05, 50.0))

    def view_matrix(self) -> np.ndarray:
        return _look_at(self.position, self.forward())

    def state_signature(self) -> tuple:
        return (
            "free",
            float(self.position[0]), float(self.position[1]), float(self.position[2]),
            float(self.yaw), float(self.pitch),
        ) + self._common_signature()
