"""Mouse / wheel / key dispatch onto ``renderer.camera``.

Same dispatch shape ``web_app.SkinnySession.handle_camera`` uses, so the
two backends produce identical camera behaviour from identical deltas.
"""

from __future__ import annotations


class CameraDispatcher:
    """Stateless wrapper that mirrors the GLFW input handler's camera path
    onto a renderer. The host (Qt viewport) feeds raw events; this picks
    the right ``camera`` method based on ``camera_mode`` + which button is
    held.
    """

    def __init__(self, renderer) -> None:
        self.renderer = renderer

    def drag(self, dx: float, dy: float, *,
             left: bool, right: bool, middle: bool) -> None:
        cam = self.renderer.camera
        mode = getattr(self.renderer, "camera_mode", "orbit")
        if mode == "orbit":
            if left:
                cam.orbit(float(dx), float(dy))
            elif right or middle:
                cam.pan(float(dx), float(dy))
        else:  # free-look
            if left:
                cam.look(float(dx), float(dy))

    def zoom(self, delta: float) -> None:
        self.renderer.camera.zoom(float(delta))

    def move(self, forward: float, right: float, up: float, dt: float) -> None:
        cam = self.renderer.camera
        if hasattr(cam, "move"):
            cam.move(float(forward), float(right), float(up), float(dt))

    def reset(self) -> None:
        self.renderer.reset_camera()

    def toggle_mode(self) -> None:
        self.renderer.toggle_camera_mode()
