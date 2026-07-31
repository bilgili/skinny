"""Web concurrency smoke for change `renderer-command-interface` (task 4.3).

Stands up a real `SkinnySession` — real GPU context, real renderer, real
background render thread — and hammers it from two other threads with the
commands a browser produces: parameter writes (the slider drag), camera
gestures, control toggles and an awaited screenshot. Then it checks that the
render thread stayed alive, produced frames throughout, and that every awaited
command settled.

This is the scripted stand-in for dragging a slider in a browser: same code
path, same threads, no browser. It does not cover the JavaScript client.

Run:
    PYTHONPATH=src SKINNY_BACKEND=metal ./bin/python3.13 tools/rci_web_smoke.py
"""
from __future__ import annotations

import threading
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCENE = REPO / "tests/assets/suite/mat_diffuse/mat_diffuse.usda"
DURATION_S = 6.0


def main() -> int:
    from dataclasses import replace

    from skinny import web_app
    from skinny.backend_select import select_backend
    from skinny.web_app import SkinnySession

    # main() normally builds the plan; this script skips main(), so resolve the
    # backend the way every front-end does rather than taking the module
    # default (which is the pre-plan "vulkan").
    web_app._PLAN = replace(web_app._PLAN, backend=select_backend(None))
    web_app._USD_PATH = SCENE  # a Path; the renderer reads `.name` off it
    print(f"backend: {web_app._PLAN.backend}", flush=True)
    session = SkinnySession("smoke")
    errors: list[str] = []
    try:
        session.initialize()
        if not session.ready:
            print(f"FAIL: session did not initialise: {session._init_error}")
            for line in session._init_log:
                print(f"  {line}")
            return 1
        print(f"session ready on {session.ctx.gpu_info.name}", flush=True)

        stop = threading.Event()
        posted = {"param": 0, "camera": 0, "control": 0}

        def slider_drag() -> None:
            """What a browser slider drag looks like on the IOLoop thread."""
            i = 0
            while not stop.is_set():
                i += 1
                session.set_param("exposure", 1.0 + (i % 20) * 0.05)
                posted["param"] += 1
                time.sleep(0.002)

        def camera_drag() -> None:
            while not stop.is_set():
                session.handle_camera("orbit", {"dx": 1.0, "dy": 0.0})
                posted["camera"] += 1
                if posted["camera"] % 50 == 0:
                    session.handle_control("toggle_hud")
                    posted["control"] += 1
                time.sleep(0.003)

        threads = [
            threading.Thread(target=slider_drag, daemon=True),
            threading.Thread(target=camera_drag, daemon=True),
        ]
        for t in threads:
            t.start()

        # Liveness during the storm is *frames produced*, NOT `accum_frame`:
        # the camera moves every few ms, so accumulation resets every frame and
        # the counter correctly sits at 0 the whole time. Count the pushes
        # instead.
        pushed = {"n": 0}
        real_push = session._push_frame

        def counting_push(frame_type, data, _real=real_push) -> None:
            pushed["n"] += 1
            _real(frame_type, data)

        session._push_frame = counting_push

        samples: list[int] = []
        deadline = time.time() + DURATION_S
        while time.time() < deadline:
            time.sleep(0.4)
            samples.append(pushed["n"])
            if not session._running:
                errors.append(f"render thread died: {session._render_error!r}")
                break

        # An awaited command mid-storm: it must settle, not time out.
        try:
            started = time.time()
            shot = session.screenshot("png")
            waited = time.time() - started
            print(f"screenshot: {len(shot)} bytes, waited {waited:.2f}s", flush=True)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"awaited screenshot failed: {exc!r}")

        stop.set()
        for t in threads:
            t.join(timeout=5)

        if session._render_error is not None:
            errors.append(f"render thread reported: {session._render_error!r}")
        if len(session._commands) > 200:
            errors.append(f"command queue grew unbounded: {len(session._commands)}")

        print(f"posted: {posted}", flush=True)
        print(f"frames pushed: {samples}", flush=True)
        if len(set(samples)) <= 1:
            errors.append(f"render stalled during the storm: {samples}")

        # Quiet phase: with nothing posting, accumulation must climb from 0.
        # A render thread that survived the storm but stopped converging would
        # pass every check above and fail this one.
        time.sleep(2.0)
        quiet_accum = session.accum_frame
        print(f"accum after 2s quiet: {quiet_accum}", flush=True)
        if quiet_accum <= 0:
            errors.append("accumulation did not resume after the storm")

        # The last posted slider value must be the one the renderer holds — a
        # dropped or torn write shows up here.
        session.set_param("exposure", 1.75)
        time.sleep(0.5)
        if abs(float(session.renderer.exposure) - 1.75) > 1e-6:
            errors.append(
                f"slider write lost: exposure={session.renderer.exposure}"
            )
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        return 1
    finally:
        # Metal dispatch hygiene: the context is destroyed on every exit path.
        session.cleanup()

    if errors:
        for e in errors:
            print(f"FAIL: {e}")
        return 1
    print("PASS: no torn state, no stall, awaited command settled")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
