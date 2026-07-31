"""Identity gate for change `renderer-command-interface` (tasks 1.2 / 4.2).

Renders a fixed set of suite scenes through `HeadlessRenderer` and writes a
SHA-256 per rendered buffer. Task 1.2 records the pre-change hashes; task 4.2
re-runs the same script after the headless driver posts through the command
queue and requires the hashes to be IDENTICAL, not close.

Run:
    PYTHONPATH=src SKINNY_BACKEND=metal ./bin/python3.13 tools/rci_identity_gate.py <out.json>
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Megakernel path tracer on flat materials — deterministic on both backends
# (the recorded nondeterministic combos are bdpt|wavefront|spectral, ReSTIR and
# neural, none of which this gate touches).
SCENES = [
    "tests/assets/suite/mat_diffuse/mat_diffuse.usda",
    "tests/assets/suite/mat_conductor/mat_conductor.usda",
    "tests/assets/suite/int_caustic/int_caustic.usda",
]
WIDTH = HEIGHT = 128
SAMPLES = 32


BASELINE = (
    REPO / "openspec/changes/renderer-command-interface/identity_before.json"
)


def main(argv: list[str]) -> int:
    out = Path(argv[1]) if len(argv) > 1 else BASELINE
    from skinny.headless import HeadlessRenderer

    results: dict[str, str] = {}
    for rel in SCENES:
        scene = REPO / rel
        started = time.time()
        print(f"START {rel}", flush=True)
        with HeadlessRenderer(WIDTH, HEIGHT) as r:
            arr = r.render_to_array(str(scene), samples=SAMPLES)
            results[rel] = hashlib.sha256(arr.tobytes()).hexdigest()
            # Second call on the SAME renderer: exercises the between-render
            # mutation path (`_prepare` re-applying scene + options), which is
            # what the queue changes.
            arr2 = r.render_to_array(str(scene), samples=SAMPLES, exposure=1.5)
            results[rel + "#exposure1.5"] = hashlib.sha256(arr2.tobytes()).hexdigest()
        print(f"DONE  {rel} ({time.time() - started:.1f}s)", flush=True)

    out.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
