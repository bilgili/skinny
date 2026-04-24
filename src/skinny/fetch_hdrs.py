"""Download well-known CC0 HDRIs from Poly Haven into ./hdrs/.

Run via:  python -m skinny.fetch_hdrs

Skinny reads any `.hdr` in that folder on startup and makes them available
via the `Environment` parameter. All listed assets are CC0 (public domain)
from polyhaven.com.
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

# Poly Haven 1K HDRIs — curated for skin/portrait rendering.
# A mix of studios (neutral), outdoor daylight, dusk/sunset, and night scenes.
POLYHAVEN_1K: dict[str, str] = {
    # Studios
    "studio_small_09":              "https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/1k/studio_small_09_1k.hdr",
    "brown_photostudio_02":         "https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/1k/brown_photostudio_02_1k.hdr",
    "brown_photostudio_06":         "https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/1k/brown_photostudio_06_1k.hdr",
    "photo_studio_loft_hall":       "https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/1k/photo_studio_loft_hall_1k.hdr",
    "studio_country_hall":          "https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/1k/studio_country_hall_1k.hdr",

    # Outdoor — daylight / overcast
    "kloofendal_48d_partly_cloudy": "https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/1k/kloofendal_48d_partly_cloudy_1k.hdr",
    "cape_hill":                    "https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/1k/cape_hill_1k.hdr",
    "autumn_field":                 "https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/1k/autumn_field_1k.hdr",
    "symmetrical_garden":           "https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/1k/symmetrical_garden_02_1k.hdr",
    "drakensberg_solitary_mountain":"https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/1k/drakensberg_solitary_mountain_1k.hdr",
    "royal_esplanade":              "https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/1k/royal_esplanade_1k.hdr",
    "empty_warehouse_01":           "https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/1k/empty_warehouse_01_1k.hdr",

    # Dawn / dusk / sunset
    "kiara_1_dawn":                 "https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/1k/kiara_1_dawn_1k.hdr",
    "venice_sunset":                "https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/1k/venice_sunset_1k.hdr",
    "the_sky_is_on_fire":           "https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/1k/the_sky_is_on_fire_1k.hdr",
    "kloppenheim_02":               "https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/1k/kloppenheim_02_1k.hdr",

    # Night
    "dikhololo_night":              "https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/1k/dikhololo_night_1k.hdr",
    "moonless_golf":                "https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/1k/moonless_golf_1k.hdr",
}


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    out_dir = repo_root / "hdrs"
    out_dir.mkdir(exist_ok=True)

    for name, url in POLYHAVEN_1K.items():
        target = out_dir / f"{name}.hdr"
        if target.exists():
            print(f"[skip] {name}.hdr already present")
            continue
        print(f"[get ] {name}.hdr  <- {url}")
        try:
            urllib.request.urlretrieve(url, target)
        except Exception as exc:
            print(f"       failed: {exc}")
            if target.exists():
                target.unlink()

    print(f"\nDone. HDRs in: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
