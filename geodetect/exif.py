"""Odczyt GPS z EXIF. Potensic Atom zapisuje tylko GPS (bez yaw/pitch/roll),
a GPS Altitude to wysokosc AGL od punktu startu, nie npm."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import exifread


@dataclass(frozen=True)
class GpsFix:
    lat: float
    lon: float
    alt: Optional[float]  # dla Potensica: AGL od punktu startu


def _ratio_to_float(v) -> float:
    """exifread zwraca Ratio lub list[Ratio]. Dla Lat/Lon lista = [d,m,s];
    dla Altitude lista bywa jednoelementowa [Ratio]."""
    if isinstance(v, list):
        nums = [float(x.num) / float(x.den) for x in v]
        if len(nums) == 3:
            d, m, s = nums
            return d + m / 60.0 + s / 3600.0
        if len(nums) == 1:
            return nums[0]
        raise ValueError(f"Nieznany ksztalt GPS Ratio: {v}")
    return float(v.num) / float(v.den)


def read_gps(image_path: Path) -> GpsFix:
    with open(image_path, "rb") as f:
        tags = exifread.process_file(f, details=False)

    if "GPS GPSLatitude" not in tags or "GPS GPSLongitude" not in tags:
        raise ValueError(f"Brak GPS w EXIF: {image_path}")

    lat = _ratio_to_float(tags["GPS GPSLatitude"].values)
    lon = _ratio_to_float(tags["GPS GPSLongitude"].values)
    if str(tags.get("GPS GPSLatitudeRef", "N")) == "S":
        lat = -lat
    if str(tags.get("GPS GPSLongitudeRef", "E")) == "W":
        lon = -lon

    alt = None
    if "GPS GPSAltitude" in tags:
        alt = _ratio_to_float(tags["GPS GPSAltitude"].values)
        if str(tags.get("GPS GPSAltitudeRef", "0")) in ("1", "\x01"):
            alt = -alt
    return GpsFix(lat, lon, alt)
