"""Przeglad folderu zdjec: lista zdjec z pozycja GPS i wysokoscia z EXIF (GUI: strona Projekt)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

PHOTO_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


@dataclass
class PhotoInfo:
    name: str
    path: Path
    lat: Optional[float]
    lon: Optional[float]
    alt: Optional[float]           # wysokosc z EXIF (Potensic: wzgledem startu)
    error: Optional[str] = None

    @property
    def has_gps(self) -> bool:
        return self.lat is not None and self.lon is not None


def list_photos(folder: Path) -> list[Path]:
    folder = Path(folder)
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in PHOTO_SUFFIXES)


def scan_photos(folder: Path) -> list[PhotoInfo]:
    """Zdjecia z folderu z GPS z EXIF (zdjecia bez GPS: error zamiast pozycji)."""
    from .exif import read_gps
    out = []
    for p in list_photos(folder):
        try:
            g = read_gps(p)
            out.append(PhotoInfo(p.name, p, g.lat, g.lon, g.alt))
        except Exception as exc:          # brak EXIF / GPS / uszkodzony plik
            out.append(PhotoInfo(p.name, p, None, None, None, str(exc)))
    log.info("%s: %d zdjec, z GPS %d", folder, len(out), sum(p.has_gps for p in out))
    return out


def suggest_start_image(photos: list[PhotoInfo], max_alt_m: float = 3.0) -> Optional[str]:
    """Kandydat na zdjecie startowe z ziemi: pierwsze zdjecie z wysokoscia EXIF <= max_alt_m
    (barometr zerowany na starcie, zdjecie ~1 m nad punktem startu)."""
    for p in photos:
        if p.alt is not None and p.alt <= max_alt_m:
            return p.name
    return None
