"""Zrodla modelu wysokosciowego (NMT/NMPT) za wspolnym interfejsem DemSampler.

Konwencja wspolrzednych w calym pakiecie: EPSG:2180 jako (x, y) = (easting, northing),
czyli kolejnosc "GIS" (always_xy=True w pyproj, kolejnosc w GeoTIFF/rasterio).
UWAGA: REST GUGiK uzywa konwencji geodezyjnej — jego parametr `x` to northing,
a `y` to easting. Zamiana odbywa sie wylacznie w GugikDem.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Optional, Protocol

import requests

from .config import DemConfig

log = logging.getLogger(__name__)


class DemSampler(Protocol):
    """Zwraca wysokosc npm [m] w punkcie (easting, northing) EPSG:2180
    lub None, gdy brak danych."""
    def sample(self, x: float, y: float) -> Optional[float]: ...


class GugikDem:
    """NMT z otwartego REST GUGiK (services.gugik.gov.pl/nmt/?request=GetHByXY).
    Wynik cache'owany w pamieci po zaokragleniu do 1 m."""

    def __init__(self, url: str, timeout: float = 10.0,
                 session: Optional[requests.Session] = None):
        self.url = url
        self.timeout = timeout
        self.session = session or requests.Session()
        self._cache: dict[tuple[int, int], Optional[float]] = {}
        self.hits = 0
        self.misses = 0

    def sample(self, x: float, y: float) -> Optional[float]:
        key = (int(round(x)), int(round(y)))
        if key in self._cache:
            self.hits += 1
            return self._cache[key]
        self.misses += 1
        z = self._query(x, y)
        self._cache[key] = z
        return z

    def _query(self, easting: float, northing: float) -> Optional[float]:
        try:
            # GetHByXY: x = northing, y = easting (konwencja geodezyjna PUWG 1992).
            r = self.session.get(
                self.url,
                params={"request": "GetHByXY", "x": northing, "y": easting},
                timeout=self.timeout,
            )
        except requests.RequestException as e:
            log.warning("GUGiK NMT: blad zapytania (%.1f, %.1f): %s", easting, northing, e)
            return None
        if not r.ok:
            return None
        try:
            z = float(r.text.strip())
        except ValueError:
            return None
        # "0" zwraca GUGiK, gdy punkt jest poza zasiegiem
        return None if z == 0.0 else z


class RasterDem:
    """NMT/NMPT z lokalnego GeoTIFF w EPSG:2180."""

    def __init__(self, path: Path):
        import rasterio
        self.path = Path(path)
        self._ds = rasterio.open(self.path)
        epsg = self._ds.crs.to_epsg() if self._ds.crs else None
        if epsg != 2180:
            log.warning("Raster %s ma CRS %s, oczekiwano EPSG:2180", self.path, self._ds.crs)

    def sample(self, x: float, y: float) -> Optional[float]:
        for val in self._ds.sample([(x, y)]):
            z = float(val[0])
            if self._ds.nodata is not None and z == self._ds.nodata:
                return None
            if math.isnan(z):
                return None
            return z
        return None

    def close(self) -> None:
        self._ds.close()


def resolve_dem(cfg: DemConfig) -> DemSampler:
    if cfg.source == "raster":
        if cfg.raster_path is None or not Path(cfg.raster_path).exists():
            raise FileNotFoundError(f"dem.source=raster, ale brak pliku: {cfg.raster_path}")
        log.info("DEM: lokalny GeoTIFF %s", cfg.raster_path)
        return RasterDem(cfg.raster_path)
    if cfg.source == "gugik":
        log.info("DEM: GUGiK REST %s (point-query + cache)", cfg.gugik_url)
        return GugikDem(cfg.gugik_url, timeout=cfg.timeout_s)
    raise ValueError(f"Nieznane zrodlo DEM: {cfg.source!r} (dozwolone: gugik, raster)")
