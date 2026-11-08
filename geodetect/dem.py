"""Zrodla modelu wysokosciowego (NMT/NMPT) za wspolnym interfejsem DemSampler.

Konwencja wspolrzednych w calym pakiecie: EPSG:2180 jako (x, y) = (easting, northing),
czyli kolejnosc "GIS" (always_xy=True w pyproj, kolejnosc w GeoTIFF/rasterio).
UWAGA: REST GUGiK uzywa konwencji geodezyjnej — jego parametr `x` to northing,
a `y` to easting. Zamiana odbywa sie wylacznie w GugikDem.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Optional, Protocol

import requests

from .config import DemConfig

log = logging.getLogger(__name__)


class _TransientError(Exception):
    """Chwilowy blad zrodla danych (siec, 5xx) — wyniku nie nalezy cache'owac."""


class DemSampler(Protocol):
    """Zwraca wysokosc npm [m] w punkcie (easting, northing) EPSG:2180
    lub None, gdy brak danych."""
    def sample(self, x: float, y: float) -> Optional[float]: ...


class GugikDem:
    """NMT z otwartego REST GUGiK (services.gugik.gov.pl/nmt/?request=GetHByXY).
    Wynik cache'owany po zaokragleniu do 1 m: w pamieci oraz opcjonalnie na dysku
    (JSON w `cache_path`, zapisywany co `autosave_every` nowych zapytan i w save())."""

    def __init__(self, url: str, timeout: float = 10.0,
                 session: Optional[requests.Session] = None,
                 cache_path: Optional[Path] = None, autosave_every: int = 200):
        self.url = url
        self.timeout = timeout
        self.session = session or requests.Session()
        self.cache_path = Path(cache_path) if cache_path else None
        self.autosave_every = autosave_every
        self._cache: dict[tuple[int, int], Optional[float]] = {}
        self._unsaved = 0
        self.hits = 0
        self.misses = 0
        if self.cache_path and self.cache_path.exists():
            raw = json.loads(self.cache_path.read_text(encoding="utf-8"))
            self._cache = {tuple(map(int, k.split("_"))): v for k, v in raw.items()}
            log.info("NMT cache: wczytano %d punktow z %s", len(self._cache), self.cache_path)

    def save(self) -> None:
        if not self.cache_path or self._unsaved == 0:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.cache_path.with_suffix(".tmp")
        tmp.write_text(json.dumps({f"{x}_{y}": z for (x, y), z in self._cache.items()}),
                       encoding="utf-8")
        tmp.replace(self.cache_path)
        self._unsaved = 0

    def sample(self, x: float, y: float) -> Optional[float]:
        key = (int(round(x)), int(round(y)))
        if key in self._cache:
            self.hits += 1
            return self._cache[key]
        self.misses += 1
        try:
            z = self._query(x, y)
        except _TransientError:
            return None  # blad sieci/serwera — nie cache'ujemy, sprobujemy ponownie
        self._cache[key] = z
        self._unsaved += 1
        if self._unsaved >= self.autosave_every:
            self.save()
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
            raise _TransientError from e
        if r.status_code >= 500 or r.status_code == 429:
            log.warning("GUGiK NMT: HTTP %d (%.1f, %.1f)", r.status_code, easting, northing)
            raise _TransientError
        if not r.ok:
            return None
        try:
            z = float(r.text.strip())
        except ValueError:
            return None
        # "0" zwraca GUGiK, gdy punkt jest poza zasiegiem
        return None if z == 0.0 else z


def save_dem_cache(dem: DemSampler) -> None:
    """Zapisuje cache na dysk, jesli zrodlo go ma (GugikDem)."""
    if hasattr(dem, "save"):
        dem.save()


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
        return GugikDem(cfg.gugik_url, timeout=cfg.timeout_s, cache_path=cfg.cache_path)
    raise ValueError(f"Nieznane zrodlo DEM: {cfg.source!r} (dozwolone: gugik, raster)")
