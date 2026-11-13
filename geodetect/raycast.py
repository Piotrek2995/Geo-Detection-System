"""Ray casting: piksel -> promien w lokalnym ENU -> PIERWSZE przeciecie z powierzchnia.

Promien liczony jest w ENU (uklad rekonstrukcji ODM). Punkty promienia przeliczamy scisle
(EnuFrame) do ukladu powierzchni i porownujemy wysokosci w ukladzie, ktory powierzchnia
deklaruje: "npm" -> H = h_ODM + z_offset (+tilt), "odm_local" -> h_ODM bez z_offset.
Probkowanie wsadowe: `chunk` punktow promienia na raz.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from pyproj import CRS, Transformer

from .config import RaycastConfig
from .geo import EnuFrame, transform
from .odm import CameraEO, CameraIO
from .surface import VERTICAL_NPM, VERTICAL_ODM, Surface, as_surface


@dataclass
class RayHit:
    enu: np.ndarray            # (3,) punkt trafienia w lokalnym ENU
    x_2180: float              # easting
    y_2180: float              # northing
    h: float                   # [m npm] wysokosc punktu trafienia (h_ODM + z_offset)
    lon: float
    lat: float
    range_m: float             # odleglosc kamera -> punkt
    surface: str = ""          # zrodlo powierzchni (cfg.surface.source)
    surface_h: float = float("nan")    # wartosc powierzchni w punkcie (w jej ukladzie wysokosci)
    ground_h: Optional[float] = None   # NMT [m npm] w tym punkcie (dla NMPT/DSM), jesli dostepny

    @property
    def above_ground_m(self) -> Optional[float]:
        """Wysokosc trafienia nad NMT (pokrycie/obiekt) — tylko gdy znany ground_h."""
        return None if self.ground_h is None else self.h - self.ground_h


def pixel_to_ray(uv: tuple[float, float], io: CameraIO, eo: CameraEO
                 ) -> tuple[np.ndarray, np.ndarray]:
    """(origin, direction) promienia w lokalnym ENU; direction jednostkowy."""
    pts = np.array([[[uv[0], uv[1]]]], dtype=np.float64)
    x, y = cv2.undistortPoints(pts, io.K, io.dist)[0, 0]
    d_cam = np.array([x, y, 1.0])  # uklad kamery: z w przod, v w dol
    d_world = eo.R_wc.T @ d_cam
    return eo.C_enu.copy(), d_world / np.linalg.norm(d_world)


class SurfaceProjector:
    """Punkty ENU -> wspolrzedne powierzchni + wysokosc do porownania z jej wartosciami."""

    def __init__(self, frame: EnuFrame, surface: Surface):
        if surface.vertical not in (VERTICAL_NPM, VERTICAL_ODM):
            raise ValueError(f"Nieznany uklad wysokosci powierzchni: {surface.vertical!r}")
        self.frame = frame
        self.surface = surface
        self._is_2180 = CRS.from_user_input(surface.crs) == CRS.from_epsg(2180)
        self._to_surface = None if self._is_2180 else \
            Transformer.from_crs("EPSG:4326", surface.crs, always_xy=True)

    def project(self, P: np.ndarray):
        """P (n,3) ENU -> (sx, sy, H_cmp, x_2180, y_2180, H_npm, lon, lat)."""
        e, n, u = (np.ascontiguousarray(P[:, k]) for k in range(3))
        lon, lat, h = self.frame.enu_to_geodetic(e, n, u)
        lon, lat, h = np.asarray(lon), np.asarray(lat), np.asarray(h)
        x, y = self.frame.lonlat_to_puwg92(lon, lat)
        x, y = np.asarray(x), np.asarray(y)
        H_npm = self.frame.npm_height(h, x, y)
        sx, sy = (x, y) if self._is_2180 else transform(self._to_surface, lon, lat)
        H_cmp = H_npm if self.surface.vertical == VERTICAL_NPM else h
        return np.asarray(sx), np.asarray(sy), H_cmp, x, y, H_npm, lon, lat

    def height_above(self, origin: np.ndarray, direction: np.ndarray, ts: np.ndarray) -> np.ndarray:
        """Wysokosc promienia nad powierzchnia w parametrach t (NaN = brak danych)."""
        P = origin[None, :] + ts[:, None] * direction[None, :]
        sx, sy, H_cmp, *_ = self.project(P)
        return H_cmp - self.surface.sample_many(sx, sy)


def raycast(origin: np.ndarray, direction: np.ndarray, surface, frame: EnuFrame,
            cfg: RaycastConfig) -> Optional[RayHit]:
    """Pierwsze przeciecie promienia z powierzchnia (marsz + bisekcja); None, gdy brak.

    `surface`: Surface albo punktowy DemSampler w EPSG:2180/npm (adapter as_surface).
    Krok: cfg.step_for(surface.resolution_m) — dla rastrow nie wiekszy niz piksel,
    zeby nie przeskoczyc cienkiej przeszkody (drzewo, krawedz dachu) na NMPT/DSM.
    Promien musi isc w dol (direction[2] < 0).
    """
    surface = as_surface(surface)
    if direction[2] >= 0:
        return None
    proj = SurfaceProjector(frame, surface)
    step = cfg.step_for(surface.resolution_m)
    chunk = cfg.rest_chunk if surface.resolution_m is None else cfg.chunk
    ts_all = np.arange(cfg.t_min_m, cfg.t_max_m + 1e-9, step)

    prev_t: Optional[float] = None
    prev_d: Optional[float] = None
    for i in range(0, len(ts_all), chunk):
        ts = ts_all[i:i + chunk]
        diffs = proj.height_above(origin, direction, ts)
        for t, d in zip(ts, diffs):
            if np.isnan(d):
                continue
            if prev_d is not None and d <= 0 < prev_d:
                return _bisect(proj, origin, direction, prev_t, float(t), cfg.bisect_iters)
            prev_t, prev_d = float(t), float(d)
    return None


def _bisect(proj: SurfaceProjector, origin, direction, lo: float, hi: float,
            iters: int) -> RayHit:
    for _ in range(iters):
        mid = (lo + hi) / 2.0
        dm = proj.height_above(origin, direction, np.array([mid]))[0]
        if np.isnan(dm):
            break
        if dm > 0:
            lo = mid
        else:
            hi = mid
    # `hi` = pierwszy punkt na/pod powierzchnia (blad <= krok/2^iters). Przy pionowym licu
    # (sciana, krawedz dachu na NMPT/DSM) srodek przedzialu moglby wypasc jeszcze nad gruntem
    # przed przeszkoda i zwrocic surface_h gruntu zamiast przeszkody.
    return _make_hit(proj, origin, direction, hi)


def _make_hit(proj: SurfaceProjector, origin, direction, t: float) -> RayHit:
    p = origin + t * direction
    sx, sy, _, x, y, H, lon, lat = proj.project(p[None, :])
    z_surf = float(proj.surface.sample_many(sx, sy)[0])
    return RayHit(p, float(x[0]), float(y[0]), float(H[0]), float(lon[0]), float(lat[0]),
                  float(t), proj.surface.name, z_surf)


def median_hit(hits: list[RayHit], frame: EnuFrame) -> RayHit:
    """Trafienie zbiorcze (mask_footprint_median): mediana XY/H/zasiegu z wielu promieni."""
    x = float(np.median([h.x_2180 for h in hits]))
    y = float(np.median([h.y_2180 for h in hits]))
    lon, lat = frame.puwg92_to_lonlat(x, y)
    ground = [h.ground_h for h in hits if h.ground_h is not None]
    return RayHit(np.median([h.enu for h in hits], axis=0), x, y,
                  float(np.median([h.h for h in hits])), float(lon), float(lat),
                  float(np.median([h.range_m for h in hits])), hits[0].surface,
                  float(np.nanmedian([h.surface_h for h in hits])),
                  float(np.median(ground)) if ground else None)
