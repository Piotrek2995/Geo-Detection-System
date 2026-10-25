"""Ray casting: piksel -> promien w lokalnym ENU -> przeciecie z DEM.

Promien liczony jest w ENU (dokladnie w ukladzie rekonstrukcji ODM). W kazdym kroku
punkt promienia przeliczamy scisle do EPSG:2180 (EnuFrame) i tam probkujemy DEM.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .config import RaycastConfig
from .dem import DemSampler
from .geo import EnuFrame
from .odm import CameraEO, CameraIO


@dataclass
class RayHit:
    enu: np.ndarray            # (3,) punkt trafienia w lokalnym ENU
    x_2180: float              # easting
    y_2180: float              # northing
    h: float                   # [m npm] = wysokosc DEM w punkcie trafienia (z dokl. bisekcji)
    lon: float
    lat: float
    range_m: float             # odleglosc kamera -> punkt


def pixel_to_ray(uv: tuple[float, float], io: CameraIO, eo: CameraEO
                 ) -> tuple[np.ndarray, np.ndarray]:
    """(origin, direction) promienia w lokalnym ENU; direction jednostkowy."""
    pts = np.array([[[uv[0], uv[1]]]], dtype=np.float64)
    x, y = cv2.undistortPoints(pts, io.K, io.dist)[0, 0]
    d_cam = np.array([x, y, 1.0])  # uklad kamery: z w przod, v w dol
    d_world = eo.R_wc.T @ d_cam
    return eo.C_enu.copy(), d_world / np.linalg.norm(d_world)


def raycast(origin: np.ndarray, direction: np.ndarray, dem: DemSampler, frame: EnuFrame,
            cfg: RaycastConfig) -> Optional[RayHit]:
    """Marsz po promieniu co cfg.step_m + bisekcja. None, gdy brak przeciecia.

    Promien musi isc w dol (direction[2] < 0); dla wiekszych odleglosci od originu
    "w dol" w ENU ≈ w dol lokalnie, bo krzywizna na setkach metrow jest pomijalna.
    """
    if direction[2] >= 0:
        return None

    def height_above_dem(t: float) -> Optional[float]:
        p = origin + t * direction
        x, y, h = frame.enu_to_puwg92(p[0], p[1], p[2])
        z_dem = dem.sample(x, y)
        return None if z_dem is None else h - z_dem  # >0 nad terenem

    prev_t: Optional[float] = None
    prev_diff: Optional[float] = None
    t = cfg.t_min_m
    while t <= cfg.t_max_m:
        diff = height_above_dem(t)
        if diff is not None:
            if prev_diff is not None and diff <= 0 < prev_diff:
                lo, hi = prev_t, t
                for _ in range(cfg.bisect_iters):
                    mid = (lo + hi) / 2.0
                    dm = height_above_dem(mid)
                    if dm is None:
                        break
                    if dm > 0:
                        lo = mid
                    else:
                        hi = mid
                return _make_hit(origin, direction, (lo + hi) / 2.0, frame)
            prev_t, prev_diff = t, diff
        t += cfg.step_m
    return None


def _make_hit(origin: np.ndarray, direction: np.ndarray, t: float, frame: EnuFrame) -> RayHit:
    p = origin + t * direction
    x, y, h = frame.enu_to_puwg92(p[0], p[1], p[2])
    lon, lat = frame.puwg92_to_lonlat(x, y)
    return RayHit(p, float(x), float(y), float(h), float(lon), float(lat), float(t))
