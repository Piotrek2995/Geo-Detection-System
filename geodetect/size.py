"""Fizyczny rozmiar obiektu z bboxa detekcji i filtr rozmiaru per klasa.

Przyblizenie (obiekt plaski, poziomy, na wysokosci trafienia):
  Najprosciej rozmiar = bbox_px * odleglosc / f_px (f_px — ogniskowa w pikselach) — to skala
  w plaszczyznie prostopadlej do promienia. Obiekt lezacy na ziemi ogladany ukosnie (kat
  padania theta od pionu) jest wzdluz kierunku pochylenia promienia skrocony o cos(theta),
  wiec jego wymiar poziomy w tym kierunku to bbox_px * R / f / cos(theta), a w poprzek
  bez zmian. Zamiast rozkladac bbox na te kierunki, promienie przez srodki czterech krawedzi
  bboxa (lewa/prawa, gorna/dolna) przecinane sa z pozioma plaszczyzna na wysokosci trafienia:
  odleglosci lewa-prawa i gora-dol to wymiary poziome obiektu z uwzglednieniem kata padania
  (i dystorsji obiektywu). size_m = wiekszy z nich.
  Ograniczenia: dla obiektow wysokich (stojacy czlowiek) widzianych ukosnie czesc pionowa
  rzutuje sie na plaszczyzne i zawyza rozmiar o ~H*tan(theta); przy duzym theta
  (cos < min_cos_incidence) wynik jest niestabilny i zastepowany wartoscia prosta
  (bbox * R / f). Obiekt uciety krawedzia kadru ma rozmiar zanizony.
Rozmiar obiektu (wielowidokowego) = mediana rozmiarow jego detekcji. Poza zakresem z configu
-> flaga size_out_of_range (obiekt nie jest usuwany).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .raycast import pixel_to_ray

log = logging.getLogger(__name__)

FLAG_SIZE = "size_out_of_range"


@dataclass
class SizeEstimate:
    width_m: float             # wymiar poziomy miedzy lewa i prawa krawedzia bboxa
    length_m: float            # miedzy gorna i dolna krawedzia
    size_m: float              # max(width_m, length_m) (albo simple_m przy duzym kacie)
    simple_m: float            # max(bbox_px) * R / f — bez kata padania
    method: str                # plane | simple

    def to_dict(self) -> dict:
        return {k: round(v, 3) if isinstance(v, float) else v for k, v in self.__dict__.items()}


def _plane_point(uv, io, eo, z0: float) -> Optional[np.ndarray]:
    o, d = pixel_to_ray(uv, io, eo)
    if d[2] > -1e-6:
        return None
    return o + (z0 - o[2]) / d[2] * d


def detection_size(bbox, io, eo, hit_enu, range_m: float, incidence_deg: float,
                   min_cos_incidence: float = 0.3) -> SizeEstimate:
    """Rozmiar detekcji (bbox [px] w calym obrazie, hit_enu — punkt trafienia w ENU)."""
    x1, y1, x2, y2 = bbox
    f_px = float(io.K[0, 0])
    simple = max(x2 - x1, y2 - y1) * range_m / f_px
    z0 = float(hit_enu[2])
    xc, yc = (x1 + x2) / 2, (y1 + y2) / 2
    pts = [_plane_point(uv, io, eo, z0) for uv in ((x1, yc), (x2, yc), (xc, y1), (xc, y2))]
    if any(p is None for p in pts) or math.cos(math.radians(incidence_deg)) < min_cos_incidence:
        return SizeEstimate(float("nan"), float("nan"), simple, simple, "simple")
    w = float(np.hypot(*(pts[1] - pts[0])[:2]))
    ln = float(np.hypot(*(pts[3] - pts[2])[:2]))
    return SizeEstimate(w, ln, max(w, ln), simple, "plane")


def size_flags(cls_name: str, size_m: float, ranges: dict) -> list[str]:
    """[FLAG_SIZE], gdy rozmiar poza zakresem [min, max] klasy z configu (brak zakresu: [])."""
    rng = ranges.get(cls_name)
    if rng is None or not np.isfinite(size_m):
        return []
    lo, hi = rng
    if (lo is not None and size_m < lo) or (hi is not None and size_m > hi):
        return [FLAG_SIZE]
    return []
