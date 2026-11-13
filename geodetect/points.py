"""Punkt reprezentatywny detekcji: ktory piksel (lub piksele) rzutujemy na powierzchnie.

Metody (cfg.point.method, per_class):
  bbox_center           — srodek bboxa YOLO (punkt wyniesiony nad grunt o ~pol wysokosci
                          obiektu, co przesuwa trafienie wzdluz promienia o ~e*tan(incidence)),
  bbox_bottom_center    — srodek dolnej krawedzi bboxa (dla ludzi: stopy),
  mask_centroid         — centroid maski SAM,
  mask_ground_contact   — punkt styku maski z gruntem (geometria nizej),
  mask_footprint_median — promienie przez probke pikseli maski, mediana XY trafien.

Geometria mask_ground_contact
-----------------------------
Proste pionowe swiata rzutuja sie na obraz jako proste zbiegajace sie w punkcie nadiru
(punkt zbiegu kierunku D = (0, 0, -1) ENU): v ~ K * R_wc * D. Idac po obiekcie w dol
(w strone gruntu), jego obraz przesuwa sie w strone tego punktu. Dla punktu obrazu p
(wspolrzedne znormalizowane x, y; promien (x, y, 1)) i D_cam = R_wc * D, rzut punktu
X + s*D ma pochodna po s proporcjonalna do (D_x - x*D_z, D_y - y*D_z) — to jest lokalny
"kierunek w dol" w obrazie (liczony w znormalizowanych wspolrzednych i przenoszony
na piksele z dystorsja przez cv2.projectPoints). Styk z gruntem = skrajny punkt maski
wzdluz tego kierunku: bierzemy kwantyl 0.99 rzutu pikseli maski na kierunek (odpornosc
na pojedyncze piksele) i mediane polozenia poprzecznego w pasie `contact_band` przy tej
krawedzi. Gdy obiekt lezy blisko nadiru (kat miedzy promieniem a pionem <
min_offnadir_deg), pionowe sciany sa prawie niewidoczne, a kierunek "w dol" w obrazie
jest nieokreslony (w samym nadirze pochodna = 0) -> fallback do mask_centroid.
Uwaga: dla obiektu o niezerowym rzucie poziomym (namiot) skrajny punkt to blizsza
kamery krawedz podstawy, nie srodek podstawy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .config import MASK_METHODS, PointConfig
from .odm import CameraEO, CameraIO
from .segment import MaskQuality, ObjectMask


@dataclass
class PointChoice:
    method: str                          # metoda zadana w konfiguracji
    used: str                            # metoda faktycznie uzyta (po fallbacku)
    fallback: Optional[str]              # powod fallbacku (None = brak)
    uv: tuple[float, float]              # piksel glowny (dla footprint: centroid maski)
    footprint_uvs: Optional[np.ndarray] = None   # (n, 2) piksele dla mask_footprint_median
    mask_quality: Optional[MaskQuality] = None


def bbox_center(bbox) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def bbox_bottom_center(bbox) -> tuple[float, float]:
    x1, _, x2, y2 = bbox
    return (x1 + x2) / 2.0, float(y2)


def mask_centroid(mask: ObjectMask) -> tuple[float, float]:
    p = mask.pixels()
    return float(p[:, 0].mean()), float(p[:, 1].mean())


def down_direction(uv, io: CameraIO, eo: CameraEO, eps: float = 1e-3
                   ) -> tuple[np.ndarray, float]:
    """(kierunek "w dol" w pikselach (jednostkowy), kat promienia od pionu [deg]) w punkcie uv.
    Kierunek = obraz ruchu po pionie swiata w strone gruntu (patrz docstring modulu)."""
    pt = np.array([[[uv[0], uv[1]]]], dtype=np.float64)
    x, y = cv2.undistortPoints(pt, io.K, io.dist)[0, 0]
    D = eo.R_wc @ np.array([0.0, 0.0, -1.0])
    ray = np.array([x, y, 1.0])
    off_nadir = math.degrees(math.acos(np.clip(ray @ D / np.linalg.norm(ray), -1.0, 1.0)))
    dn = np.array([D[0] - x * D[2], D[1] - y * D[2]])
    norm = np.linalg.norm(dn)
    if norm < 1e-12:
        return np.zeros(2), off_nadir
    dn = dn / norm
    obj = np.array([[x, y, 1.0], [x + eps * dn[0], y + eps * dn[1], 1.0]])
    px, _ = cv2.projectPoints(obj, np.zeros(3), np.zeros(3), io.K, io.dist)
    d = px[1, 0] - px[0, 0]
    n = np.linalg.norm(d)
    return (d / n if n > 0 else np.zeros(2)), off_nadir


def nadir_point(io: CameraIO, eo: CameraEO) -> Optional[tuple[float, float]]:
    """Punkt zbiegu pionow w dol (piksele, bez dystorsji); None gdy w nieskonczonosci
    lub za kamera (kamera patrzy powyzej horyzontu)."""
    v = io.K @ (eo.R_wc @ np.array([0.0, 0.0, -1.0]))
    if v[2] <= 1e-9:
        return None
    return float(v[0] / v[2]), float(v[1] / v[2])


def ground_contact(mask: ObjectMask, io: CameraIO, eo: CameraEO, min_offnadir_deg: float,
                   band: float) -> Optional[tuple[float, float]]:
    """Punkt styku maski z gruntem albo None (obiekt blisko nadiru -> kierunek nieokreslony)."""
    px = mask.pixels()
    c = px.mean(axis=0)
    d, off_nadir = down_direction(c, io, eo)
    if off_nadir < min_offnadir_deg or not np.any(d):
        return None
    perp = np.array([-d[1], d[0]])
    rel = px - c
    along, lateral = rel @ d, rel @ perp
    a_edge = float(np.quantile(along, 0.99))
    width = max(float(along.max() - along.min()), 1.0)
    in_band = along >= a_edge - max(band * width, 1.0)
    p = c + d * a_edge + perp * float(np.median(lateral[in_band]))
    return float(p[0]), float(p[1])


def footprint_pixels(mask: ObjectMask, n: int, seed: int) -> np.ndarray:
    px = mask.pixels()
    if len(px) <= n:
        return px
    idx = np.random.default_rng(seed).choice(len(px), size=n, replace=False)
    return px[np.sort(idx)]


def choose_point(method: str, bbox, io: CameraIO, eo: CameraEO, cfg: PointConfig,
                 mask: Optional[ObjectMask] = None,
                 quality: Optional[MaskQuality] = None) -> PointChoice:
    """Piksel(e) dla metody; fallbacki: brak/zla maska -> bbox_center,
    ground_contact blisko nadiru -> mask_centroid."""
    if method == "bbox_center":
        return PointChoice(method, method, None, bbox_center(bbox))
    if method == "bbox_bottom_center":
        return PointChoice(method, method, None, bbox_bottom_center(bbox))
    if method not in MASK_METHODS:
        raise ValueError(f"Nieznana metoda punktu {method!r}")
    if mask is None:
        return PointChoice(method, "bbox_center", "no_mask", bbox_center(bbox), None, quality)
    if quality is not None and not quality.ok:
        return PointChoice(method, "bbox_center", f"mask_rejected:{quality.reason}",
                           bbox_center(bbox), None, quality)
    centroid = mask_centroid(mask)
    if method == "mask_centroid":
        return PointChoice(method, method, None, centroid, None, quality)
    if method == "mask_ground_contact":
        p = ground_contact(mask, io, eo, cfg.min_offnadir_deg, cfg.contact_band)
        if p is None:
            return PointChoice(method, "mask_centroid", "near_nadir", centroid, None, quality)
        return PointChoice(method, method, None, p, None, quality)
    return PointChoice(method, method, None, centroid,
                       footprint_pixels(mask, cfg.footprint_samples, cfg.seed), quality)
