"""Przesuniecie dachu na ortofotomapie (building lean) zalezne od wysokosci.

Ortofotomapa GUGiK jest rektyfikowana na NMT, wiec punkt na wysokosci H nad gruntem jest na niej
przesuniety o d = v * H (v — wektor przesuniecia na metr wysokosci; lokalnie staly, kierunek od
rzutu srodka rzutow zdjecia lotniczego). Dotychczasowa korekta odejmowala jeden wektor (lean
zmierzony dopasowaniem obrysu EGiB do maski dachu) od kazdego punktu niezaleznie od jego wysokosci.

Tu: v = lean / H_kraw, gdzie H_kraw to wysokosc krawedzi dachu (mediana NMPT - NMT w pasie
wewnatrz obrysu EGiB — dopasowanie obrysu wyznaczaja glownie krawedzie, czyli okapy i szczyty),
a kazdy punkt koryguje sie jego wlasna wysokoscia: p_true = p_orto - v * H(p_true) (iteracja,
bo wysokosc trzeba wziac w polozeniu prawdziwym).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

log = logging.getLogger(__name__)

HeightFn = Callable[[float, float], float]     # (x, y) EPSG:2180 -> wysokosc nad gruntem [m]


@dataclass
class LeanModel:
    vx: float                  # przesuniecie [m] na 1 m wysokosci, E
    vy: float                  # N
    h_edges_m: float           # wysokosc krawedzi, z ktorej wyznaczono v
    lean_dx: float             # lean zmierzony (dla H_kraw)
    lean_dy: float
    n_edge_px: int = 0

    def shift(self, h: float) -> tuple[float, float]:
        return self.vx * h, self.vy * h

    def correct(self, x: float, y: float, height: HeightFn, extra_h: float = 0.0,
                iters: int = 8) -> tuple[float, float, float]:
        """Polozenie prawdziwe punktu odczytanego z ortofotomapy w (x, y): (x_t, y_t, H)."""
        xt, yt, h = x, y, 0.0
        for _ in range(iters):
            h = float(height(xt, yt))
            if not np.isfinite(h):
                h = 0.0
            h = max(h, 0.0) + extra_h
            xn, yn = x - self.vx * h, y - self.vy * h
            if abs(xn - xt) < 1e-3 and abs(yn - yt) < 1e-3:
                xt, yt = xn, yn
                break
            xt, yt = xn, yn
        return xt, yt, h


def edge_band(footprint: np.ndarray, res_m: float, band_m: tuple[float, float] = (0.5, 1.5)) -> np.ndarray:
    """Maska pasa wewnatrz obrysu w odleglosci band_m od jego krawedzi (NMPT 1 m na samej krawedzi
    miesza dach z gruntem, wiec pas zaczyna sie 0.5 m do wewnatrz)."""
    import cv2
    dist = cv2.distanceTransform(footprint.astype(np.uint8), cv2.DIST_L2, 5) * res_m
    return footprint & (dist >= band_m[0]) & (dist <= band_m[1])


def edge_height(footprint: np.ndarray, x0: float, y1: float, res_m: float, height_grid: Callable,
                band_m: tuple[float, float] = (0.5, 1.5)) -> tuple[float, int]:
    """Mediana wysokosci (NMPT - NMT) w pasie przy krawedzi obrysu. footprint: maska w siatce
    (x0, y1 — lewy gorny rog, res_m); height_grid(xs, ys) -> wysokosci (wektorowo)."""
    rows, cols = np.nonzero(edge_band(footprint, res_m, band_m))
    if len(rows) == 0:
        return float("nan"), 0
    xs = x0 + (cols + 0.5) * res_m
    ys = y1 - (rows + 0.5) * res_m
    h = np.asarray(height_grid(xs, ys), float)
    h = h[np.isfinite(h)]
    return (float(np.median(h)) if len(h) else float("nan")), int(len(h))


def lean_model(lean_dx: float, lean_dy: float, h_edges: float, n_px: int = 0) -> Optional[LeanModel]:
    if not np.isfinite(h_edges) or h_edges < 0.5:
        log.warning("Lean zalezny od wysokosci: wysokosc krawedzi %.2f m — za mala, pomijam", h_edges)
        return None
    return LeanModel(lean_dx / h_edges, lean_dy / h_edges, h_edges, lean_dx, lean_dy, n_px)
