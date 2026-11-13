"""Maski obiektow z SAM (ultralytics), prompt = bbox YOLO, cache na dysku + kontrola jakosci.

Model SAM ladowany leniwie przy pierwszym braku w cache — gdy metoda punktu nie wymaga
masek, SAM nie jest w ogole importowany. Maska zapisywana jest jako wycinek wokol bboxa
(npz, bool), klucz: zdjecie + bbox (zaokraglony do 0.1 px) + nazwa modelu.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .config import PointConfig

log = logging.getLogger(__name__)


@dataclass
class ObjectMask:
    """Maska obiektu jako wycinek obrazu: mask[r, c] odpowiada pikselowi (x0 + c, y0 + r)."""
    mask: np.ndarray            # bool (h, w)
    x0: int
    y0: int

    def pixels(self) -> np.ndarray:
        """(n, 2) wspolrzedne (u, v) srodkow pikseli maski w calym obrazie."""
        r, c = np.nonzero(self.mask)
        return np.c_[c + self.x0 + 0.5, r + self.y0 + 0.5].astype(np.float64)

    @property
    def area(self) -> int:
        return int(self.mask.sum())


@dataclass
class MaskQuality:
    ok: bool
    reason: Optional[str]       # None gdy ok; too_small / fills_bbox / fragmented / empty
    area_ratio: float           # pole najwiekszej skladowej / pole bboxa
    n_components: int
    largest_frac: float         # najwieksza skladowa / cala maska
    touches_all_edges: bool

    def as_dict(self) -> dict:
        return {"ok": self.ok, "reason": self.reason, "area_ratio": round(self.area_ratio, 3),
                "n_components": self.n_components, "largest_frac": round(self.largest_frac, 3),
                "touches_all_edges": self.touches_all_edges}


def clean_mask(m: ObjectMask, bbox, cfg: PointConfig) -> tuple[ObjectMask, MaskQuality]:
    """Zostawia najwieksza spojna skladowa i ocenia maske.

    Uwaga: bbox YOLO jest ciasny, wiec poprawna maska zwykle dotyka wszystkich 4 krawedzi —
    samo dotykanie nie jest bledem. Podejrzana jest maska, ktora dotyka wszystkich krawedzi
    I wypelnia prawie caly bbox (SAM zwrocil prostokat / tlo zamiast obiektu)."""
    x1, y1, x2, y2 = bbox
    bbox_area = max((x2 - x1) * (y2 - y1), 1.0)
    total = m.area
    if total == 0:
        return m, MaskQuality(False, "empty", 0.0, 0, 0.0, False)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m.mask.astype(np.uint8), connectivity=8)
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    largest = labels == big
    m2 = ObjectMask(largest, m.x0, m.y0)
    area = int(largest.sum())
    px = m2.pixels()
    tol = cfg.edge_tol_px + 0.5
    touches = bool(px[:, 0].min() <= x1 + tol and px[:, 0].max() >= x2 - tol
                   and px[:, 1].min() <= y1 + tol and px[:, 1].max() >= y2 - tol)
    ratio, frac = area / bbox_area, area / total
    reason = None
    if ratio < cfg.min_area_ratio:
        reason = "too_small"
    elif touches and ratio > cfg.max_area_ratio:
        reason = "fills_bbox"
    elif frac < cfg.min_largest_frac:
        reason = "fragmented"
    return m2, MaskQuality(reason is None, reason, ratio, n - 1, frac, touches)


class Segmenter:
    """SAM z promptem bbox. `masks(image, bboxes)` -> lista ObjectMask (cache-first)."""

    def __init__(self, cfg: PointConfig):
        self.cfg = cfg
        self.model_name = Path(cfg.sam_model).name
        self.cache_dir = Path(cfg.mask_cache_dir)
        self._model = None
        self.infer_s = 0.0          # laczny czas inferencji SAM (bez cache)
        self.n_infer = 0            # liczba wywolan modelu (zdjec)
        self.n_cached = 0

    def _key(self, image_path: Path, bbox) -> Path:
        s = f"{Path(image_path).name}|{','.join(f'{v:.1f}' for v in bbox)}|{self.model_name}"
        return self.cache_dir / f"{hashlib.sha1(s.encode()).hexdigest()[:20]}.npz"

    def _load_model(self):
        if self._model is None:
            from ultralytics import SAM  # lazy — ciezki import
            log.info("SAM: ladowanie %s", self.cfg.sam_model)
            self._model = SAM(str(self.cfg.sam_model))
        return self._model

    def masks(self, image_path: Path, bboxes: list) -> list[Optional[ObjectMask]]:
        out: list[Optional[ObjectMask]] = [None] * len(bboxes)
        todo = []
        for i, bb in enumerate(bboxes):
            p = self._key(image_path, bb)
            if p.exists():
                with np.load(p) as z:
                    out[i] = ObjectMask(z["mask"].astype(bool), int(z["x0"]), int(z["y0"]))
                self.n_cached += 1
            else:
                todo.append(i)
        if not todo:
            return out
        model = self._load_model()
        t0 = time.perf_counter()
        res = model(str(image_path), bboxes=[list(bboxes[i]) for i in todo],
                    device=self.cfg.sam_device, verbose=False)[0]
        self.infer_s += time.perf_counter() - t0
        self.n_infer += 1
        data = res.masks.data.cpu().numpy().astype(bool) if res.masks is not None else []
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        for j, i in enumerate(todo):
            if j >= len(data):
                continue
            full = data[j]
            om = crop_mask(full, bboxes[i])
            np.savez_compressed(self._key(image_path, bboxes[i]), mask=om.mask, x0=om.x0, y0=om.y0)
            out[i] = om
        return out


def crop_mask(full: np.ndarray, bbox, margin: float = 0.25) -> ObjectMask:
    """Wycinek pelnej maski wokol bboxa (+ margines), zeby cache byl maly."""
    H, W = full.shape
    x1, y1, x2, y2 = bbox
    mx, my = (x2 - x1) * margin, (y2 - y1) * margin
    c0, r0 = max(0, int(np.floor(x1 - mx))), max(0, int(np.floor(y1 - my)))
    c1, r1 = min(W, int(np.ceil(x2 + mx))), min(H, int(np.ceil(y2 + my)))
    return ObjectMask(full[r0:r1, c0:c1].copy(), c0, r0)


def mask_from_polygons(polygons: list, bbox, image_size: tuple[int, int],
                       margin: float = 0.25) -> ObjectMask:
    """Maska z konturow modelu segmentacyjnego (wspolrzedne zdjecia) jako wycinek wokol
    bboxa — ten sam format co maski SAM, wiec metody mask_* dzialaja bez SAM.
    image_size = (szerokosc, wysokosc)."""
    W, H = image_size
    x1, y1, x2, y2 = bbox
    mx, my = (x2 - x1) * margin, (y2 - y1) * margin
    c0, r0 = max(0, int(np.floor(x1 - mx))), max(0, int(np.floor(y1 - my)))
    c1, r1 = min(W, int(np.ceil(x2 + mx))), min(H, int(np.ceil(y2 + my)))
    m = np.zeros((r1 - r0, c1 - c0), np.uint8)
    for p in polygons:
        pts = np.round(np.asarray(p, dtype=np.float64) - [c0, r0]).astype(np.int32)
        if len(pts) >= 3:
            cv2.fillPoly(m, [pts.reshape(-1, 1, 2)], 1)
    return ObjectMask(m.astype(bool), c0, r0)
