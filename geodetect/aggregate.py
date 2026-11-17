"""Agregacja wielowidokowa: ten sam obiekt widziany na kilku zdjeciach -> jeden punkt.

DBSCAN w EPSG:2180 (metry), osobno dla kazdej klasy. Rozrzut pozycji z roznych
ujec (RMS i max odleglosc od srodka) jest miara spojnosci georeferencji.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
from pyproj import Transformer

from typing import Optional

from .config import AggregateConfig, SizeConfig
from .pipeline import GeoDetection

log = logging.getLogger(__name__)

WEIGHT_MODES = ("equal", "conf", "cos_incidence")


@dataclass
class AggregatedObject:
    object_id: int
    cls_name: str
    x_2180: float
    y_2180: float
    h: float                   # [m npm], srednia wazona wysokosci trafien
    lon: float
    lat: float
    n_views: int               # liczba detekcji w klastrze
    images: list[str]          # unikalne zdjecia
    rms_m: float               # RMS odleglosci poziomej detekcji od srodka
    max_m: float               # maks. odleglosc pozioma od srodka
    mean_conf: float
    mean_incidence_deg: float
    mean_range_m: float
    duplicate_images: list[str] = field(default_factory=list)  # zdjecia >1 raz w klastrze
    status: str = "confirmed"  # confirmed (>= aggregate.min_views widokow) | single_view
    size_m: float = float("nan")       # mediana rozmiaru fizycznego detekcji (geodetect/size.py)
    size_n: int = 0                    # liczba detekcji uzytych w medianie
    flags: list[str] = field(default_factory=list)   # np. size_out_of_range

    @property
    def confirmed(self) -> bool:
        return self.status == "confirmed"


def dbscan(points: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    """Klasyczny DBSCAN (metryka euklidesowa). Zwraca etykiety, -1 = szum.
    O(n^2) pamieci — wystarczajace dla setek-tysiecy detekcji z jednego lotu."""
    n = len(points)
    labels = np.full(n, -1, dtype=int)
    if n == 0:
        return labels
    diff = points[:, None, :] - points[None, :, :]
    neighbors = np.sqrt((diff ** 2).sum(-1)) <= eps
    core = neighbors.sum(1) >= min_samples
    cid = 0
    for i in range(n):
        if labels[i] != -1 or not core[i]:
            continue
        labels[i] = cid
        stack = [i]
        while stack:
            j = stack.pop()
            if not core[j]:
                continue
            for k in np.flatnonzero(neighbors[j]):
                if labels[k] == -1:
                    labels[k] = cid
                    stack.append(k)
        cid += 1
    return labels


def detection_weights(dets: list[GeoDetection], mode: str) -> np.ndarray:
    if mode == "equal":
        return np.ones(len(dets))
    if mode == "conf":
        return np.array([d.detection.conf for d in dets])
    if mode == "cos_incidence":
        return np.cos(np.radians([d.incidence_deg for d in dets]))
    raise ValueError(f"Nieznany tryb wag: {mode!r} (dozwolone: {WEIGHT_MODES})")


def aggregate(detections: list[GeoDetection], cfg: AggregateConfig,
              size_cfg: Optional[SizeConfig] = None) -> list[AggregatedObject]:
    """Klastruje przyjete detekcje i ustawia im `object_id`. Detekcje odrzucone oraz
    szum DBSCAN (gdy min_samples > 1) zostaja z object_id = None."""
    to_lonlat = Transformer.from_crs("EPSG:2180", "EPSG:4326", always_xy=True)
    accepted = [d for d in detections if d.accepted]
    for d in detections:
        d.object_id = None

    objects: list[AggregatedObject] = []
    for cls_name in sorted({d.detection.cls_name for d in accepted}):
        dets = [d for d in accepted if d.detection.cls_name == cls_name]
        xy = np.array([[d.hit.x_2180, d.hit.y_2180] for d in dets])
        labels = dbscan(xy, cfg.eps_m, cfg.min_samples)
        for lab in range(labels.max() + 1):
            members = [d for d, l in zip(dets, labels) if l == lab]
            obj = _make_object(len(objects), cls_name, members, cfg.weights, to_lonlat)
            obj.status = "confirmed" if obj.n_views >= cfg.min_views else "single_view"
            if size_cfg is not None and size_cfg.enabled:
                apply_size(obj, members, size_cfg)
            for d in members:
                d.object_id = obj.object_id
            objects.append(obj)
        if (labels == -1).any():
            log.info("%s: %d detekcji jako szum DBSCAN (bez obiektu)",
                     cls_name, int((labels == -1).sum()))
    for o in objects:
        if o.duplicate_images:
            log.warning("Obiekt %d (%s): zdjecia %s wystepuja w klastrze wiecej niz raz — "
                        "mozliwe zle scalenie blisko polozonych obiektow",
                        o.object_id, o.cls_name, o.duplicate_images)
    return objects


def _make_object(object_id: int, cls_name: str, members: list[GeoDetection],
                 weight_mode: str, to_lonlat: Transformer) -> AggregatedObject:
    w = detection_weights(members, weight_mode)
    xyh = np.array([[d.hit.x_2180, d.hit.y_2180, d.hit.h] for d in members])
    cx, cy, ch = (w[:, None] * xyh).sum(0) / w.sum()
    dist = np.hypot(xyh[:, 0] - cx, xyh[:, 1] - cy)
    lon, lat = to_lonlat.transform(cx, cy)
    counts = Counter(d.image_name for d in members)
    return AggregatedObject(
        object_id=object_id, cls_name=cls_name,
        x_2180=float(cx), y_2180=float(cy), h=float(ch), lon=float(lon), lat=float(lat),
        n_views=len(members), images=sorted(counts),
        rms_m=float(np.sqrt(np.mean(dist ** 2))), max_m=float(dist.max()),
        mean_conf=float(np.mean([d.detection.conf for d in members])),
        mean_incidence_deg=float(np.mean([d.incidence_deg for d in members])),
        mean_range_m=float(np.mean([d.hit.range_m for d in members])),
        duplicate_images=sorted(img for img, c in counts.items() if c > 1),
    )


def apply_size(obj: AggregatedObject, members: list[GeoDetection], cfg: SizeConfig) -> None:
    """Rozmiar obiektu = mediana rozmiarow detekcji (bez uciętych krawedzia kadru, jesli sa inne)
    i flaga size_out_of_range wg cfg.ranges_m."""
    from .size import size_flags
    sized = [d for d in members if d.size is not None and np.isfinite(d.size.size_m)]
    full = [d for d in sized if not d.touches_border]
    use = full if (cfg.exclude_border and full) else sized
    if not use:
        return
    obj.size_m = float(np.median([d.size.size_m for d in use]))
    obj.size_n = len(use)
    for f in size_flags(obj.cls_name, obj.size_m, cfg.ranges_m):
        if f not in obj.flags:
            obj.flags.append(f)
            rng = cfg.ranges_m.get(obj.cls_name)
            log.info("Obiekt %d (%s): rozmiar %.2f m poza zakresem %s -> %s",
                     obj.object_id, obj.cls_name, obj.size_m, rng, f)
