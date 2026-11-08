"""Diagnostyka geometrii: rozklad residuow wzgledem promieni i skan z_offset (multiview_check).

Odchylenie detekcji od srodka obiektu rozkladamy na skladowa wzdluz poziomego kierunku
promienia (kamera -> trafienie) i w poprzek. Blad wysokosci kamery dz przesuwa trafienie
wzdluz promienia o ~dz * tan(incidence), wiec przy zlym z_offset skladowa wzdluzna
koreluje z tan(incidence); w poprzek promienia zostaja bledy orientacji (EO).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .aggregate import AggregatedObject, aggregate
from .batch import BatchResult
from .config import AggregateConfig
from .pipeline import GeoDetection, PipelineContext
from .raycast import pixel_to_ray, raycast

log = logging.getLogger(__name__)


@dataclass
class RayResidual:
    detection: GeoDetection
    along: float        # [m] wzdluz poziomego kierunku promienia (+ = dalej od kamery)
    across: float       # [m] w poprzek (+ = w prawo patrzac od kamery)
    tan_incidence: float


@dataclass
class AlongFit:
    """along = slope * tan(incidence) + intercept."""
    slope: float
    intercept: float
    r: float
    n: int


def ray_residuals(batch: BatchResult, objects: list[AggregatedObject]) -> list[RayResidual]:
    """Residua detekcji przypisanych do `objects` (wg object_id) wzgledem srodkow obiektow."""
    cams = {r.image_name: r.camera for r in batch.images}
    centers = {o.object_id: o for o in objects}
    out = []
    for gd in batch.detections:
        o = centers.get(gd.object_id)
        if o is None:
            continue
        c = cams[gd.image_name]
        u = np.array([gd.hit.x_2180 - c.x_2180, gd.hit.y_2180 - c.y_2180])
        u /= np.linalg.norm(u)
        r = np.array([gd.hit.x_2180 - o.x_2180, gd.hit.y_2180 - o.y_2180])
        along = float(r @ u)
        across = float(u[1] * r[0] - u[0] * r[1])
        out.append(RayResidual(gd, along, across, math.tan(math.radians(gd.incidence_deg))))
    return out


def fit_along_vs_tan(res: list[RayResidual]) -> Optional[AlongFit]:
    if len(res) < 3:
        return None
    t = np.array([r.tan_incidence for r in res])
    a = np.array([r.along for r in res])
    (slope, icpt), *_ = np.linalg.lstsq(np.c_[t, np.ones_like(t)], a, rcond=None)
    r = float(np.corrcoef(t, a)[0, 1]) if t.std() > 0 and a.std() > 0 else float("nan")
    return AlongFit(float(slope), float(icpt), r, len(res))


def static_objects(batch: BatchResult, eps_m: float = 10.0, cls_name: str = "namiot",
                   min_views: int = 3) -> list[AggregatedObject]:
    """Ponowna agregacja (nadpisuje object_id w batchu!) i wybor obiektow statycznych."""
    objs = aggregate(batch.detections, AggregateConfig(eps_m=eps_m, min_samples=1))
    return [o for o in objs if o.cls_name == cls_name and o.n_views >= min_views]


# ───────────── multiview_check ─────────────

@dataclass
class ZScan:
    """RMS rozrzutu obiektow statycznych w funkcji z_offset (TYLKO diagnostyka)."""
    offsets: np.ndarray          # bezwzgledne z_offset [m]
    rms: np.ndarray              # zbiorczy RMS odleglosci od srodkow obiektow [m]
    n_hits: np.ndarray           # liczba trafien uzytych dla danego offsetu
    base_offset: float
    n_objects: int
    n_detections: int

    @property
    def best_offset(self) -> float:
        return float(self.offsets[int(np.nanargmin(self.rms))])

    @property
    def best_rms(self) -> float:
        return float(np.nanmin(self.rms))


def multiview_z_scan(batch: BatchResult, ctx: PipelineContext, span_m: float = 15.0,
                     step_m: float = 0.25, eps_m: float = 10.0, cls_name: str = "namiot",
                     min_views: int = 3) -> ZScan:
    """Przeszukuje z_offset w [base - span, base + span]; dla kazdego przelicza trafienia
    detekcji z obiektow statycznych (sklad klastrow staly — z bazowego offsetu) i liczy RMS.

    NIE stosowac do kalibracji wynikow: minimalizacja rozrzutu, ktorego potem uzywamy jako
    miary jakosci, bylaby kolowa. Sluzy jako niezalezna kontrola trybu dem_alignment.
    Kierunki promieni nie zaleza od z_offset, wiec probki DEM wzdluz promieni sa w wiekszosci
    wspolne (cache)."""
    surface = getattr(ctx, "ray_surface", None) or ctx.dem
    objs = static_objects(batch, eps_m, cls_name, min_views)
    members = [[gd for gd in batch.detections if gd.object_id == o.object_id] for o in objs]
    rays = []
    for group in members:
        g_rays = []
        for gd in group:
            io, eo = ctx.reconstruction.camera(gd.image_name)
            g_rays.append(pixel_to_ray(gd.uv, io, eo))
        rays.append(g_rays)

    base = ctx.z_calibration.offset
    offsets = base + np.arange(-span_m, span_m + step_m / 2, step_m)
    rms = np.full(len(offsets), np.nan)
    n_hits = np.zeros(len(offsets), dtype=int)
    tilt = ctx.frame.tilt
    for i, off in enumerate(offsets):
        frame = ctx.frame.with_z_offset(float(off), tilt)
        sq = []
        for g_rays in rays:
            pts = []
            for origin, direction in g_rays:
                hit = raycast(origin, direction, surface, frame, ctx.config.raycast)
                if hit is not None:
                    pts.append((hit.x_2180, hit.y_2180))
            if len(pts) >= 2:
                p = np.array(pts)
                sq.extend(((p - p.mean(0)) ** 2).sum(1))
                n_hits[i] += len(pts)
        if sq:
            rms[i] = math.sqrt(float(np.mean(sq)))
    if hasattr(ctx.dem, "save"):
        ctx.dem.save()
    scan = ZScan(offsets, rms, n_hits, base, len(objs), sum(len(g) for g in members))
    log.info("multiview_check: %d obiektow (%d detekcji), min RMS=%.2f m przy z_offset=%.2f "
             "(bazowy %.2f)", scan.n_objects, scan.n_detections, scan.best_rms,
             scan.best_offset, base)
    return scan
