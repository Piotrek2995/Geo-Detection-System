"""Dostepnosc powierzchni ray castingu: pokrycie obszaru lotu lokalnymi arkuszami NMT/NMPT
i dostepnosc uslugi REST GUGiK (wynik jako dataclass — CLI scripts/check_surface_coverage.py, GUI)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .config import PipelineConfig

log = logging.getLogger(__name__)


@dataclass
class FolderCoverage:
    label: str                         # NMT | NMPT | DSM ODM
    folder: Optional[Path]
    ok: bool                           # obszar w calosci w zasiegu i bez dziur
    files: list[str] = field(default_factory=list)
    resolution_m: Optional[float] = None
    mosaic_bounds: Optional[tuple] = None
    missing_m: dict = field(default_factory=dict)    # W/S/E/N: o ile obszar wychodzi poza arkusze
    nodata_frac: Optional[float] = None
    message: str = ""


@dataclass
class RestStatus:
    ok: bool
    message: str
    latency_s: Optional[float] = None


def flight_bounds(cfg: PipelineConfig, buffer_m: float = 100.0) -> tuple[float, float, float, float]:
    """bbox kamer rekonstrukcji ODM (EPSG:2180) + zasieg DSM ODM + bufor."""
    from .geo import EnuFrame
    from .odm import OdmReconstruction
    rec = OdmReconstruction(cfg.reconstruction_path, cfg.reference_lla_path)
    frame = EnuFrame(rec.ref)
    xy = np.array([frame.enu_to_puwg92(*rec.camera(n)[1].C_enu)[:2] for n in rec.image_names], dtype=float)
    xmin, ymin = xy.min(axis=0)
    xmax, ymax = xy.max(axis=0)
    dsm = cfg.odm_dir / "odm_dem" / "dsm.tif"
    if dsm.exists():
        import rasterio
        from pyproj import Transformer
        with rasterio.open(dsm) as ds:
            b = Transformer.from_crs(ds.crs.to_wkt(), "EPSG:2180", always_xy=True).transform_bounds(*ds.bounds)
        xmin, ymin = min(xmin, b[0]), min(ymin, b[1])
        xmax, ymax = max(xmax, b[2]), max(ymax, b[3])
    return xmin - buffer_m, ymin - buffer_m, xmax + buffer_m, ymax + buffer_m


def check_folder(label: str, folder: Optional[Path], bounds_2180, default_crs: str = "EPSG:2180") -> FolderCoverage:
    from .surface import _bounds_in, _read_crs, load_mosaic, raster_files
    if folder is None:
        return FolderCoverage(label, None, False, message="folder nie jest ustawiony w konfiguracji")
    try:
        files = raster_files(folder)
    except FileNotFoundError as exc:
        return FolderCoverage(label, Path(folder), False, message=f"brak arkuszy: {exc}")
    crs = _read_crs(files[0], default_crs)
    b = _bounds_in(crs, bounds_2180)
    full = load_mosaic(files, default_crs)
    fb = full.bounds
    missing = {"W": max(0.0, fb[0] - b[0]), "S": max(0.0, fb[1] - b[1]),
               "E": max(0.0, b[2] - fb[2]), "N": max(0.0, b[3] - fb[3])}
    inside = not any(v > 0 for v in missing.values())
    res = FolderCoverage(label, Path(folder), False, [f.name for f in files], float(full.res_x), tuple(fb),
                         {k: v for k, v in missing.items() if v > 0})
    try:
        g = load_mosaic(files, default_crs, b)
    except ValueError as exc:
        res.message = f"arkusze nie obejmują obszaru lotu ({exc})"
        return res
    holes = ~np.isfinite(g.z)
    res.nodata_frac = float(holes.mean()) if g.z.size else 1.0
    res.ok = inside and not holes.any()
    if res.ok:
        res.message = f"pełne pokrycie ({len(files)} arkusz(e), {res.resolution_m:g} m)"
    elif not inside:
        res.message = ("obszar wychodzi poza arkusze o " +
                       ", ".join(f"{k} {v:.0f} m" for k, v in res.missing_m.items()))
    else:
        res.message = f"braki danych w {100 * res.nodata_frac:.1f}% obszaru"
    return res


def surface_coverage(cfg: PipelineConfig, buffer_m: float = 100.0) -> dict[str, FolderCoverage]:
    """Pokrycie lokalnych arkuszy NMT i NMPT dla obszaru lotu (wymaga wynikow ODM)."""
    b = flight_bounds(cfg, buffer_m)
    return {"nmt_local": check_folder("NMT", cfg.surface.nmt_dir, b, cfg.surface.default_crs),
            "nmpt_local": check_folder("NMPT", cfg.surface.nmpt_dir, b, cfg.surface.default_crs)}


def gugik_rest_status(url: str, x: float = 660300.0, y: float = 664850.0, timeout_s: float = 6.0,
                      get=None) -> RestStatus:
    """Jedno zapytanie GetHByXY — czy usluga NMT GUGiK odpowiada."""
    import time
    if get is None:
        import requests
        get = requests.get
    t0 = time.perf_counter()
    try:
        r = get(url, params={"request": "GetHByXY", "x": f"{y:.2f}", "y": f"{x:.2f}"}, timeout=timeout_s)
        dt = time.perf_counter() - t0
        text = (getattr(r, "text", "") or "").strip()
        float(text.split()[0])
        return RestStatus(True, f"usługa GUGiK odpowiada ({dt:.1f} s)", dt)
    except Exception as exc:                 # brak sieci, timeout, odpowiedz nie-liczbowa
        return RestStatus(False, f"usługa GUGiK niedostępna ({type(exc).__name__})")


def odm_dsm_status(cfg: PipelineConfig) -> FolderCoverage:
    p = cfg.surface.odm_dsm or cfg.odm_dir / "odm_dem" / "dsm.tif"
    ok = Path(p).exists()
    return FolderCoverage("DSM ODM", Path(p), ok, [Path(p).name] if ok else [],
                          message="DSM z rekonstrukcji ODM dostępny" if ok else
                          "brak dsm.tif, uruchom ODM z opcją --dsm")
