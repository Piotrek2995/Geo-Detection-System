"""Powierzchnie do ray castingu (cfg.surface.source) za wspolnym interfejsem Surface.

Kazda powierzchnia deklaruje:
  crs         — uklad, w ktorym przyjmuje wspolrzedne (x=easting, y=northing),
  vertical    — uklad wysokosci wartosci: "npm" (PL-EVRF2007-NH, jak NMT GUGiK)
                albo "odm_local" (wysokosci rekonstrukcji ODM, bez z_offset),
  resolution_m— rozdzielczosc rastra (None dla REST) — z niej wynika krok marszu.

Zrodla:
  nmt_gugik_rest — NMT 1 m z REST GUGiK (GetHByPointList, cache na dysku),
  nmt_local      — lokalne arkusze NMT z geoportalu (folder; GeoTIFF/ASC/XYZ, mozaika),
  nmpt_local     — lokalne arkusze NMPT z geoportalu (jw.; REST GUGiK nie ma NMPT),
  odm_dsm        — odm_project/odm_dem/dsm.tif (EPSG:32634, vertical=odm_local).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol

import numpy as np

from .config import SURFACE_SOURCES, PipelineConfig, SurfaceConfig
from .dem import DemSampler, GugikDem

log = logging.getLogger(__name__)

VERTICAL_NPM = "npm"
VERTICAL_ODM = "odm_local"
RASTER_SUFFIXES = {".tif", ".tiff", ".asc", ".xyz", ".txt", ".vrt"}


class Surface(Protocol):
    name: str
    crs: str
    vertical: str
    resolution_m: Optional[float]

    def sample_many(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Wysokosci w punktach (x, y) w `crs`; NaN = brak danych."""
        ...


# ───────────── REST / dowolny punktowy DemSampler ─────────────

class PointSurface:
    """Adapter punktowego DemSampler (EPSG:2180, npm) na Surface."""

    def __init__(self, dem: DemSampler, name: str = "dem", resolution_m: Optional[float] = None):
        self.dem = dem
        self.name = name
        self.crs = "EPSG:2180"
        self.vertical = VERTICAL_NPM
        self.resolution_m = resolution_m

    def sample_many(self, x, y) -> np.ndarray:
        x, y = np.atleast_1d(x), np.atleast_1d(y)
        if hasattr(self.dem, "sample_many"):
            return self.dem.sample_many(x, y)
        out = np.full(len(x), np.nan)
        for i, (xi, yi) in enumerate(zip(x, y)):
            z = self.dem.sample(float(xi), float(yi))
            if z is not None:
                out[i] = z
        return out


def as_surface(obj) -> Surface:
    """Surface bez zmian; punktowy DemSampler (np. GugikDem, FlatDem w testach) -> PointSurface."""
    if hasattr(obj, "sample_many") and hasattr(obj, "vertical"):
        return obj
    return PointSurface(obj, name=getattr(obj, "name", type(obj).__name__))


# ───────────── rastry lokalne ─────────────

@dataclass
class Grid:
    """Raster w pamieci: z[row, col], piksel (0,0) ma lewy-gorny rog w (x0, y0)."""
    z: np.ndarray            # float64, NaN = nodata
    x0: float
    y0: float
    res_x: float
    res_y: float             # dodatnie (wiersze rosna na poludnie)
    crs: str

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        h, w = self.z.shape
        return self.x0, self.y0 - h * self.res_y, self.x0 + w * self.res_x, self.y0


def sample_grid(g: Grid, x: np.ndarray, y: np.ndarray, interpolation: str = "bilinear") -> np.ndarray:
    """Probkowanie wektorowe. bilinear: gdy ktorykolwiek z 4 sasiadow to nodata, uzywa
    najblizszego piksela (jesli jest poprawny) — dzieki temu brzegi danych nie znikaja."""
    x, y = np.atleast_1d(np.asarray(x, float)), np.atleast_1d(np.asarray(y, float))
    h, w = g.z.shape
    fc = (x - g.x0) / g.res_x            # wspolrzedne ciagle w pikselach (0 = krawedz)
    fr = (g.y0 - y) / g.res_y
    out = np.full(len(x), np.nan)

    ci, ri = np.floor(fc).astype(int), np.floor(fr).astype(int)
    inside = (ci >= 0) & (ci < w) & (ri >= 0) & (ri < h)
    nearest = np.full(len(x), np.nan)
    nearest[inside] = g.z[ri[inside], ci[inside]]
    if interpolation == "nearest":
        return nearest
    if interpolation != "bilinear":
        raise ValueError(f"Nieznana interpolacja {interpolation!r} (nearest | bilinear)")

    # srodki pikseli: c - 0.5
    cc, rr = fc - 0.5, fr - 0.5
    c0, r0 = np.floor(cc).astype(int), np.floor(rr).astype(int)
    tx, ty = cc - c0, rr - r0
    ok = (c0 >= 0) & (c0 + 1 < w) & (r0 >= 0) & (r0 + 1 < h)
    if ok.any():
        a = g.z[r0[ok], c0[ok]]
        b = g.z[r0[ok], c0[ok] + 1]
        c = g.z[r0[ok] + 1, c0[ok]]
        d = g.z[r0[ok] + 1, c0[ok] + 1]
        txo, tyo = tx[ok], ty[ok]
        out[ok] = (a * (1 - txo) * (1 - tyo) + b * txo * (1 - tyo)
                   + c * (1 - txo) * tyo + d * txo * tyo)
    fallback = np.isnan(out)
    out[fallback] = nearest[fallback]
    return out


def _read_xyz(path: Path) -> tuple[np.ndarray, float, float, float, float]:
    """Plik XYZ (x y z w kolumnach, regularna siatka) -> (z, x0, y0, res_x, res_y)."""
    data = np.loadtxt(path, usecols=(0, 1, 2), delimiter=None)
    xs, ys = np.unique(data[:, 0]), np.unique(data[:, 1])
    res_x = float(np.min(np.diff(xs))) if len(xs) > 1 else 1.0
    res_y = float(np.min(np.diff(ys))) if len(ys) > 1 else 1.0
    cols = np.rint((data[:, 0] - xs[0]) / res_x).astype(int)
    rows = np.rint((ys[-1] - data[:, 1]) / res_y).astype(int)
    z = np.full((rows.max() + 1, cols.max() + 1), np.nan)
    z[rows, cols] = data[:, 2]
    # wspolrzedne w XYZ to srodki pikseli
    return z, xs[0] - res_x / 2, ys[-1] + res_y / 2, res_x, res_y


def _read_raster(path: Path, default_crs: str):
    """GeoTIFF / ASC (GDAL) / XYZ -> (z NaN-nodata, x0, y0, res_x, res_y, crs)."""
    if path.suffix.lower() in (".xyz", ".txt"):
        z, x0, y0, rx, ry = _read_xyz(path)
        return z, x0, y0, rx, ry, default_crs
    import rasterio
    with rasterio.open(path) as ds:
        z = ds.read(1, masked=True).astype(np.float64).filled(np.nan)
        t = ds.transform
        if t.b != 0 or t.d != 0:
            raise ValueError(f"{path}: obrocony raster nie jest obslugiwany")
        crs = ds.crs.to_wkt() if ds.crs else default_crs
        return z, t.c, t.f, t.a, -t.e, crs


def load_mosaic(paths: list[Path], default_crs: str = "EPSG:2180",
                bounds: Optional[tuple[float, float, float, float]] = None) -> Grid:
    """Mozaika arkuszy (ta sama rozdzielczosc i CRS) w pamieci, przycieta do `bounds`
    (xmin, ymin, xmax, ymax w ukladzie rastrow). Nakladajace sie piksele: pierwszy
    poprawny wygrywa (kolejnosc plikow wg nazwy)."""
    if not paths:
        raise FileNotFoundError("Brak plikow rastra do mozaiki")
    sheets = []
    for p in sorted(paths):
        z, x0, y0, rx, ry, crs = _read_raster(p, default_crs)
        sheets.append((p, z, x0, y0, rx, ry, crs))
    rx, ry, crs = sheets[0][4], sheets[0][5], sheets[0][6]
    for p, _, _, _, rxi, ryi, crsi in sheets[1:]:
        if not (math.isclose(rxi, rx, rel_tol=1e-6) and math.isclose(ryi, ry, rel_tol=1e-6)):
            raise ValueError(f"{p}: inna rozdzielczosc ({rxi}x{ryi}) niz {sheets[0][0]} ({rx}x{ry})")
        if not _same_crs(crsi, crs):
            raise ValueError(f"{p}: inny CRS niz {sheets[0][0]}")

    xmin = min(s[2] for s in sheets)
    ymax = max(s[3] for s in sheets)
    xmax = max(s[2] + s[1].shape[1] * rx for s in sheets)
    ymin = min(s[3] - s[1].shape[0] * ry for s in sheets)
    if bounds is not None:
        bx0, by0, bx1, by1 = bounds
        # wyrownanie do siatki pierwszego arkusza
        gx, gy = sheets[0][2], sheets[0][3]
        xmin = max(xmin, gx + math.floor((bx0 - gx) / rx) * rx)
        xmax = min(xmax, gx + math.ceil((bx1 - gx) / rx) * rx)
        ymax = min(ymax, gy - math.floor((gy - by1) / ry) * ry)
        ymin = max(ymin, gy - math.ceil((gy - by0) / ry) * ry)
        if xmin >= xmax or ymin >= ymax:
            raise ValueError(f"Arkusze nie pokrywaja obszaru {bounds}")
    w, h = int(round((xmax - xmin) / rx)), int(round((ymax - ymin) / ry))
    out = np.full((h, w), np.nan)
    for p, z, x0, y0, _, _, _ in sheets:
        c_off = int(round((x0 - xmin) / rx))
        r_off = int(round((ymax - y0) / ry))
        r0, c0 = max(0, r_off), max(0, c_off)
        r1, c1 = min(h, r_off + z.shape[0]), min(w, c_off + z.shape[1])
        if r0 >= r1 or c0 >= c1:
            continue
        src = z[r0 - r_off:r1 - r_off, c0 - c_off:c1 - c_off]
        dst = out[r0:r1, c0:c1]
        fill = np.isnan(dst) & ~np.isnan(src)
        dst[fill] = src[fill]
    log.info("Mozaika %d arkuszy: %dx%d px @ %.2f m, pokrycie %.1f%%", len(sheets), w, h, rx,
             100.0 * np.isfinite(out).mean())
    return Grid(out, xmin, ymax, rx, ry, crs)


def _same_crs(a: str, b: str) -> bool:
    from pyproj import CRS
    return CRS.from_user_input(a) == CRS.from_user_input(b)


def raster_files(folder: Path) -> list[Path]:
    if not folder.is_dir():
        raise FileNotFoundError(f"Brak folderu z arkuszami: {folder}")
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in RASTER_SUFFIXES)
    if not files:
        raise FileNotFoundError(f"W {folder} nie ma plikow rastra ({sorted(RASTER_SUFFIXES)})")
    return files


class RasterSurface:
    def __init__(self, grid: Grid, name: str, vertical: str, interpolation: str = "bilinear"):
        self.grid = grid
        self.name = name
        self.crs = grid.crs
        self.vertical = vertical
        self.resolution_m = min(grid.res_x, grid.res_y)
        self.interpolation = interpolation

    def sample_many(self, x, y) -> np.ndarray:
        return sample_grid(self.grid, x, y, self.interpolation)

    def sample(self, x: float, y: float) -> Optional[float]:
        """Zgodnosc z DemSampler (wspolrzedne w `crs`)."""
        z = float(self.sample_many([x], [y])[0])
        return None if math.isnan(z) else z


# ───────────── wybor zrodla ─────────────

def area_bounds_2180(cfg: PipelineConfig, camera_xy_2180: np.ndarray) -> tuple[float, float, float, float]:
    """Obszar dla mozaiki: bbox kamer + bufor (promienie siegaja ~t_max od kamer)."""
    b = cfg.surface.area_buffer_m
    return (float(camera_xy_2180[:, 0].min() - b), float(camera_xy_2180[:, 1].min() - b),
            float(camera_xy_2180[:, 0].max() + b), float(camera_xy_2180[:, 1].max() + b))


def _bounds_in(crs: str, bounds_2180) -> tuple[float, float, float, float]:
    from pyproj import Transformer
    t = Transformer.from_crs("EPSG:2180", crs, always_xy=True)
    return t.transform_bounds(*bounds_2180)


def build_surface(cfg: PipelineConfig, bounds_2180: tuple[float, float, float, float],
                  ref_dem: Optional[DemSampler] = None) -> Surface:
    """Powierzchnia wg cfg.surface.source. `ref_dem` — NMT referencyjny z kontekstu;
    gdy to GugikDem, nmt_gugik_rest wspoldzieli z nim cache."""
    s: SurfaceConfig = cfg.surface
    if s.source == "nmt_gugik_rest":
        dem = ref_dem if isinstance(ref_dem, GugikDem) else GugikDem(
            cfg.dem.gugik_url, timeout=cfg.dem.timeout_s, cache_path=cfg.dem.cache_path)
        return PointSurface(dem, name="nmt_gugik_rest")
    if s.source in ("nmt_local", "nmpt_local"):
        folder = s.nmt_dir if s.source == "nmt_local" else s.nmpt_dir
        files = raster_files(folder)
        crs = _read_crs(files[0], s.default_crs)
        if not _same_crs(crs, "EPSG:2180"):
            log.warning("%s: CRS arkuszy (%s) nie jest EPSG:2180 — sprawdz pliki", s.source, crs)
        grid = load_mosaic(files, s.default_crs, _bounds_in(crs, bounds_2180))
        return RasterSurface(grid, s.source, VERTICAL_NPM, s.interpolation)
    if s.source == "odm_dsm":
        path = s.odm_dsm or cfg.odm_dir / "odm_dem" / "dsm.tif"
        if not Path(path).exists():
            raise FileNotFoundError(f"surface.source=odm_dsm: brak {path} (ODM z --dsm, patrz ODM.md)")
        crs = _read_crs(Path(path), s.default_crs)
        grid = load_mosaic([Path(path)], s.default_crs, _bounds_in(crs, bounds_2180))
        return RasterSurface(grid, "odm_dsm", VERTICAL_ODM, s.interpolation)
    raise ValueError(f"Nieznane surface.source={s.source!r} (dozwolone: {SURFACE_SOURCES})")


def _read_crs(path: Path, default_crs: str) -> str:
    if path.suffix.lower() in (".xyz", ".txt"):
        return default_crs
    import rasterio
    with rasterio.open(path) as ds:
        return ds.crs.to_wkt() if ds.crs else default_crs
