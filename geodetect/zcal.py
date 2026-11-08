"""Kalibracja wysokosci (anchor Z): H [m npm] = h_ODM + z_offset (+ opcjonalny tilt).

Potensic zapisuje GPS Altitude jako AGL od punktu startu, wiec wysokosci w rekonstrukcji
ODM (kamery, chmura punktow, DEM ODM) sa w ukladzie "h wzgledem startu", nie npm.
Tryby (cfg.z_anchor.mode):
  agl_assumption — zalozona wysokosc lotu nad terenem (fallback),
  ground_frame   — klatka zrobiona na ziemi w punkcie startu,
  dem_alignment  — dopasowanie DEM z ODM do NMT (mediana roznic w otwartym terenie).
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from pyproj import CRS, Transformer

from .config import PipelineConfig, Z_ANCHOR_MODES, ZAnchorConfig
from .dem import DemSampler, save_dem_cache
from .geo import EnuFrame, ZTilt
from .odm import OdmReconstruction

log = logging.getLogger(__name__)

ODM_DEM_DOCKER_CMD = (
    'docker run --rm -v "${PWD}\\odm_project:/datasets/code" opendronemap/odm '
    "--project-path /datasets --rerun-from odm_georeferencing --dtm --dsm --dem-resolution 10"
)


@dataclass
class PlaneFit:
    """diff = c + a*(x - x0) + b*(y - y0), x/y w EPSG:2180."""
    a: float
    b: float
    c: float
    x0: float
    y0: float
    rms_resid: float
    n_used: int

    @property
    def slope_deg(self) -> float:
        return math.degrees(math.atan(math.hypot(self.a, self.b)))

    @property
    def aspect_deg(self) -> float:
        """Azymut (od polnocy, zgodnie z ruchem wskazowek), w ktorym roznica rosnie."""
        return (math.degrees(math.atan2(self.a, self.b)) + 360.0) % 360.0

    def __call__(self, x, y):
        return self.c + self.a * (np.asarray(x) - self.x0) + self.b * (np.asarray(y) - self.y0)


@dataclass
class DemAlignment:
    """Dane trybu dem_alignment (do raportu i wykresow)."""
    layer: str                     # "dtm" / "dsm" — z ktorego DEM ODM brano wysokosci
    src_crs: str
    x_2180: np.ndarray
    y_2180: np.ndarray
    h_odm: np.ndarray
    nmt: np.ndarray
    plane: PlaneFit
    n_candidates: int              # punkty po buforze (i filtrze DSM-DTM)
    camera_agl: dict = field(default_factory=dict)  # weryfikacja: h kamery - DEM ODM pod nia

    @property
    def diff(self) -> np.ndarray:
        return self.nmt - self.h_odm


@dataclass
class ZCalibration:
    offset: float
    method: str
    tilt: Optional[ZTilt] = None
    stats: dict = field(default_factory=dict)
    alignment: Optional[DemAlignment] = None

    def describe(self) -> str:
        parts = [f"{self.method}: z_offset={self.offset:.3f} m"]
        parts += [f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                  for k, v in self.stats.items()]
        if self.tilt is not None:
            parts.append(f"tilt a={self.tilt.a:.5f} b={self.tilt.b:.5f} (stosowany)")
        return ", ".join(parts)


# ───────────── tryb 1: zalozenie AGL ─────────────

def agl_assumption(shots_geojson: Path, dem: DemSampler, frame: EnuFrame,
                   agl_m: float) -> ZCalibration:
    """z_offset = NMT(mediana xy kamer) + agl_m - srednia h kamer (z shots.geojson)."""
    if not shots_geojson.exists():
        log.warning("Brak %s — z_offset=0", shots_geojson)
        return ZCalibration(0.0, "none")
    feats = json.loads(shots_geojson.read_text(encoding="utf-8"))["features"]
    coords = np.array([f["geometry"]["coordinates"] for f in feats], dtype=np.float64)
    med_lon, med_lat = float(np.median(coords[:, 0])), float(np.median(coords[:, 1]))
    mean_alt_local = float(np.mean(coords[:, 2]))

    x, y = frame.lonlat_to_puwg92(med_lon, med_lat)
    nmt_under = dem.sample(x, y)
    if nmt_under is None:
        log.warning("DEM nie zwrocil wartosci dla median_xy=(%.1f, %.1f) — z_offset=0", x, y)
        return ZCalibration(0.0, "none", stats={"mean_alt_local": mean_alt_local})

    offset = nmt_under + agl_m - mean_alt_local
    log.info("Z (agl_assumption): NMT(median_xy)=%.2f, mean_alt_local=%.2f, AGL=%.1f "
             "-> z_offset=%.2f m", nmt_under, mean_alt_local, agl_m, offset)
    return ZCalibration(offset, "agl_assumption", stats={
        "nmt_under": nmt_under, "mean_alt_local": mean_alt_local, "agl_m": agl_m})


# ───────────── tryb 2: klatka naziemna ─────────────

def ground_frame(rec: OdmReconstruction, image: Optional[str], dem: DemSampler,
                 frame: EnuFrame, lens_height_m: float) -> ZCalibration:
    """z_offset = NMT(xy anchora) + wysokosc obiektywu - h kamery anchora w ODM."""
    if not image:
        raise ValueError("z_anchor.mode=ground_frame wymaga z_anchor.image "
                         "(zdjecie zrobione na ziemi w punkcie startu)")
    if image not in rec.shots:
        raise ValueError(
            f"Zdjecie-anchor {image!r} nie zostalo zrekonstruowane przez ODM "
            f"({rec.path.name}), wiec nie znamy jego pozycji. Sprawdz, czy bylo w "
            f"odm_project/images i czy ODM je zarejestrowal (raport ODM), albo uzyj innego "
            f"trybu z_anchor.mode.")
    _, eo = rec.camera(image)
    lon, lat, h = frame.enu_to_geodetic(*eo.C_enu)
    x, y = frame.lonlat_to_puwg92(lon, lat)
    nmt = dem.sample(x, y)
    if nmt is None:
        raise ValueError(f"DEM nie zwrocil wysokosci w punkcie anchora {image} "
                         f"(x={x:.1f}, y={y:.1f} EPSG:2180)")
    offset = nmt + lens_height_m - float(h)
    log.info("Z (ground_frame): %s NMT=%.2f + %.2f - h_ODM=%.2f -> z_offset=%.2f m",
             image, nmt, lens_height_m, h, offset)
    return ZCalibration(offset, "ground_frame", stats={
        "image": image, "nmt_anchor": nmt, "h_anchor_odm": float(h),
        "lens_height_m": lens_height_m, "x_2180": float(x), "y_2180": float(y)})


# ───────────── tryb 3: dopasowanie DEM ODM do NMT ─────────────

def fit_plane(x: np.ndarray, y: np.ndarray, d: np.ndarray, clip_sigma: float = 3.0) -> PlaneFit:
    """Najmniejsze kwadraty + jednokrotne odrzucenie odstajacych (> clip_sigma * 1.4826*MAD)."""
    x0, y0 = float(np.mean(x)), float(np.mean(y))
    A = np.c_[x - x0, y - y0, np.ones_like(x)]
    keep = np.ones(len(d), dtype=bool)
    for _ in range(2):
        (a, b, c), *_ = np.linalg.lstsq(A[keep], d[keep], rcond=None)
        resid = d - A @ [a, b, c]
        sigma = 1.4826 * np.median(np.abs(resid[keep] - np.median(resid[keep])))
        if sigma == 0:
            break
        keep = np.abs(resid) <= clip_sigma * sigma
    rms = float(np.sqrt(np.mean(resid[keep] ** 2)))
    return PlaneFit(float(a), float(b), float(c), x0, y0, rms, int(keep.sum()))


def sample_odm_dem(dtm_path: Optional[Path], dsm_path: Optional[Path], n: int, buffer_m: float,
                   max_dsm_dtm_m: float, seed: int = 0):
    """Losuje n punktow w obszarze DEM ODM, min. buffer_m od krawedzi danych.
    Gdy sa oba DEM-y: tylko piksele, gdzie |DSM - DTM| < max_dsm_dtm_m (otwarty teren).
    Zwraca (x_src, y_src, h_odm, crs, layer, n_candidates)."""
    import rasterio
    from rasterio.transform import xy as pix_to_xy
    from scipy.ndimage import distance_transform_edt

    has_dtm = dtm_path is not None and Path(dtm_path).exists()
    has_dsm = dsm_path is not None and Path(dsm_path).exists()
    primary, layer = (dtm_path, "dtm") if has_dtm else (dsm_path, "dsm")
    rng = np.random.default_rng(seed)

    with rasterio.open(primary) as ds:
        z = ds.read(1, masked=True)
        valid = ~np.ma.getmaskarray(z) & np.isfinite(z.filled(np.nan))
        res = abs(ds.transform.a)
        inner = distance_transform_edt(valid) * res >= buffer_m
        rows, cols = np.nonzero(inner)
        order = rng.permutation(len(rows))
        rows, cols = rows[order], cols[order]
        xs, ys = (np.asarray(v) for v in pix_to_xy(ds.transform, rows, cols))
        h = z.data[rows, cols].astype(np.float64)
        crs = ds.crs.to_wkt()

    n_candidates = len(h)
    if has_dtm and has_dsm:
        keep_idx = []
        with rasterio.open(dsm_path) as dsm:
            if CRS.from_user_input(dsm.crs) != CRS.from_user_input(crs):
                raise ValueError("DTM i DSM ODM maja rozne CRS")
            chunk = max(10 * n, 1000)
            for start in range(0, len(h), chunk):
                sl = slice(start, start + chunk)
                for i, v in enumerate(dsm.sample(zip(xs[sl], ys[sl]), masked=True)):
                    z_dsm = _masked_to_float(v)
                    if z_dsm is not None and abs(z_dsm - h[start + i]) < max_dsm_dtm_m:
                        keep_idx.append(start + i)
                if len(keep_idx) >= n:
                    break
        n_candidates = len(keep_idx)  # dolne oszacowanie (przerywamy po n)
        idx = np.array(keep_idx[:n], dtype=int)
    else:
        idx = np.arange(min(n, len(h)))
    return xs[idx], ys[idx], h[idx], crs, layer, n_candidates


def _masked_to_float(v) -> Optional[float]:
    """Pierwsze pasmo z rasterio.sample(masked=True) -> float lub None (nodata/NaN)."""
    if np.ma.is_masked(v[0]):
        return None
    z = float(v[0])
    return z if math.isfinite(z) else None


def camera_agl_check(rec: OdmReconstruction, frame: EnuFrame, dem_path: Path) -> dict:
    """Weryfikacja wspolnego ukladu wysokosci: h kamery (ODM) - DEM ODM pod kamera."""
    import rasterio
    vals = []
    with rasterio.open(dem_path) as ds:
        to_src = Transformer.from_crs("EPSG:4326", ds.crs, always_xy=True)
        for name in rec.image_names:
            _, eo = rec.camera(name)
            lon, lat, h = frame.enu_to_geodetic(*eo.C_enu)
            sx, sy = to_src.transform(lon, lat)
            g = _masked_to_float(next(ds.sample([(sx, sy)], masked=True)))
            if g is not None:
                vals.append((name, float(h), g))
    if not vals:
        return {"n": 0}
    agl = np.array([h - g for _, h, g in vals])
    return {"n": len(vals), "agl_median": float(np.median(agl)), "agl_min": float(agl.min()),
            "agl_max": float(agl.max()), "h_cam_median": float(np.median([h for _, h, _ in vals])),
            "dem_under_median": float(np.median([g for _, _, g in vals]))}


def dem_alignment(za: ZAnchorConfig, dtm_path: Path, dsm_path: Path, dem: DemSampler,
                  rec: Optional[OdmReconstruction] = None,
                  frame: Optional[EnuFrame] = None) -> ZCalibration:
    has_dtm, has_dsm = Path(dtm_path).exists(), Path(dsm_path).exists()
    if not (has_dtm or has_dsm):
        raise FileNotFoundError(
            f"z_anchor.mode=dem_alignment: brak DEM z ODM ({dtm_path}, {dsm_path}). "
            f"Dogeneruj je bez ponownego SfM (PowerShell, katalog repo):\n  {ODM_DEM_DOCKER_CMD}\n"
            f"Rerun od odm_georeferencing, bo tam jest klasyfikacja gruntu (SMRF) — bez niej DTM "
            f"nie powstaje. Ewentualny blad ODM przy grafice raportu PDF na koncu nie wplywa "
            f"na dtm.tif/dsm.tif.")

    xs, ys, h_odm, crs, layer, n_cand = sample_odm_dem(
        dtm_path if has_dtm else None, dsm_path if has_dsm else None,
        za.n_samples, za.buffer_m, za.max_dsm_dtm_m, za.seed)
    to_2180 = Transformer.from_crs(crs, "EPSG:2180", always_xy=True)
    x2180, y2180 = to_2180.transform(xs, ys)
    nmt = np.array([dem.sample(float(x), float(y)) for x, y in zip(x2180, y2180)], dtype=object)
    save_dem_cache(dem)
    ok = np.array([v is not None for v in nmt])
    if ok.sum() < 10:
        raise ValueError(f"dem_alignment: tylko {ok.sum()} punktow z wartoscia NMT")
    x2180, y2180, h_odm = np.asarray(x2180)[ok], np.asarray(y2180)[ok], h_odm[ok]
    nmt = nmt[ok].astype(np.float64)

    diff = nmt - h_odm
    med = float(np.median(diff))
    mad = float(np.median(np.abs(diff - med)))
    plane = fit_plane(x2180, y2180, diff)
    cam = camera_agl_check(rec, frame, dtm_path if has_dtm else dsm_path) \
        if rec is not None and frame is not None else {}
    alignment = DemAlignment(layer, crs, x2180, y2180, h_odm, nmt, plane, n_cand, cam)

    stats = {"n_points": int(ok.sum()), "layer": layer, "median": med, "mad": mad,
             "sigma_mad": 1.4826 * mad, "plane_slope_deg": plane.slope_deg,
             "plane_aspect_deg": plane.aspect_deg, "plane_c": plane.c,
             "plane_rms": plane.rms_resid}
    if cam.get("n"):
        stats["camera_agl_median"] = cam["agl_median"]
    if za.apply_tilt:
        tilt = ZTilt(plane.a, plane.b, plane.x0, plane.y0)
        log.info("Z (dem_alignment+tilt): c=%.3f, nachylenie %.3f deg, azymut %.0f deg",
                 plane.c, plane.slope_deg, plane.aspect_deg)
        return ZCalibration(plane.c, "dem_alignment+tilt", tilt, stats, alignment)
    log.info("Z (dem_alignment): mediana(NMT-DEM_ODM)=%.3f m, MAD=%.3f, n=%d, "
             "plaszczyzna: %.3f deg w kierunku %.0f deg",
             med, mad, ok.sum(), plane.slope_deg, plane.aspect_deg)
    return ZCalibration(med, "dem_alignment", None, stats, alignment)


# ───────────── dispatcher ─────────────

def calibrate_z(config: PipelineConfig, dem: DemSampler, rec: OdmReconstruction,
                frame: EnuFrame) -> ZCalibration:
    """Liczy z_offset wg config.z_anchor.mode. `frame` — EnuFrame bez offsetu."""
    za = config.z_anchor
    if za.mode == "agl_assumption":
        return agl_assumption(config.shots_geojson_path, dem, frame, za.agl_m)
    if za.mode == "ground_frame":
        return ground_frame(rec, za.image, dem, frame, za.lens_height_m)
    if za.mode == "dem_alignment":
        return dem_alignment(za, config.odm_dtm_path, config.odm_dsm_path, dem, rec, frame)
    if za.mode == "multiview_check":
        raise ValueError("multiview_check to tylko diagnostyka (geodetect.diagnostics."
                         "multiview_z_scan), nie tryb kalibracji wynikow")
    raise ValueError(f"Nieznany z_anchor.mode={za.mode!r} (dozwolone: {Z_ANCHOR_MODES})")
