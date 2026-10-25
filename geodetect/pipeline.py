"""Pipeline dla pojedynczego zdjecia: EXIF -> ODM IO/EO -> YOLO -> ray casting na DEM.

Ciezkie zasoby (DEM z cache, rekonstrukcja ODM, model YOLO) trzymane sa w
PipelineContext, budowanym raz i wspoldzielonym miedzy zdjeciami (wsad, GUI).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .config import FilterConfig, PipelineConfig
from .dem import DemSampler, resolve_dem
from .detect import Detection, Detector
from .exif import GpsFix, read_gps
from .geo import EnuFrame
from .odm import CameraEO, CameraIO, OdmReconstruction, ZCalibration, compute_z_offset
from .raycast import RayHit, pixel_to_ray, raycast

log = logging.getLogger(__name__)


@dataclass
class PipelineContext:
    config: PipelineConfig
    dem: DemSampler
    reconstruction: OdmReconstruction
    frame: EnuFrame            # z ustawionym z_offset
    z_calibration: ZCalibration
    detector: Detector


@dataclass
class CameraPose:
    enu: np.ndarray
    x_2180: float
    y_2180: float
    h: float                   # [m npm] po korekcie Z
    lon: float
    lat: float


@dataclass
class GeoDetection:
    image_name: str
    detection: Detection
    uv: tuple[float, float]    # piksel reprezentatywny (na razie srodek bboxa)
    hit: RayHit
    incidence_deg: float       # kat promienia wzgledem pionu (0 = nadir)
    touches_border: bool       # bbox styka sie z krawedzia kadru (obiekt uciety)
    rejected_reason: Optional[str] = None   # None = przyjeta
    object_id: Optional[int] = None         # nadawane przy agregacji

    @property
    def accepted(self) -> bool:
        return self.rejected_reason is None


@dataclass
class ImageResult:
    image_name: str
    gps: GpsFix
    io: CameraIO
    eo: CameraEO
    camera: CameraPose
    z_calibration: ZCalibration
    detections: list[GeoDetection] = field(default_factory=list)
    misses: list[Detection] = field(default_factory=list)  # bez przeciecia z DEM


def build_context(config: PipelineConfig, detector: Optional[Detector] = None) -> PipelineContext:
    dem = resolve_dem(config.dem)
    rec = OdmReconstruction(config.reconstruction_path, config.reference_lla_path)
    base_frame = EnuFrame(rec.ref)
    z_cal = compute_z_offset(config.shots_geojson_path, dem, base_frame, config.agl_fallback_m)
    if detector is None:
        detector = Detector(config.model_path, conf=config.detect.conf,
                            classes=config.detect.classes, device=config.detect.device)
    return PipelineContext(config, dem, rec, base_frame.with_z_offset(z_cal.offset),
                           z_cal, detector)


def run_single(config: PipelineConfig, image_name: str,
               ctx: Optional[PipelineContext] = None) -> ImageResult:
    if ctx is None:
        ctx = build_context(config)
    image_path = config.photos_dir / image_name
    gps = read_gps(image_path)

    io, eo = ctx.reconstruction.camera(image_name)
    cx, cy, ch = ctx.frame.enu_to_puwg92(*eo.C_enu)
    clon, clat = ctx.frame.puwg92_to_lonlat(cx, cy)
    camera = CameraPose(eo.C_enu, float(cx), float(cy), float(ch), float(clon), float(clat))
    log.info("%s: kamera EPSG:2180 = (%.2f, %.2f, %.2f)", image_name, cx, cy, ch)

    result = ImageResult(image_name, gps, io, eo, camera, ctx.z_calibration)
    for det in ctx.detector.detect(image_path):
        uv = det.center
        origin, direction = pixel_to_ray(uv, io, eo)
        hit = raycast(origin, direction, ctx.dem, ctx.frame, config.raycast)
        if hit is None:
            log.info("%s: %s u,v=(%.0f, %.0f) — brak przeciecia z DEM",
                     image_name, det.cls_name, *uv)
            result.misses.append(det)
            continue
        incidence = float(np.degrees(np.arccos(np.clip(-direction[2], -1.0, 1.0))))
        gd = GeoDetection(image_name, det, uv, hit, incidence,
                          touches_border(det.bbox, io.width, io.height,
                                         config.filters.border_margin_px))
        gd.rejected_reason = rejection_reason(gd, config.filters)
        result.detections.append(gd)
    n_rej = sum(not gd.accepted for gd in result.detections)
    log.info("%s: %d detekcji z georeferencja (%d odrzuconych), %d bez przeciecia",
             image_name, len(result.detections), n_rej, len(result.misses))
    return result


def touches_border(bbox: tuple[float, float, float, float], width: int, height: int,
                   margin_px: float) -> bool:
    x1, y1, x2, y2 = bbox
    return (x1 <= margin_px or y1 <= margin_px
            or x2 >= width - margin_px or y2 >= height - margin_px)


def rejection_reason(gd: GeoDetection, f: FilterConfig) -> Optional[str]:
    """Pierwszy niespelniony filtr (opis) albo None, gdy detekcja przyjeta."""
    if f.exclude_border and gd.touches_border:
        return "touches_border"
    if f.max_range_m is not None and gd.hit.range_m > f.max_range_m:
        return f"range_m>{f.max_range_m:g}"
    if f.max_incidence_deg is not None and gd.incidence_deg > f.max_incidence_deg:
        return f"incidence_deg>{f.max_incidence_deg:g}"
    return None
