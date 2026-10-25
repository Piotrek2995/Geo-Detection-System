"""Pipeline dla pojedynczego zdjecia: EXIF -> ODM IO/EO -> YOLO -> ray casting na DEM.

Ciezkie zasoby (DEM z cache, rekonstrukcja ODM, model YOLO) trzymane sa w
PipelineContext, budowanym raz i wspoldzielonym miedzy zdjeciami (wsad, GUI).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .config import PipelineConfig
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
        result.detections.append(GeoDetection(image_name, det, uv, hit))
    log.info("%s: %d detekcji z georeferencja, %d bez przeciecia",
             image_name, len(result.detections), len(result.misses))
    return result
