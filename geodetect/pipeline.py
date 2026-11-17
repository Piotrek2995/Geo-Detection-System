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
from .detect import Detection, Detector, make_detector, validate_class_names
from .exif import GpsFix, read_gps
from .geo import EnuFrame
from .odm import CameraEO, CameraIO, OdmReconstruction
from .points import PointChoice, choose_point
from .raycast import RayHit, median_hit, pixel_to_ray, raycast
from .segment import Segmenter, clean_mask, mask_from_polygons
from .surface import Surface, area_bounds_2180, as_surface, build_surface
from .zcal import ZCalibration, calibrate_z

log = logging.getLogger(__name__)

# Powierzchnie pokrycia terenu (nie gruntu): dla nich liczymy wysokosc nad NMT.
COVER_SURFACES = ("nmpt_local", "odm_dsm")


@dataclass
class PipelineContext:
    config: PipelineConfig
    dem: DemSampler            # NMT referencyjny (sekcja `dem`): kalibracja Z, wysokosc gruntu
    reconstruction: OdmReconstruction
    frame: EnuFrame            # z ustawionym z_offset
    z_calibration: ZCalibration
    detector: Detector
    surface: Optional[Surface] = None   # powierzchnia ray castingu; None = `dem`
    segmenter: Optional[Segmenter] = None   # SAM, tworzony leniwie (tylko metody mask_*)

    @property
    def ray_surface(self) -> Surface:
        return self.surface if self.surface is not None else as_surface(self.dem)


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
    uv: tuple[float, float]    # piksel reprezentatywny (cfg.point; dla footprint: centroid)
    hit: RayHit
    incidence_deg: float       # kat promienia wzgledem pionu (0 = nadir)
    touches_border: bool       # bbox styka sie z krawedzia kadru (obiekt uciety)
    rejected_reason: Optional[str] = None   # None = przyjeta
    object_id: Optional[int] = None         # nadawane przy agregacji
    point: Optional[PointChoice] = None     # metoda punktu, fallback, jakosc maski

    @property
    def ray_uvs(self) -> np.ndarray:
        """Piksele, przez ktore ida promienie (footprint: wiele, inaczej jeden)."""
        if self.point is not None and self.point.footprint_uvs is not None:
            return self.point.footprint_uvs
        return np.array([self.uv], dtype=np.float64)

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


def build_context(config: PipelineConfig, detector: Optional[Detector] = None,
                  dem: Optional[DemSampler] = None, reconstruction: Optional[OdmReconstruction] = None,
                  surface: Optional[Surface] = None) -> PipelineContext:
    """Kontekst przetwarzania. dem / reconstruction / surface mozna podac gotowe
    (wspoldzielone miedzy wariantami eksperymentu), inaczej sa budowane z configu."""
    config.point.validate()
    dem = dem if dem is not None else resolve_dem(config.dem)
    rec = reconstruction if reconstruction is not None else         OdmReconstruction(config.reconstruction_path, config.reference_lla_path)
    base_frame = EnuFrame(rec.ref)
    # DECYZJA METODYCZNA: z_offset liczony raz, wzgledem NMT referencyjnego (sekcja `dem`),
    # niezaleznie od powierzchni ray castingu (sekcja `surface`). Gdyby kalibracja zalezala
    # od wybranej powierzchni (NMT / NMPT / DSM ODM), porownanie powierzchni mieszaloby sie
    # z kalibracja i roznice wynikow nie bylyby przypisywalne samej powierzchni.
    z_cal = calibrate_z(config, dem, rec, base_frame)
    log.info("Kalibracja Z — %s", z_cal.describe())
    cams = np.array([base_frame.enu_to_puwg92(*rec.camera(n)[1].C_enu)[:2]
                     for n in rec.image_names])
    if surface is None:
        surface = build_surface(config, area_bounds_2180(config, cams), ref_dem=dem)
    log.info("Powierzchnia ray castingu: %s (uklad wysokosci %s, krok %.2f m)", surface.name,
             surface.vertical, config.raycast.step_for(surface.resolution_m))
    if detector is None:
        detector = make_detector(config.model_path, config.detect)
    names = getattr(detector, "names", None)
    if names:   # nazwy klas z konfiguracji musza istniec w modelu
        validate_class_names(config.point.per_class, names.values(), "point.per_class")
    return PipelineContext(config, dem, rec, base_frame.with_z_offset(z_cal.offset, z_cal.tilt),
                           z_cal, detector, surface)


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
    dets = ctx.detector.detect(image_path)
    choices = choose_points(ctx, image_path, dets, io, eo)
    for det, choice in zip(dets, choices):
        uv = choice.uv
        origin, direction = pixel_to_ray(uv, io, eo)
        hit = cast_point(ctx, choice, io, eo)
        if hit is None:
            log.info("%s: %s u,v=(%.0f, %.0f) — brak przeciecia z powierzchnia %s",
                     image_name, det.cls_name, *uv, ctx.ray_surface.name)
            result.misses.append(det)
            continue
        attach_ground_height(hit, ctx)
        incidence = float(np.degrees(np.arccos(np.clip(-direction[2], -1.0, 1.0))))
        gd = GeoDetection(image_name, det, uv, hit, incidence,
                          touches_border(det.bbox, io.width, io.height,
                                         config.filters.border_margin_px))
        gd.point = choice
        gd.rejected_reason = rejection_reason(gd, config.filters)
        result.detections.append(gd)
    n_rej = sum(not gd.accepted for gd in result.detections)
    log.info("%s: %d detekcji z georeferencja (%d odrzuconych), %d bez przeciecia",
             image_name, len(result.detections), n_rej, len(result.misses))
    return result


def choose_points(ctx: PipelineContext, image_path, dets: list[Detection], io: CameraIO,
                  eo: CameraEO) -> list[PointChoice]:
    """Punkt reprezentatywny kazdej detekcji wg cfg.point. Maski: z modelu
    segmentacyjnego (Detection.polygons), a gdy ich brak — z SAM (liczone lub z cache)
    tylko dla detekcji, ktorych metoda ich wymaga."""
    pc = ctx.config.point
    methods = [pc.method_for(d.cls_name) for d in dets]
    masks: dict[int, object] = {}
    for i, (d, m) in enumerate(zip(dets, methods)):
        if m.startswith("mask_") and d.polygons:
            masks[i] = mask_from_polygons(d.polygons, d.bbox, (io.width, io.height))
    need = [i for i, m in enumerate(methods) if m.startswith("mask_") and i not in masks]
    if need:
        if ctx.segmenter is None:
            ctx.segmenter = Segmenter(pc)
        try:
            got = ctx.segmenter.masks(image_path, [dets[i].bbox for i in need])
        except Exception as exc:     # SAM niedostepny -> fallback per detekcja
            log.warning("SAM: %s — fallback bbox_center", exc)
            got = [None] * len(need)
        masks.update(zip(need, got))
    out = []
    for i, (det, m) in enumerate(zip(dets, methods)):
        mask, quality = masks.get(i), None
        if mask is not None:
            mask, quality = clean_mask(mask, det.bbox, pc)
        ch = choose_point(m, det.bbox, io, eo, pc, mask, quality)
        if ch.fallback:
            log.info("%s %s: %s -> %s (%s)", image_path.name, det.cls_name, ch.method, ch.used,
                     ch.fallback)
        out.append(ch)
    return out


def cast_point(ctx: PipelineContext, choice: PointChoice, io: CameraIO, eo: CameraEO
               ) -> Optional[RayHit]:
    """Ray casting piksela; dla mask_footprint_median wiele promieni -> mediana trafien."""
    uvs = [choice.uv] if choice.footprint_uvs is None else choice.footprint_uvs
    hits = []
    for uv in uvs:
        o, d = pixel_to_ray(tuple(uv), io, eo)
        h = raycast(o, d, ctx.ray_surface, ctx.frame, ctx.config.raycast)
        if h is not None:
            hits.append(h)
    if not hits:
        return None
    return hits[0] if choice.footprint_uvs is None else median_hit(hits, ctx.frame)


def attach_ground_height(hit: RayHit, ctx: PipelineContext) -> None:
    """Dla powierzchni pokrycia (NMPT, DSM ODM) zapisuje NMT [m npm] w punkcie trafienia,
    zeby hit.above_ground_m dawalo wysokosc obiektu/pokrycia nad gruntem. Dla NMT pomijane
    (trafienie jest na gruncie z definicji)."""
    if hit.surface not in COVER_SURFACES:
        return
    try:
        hit.ground_h = ctx.dem.sample(hit.x_2180, hit.y_2180)
    except Exception as exc:  # NMT referencyjny niedostepny -> tylko brak tej informacji
        log.warning("Brak NMT w (%.1f, %.1f): %s", hit.x_2180, hit.y_2180, exc)


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
