"""Przetwarzanie wsadowe: wiele zdjec ze wspolnym PipelineContext + agregacja."""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Optional

from .aggregate import AggregatedObject, aggregate
from .config import PipelineConfig
from .detect import Detection
from .dem import save_dem_cache
from .pipeline import GeoDetection, ImageResult, PipelineContext, build_context, run_single
from .zcal import ZCalibration

log = logging.getLogger(__name__)

PHOTO_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

ProgressCallback = Callable[[int, int, str], None]  # (i, n, image_name), i od 1


@dataclass
class BatchResult:
    images: list[ImageResult]
    skipped: dict[str, str]              # nazwa zdjecia -> powod pominiecia
    objects: list[AggregatedObject] = field(default_factory=list)
    z_calibration: Optional[ZCalibration] = None

    @property
    def detections(self) -> list[GeoDetection]:
        return [gd for r in self.images for gd in r.detections]

    @property
    def accepted(self) -> list[GeoDetection]:
        return [gd for gd in self.detections if gd.accepted]

    @property
    def rejected(self) -> list[GeoDetection]:
        return [gd for gd in self.detections if not gd.accepted]

    @property
    def misses(self) -> list[tuple[str, Detection]]:
        return [(r.image_name, d) for r in self.images for d in r.misses]

    @property
    def suspicious_objects(self) -> list[AggregatedObject]:
        """Klastry, w ktorych to samo zdjecie wystepuje wiecej niz raz."""
        return [o for o in self.objects if o.duplicate_images]


def summary_text(batch: BatchResult) -> str:
    """Czytelne podsumowanie wsadu (do CLI / logu / raportu)."""
    lines = []
    if batch.z_calibration is not None:
        lines.append(f"Kalibracja Z — {batch.z_calibration.describe()}")
    lines.append(f"Zdjecia przetworzone: {len(batch.images)}, pominiete: {len(batch.skipped)}")
    for name, reason in sorted(batch.skipped.items()):
        lines.append(f"  pominiete {name}: {reason}")
    lines.append(f"Detekcje: {len(batch.detections)} z georeferencja "
                 f"(przyjete {len(batch.accepted)}, odrzucone {len(batch.rejected)}), "
                 f"bez przeciecia z DEM: {len(batch.misses)}")
    for reason, n in Counter(gd.rejected_reason for gd in batch.rejected).most_common():
        lines.append(f"  odrzucone [{reason}]: {n}")
    n_border = sum(gd.touches_border for gd in batch.detections)
    lines.append(f"  touches_border: {n_border}")
    per_cls = Counter(o.cls_name for o in batch.objects)
    lines.append("Obiekty: " + ", ".join(f"{c}={n}" for c, n in sorted(per_cls.items())))
    lines.append(f"  {'id':>3} {'klasa':8} {'widoki':>6} {'RMS[m]':>7} {'max[m]':>7} "
                 f"{'inc[deg]':>8} {'range[m]':>8} {'conf':>5}  zdjecia")
    for o in batch.objects:
        lines.append(f"  {o.object_id:>3} {o.cls_name:8} {o.n_views:>6} {o.rms_m:>7.2f} "
                     f"{o.max_m:>7.2f} {o.mean_incidence_deg:>8.1f} {o.mean_range_m:>8.1f} "
                     f"{o.mean_conf:>5.2f}  {','.join(i.removesuffix('.JPG') for i in o.images)}")
    if batch.suspicious_objects:
        lines.append("UWAGA — mozliwe zle scalenie (to samo zdjecie >1 raz w klastrze):")
        for o in batch.suspicious_objects:
            lines.append(f"  obiekt {o.object_id} ({o.cls_name}): {', '.join(o.duplicate_images)}")
    return "\n".join(lines)


def run_batch(cfg: PipelineConfig, ctx: Optional[PipelineContext] = None,
              images: Optional[list[str]] = None,
              progress_cb: Optional[ProgressCallback] = None) -> BatchResult:
    """images=None: wszystkie zdjecia z rekonstrukcji ODM. Zdjecia z folderu (lub z listy),
    ktorych nie ma w rekonstrukcji albo na dysku, sa pomijane i raportowane w `skipped`."""
    if ctx is None:
        ctx = build_context(cfg)
    in_rec = set(ctx.reconstruction.image_names)
    skipped: dict[str, str] = {}

    if images is None:
        candidates = ctx.reconstruction.image_names
        on_disk = sorted(p.name for p in cfg.photos_dir.iterdir()
                         if p.suffix.lower() in PHOTO_SUFFIXES) if cfg.photos_dir.is_dir() else []
        for name in on_disk:
            if name not in in_rec:
                skipped[name] = "brak w rekonstrukcji ODM"
    else:
        candidates = []
        for name in images:
            if name not in in_rec:
                skipped[name] = "brak w rekonstrukcji ODM"
            else:
                candidates.append(name)

    todo = []
    for name in candidates:
        if not (cfg.photos_dir / name).exists():
            skipped[name] = f"brak pliku w {cfg.photos_dir}"
        else:
            todo.append(name)
    for name, reason in skipped.items():
        log.warning("Pomijam %s: %s", name, reason)

    results: list[ImageResult] = []
    for i, name in enumerate(todo, start=1):
        if progress_cb is not None:
            progress_cb(i, len(todo), name)
        try:
            results.append(run_single(cfg, name, ctx))
        except Exception as e:  # jedno uszkodzone zdjecie nie przerywa wsadu
            log.exception("Blad przetwarzania %s", name)
            skipped[name] = f"blad: {e}"

    save_dem_cache(ctx.dem)
    batch = BatchResult(results, skipped, z_calibration=ctx.z_calibration)
    batch.objects = aggregate(batch.detections, cfg.aggregate)
    log.info("Batch: %d zdjec, %d pominietych, %d detekcji (%d odrzuconych), %d obiektow",
             len(results), len(skipped), len(batch.detections), len(batch.rejected),
             len(batch.objects))
    return batch
