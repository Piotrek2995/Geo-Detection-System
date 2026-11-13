"""Szybkie zestawienie zrodel powierzchni ray castingu (zadanie 4, nie pelny modul porownan).

Dla kazdego zrodla: batch na wszystkich zdjeciach, agregacja DBSCAN (eps), glowny namiot
(najwiecej widokow): RMS/max rozrzutu, pozycja i przesuniecie wzgledem nmt_gugik_rest,
czas (budowa powierzchni + ray casting; detekcje YOLO liczone raz, przed pomiarem).
Kalibracja Z jest wspolna (dem_alignment wzgledem NMT) — rozni sie tylko powierzchnia.

    python scripts/surface_compare.py                       # wszystkie dostepne zrodla
    python scripts/surface_compare.py --sources nmt_local nmpt_local --eps 10
"""

from __future__ import annotations

import argparse
import copy
import logging
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from geodetect.batch import run_batch  # noqa: E402
from geodetect.config import SURFACE_SOURCES, PipelineConfig  # noqa: E402
from geodetect.detect import Detector  # noqa: E402
from geodetect.diagnostics import static_objects  # noqa: E402
from geodetect.pipeline import build_context  # noqa: E402
from zcal_compare import CachedDetector  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    ap.add_argument("--sources", nargs="+", default=list(SURFACE_SOURCES), choices=SURFACE_SOURCES)
    ap.add_argument("--eps", type=float, default=10.0)
    ap.add_argument("--cls", default="namiot")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    base = PipelineConfig.from_file(args.config)
    detector = CachedDetector(Detector(base.model_path, base.detect))
    rows, ref = [], None
    for src in args.sources:
        cfg = copy.deepcopy(base)
        cfg.surface.source = src
        cfg.aggregate.eps_m = args.eps
        try:
            ctx = build_context(cfg, detector=detector)
        except (FileNotFoundError, ValueError) as exc:
            print(f"[{src}] POMINIETE: {exc}")
            continue
        if not detector._cache:            # rozgrzewka YOLO poza pomiarem czasu
            for n in ctx.reconstruction.image_names:
                if (cfg.photos_dir / n).exists():
                    detector.detect(cfg.photos_dir / n)
            ctx = build_context(cfg, detector=detector)
        t0 = time.perf_counter()
        ctx = build_context(cfg, detector=detector)
        batch = run_batch(cfg, ctx, images=None)
        dt = time.perf_counter() - t0
        objs = static_objects(batch, eps_m=args.eps, cls_name=args.cls, min_views=3)
        main_obj = max(objs, key=lambda o: o.n_views) if objs else None
        above = [gd.hit.above_ground_m for gd in batch.detections
                 if main_obj is not None and gd.object_id == main_obj.object_id
                 and gd.hit.above_ground_m is not None]
        if src == "nmt_gugik_rest":
            ref = main_obj
        rows.append((src, len(batch.detections), len(batch.misses), main_obj, dt,
                     float(np.median(above)) if above else None))
        print(f"[{src}] {len(batch.detections)} trafien, {len(batch.misses)} bez przeciecia, "
              f"{dt:.1f} s")

    print(f"\nzrodlo | trafienia (brak) | glowny {args.cls}: widoki, RMS / max [m] | "
          f"pozycja EPSG:2180 (H npm) | przesuniecie wzgl. nmt_gugik_rest [m] | "
          f"nad NMT [m] | czas [s]")
    for src, n_hit, n_miss, o, dt, above in rows:
        if o is None:
            print(f"{src} | {n_hit} ({n_miss}) | brak | - | - | - | {dt:.1f}")
            continue
        shift = "-" if ref is None else \
            f"{math.hypot(o.x_2180 - ref.x_2180, o.y_2180 - ref.y_2180):.2f} " \
            f"(dx {o.x_2180 - ref.x_2180:+.2f}, dy {o.y_2180 - ref.y_2180:+.2f}, dH {o.h - ref.h:+.2f})"
        print(f"{src} | {n_hit} ({n_miss}) | {o.n_views}, {o.rms_m:.2f} / {o.max_m:.2f} | "
              f"{o.x_2180:.2f}, {o.y_2180:.2f} ({o.h:.2f}) | {shift} | "
              f"{'-' if above is None else f'{above:.2f}'} | {dt:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
