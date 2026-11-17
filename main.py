"""Geo-Detection-System — CLI (ta sama funkcja run_pipeline co GUI).

Przyklady:
    python main.py                               # zdjecia z config.yaml (images)
    python main.py --all                         # wszystkie zdjecia z rekonstrukcji ODM
    python main.py --config config.yaml
    python main.py --image SING0304.JPG --image SING0316.JPG
    python main.py --all --surface nmpt_local --eps 10
    python main.py --all --out output/moj_przebieg

Wyniki w <output_dir>/runs/<data_czas>/ (albo --out): run_report.html (raport dla odbiorcy),
run_report.json, map.html, results.gpkg (warstwy objects/detections/cameras/rays),
objects.geojson, detections.geojson, objects.csv, detections.csv, objects.kml, thumbs/, spread.png.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from geodetect.batch import summary_text
from geodetect.config import SURFACE_SOURCES, PipelineConfig
from geodetect.run import Progress, run_pipeline

DEFAULT_CONFIG = Path(__file__).parent / "config.yaml"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Detekcja YOLO + georeferencja ray castingiem na NMT.")
    p.add_argument("--config", type=Path, default=None,
                   help="plik YAML/JSON z konfiguracja (domyslnie config.yaml, jesli istnieje)")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--image", action="append", default=None,
                   help="nazwa zdjecia (mozna podac wielokrotnie); nadpisuje 'images' z configu")
    g.add_argument("--all", action="store_true",
                   help="wszystkie zdjecia z rekonstrukcji ODM")
    p.add_argument("--surface", choices=SURFACE_SOURCES, default=None,
                   help="powierzchnia ray castingu; nadpisuje surface.source z configu")
    p.add_argument("--eps", type=float, default=None,
                   help="promien DBSCAN [m]; nadpisuje aggregate.eps_m z configu")
    p.add_argument("--out", type=Path, default=None,
                   help="katalog wynikow przebiegu (domyslnie <output_dir>/runs/<data_czas>)")
    p.add_argument("-v", "--verbose", action="store_true", help="logi DEBUG")
    return p.parse_args()


def load_config(path: Path | None) -> PipelineConfig:
    if path is not None:
        return PipelineConfig.from_file(path)
    if DEFAULT_CONFIG.exists():
        return PipelineConfig.from_file(DEFAULT_CONFIG)
    return PipelineConfig.from_dict({}, base_dir=Path(__file__).parent)


def print_progress(p: Progress) -> None:
    if p.stage == "images":
        print(f"[{p.i}/{p.n}] {p.message}", flush=True)
    elif p.i == 0 or p.stage == "done":
        print(f"== {p.message}", flush=True)


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    cfg = load_config(args.config)
    if args.image:
        cfg.images = args.image
    if args.surface:
        cfg.surface.source = args.surface
    if args.eps is not None:
        cfg.aggregate.eps_m = args.eps

    res = run_pipeline(cfg, progress_cb=print_progress, all_images=args.all, out_dir=args.out,
                       raise_errors=True)
    if res.cancelled:
        print("Przerwano.")
        return 1
    print("\n" + summary_text(res.batch))
    print(f"\nWyniki ({res.duration_s:.0f} s): {res.out_dir}")
    for k, p in res.export.items():
        print(f"  {k:20s} {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
