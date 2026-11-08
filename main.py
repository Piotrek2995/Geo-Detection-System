"""Geo-Detection-System — CLI.

Przyklady:
    python main.py                               # zdjecia z config.yaml (images)
    python main.py --all                         # wszystkie zdjecia z rekonstrukcji ODM
    python main.py --config config.yaml
    python main.py --image SING0304.JPG --image SING0316.JPG
    python main.py --all --surface nmpt_local --eps 10

Wyniki w <output_dir>: detections.geojson, objects.geojson, objects.csv,
detections_map.html, spread.png.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from geodetect.batch import run_batch, summary_text
from geodetect.config import SURFACE_SOURCES, PipelineConfig
from geodetect.dem import GugikDem
from geodetect.export import export_all
from geodetect.pipeline import build_context

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
    p.add_argument("-v", "--verbose", action="store_true", help="logi DEBUG")
    return p.parse_args()


def load_config(path: Path | None) -> PipelineConfig:
    if path is not None:
        return PipelineConfig.from_file(path)
    if DEFAULT_CONFIG.exists():
        return PipelineConfig.from_file(DEFAULT_CONFIG)
    return PipelineConfig.from_dict({}, base_dir=Path(__file__).parent)


def print_progress(i: int, n: int, name: str) -> None:
    print(f"[{i}/{n}] {name}", flush=True)


def main() -> None:
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

    ctx = build_context(cfg)
    batch = run_batch(cfg, ctx, images=None if args.all else cfg.images,
                      progress_cb=print_progress)
    paths = export_all(batch, cfg.output_dir)

    print("\n" + summary_text(batch))
    print("\nPliki wynikowe:")
    for p in paths.values():
        print(f"  {p}")
    if isinstance(ctx.dem, GugikDem):
        logging.info("NMT cache: hits=%d, http_requests=%d", ctx.dem.hits, ctx.dem.misses)


if __name__ == "__main__":
    main()
