"""Geo-Detection-System — CLI.

Przyklady:
    python main.py                               # config.yaml (jesli istnieje) lub domyslne
    python main.py --config config.yaml
    python main.py --image SING0304.JPG --image SING0316.JPG

Wynik: <output_dir>/detections.geojson + <output_dir>/detections_map.html
(wszystkie podane zdjecia w jednym pliku).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from geodetect.config import PipelineConfig
from geodetect.dem import GugikDem
from geodetect.export import export_folium_map, export_geojson
from geodetect.pipeline import build_context, run_single

DEFAULT_CONFIG = Path(__file__).parent / "config.yaml"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Detekcja YOLO + georeferencja ray castingiem na NMT.")
    p.add_argument("--config", type=Path, default=None,
                   help="plik YAML/JSON z konfiguracja (domyslnie config.yaml, jesli istnieje)")
    p.add_argument("--image", action="append", default=None,
                   help="nazwa zdjecia (mozna podac wielokrotnie); nadpisuje 'images' z configu")
    p.add_argument("-v", "--verbose", action="store_true", help="logi DEBUG")
    return p.parse_args()


def load_config(path: Path | None) -> PipelineConfig:
    if path is not None:
        return PipelineConfig.from_file(path)
    if DEFAULT_CONFIG.exists():
        return PipelineConfig.from_file(DEFAULT_CONFIG)
    return PipelineConfig.from_dict({}, base_dir=Path(__file__).parent)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    cfg = load_config(args.config)
    if args.image:
        cfg.images = args.image

    ctx = build_context(cfg)
    results = [run_single(cfg, name, ctx) for name in cfg.images]

    for res in results:
        c = res.camera
        print(f"\n{res.image_name}: kamera EPSG:2180 X={c.x_2180:.2f} Y={c.y_2180:.2f} "
              f"H={c.h:.2f} m npm")
        for gd in res.detections:
            h = gd.hit
            print(f"  {gd.detection.cls_name:8s} conf={gd.detection.conf:.3f} "
                  f"uv=({gd.uv[0]:.0f},{gd.uv[1]:.0f}) -> X={h.x_2180:.2f} Y={h.y_2180:.2f} "
                  f"H={h.h:.2f}  ({h.lat:.7f}N, {h.lon:.7f}E)")
        for det in res.misses:
            print(f"  {det.cls_name:8s} conf={det.conf:.3f} — brak przeciecia z DEM")

    out = cfg.output_dir
    export_geojson(results, out / "detections.geojson")
    export_folium_map(results, out / "detections_map.html")
    if isinstance(ctx.dem, GugikDem):
        logging.info("NMT cache: hits=%d, http_requests=%d", ctx.dem.hits, ctx.dem.misses)


if __name__ == "__main__":
    main()
