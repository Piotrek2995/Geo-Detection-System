"""Sprawdza, czy lokalne arkusze NMT/NMPT pokrywaja obszar lotu.

Obszar = bbox kamer z rekonstrukcji ODM (EPSG:2180) + bufor (domyslnie 100 m),
rozszerzony o zasieg DEM z ODM (jesli jest). Dla kazdego folderu (surface.nmt_dir,
surface.nmpt_dir) raportuje: pliki, CRS, rozdzielczosc, zasieg mozaiki, czy obszar jest
w calosci w zasiegu, odsetek pikseli bez danych (dziury) i bbox dziur.

    python scripts/check_surface_coverage.py [--config config.yaml] [--buffer 100]
Kod wyjscia: 0 = oba foldery pokrywaja obszar bez dziur, 1 = brak/niepelne pokrycie.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from geodetect.config import PipelineConfig  # noqa: E402
from geodetect.geo import EnuFrame  # noqa: E402
from geodetect.odm import OdmReconstruction  # noqa: E402
from geodetect.surface import _bounds_in, _read_crs, load_mosaic, raster_files  # noqa: E402

log = logging.getLogger("check_surface_coverage")


def flight_bounds(cfg: PipelineConfig, buffer_m: float) -> tuple[float, float, float, float]:
    rec = OdmReconstruction(cfg.reconstruction_path, cfg.reference_lla_path)
    frame = EnuFrame(rec.ref)
    xy = np.array([frame.enu_to_puwg92(*rec.camera(n)[1].C_enu)[:2] for n in rec.image_names],
                  dtype=float)
    xmin, ymin = xy.min(axis=0)
    xmax, ymax = xy.max(axis=0)
    dsm = cfg.odm_dir / "odm_dem" / "dsm.tif"
    if dsm.exists():
        import rasterio
        with rasterio.open(dsm) as ds:
            from pyproj import Transformer
            b = Transformer.from_crs(ds.crs.to_wkt(), "EPSG:2180", always_xy=True) \
                .transform_bounds(*ds.bounds)
        xmin, ymin = min(xmin, b[0]), min(ymin, b[1])
        xmax, ymax = max(xmax, b[2]), max(ymax, b[3])
    return xmin - buffer_m, ymin - buffer_m, xmax + buffer_m, ymax + buffer_m


def check_folder(label: str, folder: Path, bounds_2180, default_crs: str) -> bool:
    print(f"\n[{label}] {folder}")
    try:
        files = raster_files(folder)
    except FileNotFoundError as exc:
        print(f"  BRAK: {exc}")
        return False
    for f in files:
        print(f"  plik: {f.name}")
    crs = _read_crs(files[0], default_crs)
    b = _bounds_in(crs, bounds_2180)
    full = load_mosaic(files, default_crs)
    fb = full.bounds
    print(f"  rozdzielczosc: {full.res_x:g} x {full.res_y:g} m")
    print(f"  zasieg mozaiki: x {fb[0]:.0f}..{fb[2]:.0f}, y {fb[1]:.0f}..{fb[3]:.0f}")
    inside = fb[0] <= b[0] and fb[1] <= b[1] and fb[2] >= b[2] and fb[3] >= b[3]
    if not inside:
        print("  NIEPELNE: obszar wychodzi poza arkusze o "
              f"W {max(0, fb[0] - b[0]):.0f} m, S {max(0, fb[1] - b[1]):.0f} m, "
              f"E {max(0, b[2] - fb[2]):.0f} m, N {max(0, b[3] - fb[3]):.0f} m")
    try:
        g = load_mosaic(files, default_crs, b)
    except ValueError as exc:
        print(f"  BRAK POKRYCIA: {exc}")
        return False
    # piksele obszaru poza mozaika tez sa dziura
    h = int(round((b[3] - b[1]) / g.res_y))
    w = int(round((b[2] - b[0]) / g.res_x))
    holes = ~np.isfinite(g.z)
    n_nan, n_all = int(holes.sum()), g.z.size
    frac = n_nan / n_all if n_all else 1.0
    print(f"  wyciety obszar: {g.z.shape[1]}x{g.z.shape[0]} px (oczekiwane ~{w}x{h}), "
          f"bez danych: {n_nan} px ({100 * frac:.2f}%)")
    if n_nan:
        rows, cols = np.nonzero(holes)
        x0 = g.x0 + cols.min() * g.res_x
        x1 = g.x0 + (cols.max() + 1) * g.res_x
        y1 = g.y0 - rows.min() * g.res_y
        y0 = g.y0 - (rows.max() + 1) * g.res_y
        print(f"  dziury w bbox: x {x0:.0f}..{x1:.0f}, y {y0:.0f}..{y1:.0f}")
    return inside and n_nan == 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--buffer", type=float, default=100.0, help="bufor wokol obszaru lotu [m]")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    cfg = PipelineConfig.from_file(args.config)
    b = flight_bounds(cfg, args.buffer)
    print(f"Obszar lotu EPSG:2180 (+{args.buffer:g} m): "
          f"xmin={b[0]:.0f} ymin={b[1]:.0f} xmax={b[2]:.0f} ymax={b[3]:.0f} "
          f"({b[2] - b[0]:.0f} x {b[3] - b[1]:.0f} m)")
    ok = [check_folder(lbl, d, b, cfg.surface.default_crs)
          for lbl, d in (("NMT", cfg.surface.nmt_dir), ("NMPT", cfg.surface.nmpt_dir))]
    print("\nWynik:", "OK" if all(ok) else "pokrycie niepelne / brak plikow")
    return 0 if all(ok) else 1


if __name__ == "__main__":
    sys.exit(main())
