"""Sprawdza, czy lokalne arkusze NMT/NMPT pokrywaja obszar lotu (geodetect/coverage.py).

Obszar = bbox kamer z rekonstrukcji ODM (EPSG:2180) + bufor (domyslnie 100 m),
rozszerzony o zasieg DSM z ODM (jesli jest). Dla kazdego folderu (surface.nmt_dir,
surface.nmpt_dir): pliki, rozdzielczosc, zasieg mozaiki, czy obszar jest w calosci
w zasiegu, odsetek pikseli bez danych.

    python scripts/check_surface_coverage.py [--config config.yaml] [--buffer 100]
Kod wyjscia: 0 = oba foldery pokrywaja obszar bez dziur, 1 = brak/niepelne pokrycie.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from geodetect.config import PipelineConfig  # noqa: E402
from geodetect.coverage import check_folder, flight_bounds  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--buffer", type=float, default=100.0, help="bufor wokol obszaru lotu [m]")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    cfg = PipelineConfig.from_file(args.config)
    b = flight_bounds(cfg, args.buffer)
    print(f"Obszar lotu EPSG:2180 (+{args.buffer:g} m): xmin={b[0]:.0f} ymin={b[1]:.0f} "
          f"xmax={b[2]:.0f} ymax={b[3]:.0f} ({b[2] - b[0]:.0f} x {b[3] - b[1]:.0f} m)")
    ok = []
    for lbl, d in (("NMT", cfg.surface.nmt_dir), ("NMPT", cfg.surface.nmpt_dir)):
        r = check_folder(lbl, d, b, cfg.surface.default_crs)
        print(f"\n[{lbl}] {d}")
        for f in r.files:
            print(f"  plik: {f}")
        if r.resolution_m:
            print(f"  rozdzielczosc: {r.resolution_m:g} m")
        print(f"  {'OK' if r.ok else 'PROBLEM'}: {r.message}")
        ok.append(r.ok)
    print("\nWynik:", "OK" if all(ok) else "pokrycie niepelne / brak plikow")
    return 0 if all(ok) else 1


if __name__ == "__main__":
    sys.exit(main())
