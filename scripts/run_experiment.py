"""Uruchamia eksperyment(y) z experiments/*.yaml (geodetect/experiments.py).

    python scripts/run_experiment.py experiments/surface.yaml
    python scripts/run_experiment.py experiments/*.yaml --images SING0304.JPG SING0316.JPG
    python scripts/run_experiment.py experiments/detect.yaml --out output/experiments_test

Wynik: output/experiments/<nazwa>/<data_czas>/summary.md (+ CSV, wykresy, meta.json).
Opis eksperymentow i metryk: docs/experiments.md.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from geodetect.experiments import DetectionStore, ExperimentRunner, load_experiment  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiments", nargs="+", type=Path, help="pliki experiments/*.yaml")
    ap.add_argument("--out", type=Path, default=None, help="katalog nadrzedny (domyslnie output/experiments)")
    ap.add_argument("--images", nargs="+", default=None, help="tylko te zdjecia (domyslnie wszystkie)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    store = DetectionStore()     # detekcje wspolne takze miedzy plikami eksperymentow
    for path in args.experiments:
        exp = load_experiment(path)
        out = ExperimentRunner(exp, args.out, args.images, store=store).run()
        print(f"[{exp.name}] {len(exp.variants)} wariantow -> {out / 'summary.md'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
