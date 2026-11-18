"""Przygotowuje katalog projektu ODM: kopiuje zdjecia z photos_dir do <odm_dir>/images
z pominieciem zdjec wykluczonych (zdjecie startowe z ziemi: z_anchor.start.image).
Zdjecie wykluczone, ktore zostalo w images/ z wczesniejszego kopiowania, jest usuwane,
bo ODM uzylby go w rekonstrukcji. Na koniec wypisuje komende docker dla ODM.

    python scripts/prepare_odm.py --config config_pilka.yaml
    python scripts/prepare_odm.py --config config_pilka.yaml --dry-run
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from geodetect.config import PipelineConfig  # noqa: E402
from geodetect.odmrun import odm_command, prepare_odm  # noqa: E402

log = logging.getLogger("prepare_odm")


def docker_command(cfg: PipelineConfig) -> str:
    return " ".join(f'"{a}"' if " " in a else a for a in odm_command(cfg.odm_dir))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = PipelineConfig.from_file(args.config)
    r = prepare_odm(cfg, args.dry_run)
    n = len(r.copied) + len(r.unchanged)
    print(f"{r.images_dir}: {n} zdjec do rekonstrukcji (skopiowane {len(r.copied)}), "
          f"wykluczone: {', '.join(r.excluded) or '-'}")
    print("\nODM (PowerShell, z katalogu repo):\n  " + docker_command(cfg))
    return 0


if __name__ == "__main__":
    sys.exit(main())
