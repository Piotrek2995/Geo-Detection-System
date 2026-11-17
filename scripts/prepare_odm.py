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
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from geodetect.batch import PHOTO_SUFFIXES  # noqa: E402
from geodetect.config import PipelineConfig  # noqa: E402

log = logging.getLogger("prepare_odm")

ODM_FLAGS = "--dtm --dsm --pc-classify --dem-resolution 2 --orthophoto-resolution 1"


@dataclass
class PrepareResult:
    images_dir: Path
    copied: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    excluded: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)


def prepare_odm(cfg: PipelineConfig, dry_run: bool = False) -> PrepareResult:
    excluded = cfg.excluded_images()
    dst = cfg.odm_dir / "images"
    res = PrepareResult(dst)
    if not cfg.photos_dir.is_dir():
        raise FileNotFoundError(f"Brak katalogu zdjec {cfg.photos_dir}")
    photos = sorted(p for p in cfg.photos_dir.iterdir() if p.suffix.lower() in PHOTO_SUFFIXES)
    if not dry_run:
        dst.mkdir(parents=True, exist_ok=True)
    for p in photos:
        if p.name in excluded:
            res.excluded.append(p.name)
            continue
        target = dst / p.name
        if target.exists() and target.stat().st_size == p.stat().st_size:
            res.unchanged.append(p.name)
            continue
        if not dry_run:
            shutil.copy2(p, target)
        res.copied.append(p.name)
    for name in sorted(excluded):
        if (dst / name).exists():
            if not dry_run:
                (dst / name).unlink()
            res.removed.append(name)
    missing = excluded - {p.name for p in photos}
    if missing:
        log.warning("Zdjecie startowe %s nie istnieje w %s", sorted(missing), cfg.photos_dir)
    log.info("ODM images: %s — skopiowane %d, bez zmian %d, wykluczone %s, usuniete %s",
             dst, len(res.copied), len(res.unchanged), res.excluded, res.removed)
    return res


def docker_command(cfg: PipelineConfig) -> str:
    rel = cfg.odm_dir.relative_to(ROOT) if cfg.odm_dir.is_relative_to(ROOT) else cfg.odm_dir
    return (f'docker run --rm -v "${{PWD}}\\{rel}:/datasets/code" opendronemap/odm '
            f"--project-path /datasets {ODM_FLAGS}")


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
