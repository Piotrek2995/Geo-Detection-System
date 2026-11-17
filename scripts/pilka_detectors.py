"""Porownanie detektorow pilki: zamkniety slownik (COCO, yolo26l "sports ball") vs otwarty
slownik (YOLOE z promptem tekstowym i wizualnym, YOLO-World z promptem tekstowym).

Bez trenowania na tych zdjeciach. Dla kazdego detektora: najlepszy kandydat na zdjecie (top-1),
trafienie = srodek boxa w promieniu --tol px od punktu z data/pilka_pixels.csv (visible=1).
Zdjecie, z ktorego wzieto prompt wizualny, jest wykluczone z oceny WSZYSTKICH detektorow
(porownanie na tym samym zbiorze).

Wyniki: output/pilka/detectors/{comparison.csv, per_image.csv, summary.md, best.json}
        best.json -> detect.* dla wariantu "detection_best" w experiments/pilka.yaml.

    python scripts/pilka_detectors.py [--config config_pilka.yaml] [--tol 15]
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from geodetect.config import PipelineConfig  # noqa: E402
from geodetect.detect import read_manual_points  # noqa: E402

log = logging.getLogger("pilka_detectors")
TEXT_PROMPT = "soccer ball"


@dataclass
class Candidate:
    image: str
    conf: Optional[float]
    box: Optional[tuple[float, float, float, float]]
    dist_px: Optional[float] = None
    hit: Optional[bool] = None          # None = brak punktu referencyjnego


@dataclass
class DetectorResult:
    name: str
    family: str                         # "zamkniety (COCO)" | "otwarty slownik"
    prompt: str
    weights: str
    n_eval: int = 0
    n_hit: int = 0
    n_none: int = 0                     # brak jakiegokolwiek kandydata
    conf_hit_median: float = float("nan")
    conf_miss_median: float = float("nan")
    ms_per_image: float = float("nan")
    note: str = ""
    candidates: list[Candidate] = field(default_factory=list)


def top1(res) -> tuple[Optional[float], Optional[tuple]]:
    b = res.boxes
    if b is None or len(b) == 0:
        return None, None
    k = int(b.conf.argmax())
    return float(b.conf[k]), tuple(float(v) for v in b.xyxy[k].tolist())


def run_detector(name: str, family: str, prompt: str, weights: str,
                 predict: Callable[[Path], object], images: list[Path]) -> DetectorResult:
    r = DetectorResult(name, family, prompt, weights)
    predict(images[0])                                   # rozgrzewka (bez pomiaru czasu)
    t0 = time.perf_counter()
    for p in images:
        c, box = top1(predict(p))
        r.candidates.append(Candidate(p.name, c, box))
    r.ms_per_image = 1000 * (time.perf_counter() - t0) / len(images)
    return r


def score(r: DetectorResult, gt: dict, tol: float, exclude: set[str]) -> None:
    hits, misses = [], []
    for c in r.candidates:
        g = gt.get(c.image)
        if c.image in exclude or g is None or not g.visible:
            continue
        r.n_eval += 1
        if c.box is None:
            r.n_none += 1
            c.hit = False
            continue
        u, v = (c.box[0] + c.box[2]) / 2, (c.box[1] + c.box[3]) / 2
        c.dist_px = float(np.hypot(u - g.u, v - g.v))
        c.hit = c.dist_px <= tol
        (hits if c.hit else misses).append(c.conf)
    r.n_hit = len(hits)
    r.conf_hit_median = float(np.median(hits)) if hits else float("nan")
    r.conf_miss_median = float(np.median(misses)) if misses else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=ROOT / "config_pilka.yaml")
    ap.add_argument("--tol", type=float, default=15.0, help="tolerancja trafienia [px]")
    ap.add_argument("--vp-image", default="SING0381.JPG", help="zdjecie z boxem prompu wizualnego")
    ap.add_argument("--vp-box", default="output/pilka/detections.csv",
                    help="CSV z boxem pilki (variant bez_kafli) dla --vp-image")
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--out", type=Path, default=ROOT / "output/pilka/detectors")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("ultralytics").setLevel(logging.WARNING)

    from ultralytics import YOLO, YOLOE, YOLOWorld
    from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

    cfg = PipelineConfig.from_file(args.config)
    excl = cfg.excluded_images()
    images = sorted(p for p in cfg.photos_dir.glob("*.JPG") if p.name not in excl)
    gt = read_manual_points(cfg.detect.manual_csv)
    n_gt = sum(1 for g in gt.values() if g.visible)
    log.info("%d zdjec, punktow referencyjnych (visible): %d", len(images), n_gt)

    vp_rows = [r for r in csv.DictReader(open(ROOT / args.vp_box, encoding="utf-8"))
               if r["image"] == args.vp_image and r["variant"] == "bez_kafli" and r["rank"] == "0"]
    if not vp_rows:
        raise SystemExit(f"Brak boxa pilki dla {args.vp_image} w {args.vp_box}")
    vp_box = np.array([[float(vp_rows[0][k]) for k in ("x1", "y1", "x2", "y2")]])
    kw = dict(conf=args.conf, imgsz=args.imgsz, verbose=False)
    models_dir = ROOT / "models"
    results: list[DetectorResult] = []

    # 1) zamkniety slownik: yolo26l COCO, klasa "sports ball"
    m = YOLO(str(cfg.model_path))
    sb = [k for k, v in m.names.items() if v == "sports ball"]
    results.append(run_detector("yolo26l COCO", "zamkniety (COCO)", "klasa 'sports ball'", cfg.model_path.name,
                                lambda p: m.predict(str(p), classes=sb, max_det=300, **kw)[0], images))

    # 2) YOLOE tekst / wizualny, 3) YOLO-World tekst
    for w in ("yoloe-26l-seg.pt", "yoloe-11l-seg.pt"):
        try:
            ye = YOLOE(str(models_dir / w) if (models_dir / w).exists() else w)
            names = [TEXT_PROMPT]
            try:
                ye.set_classes(names, ye.get_text_pe(names))
            except TypeError:
                ye.set_classes(names)
            results.append(run_detector(f"YOLOE ({w.split('-seg')[0]}) tekst", "otwarty slownik",
                                        f"tekst '{TEXT_PROMPT}'", w,
                                        lambda p: ye.predict(str(p), **kw)[0], images))
            yv = YOLOE(str(models_dir / w) if (models_dir / w).exists() else w)
            vp = dict(bboxes=vp_box, cls=np.array([0]))
            ref = str(cfg.photos_dir / args.vp_image)
            results.append(run_detector(f"YOLOE ({w.split('-seg')[0]}) wizualny", "otwarty slownik",
                                        f"box pilki z {args.vp_image}", w,
                                        lambda p: yv.predict(str(p), refer_image=ref, visual_prompts=vp,
                                                             predictor=YOLOEVPSegPredictor, **kw)[0], images))
        except Exception as e:                               # noqa: BLE001
            log.warning("YOLOE %s niedostepny: %s", w, e)
    for w in ("yolov8l-worldv2.pt",):
        try:
            yw = YOLOWorld(str(models_dir / w) if (models_dir / w).exists() else w)
            yw.set_classes([TEXT_PROMPT])
            results.append(run_detector(f"YOLO-World ({w.split('.pt')[0]}) tekst", "otwarty slownik",
                                        f"tekst '{TEXT_PROMPT}'", w,
                                        lambda p: yw.predict(str(p), **kw)[0], images))
        except Exception as e:                               # noqa: BLE001
            log.warning("YOLO-World %s niedostepny: %s", w, e)

    for r in results:
        score(r, gt, args.tol, {args.vp_image})
        log.info("%-28s trafienia %d/%d  conf traf. %.3f  %.0f ms/zdj", r.name, r.n_hit, r.n_eval,
                 r.conf_hit_median, r.ms_per_image)

    args.out.mkdir(parents=True, exist_ok=True)
    flds = ["name", "family", "prompt", "weights", "n_eval", "n_hit", "n_none",
            "conf_hit_median", "conf_miss_median", "ms_per_image"]
    with open(args.out / "comparison.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=flds)
        w.writeheader()
        for r in results:
            w.writerow({k: getattr(r, k) for k in flds})
    with open(args.out / "per_image.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["detector", "image", "conf", "u", "v", "dist_px", "hit"])
        for r in results:
            for c in r.candidates:
                uv = ((c.box[0] + c.box[2]) / 2, (c.box[1] + c.box[3]) / 2) if c.box else ("", "")
                w.writerow([r.name, c.image, c.conf if c.conf is not None else "", *uv,
                            "" if c.dist_px is None else round(c.dist_px, 1),
                            "" if c.hit is None else int(c.hit)])
    best = max(results, key=lambda r: (r.n_hit, -r.n_none, -r.ms_per_image))
    # detekcje najlepszego detektora -> CSV dla detect.source=csv (wariant detection_best);
    # zdjecie promptu wizualnego pominiete (jego "detekcja" bylaby samoodniesieniem)
    best_csv = cfg.detect.detections_csv or ROOT / "data" / "pilka_detections_best.csv"
    best_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(best_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image", "u", "v", "visible", "conf", "x1", "y1", "x2", "y2", "source"])
        for c in best.candidates:
            if c.box is None or (c.image == args.vp_image and "wizualny" in best.name):
                w.writerow([c.image, "", "", 0, "", "", "", "", "",
                            "prompt wizualny" if c.box else "brak kandydata"])
                continue
            w.writerow([c.image, f"{(c.box[0] + c.box[2]) / 2:.1f}", f"{(c.box[1] + c.box[3]) / 2:.1f}", 1,
                        f"{c.conf:.4f}", *(f"{v:.1f}" for v in c.box), best.name])
    log.info("Detekcje najlepszego detektora -> %s", best_csv)
    (args.out / "best.json").write_text(json.dumps({k: v for k, v in asdict(best).items()
                                                     if k != "candidates"}, indent=2, ensure_ascii=False),
                                        encoding="utf-8")
    L = [f"# Porownanie detektorow pilki", "",
         f"Ocena: top-1 na zdjecie, trafienie gdy srodek boxa <= {args.tol:.0f} px od punktu "
         f"z {cfg.detect.manual_csv.name}; wykluczone zdjecie promptu wizualnego {args.vp_image}; "
         f"prog conf {args.conf}, imgsz {args.imgsz}.", "",
         "| detektor | slownik | prompt | trafienia | brak kandydata | conf (trafienia, mediana) | "
         "conf (pudla, mediana) | ms/zdj |", "|---|---|---|---|---|---|---|---|"]
    for r in results:
        L.append(f"| {r.name} | {r.family} | {r.prompt} | {r.n_hit}/{r.n_eval} | {r.n_none} | "
                 f"{r.conf_hit_median:.3f} | {r.conf_miss_median:.3f} | {r.ms_per_image:.0f} |")
    L += ["", f"Najlepszy: **{best.name}** ({best.n_hit}/{best.n_eval})."]
    (args.out / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
