"""Ocena wariantow detekcji (zadanie 6): bez kafli / kafle / kafle + pelny obraz, 2 progi conf.

Dla kazdego wariantu i progu: detekcje i obiekty per klasa (confirmed / single_view),
rozklad conf, RMS glownego namiotu (powierzchnia nmt_local, eps 10 m), czas na zdjecie.
Jesli podano --gt (eksport Roboflow z etykietami tych zdjec), liczy precision/recall
per klasa (IoU 0.5; dla people takze 0.3 — male boxy). Detekcje wystepujace tylko
w jednym wariancie -> podglady w output/detect/ (z boxami GT, jesli sa).
Pomiar czasu: fp32/fp16, z kaflami i bez, batch 1 vs cfg.batch.

    python scripts/detect_compare.py --gt C:/.../28sierpniaMODEL5
"""

from __future__ import annotations

import argparse
import copy
import logging
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from geodetect.batch import run_batch  # noqa: E402
from geodetect.config import PipelineConfig  # noqa: E402
from geodetect.detect import Detector  # noqa: E402
from geodetect.diagnostics import static_objects  # noqa: E402
from geodetect.gt_labels import load_ground_truth, match_detections  # noqa: E402
from geodetect.pipeline import build_context  # noqa: E402
from geodetect.tiling import ios, iou  # noqa: E402

VARIANTS = {"bez_kafli": (False, False, None), "kafle": (True, False, None),
            "kafle+pelny": (True, True, None),
            "kafle(people)+pelny": (True, True, ["people"])}


class FixedDetector:
    """Detekcje policzone wczesniej (per nazwa zdjecia), interfejs Detector."""

    def __init__(self, dets: dict, names: dict):
        self.dets, self.names = dets, names

    def detect(self, image_path):
        return self.dets.get(Path(image_path).name, [])


def make_cfg(base: PipelineConfig, tiles: bool, full: bool, half=False, batch=None,
             tile_classes=None):
    c = copy.deepcopy(base.detect)
    c.tiling.enabled, c.tiling.include_full, c.half = tiles, full, half
    c.tiling.classes = tile_classes
    c.conf = min(c.conf, 0.25)
    c.conf_per_class = {}
    if batch is not None:
        c.batch = batch
    return c


def run_variant(base, images, tiles, full, half=False, batch=None, tile_classes=None):
    det = Detector(base.model_path, make_cfg(base, tiles, full, half, batch, tile_classes))
    det.detect(base.photos_dir / images[0])             # rozgrzewka (CUDA, cudnn)
    det.infer_s, det.n_images = 0.0, 0
    t0 = time.perf_counter()
    if tiles:
        res = [det.detect(base.photos_dir / n) for n in images]
    else:
        res = det.detect_many([base.photos_dir / n for n in images])
    wall = time.perf_counter() - t0
    return det, dict(zip(images, res)), wall / len(images)


def prf(dets_by_img, gt, thr, iou_thr, cls):
    tp = fp = fn = 0
    for name, g in gt.items():
        d = [x for x in dets_by_img.get(name, []) if x.conf >= thr and x.cls_name == cls]
        pairs, missed = match_detections(d, [b for b in g.boxes if b.cls_name == cls], iou_thr)
        tp += sum(1 for _, m, _ in pairs if m is not None)
        fp += sum(1 for _, m, _ in pairs if m is None)
        fn += len(missed)
    p = tp / (tp + fp) if tp + fp else float("nan")
    r = tp / (tp + fn) if tp + fn else float("nan")
    return tp, fp, fn, p, r


def georef(base, dets_by_img, names, thr):
    cfg = copy.deepcopy(base)
    cfg.surface.source, cfg.aggregate.eps_m = "nmt_local", 10.0
    filtered = {k: [d for d in v if d.conf >= thr] for k, v in dets_by_img.items()}
    ctx = build_context(cfg, detector=FixedDetector(filtered, names))
    batch = run_batch(cfg, ctx, images=None)
    tents = static_objects(batch, 10.0, "namiot", min_views=2)
    # static_objects nadpisuje object_id -> przywracamy agregacje z configu (min_views/status)
    from geodetect.aggregate import aggregate
    batch.objects = aggregate(batch.detections, cfg.aggregate)
    main = max(tents, key=lambda o: o.n_views) if tents else None
    return batch, main


def conf_stats(dets_by_img, cls, thr):
    c = np.array([d.conf for v in dets_by_img.values() for d in v
                  if d.cls_name == cls and d.conf >= thr])
    if not len(c):
        return "-"
    return (f"n={len(c)}, min {c.min():.2f}, med {np.median(c):.2f}, max {c.max():.2f}, "
            f"<0.5: {int((c < 0.5).sum())}")


def _has_match(d, others, thr):
    return any(o.cls_name == d.cls_name and o.conf >= thr
               and (iou(o.bbox, d.bbox) >= 0.3 or ios(o.bbox, d.bbox) >= 0.6) for o in others)


def unique_detections(all_dets: dict, thr: float, ref: str = "bez_kafli"):
    """Roznice wzgledem wariantu `ref`: (wariant(y), zdjecie, detekcja).
    - detekcje z wariantow kaflowych bez odpowiednika w `ref` (wspolne dla obu wariantow
      kaflowych liczone raz, z lista wariantow),
    - detekcje z `ref`, ktorych nie ma w zadnym wariancie kaflowym."""
    others = [v for v in all_dets if v != ref]
    out = []
    for name in sorted({n for d in all_dets.values() for n in d}):
        seen = []
        for v in others:
            for d in all_dets[v].get(name, []):
                if d.conf < thr or _has_match(d, all_dets[ref].get(name, []), thr):
                    continue
                dup = next((s for s in seen if s[2].cls_name == d.cls_name
                            and iou(s[2].bbox, d.bbox) >= 0.3), None)
                if dup is not None:
                    dup[0].append(v)
                else:
                    seen.append(([v], name, d))
        out += [("+".join(vs) if len(vs) == 1 else " i ".join(vs), n, d) for vs, n, d in seen]
        for d in all_dets[ref].get(name, []):
            if d.conf >= thr and not any(_has_match(d, all_dets[v].get(name, []), thr)
                                         for v in others):
                out.append((f"tylko {ref}", name, d))
    return out


def save_preview(photos_dir, out_dir, k, variant, name, d, gt):
    img = cv2.imread(str(photos_dir / name))
    x1, y1, x2, y2 = d.bbox
    s = max(x2 - x1, y2 - y1, 40)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    c0, r0 = max(0, int(cx - 3 * s)), max(0, int(cy - 3 * s))
    c1, r1 = min(img.shape[1], int(cx + 3 * s)), min(img.shape[0], int(cy + 3 * s))
    crop = img[r0:r1, c0:c1].copy()
    sc = 420.0 / max(crop.shape[:2])
    crop = cv2.resize(crop, None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC)

    def P(u, v):
        return int((u - c0) * sc), int((v - r0) * sc)
    status = "brak GT"
    if gt is not None:
        for b in gt.boxes:
            cv2.rectangle(crop, P(*b.bbox[:2]), P(*b.bbox[2:]), (122, 175, 27), 1)
        pairs, _ = match_detections([d], gt.boxes, 0.3)
        status = "zgodna z GT (TP)" if pairs[0][1] is not None else "brak w GT (FP?)"
    cv2.rectangle(crop, P(x1, y1), P(x2, y2), (52, 104, 235), 2)
    panel = np.full((crop.shape[0], 260, 3), 252, np.uint8)
    lines = [f"{name}", f"wariant: {variant}", f"{d.cls_name} conf {d.conf:.2f}",
             f"box {x2 - x1:.0f}x{y2 - y1:.0f} px, zrodlo {d.source}",
             f"GT: {status}", "pomaranczowy = detekcja", "zielony = etykieta GT"]
    for i, t in enumerate(lines):
        cv2.putText(panel, t, (8, 22 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (11, 11, 11), 1,
                    cv2.LINE_AA)
    p = out_dir / f"unique_{k:02d}_{d.cls_name}_{Path(name).stem}.png"
    cv2.imwrite(str(p), np.hstack([crop, panel]))
    return p, status


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    ap.add_argument("--gt", type=Path, default=None, help="folder eksportu Roboflow (train/valid/test)")
    ap.add_argument("--thresholds", nargs="+", type=float, default=[0.25, 0.40, 0.50])
    ap.add_argument("--out", type=Path, default=ROOT / "output" / "detect")
    ap.add_argument("--no-timing", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    base = PipelineConfig.from_file(args.config)
    rec_images = build_context(base, detector=FixedDetector({}, {})).reconstruction.image_names
    images = [n for n in rec_images if (base.photos_dir / n).exists()]
    all_dets, per_img_s, names = {}, {}, None
    for v, (tiles, full, tcls) in VARIANTS.items():
        det, dets, t = run_variant(base, images, tiles, full, tile_classes=tcls)
        all_dets[v], per_img_s[v], names = dets, t, det.names
        print(f"[{v}] {sum(map(len, dets.values()))} detekcji, {t * 1000:.0f} ms/zdjecie", flush=True)
    classes = list(names.values())

    gt = {}
    if args.gt:
        gt = load_ground_truth(args.gt, images, base.photos_dir, classes)
        print(f"\nGT: {len(gt)}/{len(images)} zdjec z etykietami z {args.gt} "
              f"(splity: {sorted({g.split for g in gt.values()})})")

    print("\nwariant | prog | detekcje per klasa | obiekty confirmed/single_view per klasa | "
          "glowny namiot: widoki, RMS / max [m] | ms/zdjecie")
    for v in VARIANTS:
        for thr in args.thresholds:
            batch, main = georef(base, all_dets[v], names, thr)
            nd = {c: sum(1 for d in batch.detections if d.detection.cls_name == c) for c in classes}
            no = {c: (sum(1 for o in batch.objects if o.cls_name == c and o.confirmed),
                      sum(1 for o in batch.objects if o.cls_name == c and not o.confirmed))
                  for c in classes}
            print(f"{v} | {thr:.2f} | " + ", ".join(f"{c} {nd[c]}" for c in classes) + " | "
                  + ", ".join(f"{c} {a}/{b}" for c, (a, b) in no.items()) + " | "
                  + ("-" if main is None else f"{main.n_views}, {main.rms_m:.2f} / {main.max_m:.2f}")
                  + f" | {per_img_s[v] * 1000:.0f}")

    print("\nRozklad conf (prog 0.25):")
    for v in VARIANTS:
        for c in classes:
            print(f"  {v:12} {c:7} {conf_stats(all_dets[v], c, 0.25)}")

    if gt:
        print(f"\nPrecision / recall na {len(gt)} zdjeciach z GT "
              "(TP/FP/FN, P, R; IoU 0.5, people takze 0.3):")
        for v in VARIANTS:
            for thr in args.thresholds:
                parts = []
                for c in classes:
                    for it in ((0.5, 0.3) if c == "people" else (0.5,)):
                        tp, fp, fn, p, r = prf(all_dets[v], gt, thr, it, c)
                        parts.append(f"{c}@{it}: {tp}/{fp}/{fn} P {p:.2f} R {r:.2f}")
                print(f"  {v:12} conf>={thr:.2f}  " + " | ".join(parts))

    args.out.mkdir(parents=True, exist_ok=True)
    for p in args.out.glob("unique_*.png"):
        p.unlink()
    uniq = unique_detections(all_dets, min(args.thresholds))
    print(f"\nDetekcje rozne wzgledem bez_kafli (conf>={min(args.thresholds):.2f}): {len(uniq)} "
          f"(podglady w {args.out})")
    for k, (v, name, d) in enumerate(uniq, start=1):
        p, status = save_preview(base.photos_dir, args.out, k, v, name, d, gt.get(name))
        print(f"  {p.name} [{v}]: {d.cls_name} conf {d.conf:.2f}, {d.bbox[2] - d.bbox[0]:.0f}x"
              f"{d.bbox[3] - d.bbox[1]:.0f} px — {status}")

    if not args.no_timing:
        print("\nCzas inferencji [ms/zdjecie] (26 zdjec, po rozgrzewce):")
        for label, tiles, full, half, batch in [
                ("bez kafli, fp32, batch 1", False, False, False, 1),
                ("bez kafli, fp32, batch 8", False, False, False, 8),
                ("bez kafli, fp16, batch 8", False, False, True, 8),
                ("kafle, fp32", True, False, False, 8),
                ("kafle, fp16", True, False, True, 8),
                ("kafle+pelny, fp32", True, True, False, 8),
                ("kafle+pelny, fp16", True, True, True, 8),
                ("kafle(people)+pelny, fp16", True, True, True, 8)]:
            det, dets, t = run_variant(base, images, tiles, full, half, batch,
                                       ["people"] if "(people)" in label else None)
            n = sum(map(len, dets.values()))
            print(f"  {label:26} {t * 1000:6.0f} (model {det.infer_s / len(images) * 1000:.0f}), "
                  f"{n} detekcji")
    return 0


if __name__ == "__main__":
    sys.exit(main())
