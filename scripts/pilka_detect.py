"""Detekcja pilki (eksperyment walidacyjny, docs/eksperyment_pilka.md).

1. Detekcja klasy z config_pilka.yaml (sports ball, yolo26l COCO) w dwoch wariantach:
   bez kafli i kafle + pelny obraz. Podglady wszystkich detekcji:
   output/pilka/detections/<wariant>/.
2. Wstepny plik punktow recznych data/pilka_pixels.csv (image,u,v,visible,source,conf):
   srodek najpewniejszej detekcji wariantu --prefill-variant (domyslnie bez kafli),
   puste pola dla zdjec bez detekcji.
   Istniejacy plik NIE jest nadpisywany (moze zawierac poprawki reczne) — chyba ze --force.
3. Podglady punktu do recznej weryfikacji: output/pilka/pixels/ (powiekszenie wokol
   punktu + miniatura calego kadru; dla zdjec bez punktu sama miniatura z siatka pikseli).
   Opcjonalnie (--project-gt, po ODM): rzut punktu GT pilki na zdjecie jako podpowiedz.

    python scripts/pilka_detect.py
    python scripts/pilka_detect.py --project-gt        # po ODM: + rzut GT na zdjecia
    python scripts/pilka_detect.py --previews-only     # tylko podglady z istniejacego CSV
"""

from __future__ import annotations

import argparse
import copy
import csv
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from geodetect.batch import PHOTO_SUFFIXES  # noqa: E402
from geodetect.config import PipelineConfig  # noqa: E402
from geodetect.detect import Detector, read_manual_points  # noqa: E402

log = logging.getLogger("pilka_detect")
VARIANTS = {"bez_kafli": False, "kafle+pelny": True}
ORANGE, GREEN, BLUE = (52, 104, 235), (0, 131, 0), (214, 120, 42)


def project_points(cfg: PipelineConfig, latlon_h: tuple[float, float, float]) -> dict[str, tuple]:
    """Rzut punktu (lat, lon, H npm) na kazde zarejestrowane zdjecie (u, v) — po ODM."""
    from geodetect.pipeline import build_context
    from geodetect.detect import ManualDetector
    c = copy.deepcopy(cfg)
    c.detect.source = "manual"
    ctx = build_context(c, detector=ManualDetector(c.detect))
    fr = ctx.frame
    lat, lon, H = latlon_h
    # H npm -> h ODM: odwrotnosc npm_height (h + z_offset [+ tilt])
    x, y = fr.lonlat_to_puwg92(lon, lat)
    h = H - (float(fr.npm_height(0.0, x, y)))
    e, n, u = fr.geodetic_to_enu(lon, lat, h)
    P = np.array([e, n, u], dtype=np.float64)
    out = {}
    for name in ctx.reconstruction.image_names:
        io, eo = ctx.reconstruction.camera(name)
        pc = eo.R_wc @ (P - eo.C_enu)
        if pc[2] <= 0:
            continue
        uv, _ = cv2.projectPoints(pc.reshape(1, 1, 3), np.zeros(3), np.zeros(3), io.K, io.dist)
        uu, vv = uv[0, 0]
        if 0 <= uu < io.width and 0 <= vv < io.height:
            out[name] = (float(uu), float(vv))
    return out


def detection_preview(img, det, out: Path, title: str) -> None:
    x1, y1, x2, y2 = det.bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    r = max(60, int(3 * max(x2 - x1, y2 - y1)))
    H, W = img.shape[:2]
    a, b = int(max(0, cx - r)), int(max(0, cy - r))
    crop = img[b:int(min(H, cy + r)), a:int(min(W, cx + r))].copy()
    cv2.rectangle(crop, (int(x1 - a), int(y1 - b)), (int(x2 - a), int(y2 - b)), ORANGE, 1)
    crop = cv2.resize(crop, None, fx=400 / crop.shape[1], fy=400 / crop.shape[1],
                      interpolation=cv2.INTER_NEAREST)
    thumb = cv2.resize(img, (711, 400), interpolation=cv2.INTER_AREA)
    s = 711 / W
    cv2.rectangle(thumb, (int(x1 * s) - 6, int(y1 * s) - 6), (int(x2 * s) + 6, int(y2 * s) + 6), ORANGE, 2)
    canvas = np.full((max(crop.shape[0], 400) + 30, crop.shape[1] + 711 + 10, 3), 252, np.uint8)
    canvas[30:30 + crop.shape[0], :crop.shape[1]] = crop
    canvas[30:430, crop.shape[1] + 10:] = thumb
    cv2.putText(canvas, title, (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (11, 11, 11), 1, cv2.LINE_AA)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), canvas)


def point_preview(img, uv, gt_uv, out: Path, title: str) -> None:
    """Powiekszenie 8x wokol punktu (siatka co 10 px z opisem wspolrzednych) + miniatura."""
    H, W = img.shape[:2]
    thumb = cv2.resize(img, (1280, 720), interpolation=cv2.INTER_AREA)
    s = 1280 / W
    for k in range(0, W, 200):                       # siatka do odczytu wspolrzednych
        cv2.line(thumb, (int(k * s), 0), (int(k * s), 719), (200, 200, 200), 1)
        cv2.putText(thumb, str(k), (int(k * s) + 2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    for k in range(0, H, 200):
        cv2.line(thumb, (0, int(k * s)), (1279, int(k * s)), (200, 200, 200), 1)
        cv2.putText(thumb, str(k), (2, int(k * s) + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    panels = []
    for p, col in ((uv, ORANGE), (gt_uv, GREEN)):
        if p is None:
            continue
        cv2.circle(thumb, (int(p[0] * s), int(p[1] * s)), 12, col, 2)
    center = uv or gt_uv
    if center is not None:
        r = 40
        a, b = int(np.clip(center[0] - r, 0, W - 2 * r)), int(np.clip(center[1] - r, 0, H - 2 * r))
        crop = cv2.resize(img[b:b + 2 * r, a:a + 2 * r], (640, 640), interpolation=cv2.INTER_NEAREST)
        for k in range(0, 2 * r + 1, 10):
            cv2.line(crop, (k * 8, 0), (k * 8, 639), (180, 180, 180), 1)
            cv2.line(crop, (0, k * 8), (639, k * 8), (180, 180, 180), 1)
            cv2.putText(crop, str(a + k), (k * 8 + 2, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            cv2.putText(crop, str(b + k), (2, k * 8 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        for p, col in ((uv, ORANGE), (gt_uv, GREEN)):
            if p is not None:
                q = (int((p[0] - a) * 8), int((p[1] - b) * 8))
                cv2.drawMarker(crop, q, col, cv2.MARKER_CROSS, 24, 2)
        panels.append(crop)
    h = 720 + 30
    canvas = np.full((h, 1280 + (650 if panels else 0), 3), 252, np.uint8)
    canvas[30:750, :1280] = thumb
    if panels:
        canvas[30:670, 1290:1930] = panels[0]
    cv2.putText(canvas, title, (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (11, 11, 11), 1, cv2.LINE_AA)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), canvas)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=ROOT / "config_pilka.yaml")
    ap.add_argument("--out", type=Path, default=ROOT / "output" / "pilka")
    ap.add_argument("--force", action="store_true", help="nadpisz data/pilka_pixels.csv")
    ap.add_argument("--previews-only", action="store_true")
    ap.add_argument("--prefill-variant", default="bez_kafli", choices=list(VARIANTS),
                    help="z ktorego wariantu brac wstepne punkty do CSV")
    ap.add_argument("--project-gt", action="store_true", help="rzut GT pilki (wymaga ODM)")
    ap.add_argument("--gt", nargs=3, type=float, default=None, metavar=("LAT", "LON", "H"),
                    help="punkt GT do rzutu (H m npm); domyslnie z experiments/pilka.yaml")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    cfg = PipelineConfig.from_file(args.config)
    excluded = cfg.excluded_images()
    images = sorted(p for p in cfg.photos_dir.iterdir()
                    if p.suffix.lower() in PHOTO_SUFFIXES and p.name not in excluded)
    csv_path = cfg.detect.manual_csv

    best: dict[str, tuple] = {}
    if not args.previews_only:
        rows = []
        for v, tiles in VARIANTS.items():
            dc = copy.deepcopy(cfg.detect)
            dc.tiling.enabled = tiles
            det = Detector(cfg.model_path, dc)
            vdir = args.out / "detections" / v
            for p in vdir.glob("*.png"):
                p.unlink()
            for img_path in images:
                dets = det.detect(img_path)
                img = cv2.imread(str(img_path))
                for k, d in enumerate(sorted(dets, key=lambda d: -d.conf)):
                    w, h = d.bbox[2] - d.bbox[0], d.bbox[3] - d.bbox[1]
                    rows.append([v, img_path.name, k, f"{d.conf:.3f}", *(f"{x:.1f}" for x in d.bbox),
                                 f"{w:.0f}", f"{h:.0f}", d.source])
                    detection_preview(img, d, vdir / f"{img_path.stem}_{k}_{d.conf:.2f}.png",
                                      f"{img_path.name} [{v}] {d.cls_name} conf {d.conf:.2f}, "
                                      f"{w:.0f}x{h:.0f} px, zrodlo {d.source}")
                    cur = best.get(img_path.name)
                    if v == args.prefill_variant and (cur is None or d.conf > cur[2]):
                        best[img_path.name] = (*d.center, d.conf, v)
            n = sum(1 for r in rows if r[0] == v)
            print(f"[{v}] {n} detekcji na {len({r[1] for r in rows if r[0] == v})}/{len(images)} zdjeciach "
                  f"(model {det.infer_s:.1f} s)")
        args.out.mkdir(parents=True, exist_ok=True)
        with open(args.out / "detections.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["variant", "image", "rank", "conf", "x1", "y1", "x2", "y2", "w_px", "h_px", "source"])
            w.writerows(rows)
        if csv_path.exists() and not args.force:
            print(f"{csv_path} istnieje — nie nadpisuje (uzyj --force)")
        else:
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["image", "u", "v", "visible", "source", "conf"])
                for p in images:
                    b = best.get(p.name)
                    w.writerow([p.name, f"{b[0]:.1f}", f"{b[1]:.1f}", 1, f"detekcja:{b[3]}", f"{b[2]:.3f}"]
                               if b else [p.name, "", "", "", "", ""])
            print(f"-> {csv_path} ({len(best)}/{len(images)} zdjec z punktem z detekcji)")

    gt_uv = {}
    if args.project_gt:
        gt = args.gt
        if gt is None:
            import yaml
            spec = yaml.safe_load((ROOT / "experiments" / "pilka.yaml").read_text(encoding="utf-8"))
            ref = spec["metrics"]["reference_point"]
            h = ref.get("preview_h")
            if h is None:                     # H dachu z NMPT w punkcie GT + srodek pilki
                from pyproj import Transformer
                from geodetect.surface import load_mosaic, raster_files, sample_grid
                x, y = Transformer.from_crs(4326, 2180, always_xy=True).transform(ref["latlon"][1],
                                                                                  ref["latlon"][0])
                g = load_mosaic(raster_files(cfg.surface.nmpt_dir), "EPSG:2180", (x - 5, y - 5, x + 5, y + 5))
                h = float(sample_grid(g, np.array([x]), np.array([y]))[0]) + float(ref.get("object_height_m", 0))
            gt = (*ref["latlon"], h)
        gt_uv = project_points(cfg, tuple(gt))
        print(f"Rzut GT (H={gt[2]:.2f} m npm) na {len(gt_uv)} zdjec")

    pts = read_manual_points(csv_path) if csv_path.exists() else {}
    pdir = args.out / "pixels"
    for p in pdir.glob("*.png"):
        p.unlink()
    for img_path in images:
        mp = pts.get(img_path.name)
        uv = (mp.u, mp.v) if mp is not None and mp.visible else None
        g = gt_uv.get(img_path.name)
        title = (f"{img_path.name}: punkt ({uv[0]:.1f}, {uv[1]:.1f})" if uv else f"{img_path.name}: BRAK punktu")
        if g is not None:
            title += f" | rzut GT (zielony) ({g[0]:.0f}, {g[1]:.0f})"
        point_preview(cv2.imread(str(img_path)), uv, g, pdir / f"{img_path.stem}.png", title)
    print(f"Podglady punktow: {pdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
