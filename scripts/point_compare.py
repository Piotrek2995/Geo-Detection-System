"""Ocena metod punktu reprezentatywnego (zadanie 5).

Dla kazdej metody z POINT_METHODS (wszystkie klasy ta sama metoda), powierzchnia
nmt_local (domyslnie), eps 10 m:
  - glowny i drugi namiot: RMS/max, skladowe wzdluz/w poprzek promienia, nachylenie
    skladowej wzdluznej vs tan(incidence),
  - multiview_check: z_offset minimalizujacy rozrzut namiotow i roznica wzgledem
    dem_alignment (hipoteza: punkty przy gruncie zmniejszaja te roznice),
  - ludzie: tylko przesuniecie pozycji miedzy metodami (klaster laczy kilka osob),
  - czas (z SAM: zimny cache masek / bez SAM).
Wyniki: tabela na stdout, output/points/*.png (wykresy + podglady detekcji).

    python scripts/point_compare.py [--surface nmt_local] [--eps 10] [--cold]
"""

from __future__ import annotations

import argparse
import copy
import logging
import math
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from geodetect.batch import run_batch  # noqa: E402
from geodetect.config import POINT_METHODS, SURFACE_SOURCES, PipelineConfig  # noqa: E402
from geodetect.detect import Detector  # noqa: E402
from geodetect.diagnostics import (fit_along_vs_tan, multiview_z_scan, ray_residuals,  # noqa: E402
                                   static_objects)
from geodetect.points import choose_point, down_direction  # noqa: E402
from geodetect.segment import Segmenter, clean_mask  # noqa: E402
from geodetect.pipeline import build_context  # noqa: E402
from zcal_compare import CachedDetector  # noqa: E402

# paleta (dataviz): kategorie + tusz/siatka
BLUE, ORANGE, AQUA, PURPLE, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#8a5cd1", "#d9a21b"
INK, INK2, GRID, BG = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
METHOD_COLOR = dict(zip(POINT_METHODS, [INK2, ORANGE, AQUA, BLUE, PURPLE]))
SHORT = {"bbox_center": "bbox_center", "bbox_bottom_center": "bottom_center",
         "mask_centroid": "mask_centroid", "mask_ground_contact": "ground_contact",
         "mask_footprint_median": "footprint_median"}


def rms(v) -> float:
    v = np.asarray(v, float)
    return math.sqrt(float(np.mean(v ** 2))) if len(v) else float("nan")


def evaluate(method, ctx, batch, eps, scan_span, scan_step):
    tents = sorted(static_objects(batch, eps, "namiot", min_views=2),
                   key=lambda o: -o.n_views)
    main = tents[0]
    second = tents[1] if len(tents) > 1 else None
    res = ray_residuals(batch, [main])
    fit = fit_along_vs_tan(res)
    t0 = time.perf_counter()
    scan = multiview_z_scan(batch, ctx, span_m=scan_span, step_m=scan_step, eps_m=eps,
                            cls_name="namiot", min_views=3)
    scan_s = time.perf_counter() - t0
    people = [o for o in static_objects(batch, eps, "people", min_views=1)]
    people = max(people, key=lambda o: o.n_views) if people else None
    fallbacks = {}
    for gd in batch.detections:
        if gd.point and gd.point.fallback:
            key = gd.point.fallback
            fallbacks[key] = fallbacks.get(key, 0) + 1
    return dict(method=method, main=main, second=second, fit=fit,
                along=rms([r.along for r in res]), across=rms([r.across for r in res]),
                scan=scan, scan_s=scan_s, people=people, fallbacks=fallbacks,
                hits={(gd.image_name, tuple(round(v, 1) for v in gd.detection.bbox)): gd
                      for gd in batch.detections})


# ───────────── wykresy ─────────────

def _style(ax):
    ax.set_facecolor(BG)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def plot_spread(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = [SHORT[r["method"]] for r in rows]
    x = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=BG)
    ax = axes[0]
    _style(ax)
    ax.bar(x - 0.2, [r["main"].rms_m for r in rows], 0.38, color=BLUE, label="RMS")
    ax.bar(x + 0.2, [r["main"].max_m for r in rows], 0.38, color=ORANGE, label="max")
    for i, r in enumerate(rows):
        ax.text(i - 0.2, r["main"].rms_m, f"{r['main'].rms_m:.2f}", ha="center", va="bottom",
                fontsize=8, color=INK)
        ax.text(i + 0.2, r["main"].max_m, f"{r['main'].max_m:.2f}", ha="center", va="bottom",
                fontsize=8, color=INK)
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_ylabel("odchylenie od środka [m]", color=INK2)
    ax.set_title(f"Główny namiot ({rows[0]['main'].n_views} widoków)", color=INK, fontsize=11,
                 loc="left")
    ax.legend(frameon=False, fontsize=9)
    ax = axes[1]
    _style(ax)
    ax.bar(x - 0.2, [r["along"] for r in rows], 0.38, color=BLUE, label="wzdłuż promienia")
    ax.bar(x + 0.2, [r["across"] for r in rows], 0.38, color=AQUA, label="w poprzek")
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_ylabel("RMS składowej [m]", color=INK2)
    ax.set_title("Główny namiot: składowe residuów", color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=BG)
    plt.close(fig)


def plot_zoffset(rows, z_dem, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=BG,
                             gridspec_kw={"width_ratios": [1.3, 1]})
    ax = axes[0]
    _style(ax)
    for r in rows:
        s = r["scan"]
        ax.plot(s.offsets - z_dem, s.rms, color=METHOD_COLOR[r["method"]], lw=1.8,
                label=SHORT[r["method"]])
        ax.plot(s.refined_offset - z_dem, np.interp(s.refined_offset, s.offsets, s.rms), "o",
                color=METHOD_COLOR[r["method"]], ms=5)
    ax.axvline(0, color=INK2, lw=1, ls="--")
    ax.text(0.02, ax.get_ylim()[1], " dem_alignment", color=INK2, fontsize=8, va="top")
    ax.set_xlabel("z_offset − dem_alignment [m]", color=INK2)
    ax.set_ylabel("RMS rozrzutu namiotu [m]", color=INK2)
    ax.set_title("multiview_check: rozrzut vs z_offset", color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8)
    ax = axes[1]
    _style(ax)
    d = [r["scan"].refined_offset - z_dem for r in rows]
    x = np.arange(len(rows))
    ax.bar(x, d, 0.6, color=[METHOD_COLOR[r["method"]] for r in rows])
    for i, v in enumerate(d):
        ax.text(i, v, f"{v:+.2f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=8,
                color=INK)
    ax.axhline(0, color=INK2, lw=1)
    ax.set_xticks(x, [SHORT[r["method"]] for r in rows], rotation=20, ha="right")
    ax.set_ylabel("optimum − dem_alignment [m]", color=INK2)
    ax.set_title("Różnica z_offset per metoda", color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=BG)
    plt.close(fig)


# ───────────── podglady ─────────────

def _bgr(hex_):
    h = hex_.lstrip("#")
    return int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16)


def previews(batch, ctx, seg: Segmenter, out_dir: Path, n: int = 6):
    """Podglady: bbox, maska, piksele kazdej metody, kierunek pionu w dol."""
    cfg = ctx.config
    tents = sorted([gd for gd in batch.detections if gd.detection.cls_name == "namiot"],
                   key=lambda g: g.incidence_deg)
    people = sorted([gd for gd in batch.detections if gd.detection.cls_name == "people"],
                    key=lambda g: g.incidence_deg)
    pick = [tents[int(i)] for i in np.linspace(0, len(tents) - 1, max(1, n - 2))] if tents else []
    pick += [people[int(i)] for i in np.linspace(0, len(people) - 1, 2)] if people else []
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for k, gd in enumerate(pick):
        io, eo = ctx.reconstruction.camera(gd.image_name)
        img = cv2.imread(str(cfg.photos_dir / gd.image_name))
        bbox = gd.detection.bbox
        raw = seg.masks(cfg.photos_dir / gd.image_name, [bbox])[0]
        mask, q = clean_mask(raw, bbox, cfg.point) if raw is not None else (None, None)
        x1, y1, x2, y2 = bbox
        s = max(x2 - x1, y2 - y1)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        c0, r0 = max(0, int(cx - 1.2 * s)), max(0, int(cy - 1.2 * s))
        c1, r1 = min(img.shape[1], int(cx + 1.2 * s)), min(img.shape[0], int(cy + 1.2 * s))
        crop = img[r0:r1, c0:c1].copy()
        scale = 480.0 / max(crop.shape[:2])
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        def P(u, v):
            return int(round((u - c0) * scale)), int(round((v - r0) * scale))
        if mask is not None:
            m = np.zeros(img.shape[:2], bool)
            h, w = mask.mask.shape
            m[mask.y0:mask.y0 + h, mask.x0:mask.x0 + w] = mask.mask
            m = cv2.resize(m[r0:r1, c0:c1].astype(np.uint8), (crop.shape[1], crop.shape[0]),
                           interpolation=cv2.INTER_NEAREST).astype(bool)
            over = crop.copy()
            over[m] = _bgr(BLUE)
            crop = cv2.addWeighted(over, 0.35, crop, 0.65, 0)
            cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(crop, cnts, -1, _bgr(BLUE), 1)
        cv2.rectangle(crop, P(x1, y1), P(x2, y2), (255, 255, 255), 1)
        d, off = down_direction((cx, cy), io, eo)
        a = P(cx, cy)
        b = (int(a[0] + 60 * d[0]), int(a[1] + 60 * d[1]))
        cv2.arrowedLine(crop, a, b, (255, 255, 255), 1, tipLength=0.2)
        legend = []
        for meth in POINT_METHODS:
            ch = choose_point(meth, bbox, io, eo, cfg.point, mask, q)
            col = _bgr(METHOD_COLOR[meth])
            if ch.footprint_uvs is not None:
                for u, v in ch.footprint_uvs[::4]:
                    cv2.circle(crop, P(u, v), 1, col, -1)
            cv2.drawMarker(crop, P(*ch.uv), col, cv2.MARKER_CROSS, 14, 2)
            legend.append((f"{SHORT[meth]}{'' if not ch.fallback else ' -> ' + ch.used}", col))
        panel = np.full((crop.shape[0], 250, 3), 252, np.uint8)
        lines = [(f"{gd.image_name} {gd.detection.cls_name}", (11, 11, 11)),
                 (f"incidence {gd.incidence_deg:.1f} deg, off-nadir {off:.1f}", (78, 81, 82)),
                 (f"maska: {q.reason or 'ok'}, pole/bbox {q.area_ratio:.2f}" if q else "maska: brak",
                  (78, 81, 82)), ("strzalka: pion w dol", (78, 81, 82))] + legend
        for i, (t, col) in enumerate(lines):
            cv2.putText(panel, t, (8, 22 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1,
                        cv2.LINE_AA)
        p = out_dir / f"preview_{k + 1}_{gd.detection.cls_name}_{gd.image_name.removesuffix('.JPG')}.png"
        cv2.imwrite(str(p), np.hstack([crop, panel]))
        paths.append(p)
    return paths


# ───────────── main ─────────────

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    ap.add_argument("--surface", default="nmt_local", choices=SURFACE_SOURCES)
    ap.add_argument("--eps", type=float, default=10.0)
    ap.add_argument("--methods", nargs="+", default=list(POINT_METHODS), choices=POINT_METHODS)
    ap.add_argument("--scan-span", type=float, default=3.0)
    ap.add_argument("--scan-step", type=float, default=0.05)
    ap.add_argument("--cold", action="store_true", help="wyczysc cache masek (pomiar czasu SAM)")
    ap.add_argument("--out", type=Path, default=ROOT / "output" / "points")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    base = PipelineConfig.from_file(args.config)
    base.surface.source = args.surface
    base.aggregate.eps_m = args.eps
    if args.cold and base.point.mask_cache_dir.exists():
        shutil.rmtree(base.point.mask_cache_dir)
    detector = CachedDetector(Detector(base.model_path, base.detect))
    seg = Segmenter(base.point)
    rows, last_ctx, last_batch = [], None, None
    for m in args.methods:
        cfg = copy.deepcopy(base)
        cfg.point.method, cfg.point.per_class = m, {}
        ctx = build_context(cfg, detector=detector)
        ctx.segmenter = seg
        if not detector._cache:                 # YOLO poza pomiarem czasu
            run_batch(cfg, ctx, images=None)
        sam0 = seg.infer_s
        t0 = time.perf_counter()
        batch = run_batch(cfg, ctx, images=None)
        dt = time.perf_counter() - t0
        r = evaluate(m, ctx, batch, args.eps, args.scan_span, args.scan_step)
        r.update(time_s=dt, sam_s=seg.infer_s - sam0, n_det=len(batch.detections),
                 n_miss=len(batch.misses), z_dem=ctx.z_calibration.offset)
        rows.append(r)
        print(f"[{m}] {dt:.1f} s (SAM {r['sam_s']:.1f} s), z-scan {r['scan_s']:.1f} s", flush=True)
        if m == "mask_footprint_median" or last_ctx is None:
            last_ctx, last_batch = ctx, batch

    z_dem = rows[0]["z_dem"]
    print(f"\nPowierzchnia {args.surface}, eps {args.eps:g} m, dem_alignment z_offset = {z_dem:.3f} m")
    print("\nmetoda | trafienia | glowny namiot: RMS / max [m] | wzdluz / w poprzek [m] | "
          "nachylenie vs tan(inc) (r) | drugi namiot: widoki, RMS / max, odl. [m] | "
          "multiview z_offset | roznica vs dem_alignment | fallbacki | czas [s] (SAM)")
    for r in rows:
        mo, so, f, s = r["main"], r["second"], r["fit"], r["scan"]
        sec = "-" if so is None else (f"{so.n_views}, {so.rms_m:.2f} / {so.max_m:.2f}, "
                                      f"{math.hypot(so.x_2180 - mo.x_2180, so.y_2180 - mo.y_2180):.0f}")
        fb = ", ".join(f"{k}={v}" for k, v in sorted(r["fallbacks"].items())) or "-"
        print(f"{r['method']} | {r['n_det']} ({r['n_miss']}) | {mo.rms_m:.2f} / {mo.max_m:.2f} | "
              f"{r['along']:.2f} / {r['across']:.2f} | "
              f"{'-' if f is None else f'{f.slope:+.2f} ({f.r:+.2f})'} | {sec} | "
              f"{s.refined_offset:.2f} (RMS {s.best_rms:.2f}) | {s.refined_offset - z_dem:+.2f} | "
              f"{fb} | {r['time_s']:.1f} ({r['sam_s']:.1f})")

    # ludzie: tylko przesuniecie pozycji
    by = {r["method"]: r for r in rows}
    if "bbox_center" in by and "bbox_bottom_center" in by:
        a, b = by["bbox_center"], by["bbox_bottom_center"]
        shifts = [math.hypot(b["hits"][k].hit.x_2180 - g.hit.x_2180, b["hits"][k].hit.y_2180 - g.hit.y_2180)
                  for k, g in a["hits"].items()
                  if g.detection.cls_name == "people" and k in b["hits"]]
        print("\nLudzie (klaster laczy kilka poruszajacych sie osob — RMS nie jest miara jakosci):")
        if shifts:
            print(f"  bbox_center -> bbox_bottom_center: przesuniecie detekcji mediana "
                  f"{np.median(shifts):.2f} m, max {max(shifts):.2f} m (n={len(shifts)})")
        pa, pb = a["people"], b["people"]
        if pa and pb:
            print(f"  srodek klastra people: przesuniecie {math.hypot(pb.x_2180 - pa.x_2180, pb.y_2180 - pa.y_2180):.2f} m "
                  f"(dx {pb.x_2180 - pa.x_2180:+.2f}, dy {pb.y_2180 - pa.y_2180:+.2f})")
        for r in rows[2:]:
            if r["people"] and pa:
                p = r["people"]
                print(f"  bbox_center -> {r['method']}: srodek klastra przesuniety o "
                      f"{math.hypot(p.x_2180 - pa.x_2180, p.y_2180 - pa.y_2180):.2f} m")

    args.out.mkdir(parents=True, exist_ok=True)
    plot_spread(rows, args.out / "spread_by_method.png")
    plot_zoffset(rows, z_dem, args.out / "zoffset_by_method.png")
    pv = previews(last_batch, last_ctx, seg, args.out)
    print(f"\nWykresy: {args.out / 'spread_by_method.png'}, {args.out / 'zoffset_by_method.png'}")
    print(f"Podglady: {len(pv)} plikow w {args.out}")
    print(f"SAM: {seg.n_infer} wywolan modelu, {seg.infer_s:.1f} s inferencji, "
          f"{seg.n_cached} masek z cache")
    return 0


if __name__ == "__main__":
    sys.exit(main())
