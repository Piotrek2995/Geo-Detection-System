"""Porownanie trybow kalibracji Z na calym zbiorze (zadanie 3, punkt C).

Tryby: agl_assumption, dem_alignment, dem_alignment+tilt oraz diagnostyka multiview_check
(skan z_offset minimalizujacy rozrzut statycznych namiotow — tylko kontrola, nie kalibracja).
Detekcje YOLO liczone raz i wspolne dla wszystkich trybow. Metryki przy eps=10 m.

DEM z ODM: odm_project/odm_dem/{dtm,dsm}.tif, a gdy ich brak — WSTEPNY DEM z chmury punktow
(output/cache/odm_pc_{dtm,dsm}.tif, patrz scripts/pointcloud_to_dem.py).

Uzycie:  python scripts/zcal_compare.py [--config config.yaml] [--out output/zcal]
"""

from __future__ import annotations

import argparse
import copy
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from geodetect.batch import run_batch  # noqa: E402
from geodetect.config import PipelineConfig  # noqa: E402
from geodetect.detect import Detector  # noqa: E402
from geodetect.diagnostics import (fit_along_vs_tan, multiview_z_scan, ray_residuals,  # noqa: E402
                                   static_objects)
from geodetect.pipeline import PipelineContext, build_context  # noqa: E402
from geodetect.zcal import ZCalibration  # noqa: E402

# paleta (sloty kategoryczne + tekst/siatka) — jak w geodetect/export.py
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURFACE, NEUTRAL = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb", "#f0efec"


class CachedDetector:
    """Detekcje YOLO liczone raz na zdjecie, wspolne dla wszystkich trybow."""

    def __init__(self, detector: Detector):
        self._d = detector
        self._cache: dict[str, list] = {}

    def detect(self, image_path):
        key = str(image_path)
        if key not in self._cache:
            self._cache[key] = self._d.detect(image_path)
        return self._cache[key]


@dataclass
class ModeResult:
    name: str
    z_cal: ZCalibration
    batch: object
    main: object                 # glowny namiot (AggregatedObject)
    pooled_rms: float            # RMS po wszystkich statycznych namiotach (>=3 widoki)
    fit: object                  # AlongFit dla glownego namiotu
    along_rms: float
    across_rms: float
    residuals: list


def evaluate(name: str, ctx: PipelineContext, cfg: PipelineConfig) -> ModeResult:
    batch = run_batch(cfg, ctx, images=None)
    objs = static_objects(batch, eps_m=10.0, cls_name="namiot", min_views=3)
    main = max(objs, key=lambda o: o.n_views)
    res_main = ray_residuals(batch, [main])
    sq = []
    for o in objs:
        sq += [(gd.hit.x_2180 - o.x_2180) ** 2 + (gd.hit.y_2180 - o.y_2180) ** 2
               for gd in batch.detections if gd.object_id == o.object_id]
    return ModeResult(
        name, ctx.z_calibration, batch, main, math.sqrt(float(np.mean(sq))),
        fit_along_vs_tan(res_main),
        math.sqrt(float(np.mean([r.along ** 2 for r in res_main]))),
        math.sqrt(float(np.mean([r.across ** 2 for r in res_main]))),
        res_main)


def context_with_offset(ctx: PipelineContext, offset: float, label: str) -> PipelineContext:
    return PipelineContext(ctx.config, ctx.dem, ctx.reconstruction,
                           ctx.frame.with_z_offset(offset), ZCalibration(offset, label),
                           ctx.detector)


# ───────────── wykresy ─────────────

def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")
    ax.tick_params(colors=INK2)


def _fig(plt, *a, **kw):
    fig, axes = plt.subplots(*a, facecolor=SURFACE, **kw)
    for ax in np.atleast_1d(axes).ravel():
        _style(ax)
    return fig, axes


def _save(fig, path: Path, title: str, plt):
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"  -> {path}")


def plot_dem_hist(al, path, plt, provisional):
    fig, ax = _fig(plt, figsize=(7, 4))
    d = al.diff
    med = float(np.median(d))
    ax.hist(d, bins=40, color=BLUE, edgecolor=SURFACE, linewidth=1)
    ax.axvline(med, color=INK, linewidth=1.5)
    ax.annotate(f"mediana {med:.2f} m", (med, ax.get_ylim()[1] * 0.92), xytext=(6, 0),
                textcoords="offset points", color=INK, fontsize=10)
    ax.set_xlabel("NMT GUGiK − DEM ODM [m]")
    ax.set_ylabel("Liczba punktów")
    src = "wstępny DEM z chmury punktów" if provisional else "DEM ODM"
    _save(fig, path, f"Różnice wysokości NMT − {src} (n={len(d)}, {al.layer})", plt)


def plot_dem_plane(al, path, plt, provisional):
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    cmap = LinearSegmentedColormap.from_list("div", [BLUE, NEUTRAL, ORANGE])
    fig, axes = _fig(plt, 1, 2, figsize=(12, 5))
    p = al.plane
    d = al.diff
    x0, y0 = p.x0, p.y0
    # lewy: roznice; prawy: residua po odjeciu plaszczyzny (ta sama skala barw)
    lim = float(np.max(np.abs(d - np.median(d))))
    norm = TwoSlopeNorm(vcenter=float(np.median(d)), vmin=float(np.median(d)) - lim,
                        vmax=float(np.median(d)) + lim)
    sc = axes[0].scatter(al.x_2180 - x0, al.y_2180 - y0, c=d, cmap=cmap, norm=norm, s=22,
                         edgecolor=SURFACE, linewidth=0.6)
    gx, gy = np.meshgrid(np.linspace((al.x_2180 - x0).min(), (al.x_2180 - x0).max(), 50),
                         np.linspace((al.y_2180 - y0).min(), (al.y_2180 - y0).max(), 50))
    cs = axes[0].contour(gx, gy, p(gx + x0, gy + y0), levels=8, colors=INK2, linewidths=0.8)
    axes[0].clabel(cs, fmt="%.1f", fontsize=8)
    fig.colorbar(sc, ax=axes[0], label="NMT − DEM ODM [m]")
    axes[0].set_title(f"Różnice + płaszczyzna: nachylenie {p.slope_deg:.2f}°, "
                      f"azymut {p.aspect_deg:.0f}°", fontsize=10, color=INK, loc="left")
    resid = d - p(al.x_2180, al.y_2180)
    rl = float(np.max(np.abs(resid)))
    sc2 = axes[1].scatter(al.x_2180 - x0, al.y_2180 - y0, c=resid, cmap=cmap,
                          norm=TwoSlopeNorm(vcenter=0, vmin=-rl, vmax=rl), s=22,
                          edgecolor=SURFACE, linewidth=0.6)
    fig.colorbar(sc2, ax=axes[1], label="residuum po płaszczyźnie [m]")
    axes[1].set_title(f"Residua po płaszczyźnie (RMS {p.rms_resid:.2f} m bez odstających)",
                      fontsize=10, color=INK, loc="left")
    for ax in axes:
        ax.set_xlabel("ΔE od środka próbek [m] (EPSG:2180)")
        ax.set_aspect("equal")
    axes[0].set_ylabel("ΔN [m]")
    src = "wstępny DEM z chmury" if provisional else "DEM ODM"
    _save(fig, path, f"Przechył modelu ODM względem NMT ({src})", plt)


def plot_zscan(scan, marks, path, plt):
    fig, ax = _fig(plt, figsize=(8, 4.2))
    ax.plot(scan.offsets, scan.rms, color=BLUE, linewidth=2)
    ymax = float(np.nanmax(scan.rms))
    for k, ((label, off), color) in enumerate(zip(marks, [INK2, ORANGE, AQUA, INK])):
        ax.axvline(off, color=color, linewidth=1.2, linestyle="--" if label.startswith("agl") else "-")
        ax.annotate(f"{label}: {off:.2f}", (off, ymax * (1 - 0.1 * k)), xytext=(4, 0),
                    textcoords="offset points", va="top", fontsize=8, color=INK)
    ax.plot([scan.best_offset], [scan.best_rms], "o", color=BLUE, markersize=8,
            markeredgecolor=SURFACE, markeredgewidth=2, zorder=4)
    ax.annotate(f"min {scan.best_rms:.2f} m\nprzy {scan.best_offset:.2f}",
                (scan.best_offset, scan.best_rms), xytext=(-8, 4), textcoords="offset points",
                ha="right", va="bottom", fontsize=8, color=INK)
    ax.set_xlabel("z_offset [m]")
    ax.set_ylabel("RMS rozrzutu namiotów [m]")
    _save(fig, path, f"multiview_check (diagnostyka): RMS(z_offset), obiekty statyczne: "
                     f"{scan.n_objects}, detekcje: {scan.n_detections}", plt)


def plot_along(results, path, plt):
    fig, axes = _fig(plt, 1, len(results), figsize=(4.2 * len(results), 4), sharey=True)
    for ax, mr in zip(np.atleast_1d(axes), results):
        t = np.array([r.tan_incidence for r in mr.residuals])
        a = np.array([r.along for r in mr.residuals])
        c = np.array([r.across for r in mr.residuals])
        ax.scatter(t, c, s=26, marker="^", color=ORANGE, edgecolor=SURFACE, linewidth=0.8,
                   label="w poprzek promienia", zorder=2)
        ax.scatter(t, a, s=30, color=BLUE, edgecolor=SURFACE, linewidth=0.8,
                   label="wzdłuż promienia", zorder=3)
        if mr.fit:
            tt = np.linspace(t.min(), t.max(), 10)
            ax.plot(tt, mr.fit.slope * tt + mr.fit.intercept, color=BLUE, linewidth=1.5)
        ax.axhline(0, color=INK2, linewidth=0.8)
        ax.set_title(f"{mr.name}\nz={mr.z_cal.offset:.2f}  nachyl.={mr.fit.slope:.2f} m  "
                     f"r={mr.fit.r:.2f}", fontsize=9, color=INK, loc="left")
        ax.set_xlabel("tan(kąt promienia od pionu)")
    np.atleast_1d(axes)[0].set_ylabel("Odchylenie od środka namiotu [m]")
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=8, loc="lower left")
    _save(fig, path, "Główny namiot: odchylenie wzdłuż / w poprzek promienia vs tan(incidence)", plt)


# ───────────── main ─────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    ap.add_argument("--out", type=Path, default=ROOT / "output" / "zcal")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    base_cfg = PipelineConfig.from_file(args.config)
    provisional = not (base_cfg.odm_dtm_path.exists() or base_cfg.odm_dsm_path.exists())
    if provisional:
        base_cfg.z_anchor.odm_dtm = ROOT / "output" / "cache" / "odm_pc_dtm.tif"
        base_cfg.z_anchor.odm_dsm = ROOT / "output" / "cache" / "odm_pc_dsm.tif"
        if not base_cfg.z_anchor.odm_dtm.exists():
            sys.exit("Brak DEM ODM i wstepnego DEM — uruchom: python scripts/pointcloud_to_dem.py")
        print("UWAGA: brak odm_dem/dtm.tif — uzywam WSTEPNEGO DEM z chmury punktow ODM\n")
    detector = CachedDetector(Detector(base_cfg.model_path, conf=base_cfg.detect.conf,
                                       classes=base_cfg.detect.classes))

    results: list[ModeResult] = []
    ctxs = {}
    for name, mode, tilt in [("agl_assumption", "agl_assumption", False),
                             ("dem_alignment", "dem_alignment", False),
                             ("dem_alignment+tilt", "dem_alignment", True)]:
        cfg = copy.deepcopy(base_cfg)
        cfg.z_anchor.mode, cfg.z_anchor.apply_tilt = mode, tilt
        ctx = build_context(cfg, detector=detector)
        ctxs[name] = ctx
        print(f"[{name}] {ctx.z_calibration.describe()}")
        results.append(evaluate(name, ctx, cfg))

    print("\n[multiview_check] skan z_offset ±15 m co 0.25 m (pierwsze uruchomienie: kilka minut)...")
    agl = results[0]
    scan = multiview_z_scan(agl.batch, ctxs["agl_assumption"])
    mv_ctx = context_with_offset(ctxs["agl_assumption"], scan.best_offset, "multiview_check (diag.)")
    results.append(evaluate("multiview_check (diag.)", mv_ctx, base_cfg))

    # ── tabela ──
    ref = agl.main
    print("\nGlowny namiot (eps=10 m), RMS zbiorczy po statycznych namiotach z >=3 widokami")
    hdr = (f"{'tryb':26} {'z_offset':>9} {'widoki':>6} {'RMS':>6} {'max':>6} {'RMSall':>7} "
           f"{'r(along,tan)':>12} {'nachyl.':>8} {'along':>6} {'across':>7} {'przesun.':>8}")
    print(hdr)
    print("-" * len(hdr))
    for mr in results:
        shift = math.hypot(mr.main.x_2180 - ref.x_2180, mr.main.y_2180 - ref.y_2180)
        print(f"{mr.name:26} {mr.z_cal.offset:9.2f} {mr.main.n_views:6d} {mr.main.rms_m:6.2f} "
              f"{mr.main.max_m:6.2f} {mr.pooled_rms:7.2f} {mr.fit.r:12.2f} {mr.fit.slope:8.2f} "
              f"{mr.along_rms:6.2f} {mr.across_rms:7.2f} {shift:8.2f}")
    print("(RMS/max/along/across/przesun. w metrach; nachyl. = slope along vs tan(inc) [m];"
          " przesun. = pozycja glownego namiotu wzgledem agl_assumption)")

    al = results[1].z_cal.alignment
    print(f"\ndem_alignment: n={al.diff.size}, mediana={np.median(al.diff):.3f}, "
          f"MAD={results[1].z_cal.stats['mad']:.3f}, plaszczyzna {al.plane.slope_deg:.2f} deg "
          f"w kierunku {al.plane.aspect_deg:.0f} deg, c={al.plane.c:.3f}, "
          f"RMS residuow {al.plane.rms_resid:.3f} m ({al.plane.n_used} pkt)")
    print(f"weryfikacja ukladu (kamery nad DEM ODM): {al.camera_agl}")
    print(f"multiview_check: min RMS {scan.best_rms:.2f} m przy z_offset={scan.best_offset:.2f} "
          f"(dem_alignment {results[1].z_cal.offset:.2f}, roznica "
          f"{scan.best_offset - results[1].z_cal.offset:+.2f} m)")

    # ── wykresy ──
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.labelcolor": INK})
    args.out.mkdir(parents=True, exist_ok=True)
    print("\nWykresy:")
    plot_dem_hist(al, args.out / "dem_diff_hist.png", plt, provisional)
    plot_dem_plane(al, args.out / "dem_diff_plane.png", plt, provisional)
    plot_zscan(scan, [("agl_assumption", agl.z_cal.offset),
                      ("dem_alignment", results[1].z_cal.offset),
                      ("dem_al.+tilt (c)", results[2].z_cal.offset)], args.out / "multiview_rms.png", plt)
    plot_along(results, args.out / "along_vs_tan.png", plt)


if __name__ == "__main__":
    main()
