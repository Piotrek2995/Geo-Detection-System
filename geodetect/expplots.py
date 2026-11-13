"""Wykresy i podglady dla eksperymentow (geodetect.experiments).

Paleta jak w eksporcie: sloty kategoryczne (niebieski, pomaranczowy, akwamaryna, ...),
tekst #0b0b0b / #52514e, siatka #e6e5e0, tlo #fcfcfb.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

BLUE, ORANGE, AQUA, YELLOW, PINK, GREEN, PURPLE = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6")
SERIES = [BLUE, ORANGE, AQUA, YELLOW, PINK, GREEN, PURPLE]
INK, INK2, GRID, SURFACE, NEUTRAL, AXIS = ("#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb",
                                           "#f0efec", "#c3c2b7")


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.labelcolor": INK})
    return plt


def _style(ax, grid_axis="both"):
    ax.set_facecolor(SURFACE)
    ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=INK2)


def _fig(plt, *a, grid_axis="both", **kw):
    fig, axes = plt.subplots(*a, facecolor=SURFACE, **kw)
    for ax in np.atleast_1d(axes).ravel():
        _style(ax, grid_axis)
    return fig, axes


def _save(fig, plt, path: Path, title: str) -> Path:
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


def _bar_labels(ax, xs, vals, fmt="{:.2f}"):
    for x, v in zip(xs, vals):
        if v is not None and np.isfinite(v):
            ax.text(x, v, fmt.format(v), ha="center", va="bottom", fontsize=8, color=INK)


# ───────────── obiekty statyczne ─────────────

def plot_static(rows: list[dict], path: Path, obj_label: str = "główny namiot") -> Optional[Path]:
    """rows: {variant, rms, max, along, across} glownego obiektu statycznego per wariant."""
    rows = [r for r in rows if r.get("rms") is not None]
    if not rows:
        return None
    plt = _plt()
    labels = [r["variant"] for r in rows]
    x = np.arange(len(rows))
    fig, axes = _fig(plt, 1, 2, figsize=(max(9, 2.2 * len(rows) + 4), 4.3), grid_axis="y")
    w = 0.38
    for ax, (k1, l1, c1), (k2, l2, c2), yl in (
            (axes[0], ("rms", "RMS", BLUE), ("max", "max", ORANGE), "odchylenie od środka [m]"),
            (axes[1], ("along", "wzdłuż promienia", BLUE), ("across", "w poprzek", AQUA),
             "RMS składowej [m]")):
        v1 = [r[k1] for r in rows]
        v2 = [r[k2] for r in rows]
        ax.bar(x - w / 2, v1, w, color=c1, label=l1)
        ax.bar(x + w / 2, v2, w, color=c2, label=l2)
        _bar_labels(ax, x - w / 2, v1)
        _bar_labels(ax, x + w / 2, v2)
        ax.set_xticks(x, labels, rotation=20, ha="right")
        ax.set_ylabel(yl, color=INK2)
        ax.legend(frameon=False, fontsize=9)
    axes[0].set_title(f"Rozrzut: {obj_label}", fontsize=10, color=INK, loc="left")
    axes[1].set_title("Składowe residuów względem promienia", fontsize=10, color=INK, loc="left")
    return _save(fig, plt, path, "Obiekt statyczny — rozrzut pozycji między widokami")


def plot_along(items: list[tuple[str, list, object]], path: Path) -> Optional[Path]:
    """items: (wariant, residua RayResidual, AlongFit) — odchylenie vs tan(incidence)."""
    items = [it for it in items if it[1]]
    if not items:
        return None
    plt = _plt()
    fig, axes = _fig(plt, 1, len(items), figsize=(4.0 * len(items), 4), sharey=True)
    for ax, (name, res, fit) in zip(np.atleast_1d(axes), items):
        t = np.array([r.tan_incidence for r in res])
        a = np.array([r.along for r in res])
        c = np.array([r.across for r in res])
        ax.scatter(t, c, s=24, marker="^", color=ORANGE, edgecolor=SURFACE, linewidth=0.8,
                   label="w poprzek promienia", zorder=2)
        ax.scatter(t, a, s=28, color=BLUE, edgecolor=SURFACE, linewidth=0.8,
                   label="wzdłuż promienia", zorder=3)
        if fit is not None:
            tt = np.linspace(t.min(), t.max(), 10)
            ax.plot(tt, fit.slope * tt + fit.intercept, color=BLUE, linewidth=1.5)
            ax.set_title(f"{name}\nnachyl. {fit.slope:+.2f} m, r {fit.r:+.2f}", fontsize=9,
                         color=INK, loc="left")
        else:
            ax.set_title(name, fontsize=9, color=INK, loc="left")
        ax.axhline(0, color=INK2, linewidth=0.8)
        ax.set_xlabel("tan(kąt promienia od pionu)")
    np.atleast_1d(axes)[0].set_ylabel("odchylenie od środka obiektu [m]")
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=8, loc="lower left")
    return _save(fig, plt, path, "Odchylenie wzdłuż / w poprzek promienia vs tan(incidence)")


# ───────────── multiview_check ─────────────

def plot_multiview(scans: list[tuple[str, object]], ref_offset: float, path: Path,
                   marks: Optional[list[tuple[str, float]]] = None) -> Optional[Path]:
    """Krzywe RMS(z_offset) per wariant + roznica optimum wzgledem ref_offset."""
    scans = [(n, s) for n, s in scans if s is not None and s.valid]
    if not scans:
        return None
    plt = _plt()
    fig, axes = _fig(plt, 1, 2, figsize=(12, 4.4), gridspec_kw={"width_ratios": [1.4, 1]})
    ax = axes[0]
    for (name, s), col in zip(scans, SERIES):
        ax.plot(s.offsets, s.rms, color=col, linewidth=1.8, label=name)
        ro = s.refined_offset
        ax.plot([ro], [np.interp(ro, s.offsets, s.rms)], "o", color=col, markersize=6,
                markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=4)
    ax.axvline(ref_offset, color=INK2, linewidth=1, linestyle="--")
    ax.text(ref_offset, ax.get_ylim()[1], " z_offset wariantu", color=INK2, fontsize=8, va="top")
    for label, off in marks or []:
        ax.axvline(off, color=AXIS, linewidth=1)
        ax.text(off, ax.get_ylim()[1] * 0.85, f" {label}", color=INK2, fontsize=8, va="top")
    ax.set_xlabel("z_offset [m]")
    ax.set_ylabel("RMS rozrzutu obiektów statycznych [m]")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("RMS(z_offset)", fontsize=10, color=INK, loc="left")
    ax = axes[1]
    _style(ax, "y")
    d = [s.refined_offset - ref_offset for _, s in scans]
    xs = np.arange(len(scans))
    ax.bar(xs, d, 0.6, color=[c for _, c in zip(scans, SERIES)])
    for i, v in enumerate(d):
        ax.text(i, v, f"{v:+.2f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=8,
                color=INK)
    ax.axhline(0, color=INK2, linewidth=1)
    ax.set_xticks(xs, [n for n, _ in scans], rotation=20, ha="right")
    ax.set_ylabel("optimum − z_offset wariantu [m]")
    ax.set_title("Różnica optimum multiview", fontsize=10, color=INK, loc="left")
    return _save(fig, plt, path, "multiview_check (diagnostyka, nie kalibracja)")


def plot_dem_alignment(al, path: Path) -> Path:
    """Histogram NMT − DEM ODM i przechyl (plaszczyzna) — diagnostyka dem_alignment."""
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    plt = _plt()
    cmap = LinearSegmentedColormap.from_list("div", [BLUE, NEUTRAL, ORANGE])
    fig, axes = _fig(plt, 1, 2, figsize=(12, 4.8))
    d = al.diff
    med = float(np.median(d))
    axes[0].hist(d, bins=40, color=BLUE, edgecolor=SURFACE, linewidth=1)
    axes[0].axvline(med, color=INK, linewidth=1.5)
    axes[0].annotate(f"mediana {med:.2f} m", (med, axes[0].get_ylim()[1] * 0.92), xytext=(6, 0),
                     textcoords="offset points", color=INK, fontsize=9)
    axes[0].set_xlabel("NMT − DEM ODM [m]")
    axes[0].set_ylabel("liczba punktów")
    p = al.plane
    lim = float(np.max(np.abs(d - med))) or 1.0
    sc = axes[1].scatter(al.x_2180 - p.x0, al.y_2180 - p.y0, c=d, cmap=cmap,
                         norm=TwoSlopeNorm(vcenter=med, vmin=med - lim, vmax=med + lim),
                         s=20, edgecolor=SURFACE, linewidth=0.6)
    fig.colorbar(sc, ax=axes[1], label="NMT − DEM ODM [m]")
    axes[1].set_aspect("equal")
    axes[1].set_xlabel("ΔE [m] (EPSG:2180)")
    axes[1].set_ylabel("ΔN [m]")
    axes[1].set_title(f"płaszczyzna: nachylenie {p.slope_deg:.2f}°, azymut {p.aspect_deg:.0f}°",
                      fontsize=10, color=INK, loc="left")
    return _save(fig, plt, path, f"dem_alignment: różnice NMT − DEM ODM (n={len(d)}, {al.layer})")


# ───────────── detekcje vs GT ─────────────

def plot_pr(rows: list[dict], path: Path) -> Optional[Path]:
    """rows: {variant, subset, cls, p, r, f1} — slupki P/R/F1 per wariant, klasa, podzbior."""
    if not rows:
        return None
    plt = _plt()
    subsets = list(dict.fromkeys(r["subset"] for r in rows))
    classes = list(dict.fromkeys(r["cls"] for r in rows))
    variants = list(dict.fromkeys(r["variant"] for r in rows))
    fig, axes = _fig(plt, len(subsets), len(classes), squeeze=False, grid_axis="y",
                     figsize=(5.2 * len(classes), 3.4 * len(subsets)))
    w = 0.26
    x = np.arange(len(variants))
    for i, sub in enumerate(subsets):
        for j, cls in enumerate(classes):
            ax = axes[i][j]
            by = {r["variant"]: r for r in rows if r["subset"] == sub and r["cls"] == cls}
            for k, (key, lab, col) in enumerate((("p", "precision", BLUE), ("r", "recall", ORANGE),
                                                 ("f1", "F1", AQUA))):
                vals = [by[v][key] if v in by else np.nan for v in variants]
                ax.bar(x + (k - 1) * w, vals, w, color=col, label=lab)
            ax.set_ylim(0, 1.12)
            ax.set_xticks(x, variants, rotation=15, ha="right", fontsize=8)
            ax.set_title(f"{cls} — {sub}", fontsize=10, color=INK, loc="left")
            if j == 0:
                ax.set_ylabel("wartość")
    axes[0][0].legend(frameon=False, fontsize=8, ncol=3, loc="upper left")
    return _save(fig, plt, path, "Detekcja względem etykiet (IoU 0.5)")


def detection_preview(img_path: Path, out_path: Path, bbox, title_lines: list[str],
                      gt_boxes: Optional[list] = None, color=(52, 104, 235)) -> Path:
    """Wycinek zdjecia wokol boxa: box detekcji (pomaranczowy), boxy GT (zielone), opis."""
    img = cv2.imread(str(img_path))
    x1, y1, x2, y2 = bbox
    s = max(x2 - x1, y2 - y1, 40)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    c0, r0 = max(0, int(cx - 3 * s)), max(0, int(cy - 3 * s))
    c1, r1 = min(img.shape[1], int(cx + 3 * s)), min(img.shape[0], int(cy + 3 * s))
    crop = img[r0:r1, c0:c1].copy()
    sc = 420.0 / max(crop.shape[:2])
    crop = cv2.resize(crop, None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC)

    def P(u, v):
        return int((u - c0) * sc), int((v - r0) * sc)
    for b in gt_boxes or []:
        cv2.rectangle(crop, P(*b[:2]), P(*b[2:]), (122, 175, 27), 1)
    cv2.rectangle(crop, P(x1, y1), P(x2, y2), color, 2)
    panel = np.full((crop.shape[0], 280, 3), 252, np.uint8)
    lines = title_lines + ["pomaranczowy = detekcja / brak", "zielony = etykieta GT"]
    for i, t in enumerate(lines):
        cv2.putText(panel, t, (8, 22 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (11, 11, 11), 1,
                    cv2.LINE_AA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), np.hstack([crop, panel]))
    return out_path


# ───────────── profil promienia (NMT / NMPT / DSM) ─────────────

def plot_ray_profile(prof: dict, path: Path) -> Path:
    """prof: d (odl. pozioma od kamery), ray_h, nmt, nmpt, dsm (npm), hits {nazwa: d}, opis."""
    plt = _plt()
    fig, ax = _fig(plt, figsize=(10.5, 4.2))
    d = prof["d"]
    ax.fill_between(d, prof["nmt"], np.nanmin(prof["nmt"]) - 2, color=NEUTRAL, zorder=1)
    ax.plot(d, prof["nmt"], color=INK2, linewidth=1.6, label="NMT", zorder=3)
    ax.plot(d, prof["nmpt"], color=GREEN, linewidth=1.6, label="NMPT", zorder=3)
    if prof.get("dsm") is not None and np.isfinite(prof["dsm"]).any():
        ax.plot(d, prof["dsm"], color=YELLOW, linewidth=1.2, linestyle="--",
                label="DSM ODM (+z_offset)", zorder=3)
    ax.plot(d, prof["ray_h"], color=BLUE, linewidth=1.6, label="promień", zorder=4)
    lo = float(np.nanmin(prof["nmt"])) - 2
    hi = float(np.nanmax([np.nanmax(prof["nmpt"]), np.nanmax(prof["nmt"])])) + 8
    ax.set_ylim(lo, hi)
    for (name, dh), col in zip(prof["hits"].items(), [INK, GREEN, YELLOW, ORANGE]):
        if dh is not None:
            ax.axvline(dh, color=col, linewidth=0.8, linestyle=":")
            ax.text(dh, hi, f" {name}", rotation=90, va="top", fontsize=8, color=INK2)
    ax.set_xlabel("odległość pozioma od kamery [m]")
    ax.set_ylabel("H [m npm]")
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    return _save(fig, plt, path, prof["title"])


# ───────────── metody punktu reprezentatywnego ─────────────

POINT_COLORS = {"bbox_center": INK2, "bbox_bottom_center": ORANGE, "mask_centroid": AQUA,
                "mask_ground_contact": BLUE, "mask_footprint_median": PURPLE}


def _bgr(hex_):
    h = hex_.lstrip("#")
    return int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16)


def point_methods_preview(img_path: Path, out_path: Path, bbox, io, eo, point_cfg, mask, quality,
                          header: list[str]) -> Path:
    """Bbox, maska, piksele kazdej metody punktu i kierunek pionu w dol."""
    from .config import POINT_METHODS
    from .points import choose_point, down_direction
    img = cv2.imread(str(img_path))
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
    cv2.arrowedLine(crop, a, (int(a[0] + 60 * d[0]), int(a[1] + 60 * d[1])), (255, 255, 255), 1,
                    tipLength=0.2)
    legend = []
    for meth in POINT_METHODS:
        ch = choose_point(meth, bbox, io, eo, point_cfg, mask, quality)
        col = _bgr(POINT_COLORS[meth])
        if ch.footprint_uvs is not None:
            for u, v in ch.footprint_uvs[::4]:
                cv2.circle(crop, P(u, v), 1, col, -1)
        cv2.drawMarker(crop, P(*ch.uv), col, cv2.MARKER_CROSS, 14, 2)
        legend.append((f"{meth}{'' if not ch.fallback else ' -> ' + ch.used}", col))
    panel = np.full((crop.shape[0], 270, 3), 252, np.uint8)
    lines = [(t, (11, 11, 11)) for t in header] + [
        (f"off-nadir {off:.1f} deg; strzalka: pion w dol", (78, 81, 82)),
        (f"maska: {quality.reason or 'ok'}, pole/bbox {quality.area_ratio:.2f}" if quality
         else "maska: brak", (78, 81, 82))] + legend
    for i, (t, col) in enumerate(lines):
        cv2.putText(panel, t, (8, 22 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), np.hstack([crop, panel]))
    return out_path
