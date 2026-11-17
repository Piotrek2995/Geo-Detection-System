"""Kontrola jakosci rekonstrukcji ODM (wywolywana w build_context, wynik w podsumowaniu batcha).

Kontrole (progi w cfg.qc):
  vertical_scale  — nachylenie k regresji h_ODM kamer vs wysokosc barometryczna EXIF. Oczekiwane
                    ~1; k != 1 oznacza deformacje rekonstrukcji w pionie (np. bledna ogniskowa:
                    lot "pilka" k = 0.68 przy f = 0.374 zamiast 0.528).
  focal           — ogniskowa ODM (znormalizowana f / max(w, h)) vs kalibracja referencyjna
                    (cfg.qc.camera_focal, np. z lotu o dobrej geometrii) albo — gdy jej brak —
                    vs wartosc oczekiwana z EXIF FocalLengthIn35mmFilm (definicja po przekatnej
                    43.27 mm). EXIF moze nie uwzgledniac kadrowania obrazu (Potensic Atom LT:
                    f35 = 13 -> 0.345, a geometria wskazuje 0.528), wiec porownanie z EXIF daje
                    najwyzej OSTRZEZENIE.
  surface         — NMAD reszt DSM ODM - NMPT (po odjeciu mediany; z biezacym z_offset) — miara
                    zgodnosci georeferencji (pochylenie, obrot, przesuniecie) z danymi GUGiK.
  baseline        — rozpietosc pozioma kamer vs szum GPS: oczekiwany blad orientacji modelu
                    ~ sqrt(2) * sigma_GPS / rozpietosc [rad] (lot "pilka": 11 m, ~20 deg).

Status: OK / OSTRZEZENIE / ZLY (najgorszy z kontroli) + zalecenia.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

OK, WARN, BAD, NA = "OK", "OSTRZEŻENIE", "ZŁY", "brak danych"
_RANK = {NA: -1, OK: 0, WARN: 1, BAD: 2}
DIAG_35MM = 43.27

REC_CALIB = ("ustal kalibrację kamery: ODM --cameras <cameras.json z lotu o dobrej geometrii> "
             "--use-fixed-camera-params (7p tylko skompensuje deformację, nie usunie jej)")
REC_7P = "rozważ georef.mode: similarity_nmpt (wyrównanie 7p do NMPT)"
REC_BASE = "słaba geometria lotu: większa baza (obszar, kilka wysokości) albo punkty kontrolne (GCP)"


@dataclass
class QcCheck:
    name: str
    status: str
    value: Optional[float]
    text: str
    details: dict = field(default_factory=dict)


@dataclass
class QcReport:
    checks: list[QcCheck]
    status: str
    recommendations: list[str]

    def check(self, name: str) -> Optional[QcCheck]:
        return next((c for c in self.checks if c.name == name), None)

    def lines(self) -> list[str]:
        out = [f"Kontrola jakości rekonstrukcji: {self.status}"]
        out += [f"  [{c.status}] {c.text}" for c in self.checks]
        out += [f"  zalecenie: {r}" for r in self.recommendations]
        return out

    def to_dict(self) -> dict:
        return json.loads(json.dumps({"status": self.status, "recommendations": self.recommendations,
                                      "checks": [c.__dict__ for c in self.checks]}, default=float))


def _grade(x: float, warn: float, bad: float) -> str:
    return BAD if x > bad else WARN if x > warn else OK


# ───────────── kontrole (czyste funkcje — testowalne na danych syntetycznych) ─────────────

def check_vertical_scale(h_odm, alt_exif, cfg) -> QcCheck:
    h, a = np.asarray(h_odm, float), np.asarray(alt_exif, float)
    if len(a) < 3 or np.ptp(a) < cfg.min_alt_range_m:
        return QcCheck("vertical_scale", NA, None,
                       f"skala pionowa: nieokreślona (zakres wysokości EXIF {np.ptp(a) if len(a) else 0:.1f} m "
                       f"< {cfg.min_alt_range_m:.1f} m)")
    k, b = np.polyfit(a, h, 1)
    res = h - (k * a + b)
    se = float(math.sqrt(np.sum(res ** 2) / max(len(a) - 2, 1) / np.sum((a - a.mean()) ** 2)))
    r = float(np.corrcoef(a, h)[0, 1])
    dev = abs(k - 1)
    # odchylenie musi przekraczac tez niepewnosc nachylenia (krotki zakres wysokosci)
    st = OK if dev <= max(cfg.vertical_scale_warn, 2 * se) else (
        BAD if dev > max(cfg.vertical_scale_bad, 3 * se) else WARN)
    return QcCheck("vertical_scale", st, float(k),
                   f"skala pionowa: h_ODM = {k:.3f} · alt_EXIF {b:+.2f} m (±{se:.3f}, r = {r:.3f}, "
                   f"zakres {np.ptp(a):.1f} m); oczekiwane ~1",
                   {"k": float(k), "b": float(b), "se": se, "r": r, "alt_range_m": float(np.ptp(a)), "n": len(a)})


def expected_focal_exif(f35: float, width: int, height: int) -> float:
    """f / max(w, h) z ekwiwalentu 35 mm (definicja po przekatnej 43.27 mm)."""
    return f35 * math.hypot(width, height) / DIAG_35MM / max(width, height)


def check_focal(focal: float, cfg, f35: Optional[float] = None, width: int = 0, height: int = 0) -> QcCheck:
    if cfg.camera_focal:
        ref, src, cap = float(cfg.camera_focal), "kalibracja referencyjna (qc.camera_focal)", BAD
    elif f35 and width and height:
        ref, src, cap = expected_focal_exif(f35, width, height), f"EXIF f35 = {f35:g} mm", WARN
    else:
        return QcCheck("focal", NA, focal, f"ogniskowa ODM {focal:.4f}: brak odniesienia (EXIF f35 / qc.camera_focal)")
    d = 100 * (focal / ref - 1)
    st = _grade(abs(d), cfg.focal_warn_pct, cfg.focal_bad_pct)
    if _RANK[st] > _RANK[cap]:
        st = cap
    note = "" if cap == BAD else " (EXIF nie uwzględnia kadrowania — najwyżej ostrzeżenie)"
    return QcCheck("focal", st, d, f"ogniskowa ODM {focal:.4f} vs {ref:.4f} ({src}): {d:+.1f} %{note}",
                   {"focal_odm": focal, "focal_ref": ref, "source": src, "diff_pct": d})


def check_surface(residuals, cfg, ref_name: str = "NMPT") -> QcCheck:
    r = np.asarray(residuals, float)
    r = r[np.isfinite(r)]
    if len(r) < 20:
        return QcCheck("surface", NA, None, f"zgodność DSM ODM z {ref_name}: za mało punktów ({len(r)})")
    med = float(np.median(r))
    nmad = float(1.4826 * np.median(np.abs(r - med)))
    frac = float(np.mean(np.abs(r - med) <= 1.0))
    return QcCheck("surface", _grade(nmad, cfg.surface_nmad_warn_m, cfg.surface_nmad_bad_m), nmad,
                   f"DSM ODM − {ref_name}: NMAD {nmad:.2f} m (mediana {med:+.2f} m, |r − med| ≤ 1 m: "
                   f"{100 * frac:.0f} % z {len(r)})",
                   {"nmad": nmad, "median": med, "frac_1m": frac, "n": len(r), "ref": ref_name})


def check_baseline(xy, sigma_gps_m: float, cfg) -> QcCheck:
    xy = np.asarray(xy, float)
    if len(xy) < 2:
        return QcCheck("baseline", NA, None, "baza kamer: za mało zdjęć")
    d = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))
    span = float(d.max())
    rot = math.degrees(math.sqrt(2) * sigma_gps_m / max(span, 1e-6))
    return QcCheck("baseline", _grade(rot, cfg.rotation_warn_deg, cfg.rotation_bad_deg), rot,
                   f"baza kamer {span:.1f} m przy σ GPS {sigma_gps_m:.1f} m → oczekiwany błąd orientacji "
                   f"modelu ~{rot:.1f}°", {"span_m": span, "sigma_gps_m": sigma_gps_m, "rotation_deg": rot})


def summarize(checks: list[QcCheck]) -> QcReport:
    status = max((c.status for c in checks), key=lambda s: _RANK[s], default=NA)
    if status == NA:
        status = OK
    by = {c.name: c for c in checks}
    rec = []
    geom_bad = any(by.get(n) and by[n].status in (WARN, BAD) for n in ("vertical_scale",)) or (
        by.get("focal") and by["focal"].status == BAD)
    if geom_bad:
        rec.append(REC_CALIB)
    if any(by.get(n) and by[n].status in (WARN, BAD) for n in ("surface", "baseline")):
        rec.append(REC_7P + (" — po ustaleniu kalibracji" if geom_bad else ""))
    if by.get("baseline") and by["baseline"].status == BAD:
        rec.append(REC_BASE)
    return QcReport(checks, status, rec)


# ───────────── uruchomienie na rekonstrukcji ─────────────

_CACHE: dict = {}


def run_qc(config, rec, frame, dem=None) -> QcReport:
    """rec — rekonstrukcja w georeferencji gps_only; frame — z_offset z z_anchor."""
    from .exif import read_gps
    q = config.qc
    key = json.dumps([str(rec.path), str(config.odm_dsm_path), round(frame.z_offset, 4),
                      json.loads(json.dumps(q.__dict__, default=str)), str(config.surface.nmpt_dir)])
    if key in _CACHE:
        return _CACHE[key]
    names = [n for n in rec.image_names if n not in config.excluded_images()]
    checks = []
    h, alt, xy = [], [], []
    for n in names:
        C = rec.camera(n)[1].C_enu
        x, y, _ = frame.enu_to_puwg92(*C)
        xy.append((float(x), float(y)))
        h.append(float(C[2]))
        try:
            alt.append(read_gps(config.photos_dir / n).alt)
        except Exception:                      # brak zdjecia / EXIF: pomin w regresji
            alt.append(float("nan"))
    h, alt = np.array(h), np.array(alt)
    ok = np.isfinite(alt)
    checks.append(check_vertical_scale(h[ok], alt[ok], q))

    cam = next(iter(rec.cameras.values()))
    w, hh = int(cam["width"]), int(cam["height"])
    f35 = None
    try:
        from PIL import Image
        with Image.open(config.photos_dir / names[0]) as im:
            f35 = im.getexif().get_ifd(0x8769).get(0xA405)     # FocalLengthIn35mmFilm
    except Exception:
        pass
    checks.append(check_focal(float(cam.get("focal_x", cam.get("focal", 0.0))), q, f35, w, hh))

    try:
        checks.append(_surface_check(config, frame, np.array(xy)))
    except (FileNotFoundError, ValueError, IndexError) as exc:
        checks.append(QcCheck("surface", NA, None, f"zgodność DSM ODM z NMPT: pominięta ({exc})"))
    checks.append(check_baseline(np.array(xy), config.georef.sigma_gps_xy_m, q))
    rep = summarize(checks)
    for line in rep.lines():
        log.info(line)
    _CACHE[key] = rep
    return rep


def _surface_check(config, frame, cams_xy) -> QcCheck:
    from .align import dsm_points_local
    from .surface import _bounds_in, _read_crs, area_bounds_2180, load_mosaic, raster_files, sample_grid
    dsm = Path(config.odm_dsm_path)
    if not dsm.exists():
        raise FileNotFoundError(f"brak {dsm}")
    src, label = (config.surface.nmpt_dir, "NMPT") if config.surface.nmpt_dir else (config.surface.nmt_dir, "NMT")
    files = raster_files(src)
    crs = _read_crs(files[0], config.surface.default_crs)
    P = dsm_points_local(dsm, frame, config.qc.sample_step_m)
    grid = load_mosaic(files, config.surface.default_crs,
                       _bounds_in(crs, area_bounds_2180(config, np.vstack([cams_xy, P[:, :2]]))))
    # DSM wygladzony do oczka referencji: porownanie 1 m z 1 m
    P = dsm_points_local(dsm, frame, config.qc.sample_step_m, smooth_m=grid.res_x)
    H = frame.npm_height(P[:, 2], P[:, 0], P[:, 1])
    return check_surface(H - sample_grid(grid, P[:, 0], P[:, 1], "bilinear"), config.qc, label)
