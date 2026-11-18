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
  surface, tilt   — reszty DSM ODM - NMPT dla AKTYWNEJ georeferencji (z_anchor z ewentualnym
                    apply_tilt albo 7p), obok wartosci "przed" (gps_only + z_offset). Do reszt
                    dopasowana odporna plaszczyzna: tilt = jej pochylenie (przechyl modelu
                    wzgledem NMPT — wplywa na polozenie trafien proporcjonalnie do H nad gruntem),
                    surface = NMAD reszt po usunieciu plaszczyzny (zgodnosc ksztaltu: deformacje,
                    obrot, przesuniecie poziome). Pochylenie ~2-3 deg typowe przy gps_only (tez
                    w dobrym locie) — OSTRZEZENIE, nie ZLY; 7p je usuwa.
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
INFO = "informacja"     # wynik do wiadomosci, nie wplywa na ocene (np. ogniskowa vs EXIF)
_RANK = {NA: -1, INFO: -1, OK: 0, WARN: 1, BAD: 2}
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

    @classmethod
    def from_dict(cls, d: dict) -> "QcReport":
        """Odwrotnosc to_dict (run_report.json -> panel QC w GUI)."""
        checks = [QcCheck(c["name"], c["status"], c.get("value"), c.get("text", ""), c.get("details") or {})
                  for c in d.get("checks", [])]
        status = d.get("status", NA)
        # raporty sprzed statusu "informacja": ogniskowa wzgledem EXIF bywala ostrzezeniem
        fixed = False
        for c in checks:
            if c.name == "focal" and c.status == WARN and "EXIF" in str(c.details.get("source", "")):
                c.status, fixed = INFO, True
        if fixed:
            status = max((c.status for c in checks), key=lambda x: _RANK[x], default=NA)
            status = OK if status in (NA, INFO) else status
        return cls(checks, status, list(d.get("recommendations", [])))


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
        # EXIF nie uwzglednia kadrowania matrycy: odchylenie to informacja, nie ostrzezenie
        ref, src, cap = expected_focal_exif(f35, width, height), f"EXIF f35 = {f35:g} mm", INFO
    else:
        return QcCheck("focal", NA, focal, f"ogniskowa ODM {focal:.4f}: brak odniesienia (EXIF f35 / qc.camera_focal)")
    d = 100 * (focal / ref - 1)
    st = _grade(abs(d), cfg.focal_warn_pct, cfg.focal_bad_pct)
    if cap == INFO and st != OK:
        st = INFO
    note = "" if cap == BAD else " (EXIF nie uwzględnia kadrowania: tylko informacja)"
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


def fit_plane(xy, r, iters: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Odporna plaszczyzna r ~ a*(x-x0) + b*(y-y0) + c (LSQ z odrzucaniem |e| > 3 NMAD).
    Zwraca (a, b, c) i maske punktow uzytych."""
    xy = np.asarray(xy, float)
    r = np.asarray(r, float)
    ok = np.isfinite(r)
    A = np.column_stack([xy - xy[ok].mean(0), np.ones(len(r))])
    # start bez punktow odstajacych (roslinnosc, zmiany terenu: ogon reszt zaburza LSQ)
    med = np.median(r[ok])
    nmad0 = 1.4826 * np.median(np.abs(r[ok] - med))
    use = ok & (np.abs(r - med) <= max(2 * nmad0, 0.5))
    coef = np.zeros(3)
    for _ in range(iters):
        coef = np.linalg.lstsq(A[use], r[use], rcond=None)[0]
        e = r - A @ coef
        nmad = 1.4826 * np.median(np.abs(e[ok] - np.median(e[ok])))
        use = ok & (np.abs(e) <= max(3 * nmad, 0.3))
    return coef, use


def surface_stats(xy, residuals) -> dict:
    """NMAD surowych reszt, pochylenie odpornej plaszczyzny i NMAD po jej usunieciu."""
    r = np.asarray(residuals, float)
    ok = np.isfinite(r)
    xy = np.asarray(xy, float)[ok]
    r = r[ok]
    if len(r) < 20:
        return {"n": int(len(r))}
    nmad = lambda v: float(1.4826 * np.median(np.abs(v - np.median(v))))  # noqa: E731
    coef, _ = fit_plane(xy, r)
    e = r - np.column_stack([xy - xy.mean(0), np.ones(len(r))]) @ coef
    return {"n": int(len(r)), "median": float(np.median(r)), "nmad_raw": nmad(r),
            "tilt_deg": float(math.degrees(math.atan(math.hypot(coef[0], coef[1])))),
            "tilt_azimuth_deg": float(math.degrees(math.atan2(coef[0], coef[1])) % 360),
            "nmad_plane": nmad(e)}


def check_surface_modes(before: Optional[dict], after: dict, cfg, ref_name: str = "NMPT",
                        mode: str = "") -> list[QcCheck]:
    """Kontrole surface (ksztalt, NMAD po usunieciu plaszczyzny) i tilt dla trybu aktywnego;
    wartosci "przed" (gps_only) w opisie i szczegolach."""
    if after.get("n", 0) < 20:
        return [QcCheck("surface", NA, None, f"zgodność DSM ODM z {ref_name}: za mało punktów")]
    b = before if before and before.get("n", 0) >= 20 else None
    fmt = lambda k, u: (f"{b[k]:.2f}{u} → " if b else "")  # noqa: E731
    lab = f" [{mode}]" if mode else ""
    surf = QcCheck("surface", _grade(after["nmad_plane"], cfg.surface_nmad_warn_m, cfg.surface_nmad_bad_m),
                   after["nmad_plane"],
                   f"DSM ODM − {ref_name}{lab}: NMAD bez pochylenia {fmt('nmad_plane', '')}{after['nmad_plane']:.2f} m, "
                   f"surowe {fmt('nmad_raw', '')}{after['nmad_raw']:.2f} m (przed → po; n {after['n']}, "
                   f"≤ {cfg.core_radius_m:.0f} m od kamer)",
                   {"before": b, "after": after, "ref": ref_name, "mode": mode})
    tilt = QcCheck("tilt", _grade(after["tilt_deg"], cfg.tilt_warn_deg, cfg.tilt_bad_deg), after["tilt_deg"],
                   f"pochylenie modelu względem {ref_name}{lab}: {fmt('tilt_deg', '°')}{after['tilt_deg']:.2f}° "
                   f"(azymut spadku {after['tilt_azimuth_deg']:.0f}°)",
                   {"before": b, "after": after, "mode": mode})
    return [surf, tilt]


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


def summarize(checks: list[QcCheck], aligned: bool = False) -> QcReport:
    """aligned=True (georef 7p aktywne): bez zalecenia 7p."""
    status = max((c.status for c in checks), key=lambda s: _RANK[s], default=NA)
    if status == NA:
        status = OK
    by = {c.name: c for c in checks}
    rec = []
    geom_bad = any(by.get(n) and by[n].status in (WARN, BAD) for n in ("vertical_scale",)) or (
        by.get("focal") and by["focal"].status == BAD)
    if geom_bad:
        rec.append(REC_CALIB)
    if not aligned and any(by.get(n) and by[n].status in (WARN, BAD) for n in ("surface", "tilt", "baseline")):
        rec.append(REC_7P + (" — po ustaleniu kalibracji" if geom_bad else ""))
    if by.get("baseline") and by["baseline"].status == BAD:
        rec.append(REC_BASE)
    return QcReport(checks, status, rec)


# ───────────── uruchomienie na rekonstrukcji ─────────────

_CACHE: dict = {}


def run_qc(config, rec, frame, dem=None, active=None, mode: str = "gps_only") -> QcReport:
    """rec — rekonstrukcja w georeferencji gps_only; frame — z_offset z z_anchor (bez pochylenia:
    stan "przed"). active(P) -> (x, y, H npm) dla punktow DSM w ukladzie L — odwzorowanie trybu
    aktywnego (apply_tilt / 7p); None = jak frame. mode — etykieta trybu aktywnego."""
    from .exif import read_gps
    q = config.qc
    probe = (np.round(np.asarray(active(np.array([[0.0, 0.0, 0.0], [100.0, 50.0, 10.0]])), float), 4).tolist()
             if active is not None else None)          # odcisk odwzorowania aktywnego (klucz cache)
    key = json.dumps([str(rec.path), str(config.odm_dsm_path), round(frame.z_offset, 4), mode, probe,
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
        checks += _surface_checks(config, frame, np.array(xy), active, mode)
    except (FileNotFoundError, ValueError, IndexError) as exc:
        checks.append(QcCheck("surface", NA, None, f"zgodność DSM ODM z NMPT: pominięta ({exc})"))
    aligned = mode.startswith("similarity")
    base = check_baseline(np.array(xy), config.georef.sigma_gps_xy_m, q)
    if aligned and base.status == BAD:
        # orientacja z wyrownania do NMPT, nie z GPS — baza sprawdzana posrednio przez tilt/surface
        base = QcCheck(base.name, WARN, base.value, base.text + " — orientacja z 7p (kontrola: pochylenie "
                       "i NMAD po wyrównaniu)", base.details)
    checks.append(base)
    rep = summarize(checks, aligned=aligned)
    for line in rep.lines():
        log.info(line)
    _CACHE[key] = rep
    return rep


def _surface_checks(config, frame, cams_xy, active, mode) -> list[QcCheck]:
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

    # strefa rdzeniowa: w poblizu kamer (polozenie z ukladu gps_only; 7p przesuwa o kilka m)
    from scipy.spatial import cKDTree
    d, _ = cKDTree(cams_xy).query(P[:, :2])
    P = P[d <= config.qc.core_radius_m]

    def resid(Q):
        return Q[:, :2], Q[:, 2] - sample_grid(grid, Q[:, 0], Q[:, 1], "bilinear")
    before = np.column_stack([P[:, :2], frame.npm_height(P[:, 2], P[:, 0], P[:, 1])])
    sb = surface_stats(*resid(before))
    sa = surface_stats(*resid(np.asarray(active(P), float))) if active is not None else sb
    return check_surface_modes(sb, sa, config.qc, label, mode)


# ───────────── opisy dla uzytkownika spoza projektu (GUI, raport) ─────────────

TITLES = {"vertical_scale": "Skala pionowa modelu", "focal": "Kalibracja kamery",
          "surface": "Zgodność z NMPT GUGiK", "tilt": "Pochylenie modelu",
          "baseline": "Zasięg lotu"}

# zalecenia w jezyku interfejsu (GUI); action: kod akcji do przycisku "Zastosuj"
ACTIONS = {
    REC_CALIB: ("Uruchom rekonstrukcję ponownie z ustaloną kalibracją kamery (strona Projekt).", "odm_fixed_cal"),
    REC_7P: ("Na stronie Georeferencja wybierz wyrównanie do NMPT.", "georef_nmpt"),
    REC_BASE: ("Przy kolejnym locie obejmij większy obszar, lataj na kilku wysokościach albo dodaj punkty "
               "kontrolne.", None),
}


def explain(c: QcCheck) -> str:
    """Jedno zdanie prostym jezykiem: co wyszlo i co zrobic."""
    d, v, st = c.details or {}, c.value, c.status
    if st == NA or v is None:
        return "Brak danych do tej kontroli."
    aligned = str(d.get("mode", "")).startswith("similarity")
    if c.name == "vertical_scale":
        if st == OK:
            return f"Wysokości w modelu zgadzają się z barometrem drona (skala {v:.2f})."
        kind = "spłaszczony" if v < 1 else "rozciągnięty"
        return (f"Model jest {kind} w pionie (skala {v:.2f} zamiast 1), zwykle przez błędną kalibrację kamery.")
    if c.name == "focal":
        if st == OK:
            return f"Ogniskowa kamery w modelu jest zgodna z oczekiwaną ({v:+.0f}%)."
        if st == INFO:
            return (f"Ogniskowa różni się o {v:+.0f}% od wartości z EXIF. To normalne, bo EXIF nie uwzględnia "
                    "kadrowania obrazu.")
        return f"Ogniskowa różni się o {v:+.0f}% od kalibracji referencyjnej, model może być zdeformowany."
    if c.name == "surface":
        if st == OK:
            return f"Kształt modelu zgadza się z NMPT GUGiK (typowa różnica {v:.2f} m)."
        return (f"Kształt modelu odbiega od NMPT GUGiK (typowa różnica {v:.2f} m): model jest zdeformowany "
                "albo teren się zmienił.")
    if c.name == "tilt":
        b = (d.get("before") or {}).get("tilt_deg")
        if aligned:
            pre = (f"Po wyrównaniu do NMPT pochylenie spadło z {b:.1f}° do {v:.1f}°." if b is not None else
                   f"Pochylenie po wyrównaniu do NMPT: {v:.1f}°.")
            return pre + ("" if st == OK else " Nadal duże, sprawdź kalibrację kamery.")
        if st == OK:
            return f"Model jest prawie poziomy (pochylenie {v:.1f}°)."
        return f"Model jest pochylony o {v:.1f}°, obiekty wysoko nad gruntem będą przesunięte."
    if c.name == "baseline":
        span = d.get("span_m")
        sp = f"Zdjęcia obejmują obszar ok. {span:.0f} m" if span else "Obszar zdjęć"
        if st == OK:
            return f"{sp}, to wystarcza do ustalenia orientacji modelu z GPS."
        return (f"{sp}; przy dokładności GPS ±{d.get('sigma_gps_m', 3):.0f} m orientacja modelu może być błędna "
                f"o ok. {v:.0f}°." + (" Orientację poprawiło wyrównanie do NMPT." if aligned else ""))
    return c.text


def explain_report(rep: QcReport, top: int = 3) -> dict:
    """Panel QC w GUI: {"status", "summary", "checks": [(tytul, status, opis)] od najwazniejszych,
    "top": ile pokazac od razu, "actions": [(tekst, kod_akcji|None)]}. Kontrole informacyjne na koncu."""
    summary = {OK: "Rekonstrukcja wygląda dobrze.",
               WARN: "Rekonstrukcja jest użyteczna, ale ma słabsze strony.",
               BAD: "Rekonstrukcja ma poważne problemy, wyniki mogą być przesunięte o kilka metrów.",
               NA: "Brak danych do oceny."}[rep.status]
    order = sorted(rep.checks, key=lambda c: -_RANK[c.status])
    actions = [ACTIONS.get(_rec_key(r), (r, None)) for r in rep.recommendations]
    return {"status": rep.status, "summary": summary,
            "checks": [(TITLES.get(c.name, c.name), c.status, explain(c)) for c in order],
            "top": min(top, sum(1 for c in order if c.status in (WARN, BAD)) or top),
            "actions": actions, "recommendations": list(rep.recommendations)}


def _rec_key(r: str) -> str:
    return next((k for k in ACTIONS if r.startswith(k)), r)


# ───────────── automatyczny wybor georeferencji ─────────────

def auto_georef_choice(rep: Optional[QcReport]) -> tuple[str, str]:
    """georef.mode "auto": (tryb, uzasadnienie jednym zdaniem) z QC wariantu gps_only.
    Wyrownanie do NMPT, gdy QC je zaleca (zly ksztalt, pochylenie albo slaba baza)."""
    if rep is None:
        return "gps_only", "Wybrano standard (GPS drona), bo brak danych do kontroli jakości."
    by = {c.name: c for c in rep.checks}
    bad = [by[n] for n in ("tilt", "surface", "baseline") if n in by and by[n].status in (WARN, BAD)]
    if not bad:
        return "gps_only", "Wybrano standard (GPS drona), bo model jest zgodny z NMPT GUGiK."
    c = bad[0]
    why = {"tilt": f"model jest pochylony o {c.value:.1f}°" if c.value is not None else "model jest pochylony",
           "surface": "kształt modelu odbiega od NMPT", "baseline": "zasięg lotu jest mały"}[c.name]
    return "similarity_nmpt", f"Wybrano wyrównanie do NMPT, bo {why}."
