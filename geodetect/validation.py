"""Walidacja wzgledem niezaleznie pomierzonego punktu (eksperyment "pilka na dachu").

Uzywane przez geodetect.experiments, gdy eksperyment ma metrics.reference_point:

    reference_point:
      latlon: [lat, lon]          # GT obiektu (WGS84)
      cls: sports ball            # klasa detekcji odpowiadajaca obiektowi GT
      object_height_m: 0.11       # srodek obiektu nad powierzchnia, na ktorej lezy
      profiles: 3                 # profile promieni (rysunek do pracy)
      reference_json: output/pilka/reference/reference.json   # scripts/pilka_reference.py
      uncertainty: [{component, sigma_m, applies_to, note}]   # budzet niepewnosci GT

Metryki per detekcja (wzgledem GT): dx, dy, blad poziomy, skladowe wzdluz / w poprzek
poziomego kierunku kamera -> GT (+ wzdluz = dalej od kamery), wysokosc trafienia vs
wysokosc dachu w punkcie GT (NMPT i DSM ODM), kat padania, wysokosc kamery nad dachem
i nad terenem, odleglosc. Per wariant: statystyki bledow, pozycja zagregowana (DBSCAN
z zadania 2) i jej blad, RMS wielowidokowy. Hipoteza NMT: blad wzdluz promienia ~
h_dachu * tan(incidence).
"""

from __future__ import annotations

import csv
import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

log = logging.getLogger(__name__)

SURFACE_LABEL = {"nmt_local": "NMT", "nmpt_local": "NMPT", "odm_dsm": "DSM ODM",
                 "nmt_gugik_rest": "NMT (REST)"}


# ───────────── geometria wzgledem GT ─────────────

def along_across(hit_xy, gt_xy, cam_xy) -> tuple[float, float]:
    """Skladowe bledu (hit - GT) wzdluz poziomego kierunku kamera -> GT (+ = dalej od
    kamery) i w poprzek (+ = w prawo patrzac od kamery)."""
    u = np.array([gt_xy[0] - cam_xy[0], gt_xy[1] - cam_xy[1]], float)
    n = np.linalg.norm(u)
    if n < 1e-9:
        return float("nan"), float("nan")
    u /= n
    e = np.array([hit_xy[0] - gt_xy[0], hit_xy[1] - gt_xy[1]], float)
    return float(e @ u), float(u[1] * e[0] - u[0] * e[1])


def predicted_overshoot(height_above_hit_surface: float, incidence_deg: float) -> float:
    """Przesuniecie poziome trafienia, gdy promien przechodzi przez obiekt na wysokosci h
    nad powierzchnia, na ktora jest rzutowany: h * tan(incidence) (od kamery)."""
    return height_above_hit_surface * math.tan(math.radians(incidence_deg))


def fit_line(x, y) -> Optional[dict]:
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3 or np.ptp(x) < 1e-9:
        return None
    a, b = np.polyfit(x, y, 1)
    r = float(np.corrcoef(x, y)[0, 1]) if np.std(y) > 1e-12 else float("nan")
    res = y - (a * x + b)
    return {"slope": float(a), "intercept": float(b), "r": r, "n": int(len(x)),
            "rms_resid": float(np.sqrt(np.mean(res ** 2)))}


def helmert2d(src, dst) -> dict:
    """Transformacja Helmerta 2D (przesuniecie, obrot, skala) src -> dst metoda najmniejszych
    kwadratow. 1 para: samo przesuniecie. Zwraca {apply, n, scale, rot_deg, shift, rms, residuals}."""
    src, dst = np.asarray(src, float).reshape(-1, 2), np.asarray(dst, float).reshape(-1, 2)
    n = len(src)
    if n == 0:
        raise ValueError("helmert2d: brak par punktow")
    cs, cd = src.mean(0), dst.mean(0)
    if n == 1:
        a, b = 1.0, 0.0
    else:
        s0, d0 = src - cs, dst - cd
        den = float((s0 ** 2).sum())
        a = float((s0[:, 0] * d0[:, 0] + s0[:, 1] * d0[:, 1]).sum()) / den
        b = float((s0[:, 0] * d0[:, 1] - s0[:, 1] * d0[:, 0]).sum()) / den
    R = np.array([[a, -b], [b, a]])

    def apply(pts):
        p = np.asarray(pts, float).reshape(-1, 2)
        return (p - cs) @ R.T + cd
    res = dst - apply(src)
    dof = 2 * n - (4 if n >= 2 else 2)
    rms = float(np.sqrt((res ** 2).sum() / dof)) if dof > 0 else float("nan")
    return {"apply": apply, "n": n, "scale": float(np.hypot(a, b)),
            "rot_deg": float(np.degrees(np.arctan2(b, a))), "shift": (cd - cs).tolist(),
            "rms": rms, "residuals": np.hypot(res[:, 0], res[:, 1]).tolist()}


def gt_transfer(pairs_csv: Path, lean: Optional[dict], height_fn=None) -> Optional[dict]:
    """GT_transfer: pilka na ortofotomapie ODM -> (Helmert z punktow dachu) -> polozenie prawdziwe.

    Tryb "height" (lean z vx_per_m/vy_per_m i height_fn(x, y) = NMPT - NMT): kazdy punkt dachu
    z ortofotomapy GUGiK jest korygowany o przesuniecie proporcjonalne do JEGO wysokosci
    (geodetect/lean.py), Helmert ODM -> polozenia prawdziwe, pilka bez dodatkowej korekty
    (ortofotomapa ODM jest rektyfikowana na wlasnym DSM). Tryb "uniform" (dawny): Helmert do
    polozen na ortofotomapie, potem jeden wektor lean odjety od pilki."""
    import csv
    if not pairs_csv.exists():
        return None
    with open(pairs_csv, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    pts = [r for r in rows if r["kind"] == "punkt_dachu"]
    ball = [r for r in rows if r["kind"] == "pilka"]
    if not pts or not ball:
        log.warning("GT_transfer: %s — potrzebne punkty dachu i pilka (jest %d / %d)",
                    pairs_csv, len(pts), len(ball))
        return None
    src = [(float(r["odm_x"]), float(r["odm_y"])) for r in pts]
    dst = [(float(r["gugik_x"]), float(r["gugik_y"])) for r in pts]
    h = helmert2d(src, dst)
    bo = (float(ball[0]["odm_x"]), float(ball[0]["odm_y"]))
    bg = h["apply"](bo)[0]
    ldx, ldy = (lean["dx_m"], lean["dy_m"]) if lean else (0.0, 0.0)
    out = {"n_pairs": h["n"], "scale": h["scale"], "rot_deg": h["rot_deg"], "helmert_rms": h["rms"],
           "residuals": h["residuals"], "ball_odm": bo, "ball_gugik": bg.tolist(),
           "lean_applied": [ldx, ldy], "x": float(bg[0] - ldx), "y": float(bg[1] - ldy),
           "odm_vs_gugik_shift": [float(bg[0] - bo[0]), float(bg[1] - bo[1])], "lean_mode": "uniform"}
    out["x_uniform"], out["y_uniform"] = out["x"], out["y"]
    if lean and lean.get("vx_per_m") is not None and height_fn is not None:
        from .lean import LeanModel
        lm = LeanModel(lean["vx_per_m"], lean["vy_per_m"], lean.get("h_edges_m") or float("nan"), ldx, ldy)
        corr = [lm.correct(x, y, height_fn) for x, y in dst]
        hc = helmert2d(src, [(c[0], c[1]) for c in corr])
        bt = hc["apply"](bo)[0]
        out.update({"lean_mode": "height", "x": float(bt[0]), "y": float(bt[1]), "helmert_rms": hc["rms"],
                    "scale": hc["scale"], "rot_deg": hc["rot_deg"], "residuals": hc["residuals"],
                    "lean_v_per_m": [lm.vx, lm.vy], "lean_h_edges_m": lm.h_edges_m,
                    "roof_point_heights": [c[2] for c in corr],
                    "roof_point_shifts": [float(math.hypot(x - c[0], y - c[1])) for (x, y), c in zip(dst, corr)],
                    "ball_height": float(height_fn(float(bt[0]), float(bt[1]))),
                    "shift_vs_uniform": [float(bt[0] - out["x_uniform"]), float(bt[1] - out["y_uniform"])]})
    log.info("GT_transfer (%s): %d par, skala %.4f, obrot %.2f deg, RMS Helmerta %.3f m -> (%.2f, %.2f)",
             out["lean_mode"], h["n"], out["scale"], out["rot_deg"], out["helmert_rms"], out["x"], out["y"])
    return out


def combined_sigma(sigmas) -> float:
    s = [float(v) for v in sigmas if v is not None and np.isfinite(v)]
    return float(math.sqrt(sum(v * v for v in s))) if s else float("nan")


def _rms(v) -> float:
    v = np.asarray([x for x in v if np.isfinite(x)], float)
    return float(np.sqrt(np.mean(v ** 2))) if len(v) else float("nan")


def _f(v, fmt="{:.2f}") -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "–"
    return fmt.format(v)


# ───────────── powierzchnie w punkcie ─────────────

class PointSampler:
    """Wartosci powierzchni (npm) w punktach EPSG:2180; DSM/DTM ODM przeliczane przez
    frame wariantu (h_ODM + z_offset)."""

    def __init__(self):
        self._tr: dict[str, Any] = {}

    def __call__(self, surface, x: float, y: float, frame=None) -> float:
        from pyproj import CRS, Transformer
        from .surface import VERTICAL_ODM
        if surface is None:
            return float("nan")
        crs = getattr(surface, "crs", "EPSG:2180")
        if CRS.from_user_input(crs) == CRS.from_epsg(2180):
            sx, sy = x, y
        else:
            if crs not in self._tr:
                self._tr[crs] = Transformer.from_crs("EPSG:2180", crs, always_xy=True)
            sx, sy = self._tr[crs].transform(x, y)
        v = float(surface.sample_many(np.array([sx]), np.array([sy]))[0])
        if surface.vertical == VERTICAL_ODM:
            if frame is None:
                return float("nan")
            v = float(frame.npm_height(v, x, y))
        return v


def odm_dtm_surface(cfg, bounds_2180):
    from .surface import VERTICAL_ODM, RasterSurface, _bounds_in, _read_crs, load_mosaic
    path = cfg.odm_dtm_path
    if not Path(path).exists():
        return None
    crs = _read_crs(Path(path), cfg.surface.default_crs)
    grid = load_mosaic([Path(path)], cfg.surface.default_crs, _bounds_in(crs, bounds_2180))
    return RasterSurface(grid, "odm_dtm", VERTICAL_ODM, cfg.surface.interpolation)


# ───────────── raport ─────────────

@dataclass
class ReferenceReport:
    spec: dict
    gt_x: float
    gt_y: float
    rows: list[dict] = field(default_factory=list)          # per wariant x detekcja
    variants: list[dict] = field(default_factory=list)      # per wariant
    calib: list[dict] = field(default_factory=list)         # kalibracje Z
    nmt_fit: list[dict] = field(default_factory=list)
    trend_fit: list[dict] = field(default_factory=list)
    roof: dict = field(default_factory=dict)                # wysokosci w GT
    reference: Optional[dict] = None                        # reference.json
    gt_corr: Optional[tuple[float, float]] = None           # GT skorygowany o przesuniecie dachu
    transfer: Optional[dict] = None                         # GT_transfer (ortofotomapa ODM -> GUGiK)
    budget: list[dict] = field(default_factory=list)
    plots: list[Path] = field(default_factory=list)
    profiles: list[Path] = field(default_factory=list)
    skipped: dict = field(default_factory=dict)


_BASE_REC: dict = {}      # rekonstrukcja bazowa eksperymentu (bez sufiksu w etykietach)


def _variant_axes(vr) -> dict:
    c = vr.cfg
    z = c.z_anchor.mode
    if c.georef.mode != "gps_only":
        z = "7p" + ("" if c.georef.use_baro_prior else " bez baro")
    rec = Path(c.odm_dir).name
    if rec != _BASE_REC.get("name", rec):
        z = f"{z} [{rec}]"
    return {"surface": c.surface.source, "z_mode": z,
            "pixel": {"manual": "manual", "csv": "detekcja_best"}.get(c.detect.source, "detekcja")}


def model_roof_height(cfg, frame, aln, ball_xy, radius_m: float = 12.0) -> float:
    """Wysokosc dachu w modelu ODM w miejscu pilki: H(DSM w pilce) - 10. percentyl H DSM w promieniu
    radius_m (grunt wokol budynku). Niezalezna od DTM ODM (pod dachem interpolowany, w fixcal
    blednie sklasyfikowany) i od przesuniecia georeferencji; po 7p liczona w ukladzie wyrownanym."""
    if not ball_xy or not Path(cfg.odm_dsm_path).exists():
        return float("nan")
    from .align import dsm_points_local, raster_h
    P = dsm_points_local(Path(cfg.odm_dsm_path), frame, 0.2)
    P = P[np.hypot(P[:, 0] - ball_xy[0], P[:, 1] - ball_xy[1]) <= radius_m]
    top = np.array([[ball_xy[0], ball_xy[1], raster_h(cfg.odm_dsm_path, *ball_xy)]])
    if aln is not None:
        P, top = aln.result.sim.apply(P), aln.result.sim.apply(top)
    if len(P) < 50 or not np.isfinite(top[0, 2]):
        return float("nan")
    return float(top[0, 2] - np.percentile(P[:, 2], 10))


def build_report(runner) -> ReferenceReport:
    import copy
    from pyproj import Transformer
    spec = runner.exp.metrics.reference_point
    lat, lon = spec["latlon"]
    gx, gy = Transformer.from_crs(4326, 2180, always_xy=True).transform(lon, lat)
    rep = ReferenceReport(spec, gx, gy)
    cls = spec.get("cls")
    obj_h = float(spec.get("object_height_m", 0.0))
    ref_vr = runner.results[runner.exp.reference_variant]
    sample = PointSampler()

    def surf(src):
        c = copy.deepcopy(ref_vr.cfg)
        c.surface.source = src
        try:
            return runner.surface(c)
        except (FileNotFoundError, ValueError) as exc:
            log.warning("Walidacja: brak powierzchni %s (%s)", src, exc)
            return None
    nmt, nmpt, dsm = surf("nmt_local"), surf("nmpt_local"), surf("odm_dsm")
    dtm = odm_dtm_surface(ref_vr.cfg, (gx - 200, gy - 200, gx + 200, gy + 200))
    rj = spec.get("reference_json")
    if rj:
        p = Path(rj) if Path(rj).is_absolute() else runner.exp.base_config.parent / rj
        if p.exists():
            rep.reference = json.loads(p.read_text(encoding="utf-8"))
    le = (rep.reference or {}).get("lean_gugik")
    if le and spec.get("lean_correction", True) and not le.get("at_search_bound"):
        # dach na ortofotomapie przesuniety o (dx, dy) wzgledem podstawy -> prawdziwe polozenie
        # punktu na dachu = odczyt z ortofotomapy - (dx, dy)
        rep.gt_corr = (gx - le["dx_m"], gy - le["dy_m"])
    tp = spec.get("ortho_pairs_csv")
    hfn = None
    if tp:
        tp = Path(tp) if Path(tp).is_absolute() else runner.exp.base_config.parent / tp
        if spec.get("lean_height_dependent", True) and nmt is not None and nmpt is not None:
            def hfn(x, y):
                return sample(nmpt, x, y) - sample(nmt, x, y)
        rep.transfer = gt_transfer(tp, le if le and not le.get("at_search_bound") else None, hfn)
    gts = {"corr": rep.gt_corr, "transfer": (rep.transfer["x"], rep.transfer["y"]) if rep.transfer else None,
           # pilka na ortofotomapie ODM: odniesienie w ukladzie ODM (spojnosc wewnetrzna, bez bledu
           # georeferencji rekonstrukcji)
           "odm": tuple(rep.transfer["ball_odm"]) if rep.transfer else None}

    def err_to(key, x, y, g_over=None):
        g = (g_over or gts)[key]
        return math.hypot(x - g[0], y - g[1]) if g else float("nan")
    rep.roof = {"nmt": sample(nmt, gx, gy), "nmpt": sample(nmpt, gx, gy)}
    rep.roof["roof_h_nmpt"] = rep.roof["nmpt"] - rep.roof["nmt"]

    base_gts = gts
    pairs_by_rec = {k: (Path(v) if Path(v).is_absolute() else runner.exp.base_config.parent / v)
                    for k, v in (spec.get("ortho_pairs_by_rec") or {}).items()}
    lean_xy = ((le["dx_m"], le["dy_m"]) if le and not le.get("at_search_bound") else (0.0, 0.0))

    def rec_pairs(vr):
        """Punkty dachu i pilka na ortofotomapie TEJ rekonstrukcji (EPSG:2180) + GUGiK - lean."""
        import csv as _csv
        rec = Path(vr.cfg.odm_dir).name
        f = pairs_by_rec.get(rec) or (tp if rec == _BASE_REC.get("name") else None)
        if not f or not Path(f).exists():
            return None, []
        rows_ = list(_csv.DictReader(open(f, encoding="utf-8")))
        ball = next(((float(r["odm_x"]), float(r["odm_y"])) for r in rows_ if r["kind"] == "pilka"), None)
        roof = [((float(r["odm_x"]), float(r["odm_y"])), roof_true(float(r["gugik_x"]), float(r["gugik_y"])))
                for r in rows_ if r["kind"] == "punkt_dachu"]
        return ball, roof

    _roof_cache: dict = {}

    def roof_true(x, y):
        """Punkt dachu z ortofotomapy GUGiK -> polozenie prawdziwe (lean zalezny od wysokosci, jesli
        dostepny, inaczej staly wektor)."""
        if (x, y) not in _roof_cache:
            if rep.transfer and rep.transfer.get("lean_mode") == "height":
                from .lean import LeanModel
                lm = LeanModel(*rep.transfer["lean_v_per_m"], rep.transfer["lean_h_edges_m"], *lean_xy)
                _roof_cache[(x, y)] = lm.correct(x, y, hfn)[:2]
            else:
                _roof_cache[(x, y)] = (x - lean_xy[0], y - lean_xy[1])
        return _roof_cache[(x, y)]

    first = next(iter(runner.results.values()))
    _BASE_REC.setdefault("name", Path(spec.get("base_odm_dir") or first.cfg.odm_dir).name)
    for name, vr in runner.results.items():
        ax = _variant_axes(vr)
        fr = vr.ctx.frame
        aln = getattr(vr.ctx, "alignment", None)
        gts = dict(base_gts)
        ball_rec, roof_rec = rec_pairs(vr)
        gts["odm"] = ball_rec
        roof_model = model_roof_height(vr.cfg, fr, aln, ball_rec, float(spec.get("roof_ground_radius_m", 12.0)))
        roof_err = []
        for (ox, oy), (gx_, gy_) in roof_rec:
            q = aln.map_odm_point(ox, oy) if aln is not None else (ox, oy)
            roof_err.append(math.hypot(q[0] - gx_, q[1] - gy_))
        if aln is not None:
            # georef.mode=similarity_nmpt: model ODM (DSM/DTM, pilka na ortofotomapie ODM) w ukladzie
            # wyrownanym — odniesienie "odm" przeliczone ta sama transformacja 7p
            dsm_gt = aln.model_H(aln.dsm_path, gx, gy)
            dtm_gt = aln.model_H(vr.cfg.odm_dtm_path, gx, gy) if Path(vr.cfg.odm_dtm_path).exists() else float("nan")
            if ball_rec:
                gts["odm"] = tuple(aln.map_odm_point(*ball_rec)[:2])
        elif Path(vr.cfg.odm_dir) == Path(ref_vr.cfg.odm_dir):
            dsm_gt, dtm_gt = sample(dsm, gx, gy, fr), sample(dtm, gx, gy, fr)
        else:                      # inna rekonstrukcja niz wariant referencyjny
            c_ = copy.deepcopy(vr.cfg)
            c_.surface.source = "odm_dsm"
            dsm_gt = sample(runner.surface(c_), gx, gy, fr)
            dtm_gt = sample(odm_dtm_surface(vr.cfg, (gx - 200, gy - 200, gx + 200, gy + 200)), gx, gy, fr)
        cams = {r.image_name: r.camera for r in vr.batch.images}
        recs = []
        for gd in vr.batch.detections:
            if cls and gd.detection.cls_name != cls:
                continue
            c = cams[gd.image_name]
            h = gd.hit
            dx, dy = h.x_2180 - gx, h.y_2180 - gy
            al, ac = along_across((h.x_2180, h.y_2180), (gx, gy), (c.x_2180, c.y_2180))
            nmt_hit = sample(nmt, h.x_2180, h.y_2180)
            roof_npm = rep.roof["nmpt"]
            rec = {"variant": name, **ax, "image": gd.image_name,
                   "u": round(gd.uv[0], 1), "v": round(gd.uv[1], 1), "conf": round(gd.detection.conf, 4),
                   "hit_x": h.x_2180, "hit_y": h.y_2180, "hit_h": h.h,
                   "dx": dx, "dy": dy, "err_m": math.hypot(dx, dy),
                   "err_corr_m": err_to("corr", h.x_2180, h.y_2180),
                   "err_transfer_m": err_to("transfer", h.x_2180, h.y_2180),
                   "err_odm_m": err_to("odm", h.x_2180, h.y_2180),
                   "along": al, "across": ac,
                   "dh_vs_nmpt": h.h - (roof_npm + obj_h), "dh_vs_dsm": h.h - (dsm_gt + obj_h),
                   "incidence_deg": gd.incidence_deg, "tan_inc": math.tan(math.radians(gd.incidence_deg)),
                   "cam_h": c.h, "agl_roof_nmpt": c.h - roof_npm, "agl_roof_dsm": c.h - dsm_gt,
                   "agl_ground": c.h - rep.roof["nmt"], "range_m": h.range_m,
                   "cam_dist_gt_m": math.hypot(c.x_2180 - gx, c.y_2180 - gy),
                   # NMT: promien przez obiekt na dachu schodzi do gruntu przy trafieniu
                   "pred_along_nmt": predicted_overshoot(roof_npm + obj_h - nmt_hit, gd.incidence_deg),
                   "pred_along_nmt_gt": predicted_overshoot(rep.roof["roof_h_nmpt"] + obj_h, gd.incidence_deg),
                   "pred_along_odm": predicted_overshoot(dsm_gt - dtm_gt + obj_h, gd.incidence_deg)}
            recs.append(rec)
        rep.rows += recs
        objs = [o for o in vr.batch.objects if (not cls or o.cls_name == cls)]
        main = max(objs, key=lambda o: o.n_views) if objs else None
        e = np.array([r["err_m"] for r in recs])
        allmean = (np.mean([r["hit_x"] for r in recs]), np.mean([r["hit_y"] for r in recs])) if recs else None
        rep.variants.append({
            "variant": name, **ax, "z_offset": vr.ctx.z_calibration.offset,
            "n": len(recs), "n_images": len(vr.batch.images),
            "err_median": float(np.median(e)) if len(e) else float("nan"),
            "err_rms": _rms(e), "err_max": float(e.max()) if len(e) else float("nan"),
            "along_mean": float(np.mean([r["along"] for r in recs])) if recs else float("nan"),
            "across_mean": float(np.mean([r["across"] for r in recs])) if recs else float("nan"),
            "along_rms": _rms([r["along"] for r in recs]), "across_rms": _rms([r["across"] for r in recs]),
            "dh_nmpt_median": float(np.median([r["dh_vs_nmpt"] for r in recs])) if recs else float("nan"),
            "dh_dsm_median": float(np.median([r["dh_vs_dsm"] for r in recs])) if recs else float("nan"),
            "obj_n_views": main.n_views if main else 0,
            "obj_err": math.hypot(main.x_2180 - gx, main.y_2180 - gy) if main else float("nan"),
            "obj_dx": main.x_2180 - gx if main else float("nan"),
            "obj_dy": main.y_2180 - gy if main else float("nan"),
            "obj_rms": main.rms_m if main else float("nan"),
            "obj_err_corr": err_to("corr", main.x_2180, main.y_2180) if main else float("nan"),
            "err_corr_median": float(np.median([r["err_corr_m"] for r in recs])) if recs else float("nan"),
            "obj_err_transfer": err_to("transfer", main.x_2180, main.y_2180) if main else float("nan"),
            "obj_err_odm": err_to("odm", main.x_2180, main.y_2180) if main else float("nan"),
            "err_odm_median": float(np.median([r["err_odm_m"] for r in recs])) if recs else float("nan"),
            "err_transfer_median": (float(np.median([r["err_transfer_m"] for r in recs]))
                                    if recs else float("nan")),
            "obj_h": main.h if main else float("nan"),
            "rec": Path(vr.cfg.odm_dir).name, "roof_ctrl": roof_err, "roof_model": roof_model,
            "roof_ctrl_rms": _rms(roof_err) if roof_err else float("nan"),
            "georef": vr.cfg.georef.mode + ("" if vr.cfg.georef.use_baro_prior else " bez baro"),
            "align": (aln.result.sim.summary() | {"nmad_all": aln.result.residuals_after["all"].get("nmad"),
                                                 "nmad_before": aln.result.residuals_before.get("nmad"),
                                                 "baro_k": aln.result.priors.get("baro_scale_k")})
                     if aln is not None else None,
            "qc_k": (vr.ctx.qc.check("vertical_scale").value
                     if getattr(vr.ctx, "qc", None) is not None and vr.ctx.qc.check("vertical_scale") else None),
            "n_objects": len(objs),
            "mean_err": math.hypot(allmean[0] - gx, allmean[1] - gy) if allmean else float("nan"),
            "dsm_gt": dsm_gt, "dtm_gt": dtm_gt})
        if ax["surface"] == "nmt_local":
            f = fit_line([r["pred_along_nmt"] for r in recs], [r["along"] for r in recs])
            if f:
                rep.nmt_fit.append({"variant": name, **f,
                                    "mean_ratio": float(np.median([r["along"] / r["pred_along_nmt"]
                                                                   for r in recs if r["pred_along_nmt"] > 0.1]))})
        f = fit_line([r["tan_inc"] for r in recs], [r["err_m"] for r in recs])
        if f:
            rep.trend_fit.append({"variant": name, **ax, **f})

    seen = set()
    for vr in runner.results.values():
        zc = vr.ctx.z_calibration
        if zc.method in seen:
            continue
        seen.add(zc.method)
        rep.calib.append({"method": zc.method, "offset": zc.offset,
                          "stats": {k: v for k, v in zc.stats.items()},
                          "alignment": (None if zc.alignment is None else
                                        {"n": len(zc.alignment.diff), "mad": float(np.median(np.abs(
                                            zc.alignment.diff - np.median(zc.alignment.diff)))),
                                         "slope_deg": zc.alignment.plane.slope_deg,
                                         "aspect_deg": zc.alignment.plane.aspect_deg})})

    rep.budget = list(spec.get("uncertainty") or [])
    if rep.reference and rep.reference.get("lean_gugik"):
        for b in rep.budget:
            if b.get("from") == "lean_gugik" and b.get("sigma_m") is None:
                b["sigma_m"] = rep.reference["lean_gugik"]["shift_m"]
    for b in rep.budget:
        if b.get("from") == "helmert_rms" and b.get("sigma_m") is None and rep.transfer:
            b["sigma_m"] = rep.transfer["helmert_rms"] if rep.transfer["n_pairs"] >= 3 else None
    return rep


# ───────────── wykresy / mapa / profile ─────────────

def plot_error_vs(rep: ReferenceReport, key: str, xlabel: str, path: Path, nmt_pred: bool) -> Optional[Path]:
    from .expplots import AQUA, BLUE, GREEN, INK2, ORANGE, PURPLE, YELLOW, _fig, _plt, _save
    ek = "err_transfer_m" if rep.transfer else "err_m"
    gt_lab = "GT_transfer" if rep.transfer else "GT"
    surfaces = [s for s in ("nmt_local", "nmpt_local", "odm_dsm") if any(r["surface"] == s for r in rep.rows)]
    if not surfaces:
        return None
    plt = _plt()
    fig, axes = _fig(plt, 1, len(surfaces), figsize=(4.6 * len(surfaces), 4.0), sharey=False)
    axes = np.atleast_1d(axes)
    style = {("start_point", "manual"): (BLUE, "o"), ("start_point", "detekcja"): (AQUA, "s"),
             ("start_point", "detekcja_best"): (YELLOW, "v"),
             ("dem_alignment", "manual"): (ORANGE, "^"), ("dem_alignment", "detekcja"): (PURPLE, "D"),
             ("dem_alignment", "detekcja_best"): (GREEN, "P")}
    for ax, s in zip(axes, surfaces):
        rows = [r for r in rep.rows if r["surface"] == s]
        for (zm, px), (col, mk) in style.items():
            rr = [r for r in rows if r["z_mode"] == zm and r["pixel"] == px]
            if s == "odm_dsm" and zm == "dem_alignment":
                continue
            if rr:
                ax.scatter([r[key] for r in rr], [r[ek] for r in rr], s=22, color=col, marker=mk,
                           label=f"{'start' if zm == 'start_point' else 'dem_al.'}, {px}", alpha=0.85)
        if nmt_pred and s == "nmt_local":
            rr = sorted([r for r in rows if r["z_mode"] == "start_point"], key=lambda r: r[key])
            if rr:
                ax.plot([r[key] for r in rr], [abs(r["pred_along_nmt"]) for r in rr], color=INK2,
                        linewidth=1.2, linestyle="--", label="h_dachu · tan(inc)")
        ax.set_title(SURFACE_LABEL.get(s, s), fontsize=10, color=INK2, loc="left")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(f"błąd poziomy względem {gt_lab} [m]")
        ax.set_ylim(bottom=0)
        ax.legend(frameon=False, fontsize=7)
    return _save(fig, plt, path, f"Błąd poziomy pozycji piłki a {xlabel.split(' [')[0]}")




def make_map(rep: ReferenceReport, runner, path: Path) -> Path:
    import folium
    from .basemaps import add_basemaps
    from pyproj import Transformer
    from .expplots import BLUE, ORANGE, AQUA, GREEN, INK2, PURPLE
    t = Transformer.from_crs(2180, 4326, always_xy=True)
    lat, lon = rep.spec["latlon"]
    m = folium.Map(location=[lat, lon], zoom_start=21, max_zoom=23, tiles=None)
    add_basemaps(m, buildings=True)
    ref = folium.FeatureGroup("GT i start", show=True)
    folium.CircleMarker([lat, lon], radius=7, color=GREEN, fill=True, fill_opacity=1,
                        tooltip="GT piłki (Google Maps)").add_to(ref)
    from pyproj import Transformer
    inv = Transformer.from_crs(2180, 4326, always_xy=True)
    for key, xy, col in (("GT − lean", rep.gt_corr, "#6250d6"),
                         ("GT_transfer", (rep.transfer["x"], rep.transfer["y"]) if rep.transfer else None,
                          "#008300")):
        if xy:
            lo, la = inv.transform(*xy)
            folium.RegularPolygonMarker([la, lo], number_of_sides=4, radius=8, color=col, fill=True,
                                        fill_opacity=0.9, tooltip=key).add_to(ref)
    zcfg = runner.base.z_anchor.start
    if zcfg.latlon:
        folium.CircleMarker(zcfg.latlon, radius=6, color=INK2, fill=True, tooltip="start (telefon)").add_to(ref)
    for c in rep.calib:
        st = c["stats"]
        if "exif_start_x_2180" in st:
            lo, la = t.transform(st["exif_start_x_2180"], st["exif_start_y_2180"])
            folium.CircleMarker([la, lo], radius=6, color=INK2, fill=False,
                                tooltip=f"start (EXIF {zcfg.image})").add_to(ref)
    ref.add_to(m)
    vr0 = runner.results[runner.exp.reference_variant]
    cams = folium.FeatureGroup("Kamery", show=True)
    for r in vr0.batch.images:
        folium.CircleMarker([r.camera.lat, r.camera.lon], radius=4, color=PURPLE, fill=True,
                            tooltip=f"{r.image_name}: H {r.camera.h:.1f} m npm").add_to(cams)
    cams.add_to(m)
    colors = {"nmt_local": ORANGE, "nmpt_local": BLUE, "odm_dsm": AQUA}
    for name, vr in runner.results.items():
        ax = _variant_axes(vr)
        show = ax["z_mode"] == "start_point" and ax["pixel"] == "manual"
        g = folium.FeatureGroup(f"trafienia: {name}", show=show)
        for rr in [x for x in rep.rows if x["variant"] == name]:
            lo, la = t.transform(rr["hit_x"], rr["hit_y"])
            folium.CircleMarker([la, lo], radius=3, color=colors.get(ax["surface"], INK2), fill=True,
                                tooltip=f"{name} {rr['image']}: błąd {rr['err_m']:.2f} m, "
                                        f"inc {rr['incidence_deg']:.0f}°").add_to(g)
        g.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.save(str(path))
    return path


def ray_profiles(rep: ReferenceReport, runner, out_dir: Path, n: int) -> list[Path]:
    """Profile promienia przez punkt obiektu (NMT, NMPT, DSM ODM, DTM ODM, promien, GT)."""
    import copy
    from .expplots import AQUA, BLUE, GREEN, INK, INK2, NEUTRAL, ORANGE, YELLOW, _fig, _plt, _save
    from .raycast import SurfaceProjector, pixel_to_ray, raycast
    pref = [v for v in runner.results.values()
            if {k: _variant_axes(v)[k] for k in ("surface", "z_mode", "pixel")}
            == {"surface": "nmt_local", "z_mode": "start_point", "pixel": "manual"}]
    pref = pref or [v for v in runner.results.values() if v.cfg.surface.source == "nmt_local"]
    if not pref:
        return []
    vr = pref[0]
    rows = sorted([r for r in rep.rows if r["variant"] == next(k for k, v in runner.results.items() if v is vr)],
                  key=lambda r: r["incidence_deg"])
    if not rows:
        return []
    pick = [rows[-1], rows[0], rows[len(rows) // 2]][:n]
    def surf(src):
        c = copy.deepcopy(vr.cfg)
        c.surface.source = src
        try:
            return runner.surface(c)
        except (FileNotFoundError, ValueError):
            return None
    S = {"NMT": surf("nmt_local"), "NMPT": surf("nmpt_local"), "DSM ODM": surf("odm_dsm"),
         "DTM ODM": odm_dtm_surface(vr.cfg, (rep.gt_x - 200, rep.gt_y - 200, rep.gt_x + 200, rep.gt_y + 200))}
    fr = vr.ctx.frame
    proj = SurfaceProjector(fr, S["NMT"])
    sample = PointSampler()
    out = []
    for k, r in enumerate(pick):
        io, eo = vr.ctx.reconstruction.camera(r["image"])
        o, d = pixel_to_ray((r["u"], r["v"]), io, eo)
        hits = {nm: raycast(o, d, s, fr, vr.cfg.raycast) for nm, s in S.items()
                if s is not None and nm != "DTM ODM"}
        t_end = max([h.range_m for h in hits.values() if h is not None] + [5.0]) * 1.15
        ts = np.linspace(0, t_end, 700)
        P = o[None, :] + ts[:, None] * d[None, :]
        _, _, _, x, y, H, _, _ = proj.project(P)
        cam = proj.project(o[None, :])
        cx, cy = cam[3][0], cam[4][0]
        dist = np.hypot(x - cx, y - cy)
        series = {nm: np.array([sample(s, xi, yi, fr) for xi, yi in zip(x, y)]) if s is not None else None
                  for nm, s in S.items()}
        plt = _plt()
        fig, ax = _fig(plt, figsize=(10.5, 4.4))
        base = np.nanmin(series["NMT"]) - 1.5
        ax.fill_between(dist, series["NMT"], base, color=NEUTRAL, zorder=1)
        styles = {"NMT": (INK2, "-", 1.6), "NMPT": (GREEN, "-", 1.6), "DSM ODM": (YELLOW, "--", 1.3),
                  "DTM ODM": (AQUA, ":", 1.3)}
        for nm, sr in series.items():
            if sr is not None and np.isfinite(sr).any():
                col, ls, lw = styles[nm]
                ax.plot(dist, sr, color=col, linestyle=ls, linewidth=lw, label=nm, zorder=3)
        ax.plot(dist, H, color=BLUE, linewidth=1.6, label="promień", zorder=4)
        gd = math.hypot(rep.gt_x - cx, rep.gt_y - cy)
        ax.scatter([gd], [rep.roof["nmpt"] + float(rep.spec.get("object_height_m", 0))], color=ORANGE,
                   s=60, marker="*", zorder=6, label="GT (wys. NMPT)")
        for nm, h in hits.items():
            if h is not None:
                ax.scatter([math.hypot(h.x_2180 - cx, h.y_2180 - cy)], [h.h], color=styles[nm][0], s=28,
                           marker="o", edgecolor=INK, zorder=5)
        lo = base
        hi = np.nanmax([np.nanmax(s) for s in series.values() if s is not None]) + 3
        ax.set_ylim(lo, max(hi, rep.roof["nmpt"] + 4))
        ax.set_xlim(max(0.0, gd - 25), dist.max())
        ax.set_xlabel("odległość pozioma od kamery [m]")
        ax.set_ylabel("H [m npm]")
        ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))
        p = _save(fig, plt, out_dir / f"ray_profile_{k + 1}_{Path(r['image']).stem}.png",
                  f"Profil promienia: {r['image']}, kąt padania {r['incidence_deg']:.0f}°, "
                  f"kamera {r['agl_roof_nmpt']:.1f} m nad dachem")
        out.append(p)
    return out


# ───────────── zapis ─────────────

def write_outputs(rep: ReferenceReport, runner, out_dir: Path) -> None:
    if rep.rows:
        keys = list(dict.fromkeys(k for r in rep.rows for k in r))
        with open(out_dir / "reference_per_image.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rep.rows)
    if rep.variants:
        keys = list(dict.fromkeys(k for r in rep.variants for k in r))
        with open(out_dir / "reference_variants.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rep.variants)
    (out_dir / "reference_calibration.json").write_text(
        json.dumps(rep.calib, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    for key, lab, nm in (("incidence_deg", "kąt padania [°]", "error_vs_incidence.png"),
                         ("agl_roof_nmpt", "wysokość kamery nad dachem [m]", "error_vs_height_roof.png"),
                         ("agl_ground", "wysokość kamery nad terenem [m]", "error_vs_height_ground.png")):
        p = plot_error_vs(rep, key, lab, out_dir / nm, key == "incidence_deg")
        if p:
            rep.plots.append(p)
    rep.plots.append(make_map(rep, runner, out_dir / "reference_map.html"))
    rep.profiles = ray_profiles(rep, runner, out_dir, int(rep.spec.get("profiles", 3)))


def _vn(name: str) -> str:
    """Nazwa wariantu bezpieczna w tabeli Markdown ("a | b | c" -> "a · b · c")."""
    return name.replace(" | ", " · ")


def summary_md(rep: ReferenceReport, runner) -> list[str]:
    s = rep.spec
    L = ["## Walidacja względem punktu odniesienia (GT)", "",
         f"GT: {s['latlon'][0]:.8f}, {s['latlon'][1]:.8f} (WGS84) = X {rep.gt_x:.2f}, Y {rep.gt_y:.2f} "
         f"(EPSG:2180). W punkcie GT: NMT {_f(rep.roof['nmt'])} m, NMPT {_f(rep.roof['nmpt'])} m npm → "
         f"wysokość dachu nad terenem wg NMPT − NMT {_f(rep.roof['roof_h_nmpt'])} m. "
         f"Środek obiektu {float(s.get('object_height_m', 0)):.2f} m nad dachem (pomijalne wobec błędów).", ""]

    L += ["### Kalibracja wysokości", "", "| tryb | z_offset [m] | szczegóły |", "|---|---|---|"]
    for c in rep.calib:
        st = c["stats"]
        if c["method"] == "start_point":
            det = (f"start {st['start_xy_source']}, NMT(start) {st['nmt_start']:.2f}, grunt = EXIF "
                   f"{st['ground_alt_exif_m']:.2f} m, mediana(h_ODM − alt_EXIF) "
                   f"{st['median_hodm_minus_exif']:.3f} m, MAD {st['mad_hodm_minus_exif']:.3f} m, "
                   f"zakres {st['min_hodm_minus_exif']:.2f}…{st['max_hodm_minus_exif']:.2f} m, n={st['n_cameras']}")
        elif c["alignment"]:
            a = c["alignment"]
            det = (f"{a['n']} punktów otwartego terenu, MAD {a['mad']:.3f} m, płaszczyzna: nachylenie "
                   f"{a['slope_deg']:.2f}°, azymut {a['aspect_deg']:.0f}°")
        else:
            det = ", ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                            for k, v in st.items() if not isinstance(v, dict))
        L.append(f"| {c['method']} | {c['offset']:.3f} | {det} |")
    offs = {c["method"]: c["offset"] for c in rep.calib}
    sp = next((c for c in rep.calib if c["method"] == "start_point"), None)
    if "start_point" in offs and "dem_alignment" in offs:
        L += ["", f"Różnica start_point − dem_alignment: **{offs['start_point'] - offs['dem_alignment']:+.3f} m**."]
    if sp:
        st = sp["stats"]
        if st.get("start_image_alt_exif") is not None:
            L += ["", f"Porównanie (wariant odrzucony): gdyby grunt miał wysokość EXIF zdjęcia startowego "
                  f"({st['start_image_alt_exif']:.2f} m), z_offset = {st['offset_if_ground_at_image_alt']:.3f} m "
                  f"(różnica {st['offset_if_ground_at_image_alt'] - sp['offset']:+.2f} m). Zdjęcie startowe "
                  f"wykonano celowo nad punktem startu, więc jego wysokość EXIF to wysokość klatki nad gruntem."]
        if st.get("scale_k") is not None:
            L += ["", f"Diagnostyka skali: h_ODM = {st['scale_k']:.3f} · alt_EXIF {st['scale_b']:+.2f} m "
                  f"(r = {st['scale_r']:.3f}). Współczynnik różny od 1 oznacza błąd skali pionowej rekonstrukcji "
                  f"względem barometru — założenie stałej różnicy (mediana) nie jest wtedy spełnione. Grunt z "
                  f"regresji: z_offset = {st['offset_linear']:.3f} m (różnica względem mediany "
                  f"{st['offset_linear'] - sp['offset']:+.2f} m" + (
                      f", względem dem_alignment {st['offset_linear'] - offs['dem_alignment']:+.2f} m"
                      if "dem_alignment" in offs else "") + ")."]
        if st.get("baro_check_m") is not None:
            L += [f"Kontrola barometru: zdjęcie startowe EXIF {st['start_image_alt_exif']:.2f} m vs deklarowane "
                  f"~{st['start_image_height_declared_m']:.1f} m nad gruntem → różnica {st['baro_check_m']:+.2f} m "
                  f"(orientacyjna precyzja barometru, łącznie z niepewnością deklaracji)."]
        if st.get("start_xy_exif_vs_used_m") is not None:
            L += [f"XY startu: telefon vs EXIF zdjęcia startowego — odległość "
                  f"{st['start_xy_exif_vs_used_m']:.2f} m, różnica NMT w tych punktach "
                  f"{_f(st.get('nmt_exif_minus_used'), '{:+.3f}')} m (wpływ na z_offset)."]
        pc = st.get("per_camera") or {}
        if pc:
            L += ["", "| zdjęcie | alt EXIF [m] | h ODM [m] | h_ODM − alt_EXIF [m] |", "|---|---|---|---|"]
            for k, v in sorted(pc.items()):
                L.append(f"| {k} | {v['alt_exif']:.2f} | {v['h_odm']:.2f} | {v['diff']:+.3f} |")
    L.append("")

    L += ["### Wyniki per wariant", "",
          "| wariant | powierzchnia | z_anchor | piksel | n | błąd poziomy: mediana / RMS / max [m] | "
          "wzdłuż / w poprzek: średnia [m] | ΔH vs dach NMPT (mediana) [m] | pozycja zagregowana: błąd [m] "
          "(dx, dy) | błąd vs GT skoryg. o lean: mediana / zagregowana [m] | RMS wielowidokowy [m] (widoki) |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for v in rep.variants:
        L.append(f"| {_vn(v['variant'])} | {SURFACE_LABEL.get(v['surface'], v['surface'])} | {v['z_mode']} | "
                 f"{v['pixel']} | {v['n']} | {_f(v['err_median'])} / {_f(v['err_rms'])} / {_f(v['err_max'])} | "
                 f"{_f(v['along_mean'], '{:+.2f}')} / {_f(v['across_mean'], '{:+.2f}')} | "
                 f"{_f(v['dh_nmpt_median'], '{:+.2f}')} | {_f(v['obj_err'])} ({_f(v['obj_dx'], '{:+.2f}')}, "
                 f"{_f(v['obj_dy'], '{:+.2f}')}) | {_f(v['err_corr_median'])} / {_f(v['obj_err_corr'])} | "
                 f"{_f(v['obj_rms'])} ({v['obj_n_views']}) |")
    L += ["", "Wzdłuż: + = trafienie dalej od kamery niż GT. ΔH: wysokość trafienia minus "
          "(NMPT w GT + wysokość środka obiektu)."]
    if rep.gt_corr:
        le = rep.reference["lean_gugik"]
        L += [f"GT skorygowany: odczyt z ortofotomapy przesunięty o ({-le['dx_m']:+.2f}, {-le['dy_m']:+.2f}) m "
              f"— zmierzone przesunięcie dachu budynku względem obrysu EGiB (building lean)."]
    if rep.transfer:
        t = rep.transfer
        L += [f"GT_transfer: piłka wskazana na ortofotomapie ODM z lotu, przeniesiona na ortofotomapę GUGiK "
              f"transformacją Helmerta 2D z {t['n_pairs']} par punktów dachu (skala {t['scale']:.4f}, obrót "
              f"{t['rot_deg']:+.2f}°, RMS {_f(t['helmert_rms'])} m), "
              + (f"z punktami dachu skorygowanymi o przesunięcie zależne od wysokości: v = "
                 f"({t['lean_v_per_m'][0]:+.3f}, {t['lean_v_per_m'][1]:+.3f}) m/m (lean "
                 f"({t['lean_applied'][0]:+.2f}, {t['lean_applied'][1]:+.2f}) m przy wysokości krawędzi "
                 f"{t['lean_h_edges_m']:.2f} m), wysokości punktów dachu "
                 f"{', '.join(f'{h:.1f}' for h in t['roof_point_heights'])} m, piłki {t['ball_height']:.2f} m; "
                 f"względem dawnej korekty stałym wektorem: ({t['shift_vs_uniform'][0]:+.2f}, "
                 f"{t['shift_vs_uniform'][1]:+.2f}) m"
                 if t.get("lean_mode") == "height" else
                 f"następnie skorygowana o przesunięcie dachu ({t['lean_applied'][0]:+.2f}, "
                 f"{t['lean_applied'][1]:+.2f}) m") + f". Pozycja: X {t['x']:.2f}, "
              f"Y {t['y']:.2f}; względem GT z Google Maps: ({t['x'] - rep.gt_x:+.2f}, {t['y'] - rep.gt_y:+.2f}) m. "
              f"Przesunięcie ortofotomapy ODM względem GUGiK w punkcie piłki: "
              f"({t['odm_vs_gugik_shift'][0]:+.2f}, {t['odm_vs_gugik_shift'][1]:+.2f}) m."]
    L += ["", "Kolumna „piłka na ortofotomapie ODM”: odniesienie w układzie rekonstrukcji ODM — mierzy spójność "
          "wewnętrzną (ray casting, powierzchnia, kalibracja Z), bez błędu georeferencji samej rekonstrukcji. "
          "Różnica względem GT_transfer to w przybliżeniu błąd georeferencji ODM w miejscu piłki."]
    if any(v.get("roof_ctrl") for v in rep.variants) or any(v.get("align") for v in rep.variants):
        L += ["", "### Rekonstrukcja, georeferencja i kontrole", "",
              "Skala pionowa k: h_ODM kamer vs barometr (QC, przed wyrównaniem). Dach w modelu: H DSM ODM w miejscu "
              "piłki (ortofotomapa tej rekonstrukcji) minus 10. percentyl H DSM w promieniu 12 m (grunt), po 7p "
              "w układzie wyrównanym — bez 7p zawyżona przez pochylenie modelu. H modelu w GT Google − NMT: "
              "porównanie z NMPT w tym samym punkcie (kalenica, " + _f(rep.roof.get("roof_h_nmpt")) + " m); "
              "sensowne tylko, gdy georeferencja modelu trafia w budynek. Punkty dachu: odległość punktu z ortofotomapy "
              "danej rekonstrukcji (po transformacji 7p, jeśli jest) od GUGiK − lean — kontrola niezależna od "
              "wyrównania i od ray castingu.", "",
              "| wariant | rekonstrukcja | georef | k (QC) | skala 7p | pochylenie 7p [°] | k baro po 7p | "
              "NMAD DSM−NMPT przed / po [m] | H modelu w GT Google − NMT [m] | dach w miejscu piłki [m] | "
              "RMS wielowidokowy [m] | błąd vs GT_transfer [m] | punkty dachu: RMS (max) [m] |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for v in rep.variants:
            a_ = v.get("align") or {}
            rc = v.get("roof_ctrl") or []
            L.append(f"| {_vn(v['variant'])} | {v.get('rec', '')} | {v.get('georef', '')} | {_f(v.get('qc_k'), '{:.3f}')} | "
                     f"{_f(a_.get('scale'), '{:.3f}')} | {_f(a_.get('tilt_deg'))} | {_f(a_.get('baro_k'), '{:.3f}')} | "
                     f"{_f(a_.get('nmad_before'))} / {_f(a_.get('nmad_all'))} | {_f(v['dsm_gt'] - rep.roof['nmt'])} | "
                     f"{_f(v.get('roof_model'))} | "
                     f"{_f(v['obj_rms'])} | {_f(v['obj_err_transfer'])} | "
                     + (f"{_f(v['roof_ctrl_rms'])} ({_f(max(rc))})" if rc else "–") + " |")
        L.append("")
    L += ["", "### Błąd pozycji zagregowanej względem wariantów GT", "",
          "| wariant | GT Google Maps [m] | GT − lean [m] | GT_transfer [m] | mediana per zdjęcie vs GT_transfer [m] "
          "| piłka na ortofotomapie ODM [m] | mediana per zdjęcie vs ODM [m] |",
          "|---|---|---|---|---|---|---|"]
    for v in rep.variants:
        L.append(f"| {_vn(v['variant'])} | {_f(v['obj_err'])} | {_f(v['obj_err_corr'])} | {_f(v['obj_err_transfer'])} | "
                 f"{_f(v['err_transfer_median'])} | {_f(v['obj_err_odm'])} | {_f(v['err_odm_median'])} |")
    L.append("")

    if rep.nmt_fit:
        L += ["### Hipoteza NMT: promień przechodzi przez dach do gruntu", "",
              "Przewidywane przesunięcie wzdłuż promienia: (H_dachu + h_obiektu − NMT w miejscu trafienia) · "
              "tan(kąt padania), skierowane od kamery.", "",
              "| wariant | nachylenie (zmierzone vs przewidywane) | wyraz wolny [m] | r | RMS reszt [m] | "
              "mediana stosunku | n |", "|---|---|---|---|---|---|---|"]
        for f in rep.nmt_fit:
            L.append(f"| {_vn(f['variant'])} | {f['slope']:.2f} | {f['intercept']:+.2f} | {f['r']:.2f} | "
                     f"{f['rms_resid']:.2f} | {_f(f['mean_ratio'])} | {f['n']} |")
        L.append("")
    if rep.trend_fit:
        L += ["### Trend błędu z kątem padania (błąd poziomy = a · tan(inc) + b)", "",
              "| wariant | powierzchnia | a [m] | b [m] | r | n |", "|---|---|---|---|---|---|"]
        for f in rep.trend_fit:
            L.append(f"| {_vn(f['variant'])} | {SURFACE_LABEL.get(f['surface'], f['surface'])} | {f['slope']:+.2f} | "
                     f"{f['intercept']:+.2f} | {_f(f['r'], '{:+.2f}')} | {f['n']} |")
        L.append("")

    v0 = rep.variants[0] if rep.variants else None
    if v0:
        L += ["### Wysokość dachu w punkcie GT", "",
              f"| źródło | H dachu [m npm] | nad NMT [m] |", "|---|---|---|",
              f"| NMPT (GUGiK) | {_f(rep.roof['nmpt'])} | {_f(rep.roof['roof_h_nmpt'])} |"]
        for v in rep.variants:
            if v["surface"] == "nmt_local" and v["pixel"] == "manual":
                L.append(f"| DSM ODM (z_offset {v['z_mode']}) | {_f(v['dsm_gt'])} | "
                         f"{_f(v['dsm_gt'] - rep.roof['nmt'])} |")
                L.append(f"| DTM ODM (z_offset {v['z_mode']}) | {_f(v['dtm_gt'])} | "
                         f"{_f(v['dtm_gt'] - rep.roof['nmt'])} |")
        L.append("")

    r = rep.reference
    if r:
        L += ["### Dane odniesienia GUGiK (scripts/pilka_reference.py)", ""]
        for k, lab in (("orto_sheets", "ortofotomapa"), ("nmpt_sheets", "NMPT"), ("nmt_sheets", "NMT")):
            sheets = sorted(r.get(k) or [], key=lambda d: d.get("aktualnosc", ""), reverse=True)
            if k == "orto_sheets":           # najnowsze arkusze; starsze tylko jako liczba
                rest = len(sheets) - 3
                sheets = sheets[:3]
            for d in sheets:
                if d.get("error"):
                    L.append(f"- {lab} ({d['layer']}): usługa niedostępna — {d['error']}")
                    continue
                extra = ", ".join(f"{kk} {d[kk]}" for kk in ("kolor", "zrodloDanych", "bladSredniPolozenia",
                                                          "bladSredniWysokosci") if d.get(kk))
                L.append(f"- {lab}: arkusz {d.get('godlo', '?')}, aktualność {d.get('aktualnosc', '?')}, "
                         f"piksel/oczko {d.get('wielkoscPiksela') or d.get('charakterystykaPrzestrzenna', '?')}"
                         + (f", {extra}" if extra else ""))
            if k == "orto_sheets" and rest > 0:
                L.append(f"- ortofotomapa: {rest} starszych arkuszy (1997–2022) w reference.json")
        for k, lab in (("lean_gugik", "ortofotomapa GUGiK, budynek z GT"),
                       ("lean_gugik_all", "ortofotomapa GUGiK, wszystkie budynki w kadrze"),
                       ("lean_odm", "ortofotomapa ODM, budynek z GT"),
                       ("lean_odm_all", "ortofotomapa ODM, wszystkie budynki w kadrze")):
            le = r.get(k)
            if le:
                L.append(f"- przesunięcie dachu względem obrysu EGiB ({lab}): {le['shift_m']:.2f} m, azymut "
                         f"{le['azimuth_deg']:.0f}° (dx {le['dx_m']:+.2f}, dy {le['dy_m']:+.2f}), IoU "
                         f"{le['iou_before']:.2f} → {le['iou_after']:.2f}")
        L.append("")

    if rep.budget:
        L += ["### Budżet niepewności punktu odniesienia", "",
              "| składnik | dotyczy | σ [m] | uzasadnienie |", "|---|---|---|---|"]
        for b in rep.budget:
            L.append(f"| {b['component']} | {b.get('applies_to', 'GT')} | {_f(b.get('sigma_m'))} | "
                     f"{b.get('note', '')} |")
        for tgt in sorted({b.get("applies_to", "GT") for b in rep.budget}):
            L.append(f"| **łącznie (pierwiastek sumy kwadratów)** | {tgt} | "
                     f"**{_f(combined_sigma(b.get('sigma_m') for b in rep.budget if b.get('applies_to', 'GT') == tgt))}** | |")
        L.append("")

    L += ["Pliki: `reference_per_image.csv`, `reference_variants.csv`, `reference_calibration.json`, "
          "`reference_map.html`, " + ", ".join(f"`{p.name}`" for p in rep.plots + rep.profiles
                                                if p.suffix == ".png"), ""]
    return L
