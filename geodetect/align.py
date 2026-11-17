"""Wyrownanie rekonstrukcji ODM do danych GUGiK transformacja przestrzennego podobienstwa
(7 parametrow: tx, ty, tz, 3 obroty, skala) — cfg.georef.mode = "similarity_nmpt".

Problem: ODM georeferuje model wylacznie z GPS drona. Przy krotkiej bazie kamer
i skwantowanym GPS skala, orientacja i pochylenie modelu sa slabo wyznaczone (lot "pilka":
skala 0.68/0.85, obrot 19 st., pochylenie 14 st.). Kalibracje z_anchor poprawiaja tylko
stala wysokosci.

Uklad estymacji L: (x, y) EPSG:2180 i h = wysokosc ODM (elipsoidalna, z_offset = 0),
przeliczane SCISLE przez EnuFrame (uwzglednia zbieznosc poludnikow). Uklad docelowy:
(x, y) EPSG:2180 i H [m npm] PL-EVRF2007-NH. Transformacja:

    P' = s * R * (P - c0) + c0 + t,     c0 = srodek ciezkosci punktow DSM (dekorelacja
                                          obrotow i przesuniec), R = R(rx, ry, rz),
    R = Rz(rz) Ry(ry) Rx(rx): rx — wokol osi x (wschod), ry — wokol y (polnoc), rz — wokol
    pionu (azymut); pochylenie modelu zalezy tylko od rx, ry.

Obserwacje:
  (1) punkty DSM ODM -> odleglosc do NMPT (pionowo albo wzdluz normalnej), odporna funkcja
      straty (Huber/Tukey), odrzucanie zmian |r| > prog (samochody, pilka, nowe obiekty,
      roslinnosc niezgodna z NMPT 2025);
  (2) pozycje kamer: XY z GPS EXIF (slaba obserwacja, sigma ~3 m) i — gdy znany punkt startu
      (z_anchor.start) — wysokosci barometryczne H = NMT(start) + alt_EXIF - alt_gruntu
      (sigma ~0.5 m).
Start: obecna georeferencja + z_offset z z_anchor; przeszukanie zgrubne (tx, ty, rz),
potem iteracje Gauss-Newtona z wagami IRLS (ICP punkt-powierzchnia).

Zastosowanie: kamera C' = G(C), G = F^-1 o T o F (F: ENU -> L), R_wc' = R_wc * Rj^T, gdzie Rj to
czesc obrotowa jakobianu G w kamerze (rozklad polarny). W ukladzie wyrownanym H = h, wiec
z_offset nie jest potrzebny (zawiera go tz). DSM ODM jest przeliczany do EPSG:2180/npm.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .geo import EnuFrame

log = logging.getLogger(__name__)

PARAM_NAMES = ("tx", "ty", "tz", "rx", "ry", "rz", "ds")
PARAM_UNITS = ("m", "m", "m", "deg", "deg", "deg", "%")


# ───────────── transformacja ─────────────

def rot_matrix(rv) -> np.ndarray:
    """Macierz obrotu z wektora obrotu (Rodrigues)."""
    rv = np.asarray(rv, float)
    th = float(np.linalg.norm(rv))
    if th < 1e-15:
        return np.eye(3)
    k = rv / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


def euler_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
    """R = Rz(rz) Ry(ry) Rx(rx). Trzeci wiersz (obraz pionu) nie zalezy od rz: pochylenie
    (rx, ry) i azymut (rz) sa rozdzielone — wazne dla raportu uwarunkowania (na plaskim terenie
    niewyznaczalny jest tylko rz, a nie mieszanina obrotow)."""
    cx, sx, cy, sy, cz, sz = math.cos(rx), math.sin(rx), math.cos(ry), math.sin(ry), math.cos(rz), math.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


@dataclass
class Similarity:
    """P' = s R (P - c0) + c0 + t; parametry p = [tx, ty, tz, rx, ry, rz, log s],
    R = Rz(rz) Ry(ry) Rx(rx) (euler_matrix)."""
    p: np.ndarray
    c0: np.ndarray

    @classmethod
    def identity(cls, c0, tz: float = 0.0) -> "Similarity":
        return cls(np.array([0, 0, tz, 0, 0, 0, 0], float), np.asarray(c0, float))

    @property
    def t(self) -> np.ndarray:
        return self.p[:3]

    @property
    def R(self) -> np.ndarray:
        return euler_matrix(*self.p[3:6])

    @property
    def s(self) -> float:
        return float(math.exp(self.p[6]))

    def apply(self, P) -> np.ndarray:
        P = np.atleast_2d(np.asarray(P, float))
        return self.s * (P - self.c0) @ self.R.T + self.c0 + self.t

    def inverse(self, Q) -> np.ndarray:
        Q = np.atleast_2d(np.asarray(Q, float))
        return (Q - self.c0 - self.t) @ self.R / self.s + self.c0

    @property
    def tilt_deg(self) -> float:
        """Kat miedzy pionem a obrazem pionu (pochylenie modelu), niezalezny od azymutu."""
        return float(np.degrees(np.arccos(np.clip(self.R[2, 2], -1, 1))))

    def summary(self) -> dict:
        return {"tx": float(self.p[0]), "ty": float(self.p[1]), "tz": float(self.p[2]),
                "rx_deg": float(np.degrees(self.p[3])), "ry_deg": float(np.degrees(self.p[4])),
                "rz_deg": float(np.degrees(self.p[5])), "scale": self.s,
                "tilt_deg": self.tilt_deg, "c0": self.c0.tolist()}


# ───────────── estymacja ─────────────

@dataclass
class AlignConfigView:
    """Podzbior cfg.georef potrzebny estymacji (testy bez PipelineConfig)."""
    residual: str = "vertical"          # vertical | normal
    loss: str = "tukey"                 # huber | tukey (etap koncowy; zgrubny zawsze Huber)
    loss_k: float = 3.0                 # prog funkcji straty w jednostkach sigma_surface_m
    sigma_surface_m: float = 0.15       # dolna granica sigma powierzchni
    reject_coarse_m: float = 3.0
    reject_m: float = 1.0
    sigma_gps_xy_m: float = 3.0
    sigma_baro_m: float = 0.5
    search_xy_m: float = 8.0
    search_step_m: float = 1.0
    search_rot_deg: float = 30.0
    search_rot_step_deg: float = 5.0
    search_scales: tuple = (1.0,)
    use_baro_prior: bool = True         # False: wysokosci baro tylko do diagnostyki (baro_scale_k)
    n_starts: int = 4                   # liczba kandydatow zgrubnych (+ start gps_only) do multi-startu
    estimate_sigma: bool = True         # sigma powierzchni z danych (NMAD reszt), dolna granica sigma_surface_m
    decorrelate: bool = True            # waga powierzchni * (krok probkowania / oczko NMPT)^2
    sample_step_m: float = 0.5
    max_iter: int = 30
    weak_ratio: float = 2.0             # std(tylko powierzchnia) / std(calosc) > weak_ratio -> slabo


@dataclass
class AlignResult:
    sim: Similarity
    cov: np.ndarray                     # 7x7, parametry jak p (rotacje w rad, log s)
    sigma0: float
    std: dict                           # w jednostkach PARAM_UNITS
    std_surface_only: dict              # inf = niewyznaczalny z samej powierzchni
    weak: list[str]                     # parametry wiazane glownie przez priory
    corr: np.ndarray
    cond_surface: float                 # liczba uwarunkowania macierzy normalnej (powierzchnia)
    n_points: int
    n_inliers: int
    residuals_before: dict              # statystyki |r| pionowo [m] przed (start) i po
    residuals_after: dict
    priors: dict                        # reszty priorow po wyrownaniu
    coarse: dict
    iterations: int
    init: dict

    def report(self) -> dict:
        d = {"transform": self.sim.summary(), "std": self.std, "std_surface_only": self.std_surface_only,
             "weak": self.weak, "sigma0": self.sigma0, "cond_surface": self.cond_surface,
             "n_points": self.n_points, "n_inliers": self.n_inliers,
             "residuals_before": self.residuals_before, "residuals_after": self.residuals_after,
             "priors": self.priors, "coarse": self.coarse, "iterations": self.iterations, "init": self.init,
             "corr": np.round(self.corr, 3).tolist()}
        return json.loads(json.dumps(d, default=float))


def _std_units(cov: np.ndarray) -> np.ndarray:
    d = np.sqrt(np.clip(np.diag(cov), 0, None))
    return np.array([d[0], d[1], d[2], math.degrees(d[3]), math.degrees(d[4]), math.degrees(d[5]),
                     100 * d[6]])


def _robust_stats(r: np.ndarray) -> dict:
    r = r[np.isfinite(r)]
    if len(r) == 0:
        return {"n": 0}
    med = float(np.median(r))
    return {"n": int(len(r)), "median": med, "nmad": float(1.4826 * np.median(np.abs(r - med))),
            "rms": float(np.sqrt(np.mean(r ** 2))), "p90_abs": float(np.percentile(np.abs(r), 90))}


def estimate_similarity(P: np.ndarray, surface: Callable[[np.ndarray, np.ndarray], np.ndarray],
                        init: Similarity, cfg: AlignConfigView,
                        cams: Optional[np.ndarray] = None, cam_gps_xy: Optional[np.ndarray] = None,
                        cam_baro_h: Optional[np.ndarray] = None,
                        surface_res_m: float = 1.0) -> AlignResult:
    """P (n,3) — punkty DSM w ukladzie L; surface(x, y) -> H docelowe (NaN = brak);
    cams (m,3) — kamery w L; cam_gps_xy (m,2) — XY z EXIF; cam_baro_h (m,) — H z barometru."""
    P = np.asarray(P, float)
    sim = Similarity(init.p.copy(), init.c0.copy())
    use_gps = cams is not None and cam_gps_xy is not None and cfg.sigma_gps_xy_m > 0
    use_baro = cams is not None and cam_baro_h is not None and cfg.sigma_baro_m > 0 and cfg.use_baro_prior
    eps = max(surface_res_m / 2, 0.05)

    def surf_res(p, idx=None):
        """Reszty powierzchni [m] (pionowe albo wzdluz normalnej) dla parametrow p."""
        Q = Similarity(p, sim.c0).apply(P if idx is None else P[idx])
        z = surface(Q[:, 0], Q[:, 1])
        r = Q[:, 2] - z
        if cfg.residual == "normal":
            gx = (surface(Q[:, 0] + eps, Q[:, 1]) - surface(Q[:, 0] - eps, Q[:, 1])) / (2 * eps)
            gy = (surface(Q[:, 0], Q[:, 1] + eps) - surface(Q[:, 0], Q[:, 1] - eps)) / (2 * eps)
            g = np.nan_to_num(gx ** 2 + gy ** 2, nan=0.0)
            r = r / np.sqrt(1 + g)
        return r

    def prior_res(p):
        out = []
        if use_gps or use_baro:
            C = Similarity(p, sim.c0).apply(cams)
            if use_gps:
                out.append(((C[:, :2] - cam_gps_xy) / cfg.sigma_gps_xy_m).ravel())
            if use_baro:
                out.append((C[:, 2] - cam_baro_h) / cfg.sigma_baro_m)
        return np.concatenate(out) if out else np.zeros(0)

    r0 = surf_res(sim.p)
    before = _robust_stats(r0)

    # -- zgrubnie: siatka (tx, ty, rz, s) z tz z mediany; ocena: MAD/s + kara priorow --
    rng = np.random.default_rng(0)
    sub = rng.choice(len(P), size=min(len(P), 1500), replace=False)

    def score(p):
        r = surf_res(p, sub)
        ok = np.isfinite(r)
        if ok.sum() < 0.3 * len(sub):
            return np.inf, 0.0
        med = float(np.median(r[ok]))
        # MAD / s: miara w jednostkach modelu ODM — bez dzielenia przez s ocena maleje przy
        # zmniejszaniu skali (mniejsza rzezba modelu), a przy s -> 0 punkty zapadaja sie w jeden.
        # + kara priorow (sredni chi^2 na obserwacje * sigma_surface): hipotezy sprzeczne
        # z GPS/barometrem przegrywaja juz w etapie zgrubnym.
        q = p.copy()
        q[2] -= med
        pr = prior_res(q)
        pen = cfg.sigma_surface_m * float(np.mean(pr ** 2)) if len(pr) else 0.0
        return float(np.median(np.abs(r[ok] - med))) / math.exp(p[6]) + pen, med

    p = sim.p.copy()
    _, med = score(p)
    p[2] -= med
    start = p.copy()
    cands = [(score(p)[0], p.copy())]
    xs = np.arange(-cfg.search_xy_m, cfg.search_xy_m + 1e-9, cfg.search_step_m) if cfg.search_xy_m > 0 else [0.0]
    ks = (np.radians(np.arange(-cfg.search_rot_deg, cfg.search_rot_deg + 1e-9, cfg.search_rot_step_deg))
          if cfg.search_rot_deg > 0 else [0.0])
    for ls in np.log(np.asarray(cfg.search_scales, float)):
        for k in ks:
            for dx in xs:
                for dy in xs:
                    q = start.copy()
                    q[0] += dx
                    q[1] += dy
                    q[5] += k
                    q[6] += ls
                    sc, m = score(q)
                    if np.isfinite(sc):
                        q[2] -= m
                        cands.append((sc, q))
    cands.sort(key=lambda c: c[0])

    def distinct(q, chosen):
        return all(np.hypot(*(q[:2] - c[:2])) > 2 * cfg.search_step_m or abs(q[5] - c[5]) > math.radians(
            cfg.search_rot_step_deg * 1.5) or abs(q[6] - c[6]) > 0.05 for c in chosen)

    seeds = [start]                              # start gps_only zawsze jest jednym z kandydatow
    for sc, q in cands:
        if len(seeds) > cfg.n_starts:
            break
        if distinct(q, seeds):
            seeds.append(q)

    def pattern(p):
        """Przeszukanie wzorcowe (Hooke-Jeeves) po (tx, ty, rz, log s, rx, ry), tz z mediany."""
        best_sc = score(p)[0]
        steps = np.array([cfg.search_step_m, cfg.search_step_m, math.radians(cfg.search_rot_step_deg / 2),
                          0.04, math.radians(1.0), math.radians(1.0)])
        dims = [0, 1, 5, 6, 3, 4]
        while steps[0] > 0.02:
            improved = False
            for k, j in enumerate(dims):
                for sgn in (1, -1):
                    q = p.copy()
                    q[j] += sgn * steps[k]
                    sc, m = score(q)
                    if sc < best_sc:
                        q[2] -= m
                        best_sc, p, improved = sc, q, True
                        break
            if not improved:
                steps = steps / 2
        return p, best_sc

    # -- dokladnie: IRLS Gauss-Newton z estymacja sigma powierzchni (komponent wariancji) --
    # Sasiednie probki DSM po wygladzeniu do oczka NMPT sa skorelowane: efektywna liczba
    # obserwacji ~ n * (krok / oczko)^2 -> waga powierzchni (i jej informacja w kowariancji)
    # mnozona przez corr_factor, inaczej priory sa pomijalne, a std zanizone.
    corr_factor = min(1.0, (cfg.sample_step_m / max(surface_res_m, 1e-9)) ** 2) if cfg.decorrelate else 1.0

    def weights(rs, sig, loss=None):
        u = np.abs(rs) / (cfg.loss_k * sig)
        if (loss or cfg.loss) == "tukey":
            w = np.where(u < 1, (1 - u ** 2) ** 2, 0.0)
        else:
            w = np.where(u <= 1, 1.0, 1.0 / np.maximum(u, 1e-12))
        return w

    def jac(fun, p, h=None):
        h = h if h is not None else np.array([1e-3, 1e-3, 1e-3, 1e-5, 1e-5, 1e-5, 1e-5])
        f0 = fun(p)
        J = np.zeros((len(f0), len(p)))
        for j in range(len(p)):
            dp = np.zeros(len(p))
            dp[j] = h[j]
            J[:, j] = (fun(p + dp) - fun(p - dp)) / (2 * h[j])
        return f0, J

    def fine(p):
        it = 0
        sig = cfg.sigma_surface_m
        # etap 1: Huber i szeroki prog odrzucania; etap 2: cfg.loss (Tukey odrzuca reszty
        # systematyczne — krawedzie budynkow, zmiany — zamiast tylko je oslabiac)
        for reject, loss in ((cfg.reject_coarse_m, "huber"), (cfg.reject_m, cfg.loss)):
            for _ in range(cfg.max_iter):
                it += 1
                r = surf_res(p)
                active = np.isfinite(r) & (np.abs(r) <= reject)
                if active.sum() < 7:
                    return p, sig, it, False
                idx = np.nonzero(active)[0]
                rs, Js = jac(lambda q: surf_res(q, idx), p)
                ok = np.isfinite(rs) & np.isfinite(Js).all(1)
                rs, Js = rs[ok], Js[ok]
                if cfg.estimate_sigma:
                    sig = max(cfg.sigma_surface_m, 1.4826 * float(np.median(np.abs(rs))))
                w = corr_factor * weights(rs, sig, loss) / sig ** 2
                N = Js.T @ (Js * w[:, None])
                b = Js.T @ (w * rs)
                rp, Jp = jac(prior_res, p)
                if len(rp):
                    N += Jp.T @ Jp
                    b += Jp.T @ rp
                N += np.eye(7) * 1e-9 * np.trace(N) / 7
                dp = -np.linalg.solve(N, b)
                p = p + dp
                if np.all(np.abs(dp[:3]) < 1e-4) and np.all(np.abs(dp[3:]) < 1e-7):
                    break
        return p, sig, it, True

    def map_cost(p, sig):
        """Ujemny log-posterior (z dokladnoscia do stalej): Tukey na resztach powierzchni (wszystkie
        punkty; odrzucone = koszt maksymalny) * corr_factor + 1/2 chi^2 priorow."""
        r = surf_res(p)
        r = r[np.isfinite(r)]
        c = min(cfg.loss_k, cfg.reject_m / sig)
        u = np.minimum(np.abs(r) / sig, c)
        rho = c ** 2 / 6 * (1 - (1 - (u / c) ** 2) ** 3)
        pr = prior_res(p)
        return float(corr_factor * rho.sum() + 0.5 * np.sum(pr ** 2))

    runs = []
    for k, seed in enumerate(seeds):
        q, csc = pattern(seed.copy())
        q, sig, it, ok = fine(q)
        if ok:
            runs.append({"p": q, "sigma": sig, "it": it, "seed": k, "coarse_score": csc})
    if not runs:
        raise ValueError("Wyrownanie: za malo punktow zgodnych z powierzchnia")
    sig_c = min(r_["sigma"] for r_ in runs)
    for r_ in runs:
        r_["cost"] = map_cost(r_["p"], sig_c)
        r_["prior_chi2"] = float(np.sum(prior_res(r_["p"]) ** 2))
    runs.sort(key=lambda r_: r_["cost"])
    best_run = runs[0]
    p, it_total = best_run["p"], sum(r_["it"] for r_ in runs)
    alternatives = []
    for r_ in runs:
        q = r_["p"]
        sim_q = Similarity(q, sim.c0)
        dmax = float(np.abs(sim_q.apply(P) - Similarity(p, sim.c0).apply(P)).max())
        alternatives.append({"seed": r_["seed"], "cost": r_["cost"], "d_cost": r_["cost"] - best_run["cost"],
                             "prior_chi2": r_["prior_chi2"], "sigma_surface": r_["sigma"],
                             "scale": sim_q.s, "rz_deg": float(np.degrees(q[5])), "tilt_deg": sim_q.tilt_deg,
                             "tx": float(q[0]), "ty": float(q[1]), "max_point_diff_m": dmax})
    coarse = {"n_seeds": len(seeds), "corr_factor": corr_factor, "sigma_common": sig_c,
              "alternatives": alternatives}
    for a in alternatives:
        log.info("  minimum (start %d): koszt %+.1f, chi2 priorow %.1f, s %.3f, rz %.1f, pochyl. %.2f, "
                 "max roznica punktow %.2f m", a["seed"], a["d_cost"], a["prior_chi2"], a["scale"], a["rz_deg"],
                 a["tilt_deg"], a["max_point_diff_m"])
    sigma_used = best_run["sigma"]

    sim = Similarity(p, sim.c0)
    r = surf_res(p)
    active = np.isfinite(r) & (np.abs(r) <= cfg.reject_m)
    idx = np.nonzero(active)[0]
    rs, Js = jac(lambda q: surf_res(q, idx), p)
    ok = np.isfinite(rs) & np.isfinite(Js).all(1)
    rs, Js = rs[ok], Js[ok]
    w = weights(rs, sigma_used)
    dof = max(len(rs) - 7, 1)
    sigma0_surf = float(np.sqrt(np.sum(w * rs ** 2) / np.sum(w) * len(rs) / dof))   # [m]
    Ns = corr_factor * Js.T @ (Js * (w / sigma0_surf ** 2)[:, None])
    rp, Jp = jac(prior_res, p)
    Np = Jp.T @ Jp if len(rp) else np.zeros((7, 7))
    Nt = Ns + Np
    cov = np.linalg.inv(Nt + np.eye(7) * 1e-12 * np.trace(Nt) / 7)
    # niepewnosc "tylko z powierzchni": rozklad wlasny macierzy normalnej (kierunki prawie
    # zerowej informacji -> bardzo duze std zamiast bledu odwracania). Macierz skalowana
    # jednostkami (m, rad, log s) — porownania wzgledne std_s/std, nie wartosci wlasnych.
    ev, V = np.linalg.eigh(Ns)
    lam = np.clip(ev, ev.max() * 1e-16, None)
    cond = float(ev.max() / max(ev.min(), ev.max() * 1e-300))
    cov_s = (V / lam) @ V.T
    std = _std_units(cov)
    std_s = _std_units(cov_s)
    weak = [n for n, a, b_ in zip(PARAM_NAMES, std, std_s) if b_ > cfg.weak_ratio * a]
    d = np.sqrt(np.clip(np.diag(cov), 1e-300, None))
    corr = cov / np.outer(d, d)
    priors = {}
    if use_gps or use_baro:
        C = sim.apply(cams)
        if use_gps:
            dxy = C[:, :2] - cam_gps_xy
            priors["gps_xy_rms_m"] = float(np.sqrt(np.mean(np.sum(dxy ** 2, 1))))
            priors["gps_xy_mean_m"] = dxy.mean(0).tolist()
        if use_baro:
            dh = C[:, 2] - cam_baro_h
            priors["baro_rms_m"] = float(np.sqrt(np.mean(dh ** 2)))
            priors["baro_mean_m"] = float(dh.mean())
            if np.ptp(cam_baro_h) > 1.0:
                # zgodnosc modelu z barometrem: H_kamery_po = k * H_baro + b; k != 1 -> rekonstrukcja
                # zdeformowana anizotropowo (np. zla ogniskowa: splaszczenie w pionie), czego
                # transformacja podobienstwa nie usunie
                k, b0 = np.polyfit(cam_baro_h, C[:, 2], 1)
                priors["baro_scale_k"] = float(k)
    if cams is not None and cam_baro_h is not None and np.ptp(cam_baro_h) > 1.0 and "baro_scale_k" not in priors:
        C = sim.apply(cams)
        priors["baro_scale_k"] = float(np.polyfit(cam_baro_h, C[:, 2], 1)[0])
        priors["baro_rms_m_unused"] = float(np.sqrt(np.mean((C[:, 2] - cam_baro_h) ** 2)))
    res = AlignResult(sim, cov, sigma0_surf, dict(zip(PARAM_NAMES, map(float, std))),
                      dict(zip(PARAM_NAMES, map(float, std_s))), weak, corr, cond, len(P), int(active.sum()),
                      before, {**_robust_stats(r[active]), "all": _robust_stats(r)}, priors, coarse, it_total,
                      init.summary())
    log.info("Wyrownanie: s=%.4f, pochylenie %.2f deg, rz %.2f deg, t=(%.2f, %.2f, %.2f); reszty NMAD %.3f m "
             "(%d/%d punktow); slabo wyznaczone z powierzchni: %s", sim.s, sim.tilt_deg,
             np.degrees(p[5]), *p[:3], res.residuals_after.get("nmad", float("nan")), res.n_inliers,
             res.n_points, ", ".join(weak) or "-")
    return res


# ───────────── zastosowanie do rekonstrukcji ─────────────

class LocalFrame:
    """F: ENU (ODM) <-> L = (x_2180, y_2180, h) — scisle, przez EnuFrame z z_offset = 0."""

    def __init__(self, frame: EnuFrame):
        self.frame = frame.with_z_offset(0.0)

    def to_local(self, E: np.ndarray) -> np.ndarray:
        E = np.atleast_2d(E)
        x, y, h = self.frame.enu_to_puwg92(E[:, 0], E[:, 1], E[:, 2])
        return np.column_stack([np.atleast_1d(x), np.atleast_1d(y), np.atleast_1d(h)])

    def to_enu(self, L: np.ndarray) -> np.ndarray:
        L = np.atleast_2d(L)
        lon, lat = self.frame.puwg92_to_lonlat(L[:, 0], L[:, 1])
        e, n, u = self.frame.geodetic_to_enu(lon, lat, L[:, 2])
        return np.column_stack([np.atleast_1d(e), np.atleast_1d(n), np.atleast_1d(u)])


def polar_rotation(J: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(J)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


class AlignedReconstruction:
    """Rekonstrukcja z kamerami przeliczonymi transformacja: C' = G(C), R_wc' = R_wc Rj^T.
    Interfejs jak OdmReconstruction (image_names, camera, ref, path, shots, cameras)."""

    def __init__(self, base, frame: EnuFrame, sim: Similarity):
        self.base, self.sim = base, sim
        self.lf = LocalFrame(frame)
        self.ref, self.path = base.ref, base.path
        self.cameras, self.shots = base.cameras, base.shots
        self._cache: dict = {}

    @property
    def image_names(self) -> list[str]:
        return self.base.image_names

    def map_enu(self, E: np.ndarray) -> np.ndarray:
        return self.lf.to_enu(self.sim.apply(self.lf.to_local(E)))

    def camera(self, image_name: str):
        if image_name not in self._cache:
            from .odm import CameraEO
            io, eo = self.base.camera(image_name)
            C = np.asarray(eo.C_enu, float)
            C2 = self.map_enu(C)[0]
            J = np.zeros((3, 3))
            h = 1.0
            for k in range(3):
                d = np.zeros(3)
                d[k] = h
                J[:, k] = (self.map_enu(C + d)[0] - self.map_enu(C - d)[0]) / (2 * h)
            Rj = polar_rotation(J)
            self._cache[image_name] = (io, CameraEO(C_enu=C2, R_wc=eo.R_wc @ Rj.T))
        return self._cache[image_name]


# ───────────── dane wejsciowe z ODM / GUGiK ─────────────

def dsm_points_local(dsm_path: Path, frame: EnuFrame, step_m: float,
                     bounds_2180: Optional[tuple] = None, smooth_m: float = 0.0) -> np.ndarray:
    """Punkty DSM ODM (srodki pikseli co ~step_m) w ukladzie L (x, y EPSG:2180, h ODM).
    smooth_m > 0: srednia ruchoma w oknie smooth_m (rozdzielczosc NMPT) — porownanie DSM 2 cm
    z rastrem 1 m bez wygladzenia daje systematyczne reszty na krawedziach budynkow."""
    import rasterio
    from pyproj import Transformer
    from scipy.ndimage import uniform_filter
    with rasterio.open(dsm_path) as ds:
        z = ds.read(1).astype(float)
        if ds.nodata is not None:
            z[z == ds.nodata] = np.nan
        z[z < -1e4] = np.nan
        w = int(round(smooth_m / abs(ds.res[0])))
        if w > 1:
            m = np.isfinite(z)
            num = uniform_filter(np.where(m, z, 0.0), w, mode="constant")
            den = uniform_filter(m.astype(float), w, mode="constant")
            z = np.where(den > 0.75, num / np.maximum(den, 1e-12), np.nan)
        k = max(1, int(round(step_m / abs(ds.res[0]))))
        rows, cols = np.mgrid[k // 2:z.shape[0]:k, k // 2:z.shape[1]:k]
        rows, cols = rows.ravel(), cols.ravel()
        h = z[rows, cols]
        ok = np.isfinite(h)
        xs, ys = rasterio.transform.xy(ds.transform, rows[ok], cols[ok])
        tr = Transformer.from_crs(ds.crs, "EPSG:2180", always_xy=True)
    x, y = tr.transform(np.asarray(xs), np.asarray(ys))
    P = np.column_stack([x, y, h[ok]])
    if bounds_2180 is not None:
        b = bounds_2180
        P = P[(P[:, 0] >= b[0]) & (P[:, 0] <= b[2]) & (P[:, 1] >= b[1]) & (P[:, 1] <= b[3])]
    return P


def aligned_dsm_surface(dsm_path: Path, frame: EnuFrame, sim: Similarity, res_m: float, name="odm_dsm"):
    """DSM ODM przeliczony transformacja do EPSG:2180 / npm (siatka res_m).

    Rasteryzacja z-buforem (maksimum H w komorce), nie interpolacja: przy duzym pochyleniu
    transformacji strome sciany przechodza "za pion" i powierzchnia po przeliczeniu jest
    wielowartosciowa — DSM to powierzchnia gorna. Puste komorki wewnatrz zasiegu wypelnia
    interpolacja liniowa."""
    from scipy.interpolate import griddata

    from .surface import VERTICAL_NPM, Grid, RasterSurface
    P = sim.apply(dsm_points_local(dsm_path, frame, res_m / 3))
    x0, x1 = math.floor(P[:, 0].min()), math.ceil(P[:, 0].max())
    y0, y1 = math.floor(P[:, 1].min()), math.ceil(P[:, 1].max())
    nx, ny = int(round((x1 - x0) / res_m)), int(round((y1 - y0) / res_m))
    col = np.clip(((P[:, 0] - x0) / res_m).astype(int), 0, nx - 1)
    row = np.clip(((y1 - P[:, 1]) / res_m).astype(int), 0, ny - 1)
    Z = np.full(ny * nx, -np.inf)
    np.maximum.at(Z, row * nx + col, P[:, 2])
    Z = Z.reshape(ny, nx)
    Z[~np.isfinite(Z)] = np.nan
    hole = np.isnan(Z)
    if hole.any() and (~hole).sum() > 3:
        rr, cc = np.nonzero(~hole)
        hr, hc = np.nonzero(hole)
        Z[hr, hc] = griddata(np.column_stack([rr, cc]), Z[rr, cc], (hr, hc), method="linear")
    log.info("DSM ODM po wyrownaniu: %dx%d px @ %.2f m (EPSG:2180, npm)", nx, ny, res_m)
    return RasterSurface(Grid(Z, float(x0), float(y1), res_m, res_m, "EPSG:2180"), name, VERTICAL_NPM)


def raster_h(path: Path, x, y) -> float:
    """Wysokosc ODM (h, uklad L) rastra ODM (DSM/DTM) w punkcie (x, y) EPSG:2180 ukladu ODM
    (mediana 5x5 px, odporna na pojedyncze piksele)."""
    import rasterio
    from pyproj import Transformer
    with rasterio.open(path) as ds:
        e, n = Transformer.from_crs("EPSG:2180", ds.crs, always_xy=True).transform(x, y)
        r, c = ds.index(e, n)
        w = ds.read(1, window=((r - 2, r + 3), (c - 2, c + 3))).astype(float)
        w = w[(w > -1e4) & np.isfinite(w)]
    return float(np.median(w)) if len(w) else float("nan")


_CACHE: dict = {}


@dataclass
class GeorefAlignment:
    """Wynik trybu similarity_nmpt dla kontekstu: rekonstrukcja, DSM i raport."""
    result: AlignResult
    reconstruction: AlignedReconstruction
    dsm_path: Path
    init_z_offset: float
    init_method: str
    dsm_res_m: float
    frame: EnuFrame
    _dsm: object = field(default=None, repr=False)

    def dsm_surface(self):
        if self._dsm is None:
            self._dsm = aligned_dsm_surface(self.dsm_path, self.frame, self.result.sim, self.dsm_res_m)
        return self._dsm

    def raster_h(self, path: Path, x, y) -> float:
        return raster_h(path, x, y)

    def map_odm_point(self, x, y, path: Optional[Path] = None) -> np.ndarray:
        """Punkt z ortofotomapy/DSM ODM (x, y w ukladzie gps_only, h z DSM) -> uklad wyrownany."""
        h = self.raster_h(path or self.dsm_path, x, y)
        return self.result.sim.apply([x, y, h])[0]

    def model_H(self, path: Path, x: float, y: float, iters: int = 6) -> float:
        """H [m npm] rastra ODM po wyrownaniu w punkcie docelowym (x, y): iteracja odwrotna."""
        sim = self.result.sim
        H = float(sim.apply([x, y, self.raster_h(path, x, y)])[0, 2])
        for _ in range(iters):
            L = sim.inverse([x, y, H])[0]
            h = self.raster_h(path, L[0], L[1])
            if not np.isfinite(h):
                return float("nan")
            Q = sim.apply([L[0], L[1], h])[0]
            H = float(Q[2])
        return H

    def stats(self) -> dict:
        """Plaskie statystyki do ZCalibration.stats (raport kalibracji)."""
        r = self.result
        s = {"init": f"{self.init_method} z_offset={self.init_z_offset:.3f}", **{
            k: v for k, v in r.sim.summary().items() if k != "c0"}}
        s.update({f"std_{k}": v for k, v in r.std.items()})
        s.update({"weak": ",".join(r.weak) or "-", "n_inliers": r.n_inliers, "n_points": r.n_points,
                  "res_nmad_before": r.residuals_before.get("nmad"), "res_nmad_after": r.residuals_after.get("nmad"),
                  "res_median_after": r.residuals_after.get("median"), "cond_surface": r.cond_surface})
        s.update({f"prior_{k}": v for k, v in r.priors.items() if not isinstance(v, list)})
        s["report"] = r.report()            # dict -> pomijany w opisie tekstowym
        return s


def align_reconstruction(config, rec, frame: EnuFrame, z_init, dem) -> GeorefAlignment:
    """cfg.georef.mode=similarity_nmpt: estymacja 7p na NMPT z priorami GPS/barometru."""
    from .exif import read_gps
    from .surface import area_bounds_2180, load_mosaic, raster_files, _bounds_in, _read_crs, sample_grid
    g = config.georef
    dsm_path = Path(config.odm_dsm_path)
    key = json.dumps([str(rec.path), str(dsm_path), json.loads(json.dumps(g.__dict__, default=str)),
                      round(z_init.offset, 6), config.z_anchor.start.latlon, str(config.photos_dir)])
    if key in _CACHE:
        return _CACHE[key]
    if not dsm_path.exists():
        raise FileNotFoundError(f"georef.mode=similarity_nmpt wymaga DSM ODM: {dsm_path}")
    lf = LocalFrame(frame)
    names = [n for n in rec.image_names if n not in config.excluded_images()]
    cams = np.vstack([lf.to_local(rec.camera(n)[1].C_enu) for n in names])
    files = raster_files(config.surface.nmpt_dir)
    crs = _read_crs(files[0], config.surface.default_crs)
    P0 = dsm_points_local(dsm_path, frame, max(g.sample_step_m, 2.0))
    bounds = area_bounds_2180(config, np.vstack([cams[:, :2], P0[:, :2]]))
    nmpt = load_mosaic(files, config.surface.default_crs, _bounds_in(crs, bounds))
    P = dsm_points_local(dsm_path, frame, g.sample_step_m, smooth_m=nmpt.res_x if g.smooth_dsm else 0.0)

    def surf(x, y):
        return sample_grid(nmpt, x, y, "bilinear")

    gps_xy, baro = None, None
    if g.use_gps_prior:
        fixes = [read_gps(config.photos_dir / n) for n in names]
        gps_xy = np.array([frame.lonlat_to_puwg92(f.lon, f.lat) for f in fixes], float)
    if config.z_anchor.start.latlon or config.z_anchor.start.image:
        from .zcal import resolve_start
        st = resolve_start(config.z_anchor, config.photos_dir, frame)
        H0 = dem.sample(st.x_2180, st.y_2180)
        if H0 is not None:
            baro = np.array([H0 + read_gps(config.photos_dir / n).alt - config.z_anchor.start.ground_alt_exif_m
                             for n in names], float)
    cv = AlignConfigView(**{k: getattr(g, k) for k in AlignConfigView.__dataclass_fields__})
    c0 = np.array([P[:, 0].mean(), P[:, 1].mean(), float(np.median(P[:, 2]))])
    init = Similarity.identity(c0, tz=z_init.offset)
    res = estimate_similarity(P, surf, init, cv, cams, gps_xy, baro, surface_res_m=nmpt.res_x)
    out = GeorefAlignment(res, AlignedReconstruction(rec, frame, res.sim), dsm_path, z_init.offset,
                          z_init.method, g.dsm_resolution_m, frame)
    _CACHE[key] = out
    return out


def alignment_md(al: "GeorefAlignment", title: str = "Wyrównanie 7p do NMPT (georef.mode = similarity_nmpt)") -> list[str]:
    """Sekcja Markdown raportu: parametry z niepewnosciami, uwarunkowanie, reszty, priory, minima."""
    r = al.result
    t = r.sim.summary()
    u = dict(zip(PARAM_NAMES, PARAM_UNITS))
    val = {"tx": t["tx"], "ty": t["ty"], "tz": t["tz"], "rx": t["rx_deg"], "ry": t["ry_deg"], "rz": t["rz_deg"],
           "ds": 100 * (t["scale"] - 1)}
    L = [f"## {title}", "",
         f"Start: {al.init_method}, z_offset {al.init_z_offset:.3f} m. Punkty DSM ODM: {r.n_points} "
         f"(po wygładzeniu do oczka NMPT), zgodne z NMPT po wyrównaniu: {r.n_inliers} "
         f"(odrzucone |r| > próg = zmiany, roślinność, krawędzie). Skala {t['scale']:.4f}, "
         f"pochylenie modelu {t['tilt_deg']:.2f}°.", "",
         "| parametr | wartość | σ (całość) | σ (tylko powierzchnia) | słabo wyznaczony z powierzchni |",
         "|---|---|---|---|---|"]
    for n in PARAM_NAMES:
        L.append(f"| {n} [{u[n]}] | {val[n]:+.3f} | {r.std[n]:.3f} | {r.std_surface_only[n]:.3g} | "
                 f"{'**tak**' if n in r.weak else 'nie'} |")
    L += ["", f"tx, ty, tz — przesunięcie środka c0 = ({t['c0'][0]:.1f}, {t['c0'][1]:.1f}, h {t['c0'][2]:.1f}); "
          f"tz zawiera przejście h ODM → H npm. Liczba uwarunkowania macierzy normalnej powierzchni: "
          f"{r.cond_surface:.3g}. σ formalne (z kowariancji) nie obejmują błędu modelu (deformacji "
          f"rekonstrukcji, której podobieństwo nie opisuje) — zob. alternatywne minima i reszty priorów.", "",
          "| reszty pionowe DSM − NMPT [m] | n | mediana | NMAD | RMS | p90 abs(r) |", "|---|---|---|---|---|---|"]
    for lab, d in (("przed (gps_only + z_offset)", r.residuals_before), ("po — punkty zgodne", r.residuals_after),
                   ("po — wszystkie", r.residuals_after.get("all", {}))):
        if d.get("n"):
            L.append(f"| {lab} | {d['n']} | {d['median']:+.3f} | {d['nmad']:.3f} | {d['rms']:.3f} | "
                     f"{d['p90_abs']:.3f} |")
    p = r.priors
    L += [""]
    if p:
        parts = []
        if "gps_xy_rms_m" in p:
            parts.append(f"XY kamer vs GPS EXIF: RMS {p['gps_xy_rms_m']:.2f} m (średnio "
                         f"{p['gps_xy_mean_m'][0]:+.2f}, {p['gps_xy_mean_m'][1]:+.2f} m)")
        if "baro_rms_m" in p:
            parts.append(f"H kamer vs barometr: RMS {p['baro_rms_m']:.2f} m (średnio {p['baro_mean_m']:+.2f} m)")
        if "baro_rms_m_unused" in p:
            parts.append(f"H kamer vs barometr (prior wyłączony): RMS {p['baro_rms_m_unused']:.2f} m")
        if "baro_scale_k" in p:
            parts.append(f"H_kamer = k·H_baro + b: k = {p['baro_scale_k']:.3f}")
        L += ["Priory po wyrównaniu: " + "; ".join(parts) + ".", ""]
    alts = r.coarse.get("alternatives") or []
    if len(alts) > 1:
        L += [f"Multi-start ({r.coarse['n_seeds']} startów, start 0 = gps_only; koszt MAP przy wspólnym σ "
              f"powierzchni {r.coarse['sigma_common']:.3f} m, waga dekorelacji {r.coarse['corr_factor']:.2f}):", "",
              "| start | Δkoszt | χ² priorów | skala | rz [°] | pochylenie [°] | max różnica punktów vs wybrane [m] |",
              "|---|---|---|---|---|---|---|"]
        for a in alts:
            L.append(f"| {a['seed']} | {a['d_cost']:+.1f} | {a['prior_chi2']:.1f} | {a['scale']:.3f} | "
                     f"{a['rz_deg']:+.1f} | {a['tilt_deg']:.2f} | {a['max_point_diff_m']:.2f} |")
        L.append("")
    return L
