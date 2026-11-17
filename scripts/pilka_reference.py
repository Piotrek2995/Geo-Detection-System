"""Dane odniesienia eksperymentu "pilka na dachu" (docs/eksperyment_pilka.md, czesc E).

1. Metadane GUGiK w punkcie GT (GetFeatureInfo skorowidzow WMS): ortofotomapa (data nalotu,
   piksel, zrodlo) i NMPT (data, zrodlo) — bez autoryzacji.
2. Wycinek ortofotomapy (WMS HighResolution, 60x60 m, 0.05 m/px) i obrysy budynkow EGiB
   (WMS KIEG, warstwa "budynki") — PNG + world file (.pgw) w EPSG:2180.
3. Przesuniecie dachu wzgledem obrysu (building lean): maska dachu z ortofotomapy
   (dachowka: kolor czerwono-pomaranczowy, HSV) vs wypelniony obrys EGiB; szukane
   przesuniecie (dx, dy) maksymalizujace IoU (siatka +-4 m co 0.05 m). Okap wystaje
   symetrycznie poza obrys, wiec zawyza pole dachu, ale nie przesuwa jego srodka.
4. Wysokosc dachu w punkcie GT: NMPT, NMT (arkusze lokalne), roznica = h budynku nad gruntem.
5. Opcjonalnie: to samo dla ortofotomapy ODM z lotu (odm_pilka/odm_orthophoto), ktora
   jest wygenerowana na DSM (bliska ortofotomapie prawdziwej) — po uruchomieniu ODM.

Wynik: output/pilka/reference/ (reference.json, ortho_*.png, lean.png).

    python scripts/pilka_reference.py
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from geodetect.config import PipelineConfig  # noqa: E402

log = logging.getLogger("pilka_reference")
UA = {"User-Agent": "geodetect-thesis/1.0"}
ORTO_INDEX = "https://mapy.geoportal.gov.pl/wss/service/PZGIK/ORTO/WMS/SkorowidzeWgAktualnosci"
ORTO_LAYERS = ["SkorowidzeOrtofotomapy2026", "SkorowidzeOrtofotomapy2025",
               "SkorowidzeOrtofotomapy2024", "SkorowidzeOrtofotomapyStarsze"]
NMPT_INDEX = "https://mapy.geoportal.gov.pl/wss/service/PZGIK/NMPT/WMS/SkorowidzeUkladEVRF2007"
NMPT_LAYERS = ["SkorowidzeNMPT2026", "SkorowidzeNMPT2025", "SkorowidzeNMPT2024",
               "SkorowidzeNMPT2023iStarsze"]
NMT_INDEX = "https://mapy.geoportal.gov.pl/wss/service/PZGIK/NMT/WMS/SkorowidzeUkladEVRF2007"
NMT_LAYERS = ["SkorowidzeNMT2026", "SkorowidzeNMT2025", "SkorowidzeNMT2024", "SkorowidzeNMT2023iStarsze"]
ORTO_WMS = "https://mapy.geoportal.gov.pl/wss/service/PZGIK/ORTO/WMS/HighResolution"
EGIB_WMS = "https://integracja.gugik.gov.pl/cgi-bin/KrajowaIntegracjaEwidencjiGruntow"


@dataclass
class LeanEstimate:
    dx_m: float
    dy_m: float
    shift_m: float
    azimuth_deg: float           # kierunek przesuniecia dachu wzgledem obrysu (0 = N, 90 = E)
    iou_before: float
    iou_after: float
    roof_px: int
    footprint_px: int
    scope: str = ""              # "wszystkie budynki w kadrze" | "budynek z GT"
    at_search_bound: bool = False   # optimum na granicy przeszukiwania -> wynik niewiarygodny


@dataclass
class Reference:
    gt_lat: float
    gt_lon: float
    gt_x_2180: float
    gt_y_2180: float
    start_lat: float
    start_lon: float
    start_x_2180: float
    start_y_2180: float
    orto_sheets: list[dict] = field(default_factory=list)
    nmpt_sheets: list[dict] = field(default_factory=list)
    nmt_sheets: list[dict] = field(default_factory=list)
    nmt_at_gt: Optional[float] = None
    nmpt_at_gt: Optional[float] = None
    roof_height_m: Optional[float] = None
    nmt_at_start: Optional[float] = None
    lean_gugik: Optional[LeanEstimate] = None          # budynek z GT (grupa stykajacych sie obrysow)
    lean_gugik_all: Optional[LeanEstimate] = None      # wszystkie budynki w kadrze (przesuniecie wspolne)
    lean_odm: Optional[LeanEstimate] = None
    lean_odm_all: Optional[LeanEstimate] = None
    files: dict = field(default_factory=dict)


def _get(url: str, params: dict, timeout: float = 60.0) -> bytes:
    q = url + "?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(urllib.request.Request(q, headers=UA), timeout=timeout) as r:
        return r.read()


def index_metadata(service: str, layers: list[str], x: float, y: float) -> list[dict]:
    """Arkusze ze skorowidza GUGiK w punkcie (x, y) — pola z JS w odpowiedzi GetFeatureInfo."""
    bbox = f"{y - 50},{x - 50},{y + 50},{x + 50}"      # WMS 1.3.0 + EPSG:2180: (N, E)
    out = []
    for lay in layers:
        try:
            html = _get(service, {"SERVICE": "WMS", "VERSION": "1.3.0", "REQUEST": "GetFeatureInfo",
                                  "LAYERS": lay, "QUERY_LAYERS": lay, "CRS": "EPSG:2180", "BBOX": bbox,
                                  "WIDTH": 101, "HEIGHT": 101, "I": 50, "J": 50, "STYLES": "",
                                  "INFO_FORMAT": "text/html", "FEATURE_COUNT": 20}).decode("utf-8", "replace")
        except Exception as exc:          # usluga niedostepna — raportujemy, nie przerywamy
            out.append({"layer": lay, "error": str(exc)})
            continue
        for obj in re.findall(r"\.push\(\{(.*?)\}\);", html, flags=re.S):
            d = dict(re.findall(r'(\w+)\s*:\s*"([^"]*)"', obj))
            d["layer"] = lay
            out.append(d)
    return out


def get_map(service: str, layer: str, bbox_2180: tuple, px: int, transparent: bool,
            fmt: str = "image/png") -> np.ndarray:
    x0, y0, x1, y1 = bbox_2180
    data = _get(service, {"SERVICE": "WMS", "VERSION": "1.3.0", "REQUEST": "GetMap", "LAYERS": layer,
                          "STYLES": "", "CRS": "EPSG:2180", "BBOX": f"{y0},{x0},{y1},{x1}",
                          "WIDTH": px, "HEIGHT": px, "FORMAT": fmt,
                          "TRANSPARENT": "TRUE" if transparent else "FALSE"}, timeout=120)
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"{service} {layer}: odpowiedz nie jest obrazem: {data[:300]!r}")
    return img


def write_world_file(png: Path, bbox_2180: tuple, px: int) -> None:
    x0, y0, x1, y1 = bbox_2180
    r = (x1 - x0) / px
    png.with_suffix(".pgw").write_text(f"{r}\n0\n0\n{-r}\n{x0 + r / 2}\n{y1 - r / 2}\n")


def roof_mask(bgr: np.ndarray) -> np.ndarray:
    """Dachowka ceramiczna/blachodachowka: czerwien-pomarancz (HSV), oczyszczona morfologia."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    # dachowka w sloncu (czerwien-pomarancz) i polacie w cieniu (ciemny braz)
    m = (((h <= 20) | (h >= 165)) & (s >= 45) & (v >= 25)).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    m = cv2.morphologyEx(cv2.morphologyEx(m, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)
    return m.astype(bool)


def footprint_masks(egib: np.ndarray) -> list[np.ndarray]:
    """Wypelnione obrysy budynkow z rastra WMS (linie obrysu -> kontury -> wypelnienie)."""
    a = egib[..., 3] if egib.ndim == 3 and egib.shape[2] == 4 else cv2.cvtColor(egib, cv2.COLOR_BGR2GRAY) < 250
    lines = (a > 0).astype(np.uint8) * 255
    lines = cv2.dilate(lines, np.ones((3, 3), np.uint8))
    cnts, hier = cv2.findContours(lines, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for i, c in enumerate(cnts):
        if hier[0][i][3] < 0:          # tylko kontury wewnetrzne (wnetrza budynkow)
            continue
        m = np.zeros(lines.shape, np.uint8)
        cv2.drawContours(m, [c], -1, 1, -1)
        if m.sum() > 400:
            out.append(m.astype(bool))
    return out


def estimate_lean(roof: np.ndarray, footprint: np.ndarray, res_m: float, max_m: float = 4.0,
                  step_m: float = 0.05, scope: str = "", exclude: Optional[np.ndarray] = None) -> LeanEstimate:
    """Przesuniecie (dx E, dy N) obrysu, przy ktorym IoU z maska dachu jest najwieksze.
    Dach rozpatrywany tylko w otoczeniu obrysu (bufor max_m), zeby nie lapac sasiadow."""
    buf = int(round(max_m / res_m))
    near = cv2.dilate(footprint.astype(np.uint8), np.ones((2 * buf + 1, 2 * buf + 1), np.uint8)).astype(bool)
    r = roof & near
    if exclude is not None:
        r &= ~exclude
    ys, xs = np.nonzero(footprint)

    def iou(sx: int, sy: int) -> float:
        yy, xx = ys + sy, xs + sx
        ok = (yy >= 0) & (yy < r.shape[0]) & (xx >= 0) & (xx < r.shape[1])
        inter = r[yy[ok], xx[ok]].sum()
        return inter / (r.sum() + len(ys) - inter)

    n = int(round(max_m / step_m))
    st = step_m / res_m
    best = (0.0, 0, 0)
    for i in range(-n, n + 1):
        for j in range(-n, n + 1):
            v = iou(int(round(i * st)), int(round(j * st)))
            if v > best[0]:
                best = (v, i, j)
    v, i, j = best
    dx, dy = i * step_m, -j * step_m          # wiersze rosna na poludnie
    return LeanEstimate(dx, dy, float(np.hypot(dx, dy)), float(np.degrees(np.arctan2(dx, dy)) % 360),
                        float(iou(0, 0)), float(v), int(r.sum()), int(footprint.sum()), scope,
                        max(abs(i), abs(j)) >= n)


def lean_figure(img: np.ndarray, fp: np.ndarray, est: LeanEstimate, res_m: float, gt_px, start_px,
                title: str, path: Path) -> None:
    vis = img[..., :3].copy()
    def outline(m, col, shift=(0, 0)):
        c, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = [cc + np.array(shift, dtype=cc.dtype) for cc in c]
        cv2.drawContours(vis, c, -1, col, 2)
    outline(fp, (255, 255, 255))
    outline(fp, (0, 200, 255), (int(round(est.dx_m / res_m)), int(round(-est.dy_m / res_m))))
    for p, col in ((gt_px, (0, 131, 0)), (start_px, (214, 120, 42))):
        if p is not None:
            cv2.drawMarker(vis, (int(p[0]), int(p[1])), col, cv2.MARKER_CROSS, 30, 3)
    cv2.putText(vis, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(vis, f"bialy: obrys EGiB, zolty: obrys przesuniety o ({est.dx_m:+.2f}, {est.dy_m:+.2f}) m",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(vis, "zielony: GT pilki, niebieski: start", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(path), vis)


def odm_ortho_crop(cfg: PipelineConfig, bbox: tuple, px: int) -> Optional[np.ndarray]:
    path = cfg.odm_dir / "odm_orthophoto" / "odm_orthophoto.tif"
    if not path.exists():
        return None
    import rasterio
    from rasterio.warp import Resampling, reproject
    from rasterio.transform import from_bounds
    dst = np.zeros((3, px, px), np.uint8)
    with rasterio.open(path) as src:
        for b in range(3):
            reproject(rasterio.band(src, b + 1), dst[b], dst_transform=from_bounds(*bbox, px, px),
                      dst_crs="EPSG:2180", resampling=Resampling.bilinear)
    return np.ascontiguousarray(np.transpose(dst, (1, 2, 0))[..., ::-1])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=ROOT / "config_pilka.yaml")
    ap.add_argument("--experiment", type=Path, default=ROOT / "experiments" / "pilka.yaml")
    ap.add_argument("--out", type=Path, default=ROOT / "output" / "pilka" / "reference")
    ap.add_argument("--size-m", type=float, default=60.0)
    ap.add_argument("--res-m", type=float, default=0.05)
    ap.add_argument("--building-radius-m", type=float, default=6.0,
                    help="budynek z GT: obrys w tej odleglosci od punktu GT")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    import yaml
    from pyproj import Transformer
    from geodetect.surface import load_mosaic, raster_files, sample_grid
    cfg = PipelineConfig.from_file(args.config)
    spec = yaml.safe_load(args.experiment.read_text(encoding="utf-8"))["metrics"]["reference_point"]
    t = Transformer.from_crs(4326, 2180, always_xy=True)
    glat, glon = spec["latlon"]
    slat, slon = cfg.z_anchor.start.latlon
    gx, gy = t.transform(glon, glat)
    sx, sy = t.transform(slon, slat)
    ref = Reference(glat, glon, gx, gy, slat, slon, sx, sy)
    args.out.mkdir(parents=True, exist_ok=True)

    ref.orto_sheets = index_metadata(ORTO_INDEX, ORTO_LAYERS, gx, gy)
    ref.nmpt_sheets = index_metadata(NMPT_INDEX, NMPT_LAYERS, gx, gy)
    ref.nmt_sheets = index_metadata(NMT_INDEX, NMT_LAYERS, gx, gy)
    for k in ("orto_sheets", "nmpt_sheets", "nmt_sheets"):
        for d in getattr(ref, k):
            log.info("%s: %s", k, {kk: d.get(kk) for kk in ("layer", "godlo", "aktualnosc", "wielkoscPiksela",
                                                            "zrodloDanych", "charakterystykaPrzestrzenna", "error")
                                   if d.get(kk)})

    b = 60.0
    bb = (min(gx, sx) - b, min(gy, sy) - b, max(gx, sx) + b, max(gy, sy) + b)
    nmt = load_mosaic(raster_files(cfg.surface.nmt_dir), "EPSG:2180", bb)
    nmpt = load_mosaic(raster_files(cfg.surface.nmpt_dir), "EPSG:2180", bb)
    ref.nmt_at_gt = float(sample_grid(nmt, np.array([gx]), np.array([gy]))[0])
    ref.nmpt_at_gt = float(sample_grid(nmpt, np.array([gx]), np.array([gy]))[0])
    ref.roof_height_m = ref.nmpt_at_gt - ref.nmt_at_gt
    ref.nmt_at_start = float(sample_grid(nmt, np.array([sx]), np.array([sy]))[0])

    h = args.size_m / 2
    bbox = (gx - h, gy - h, gx + h, gy + h)
    px = int(round(args.size_m / args.res_m))
    to_px = lambda x, y: ((x - bbox[0]) / args.res_m, (bbox[3] - y) / args.res_m)  # noqa: E731
    gt_px, st_px = to_px(gx, gy), to_px(sx, sy)
    st_px = st_px if 0 <= st_px[0] < px and 0 <= st_px[1] < px else None
    try:
        orto = get_map(ORTO_WMS, "Raster", bbox, px, False, "image/jpeg")
        egib = get_map(EGIB_WMS, "budynki", bbox, px, True)
    except Exception as exc:
        log.error("WMS niedostepny: %s", exc)
        (args.out / "reference.json").write_text(json.dumps(asdict(ref), indent=2, ensure_ascii=False),
                                                 encoding="utf-8")
        return 1
    for name, img in (("ortho_gugik.png", orto), ("egib_budynki.png", egib)):
        cv2.imwrite(str(args.out / name), img)
        write_world_file(args.out / name, bbox, px)
        ref.files[name] = str(args.out / name)

    fps = footprint_masks(egib)
    rad = args.building_radius_m / args.res_m
    def dist_to(m):
        ys, xs = np.nonzero(m)
        return float(np.min(np.hypot(xs - gt_px[0], ys - gt_px[1])))
    cand = sorted([(dist_to(m), k) for k, m in enumerate(fps)])
    if not cand or cand[0][0] > rad:
        log.error("Nie znaleziono obrysu budynku w promieniu %.1f m od GT (obrysow: %d)",
                  args.building_radius_m, len(fps))
    else:
        # grupa: obrysy stykajace sie z obrysem zawierajacym GT (wspolny dach)
        k0 = cand[0][1]
        grow = lambda m, px_: cv2.dilate(m.astype(np.uint8), np.ones((px_, px_), np.uint8)).astype(bool)  # noqa: E731
        group = fps[k0].copy()
        for m in fps:                      # obrysy stykajace sie z obrysem GT (jeden poziom)
            if (grow(m, 15) & fps[k0]).any():
                group |= m
        union = np.zeros_like(group)
        for m in fps:
            union |= m
        others = grow(union & ~grow(group, 3), int(0.6 / args.res_m))
        log.info("Budynek z GT: %d obrys(y), %.0f m2; obrysow w kadrze: %d",
                 sum(1 for m in fps if (m & group).sum() > 0.9 * m.sum()), group.sum() * args.res_m ** 2, len(fps))

        def run(img, tag, title):
            rm = roof_mask(img)
            g = estimate_lean(rm, group, args.res_m, scope="budynek z GT", exclude=others)
            a = estimate_lean(rm, union, args.res_m, scope="wszystkie budynki w kadrze")
            lean_figure(img, group, g, args.res_m, gt_px, st_px, f"{title}: budynek z GT",
                        args.out / f"lean_{tag}.png")
            lean_figure(img, union, a, args.res_m, gt_px, st_px, f"{title}: wszystkie obrysy",
                        args.out / f"lean_{tag}_all.png")
            log.info("Lean %s: GT %s | wszystkie %s", tag, g, a)
            return g, a
        ref.lean_gugik, ref.lean_gugik_all = run(orto, "gugik", "Ortofotomapa GUGiK 2025")
        odm = odm_ortho_crop(cfg, bbox, px)
        if odm is not None:
            cv2.imwrite(str(args.out / "ortho_odm.png"), odm)
            write_world_file(args.out / "ortho_odm.png", bbox, px)
            ref.lean_odm, ref.lean_odm_all = run(odm, "odm", "Ortofotomapa ODM 15.11.2026")
        else:
            log.info("Brak ortofotomapy ODM (%s) — pomijam porownanie", cfg.odm_dir / "odm_orthophoto")
    (args.out / "reference.json").write_text(json.dumps(asdict(ref), indent=2, ensure_ascii=False),
                                             encoding="utf-8")
    print(json.dumps({k: v for k, v in asdict(ref).items() if k not in ("files",)}, indent=1,
                     ensure_ascii=False)[:4000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
