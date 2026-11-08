"""Porownanie prototypu (legacy/prototype_main.py) z pakietem geodetect.

Warianty:
  A  legacy           — stary main.py bez zmian
  B  new, stare bledy — nowy kod + naiwne ENU (ref_x+E, ref_y+N) + zamienione osie GUGiK
                        + reconstruction.json (UTM z offsetem) czytany jak ENU
                        (powinno dac to samo co A -> refaktor wierny)
  (C i D czytaja reconstruction.topocentric.json; kalibracja Z: agl_assumption)
  C  new, naiwne ENU  — poprawione osie GUGiK, naiwne ENU
  D  new              — pelna poprawka (scisle ENU -> WGS84 -> EPSG:2180)

Uzycie:  python scripts/compare_legacy.py [--image SING0304.JPG ...]
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from geodetect.config import PipelineConfig  # noqa: E402
from geodetect.dem import GugikDem  # noqa: E402
from geodetect.detect import Detector  # noqa: E402
from geodetect.geo import EnuFrame  # noqa: E402
from geodetect.odm import OdmReconstruction  # noqa: E402
from geodetect.zcal import agl_assumption  # noqa: E402
from geodetect.pipeline import PipelineContext, run_single  # noqa: E402


class NaiveEnuFrame(EnuFrame):
    """Odtworzenie zalozenia prototypu: osie ENU == osie EPSG:2180."""

    def __init__(self, ref, z_offset=0.0):
        super().__init__(ref, z_offset)
        self._ref_x, self._ref_y = self.lonlat_to_puwg92(ref.lon, ref.lat)

    def with_z_offset(self, z_offset):
        return NaiveEnuFrame(self.ref, z_offset)

    def enu_to_puwg92(self, e, n, u):
        return self._ref_x + e, self._ref_y + n, self.ref.alt + u + self.z_offset


class SwappedAxesGugikDem(GugikDem):
    """Odtworzenie bledu prototypu: x=easting, y=northing w GetHByXY."""

    def _query(self, easting, northing):
        return super()._query(northing, easting)


def run_legacy(image_name: str) -> list[dict]:
    import legacy.prototype_main as old
    old.PHOTOS_DIR = ROOT / "zdjecia"
    old.MODEL_PATH = ROOT / "LudzieiNamioty.pt"
    old.ODM_RECONSTRUCTION = ROOT / "odm_project" / "opensfm" / "reconstruction.json"
    old.ODM_REFERENCE_LLA = ROOT / "odm_project" / "opensfm" / "reference_lla.json"
    old.ODM_SHOTS_GEOJSON = ROOT / "odm_project" / "odm_report" / "shots.geojson"
    dem = old.GugikDem()
    z_offset = old.compute_z_offset(dem, old.AGL_LOT_M)
    io, eo = old.parse_odm_for_image(image_name, z_offset=z_offset)
    out = []
    for d in old.detect_objects(old.PHOTOS_DIR / image_name):
        o, v = old.pixel_to_world_ray(d["center"], io, eo)
        hit = old.raycast_on_dem(o, v, dem)
        out.append({"cls": d["cls_name"], "conf": d["conf"], "uv": d["center"],
                    "xyz": None if hit is None else tuple(float(c) for c in hit),
                    "cam": tuple(float(c) for c in eo.C_xyz), "z_offset": z_offset})
    return out


def run_new(cfg: PipelineConfig, image_name: str, detector: Detector,
            naive_enu: bool, swapped_axes: bool, legacy_rec: bool = False) -> list[dict]:
    dem_cls = SwappedAxesGugikDem if swapped_axes else GugikDem
    dem = dem_cls(cfg.dem.gugik_url, timeout=cfg.dem.timeout_s)
    rec_path = cfg.odm_dir / "opensfm" / "reconstruction.json" if legacy_rec else cfg.reconstruction_path
    rec = OdmReconstruction(rec_path, cfg.reference_lla_path)
    frame = (NaiveEnuFrame if naive_enu else EnuFrame)(rec.ref)
    z_cal = agl_assumption(cfg.shots_geojson_path, dem, frame, cfg.z_anchor.agl_m)
    ctx = PipelineContext(cfg, dem, rec, frame.with_z_offset(z_cal.offset), z_cal, detector)
    res = run_single(cfg, image_name, ctx)
    cam = (res.camera.x_2180, res.camera.y_2180, res.camera.h)
    out = []
    # parowanie z legacy po uv (run_single zwraca trafione i chybione osobno)
    for gd in res.detections:
        out.append({"cls": gd.detection.cls_name, "conf": gd.detection.conf, "uv": gd.uv,
                    "xyz": (gd.hit.x_2180, gd.hit.y_2180, gd.hit.h), "cam": cam,
                    "z_offset": z_cal.offset})
    for det in res.misses:
        out.append({"cls": det.cls_name, "conf": det.conf, "uv": det.center, "xyz": None,
                    "cam": cam, "z_offset": z_cal.offset})
    return sorted(out, key=lambda r: r["uv"])


def d2(a, b) -> str:
    if a is None or b is None:
        return "   n/a"
    return f"{math.hypot(a[0] - b[0], a[1] - b[1]):6.2f}"


def fmt(xyz) -> str:
    return "brak trafienia" if xyz is None else f"X={xyz[0]:.2f} Y={xyz[1]:.2f} H={xyz[2]:.2f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", action="append", default=None)
    args = ap.parse_args()
    images = args.image or ["SING0304.JPG", "SING0316.JPG"]
    logging.basicConfig(level=logging.WARNING)

    cfg = PipelineConfig.from_dict({}, base_dir=ROOT)
    detector = Detector(cfg.model_path, conf=cfg.detect.conf)

    for name in images:
        A = sorted(run_legacy(name), key=lambda r: r["uv"])
        B = run_new(cfg, name, detector, naive_enu=True, swapped_axes=True, legacy_rec=True)
        C = run_new(cfg, name, detector, naive_enu=True, swapped_axes=False)
        D = run_new(cfg, name, detector, naive_enu=False, swapped_axes=False)

        print(f"\n===== {name} =====")
        same = [(a["cls"], round(a["conf"], 4), tuple(round(v, 1) for v in a["uv"])) for a in A] == \
               [(b["cls"], round(b["conf"], 4), tuple(round(v, 1) for v in b["uv"])) for b in D]
        print(f"detekcje: legacy={len(A)}, new={len(D)}, identyczne (klasa, conf, uv): {same}")
        print(f"z_offset: A={A[0]['z_offset'] if A else float('nan'):.3f}  "
              f"C/D={D[0]['z_offset'] if D else float('nan'):.3f}")
        if A:
            print(f"kamera  A: {fmt(A[0]['cam'])}")
            print(f"kamera  D: {fmt(D[0]['cam'])}   |A-D| poziomo = {d2(A[0]['cam'], D[0]['cam'])} m")
        for a, b, c, d in zip(A, B, C, D):
            dist_cam = math.hypot(d["xyz"][0] - d["cam"][0], d["xyz"][1] - d["cam"][1]) \
                if d["xyz"] else float("nan")
            print(f"\n  {a['cls']} conf={a['conf']:.3f} uv=({a['uv'][0]:.0f},{a['uv'][1]:.0f})"
                  f"  odl. pozioma od kamery (D) = {dist_cam:.1f} m")
            print(f"    A legacy            : {fmt(a['xyz'])}")
            print(f"    B new + stare bledy : {fmt(b['xyz'])}   |A-B| = {d2(a['xyz'], b['xyz'])} m")
            print(f"    C new, naiwne ENU   : {fmt(c['xyz'])}   |B-C| = {d2(b['xyz'], c['xyz'])} m"
                  f"  (osie GUGiK + uklad rekonstrukcji)")
            print(f"    D new (poprawne)    : {fmt(d['xyz'])}   |C-D| = {d2(c['xyz'], d['xyz'])} m"
                  f"  (efekt poprawki ENU)")
            print(f"    calkowita |A-D|     : {d2(a['xyz'], d['xyz'])} m")


if __name__ == "__main__":
    main()
