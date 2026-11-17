"""Tryb z_anchor.mode=start_point (syntetyczne kamery ze znanym offsetem), wykluczanie\nzdjecia startowego (run_batch, prepare_odm), punkty reczne, geometria walidacji."""

import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from geodetect import batch as batch_mod
from geodetect.config import PipelineConfig, StartConfig, ZAnchorConfig
from geodetect.detect import ManualDetector, make_detector, read_manual_points, top_k_per_class
from geodetect.detect import Detection
from geodetect.exif import GpsFix
from geodetect.geo import EnuFrame, GeoRef
from geodetect.validation import along_across, combined_sigma, fit_line, gt_transfer, helmert2d, predicted_overshoot
from geodetect.zcal import calibrate_z, start_point

ROOT = Path(__file__).resolve().parents[1]
REF = GeoRef(53.8238, 21.4352, 0.0)


class ConstDem:
    def __init__(self, z, slope_x=0.0):
        self.z, self.slope_x = z, slope_x
        self.calls = []

    def sample(self, x, y):
        self.calls.append((x, y))
        return self.z + self.slope_x * (x - 660000.0)


def synthetic(photos: Path, c_true: float, alts: list[float], noise: list[float], start_alt=1.4):
    """Kamery ODM: u = alt_EXIF + c_true (+ szum) w ENU; zdjecie startowe S.JPG na ziemi."""
    frame = EnuFrame(REF)
    shots, gps = {}, {}
    for k, (a, n) in enumerate(zip(alts, noise)):
        name = f"C{k}.JPG"
        (photos / name).write_bytes(b"")
        shots[name] = np.array([5.0 * k, -3.0 * k, a + c_true + n])
        lon, lat, _ = frame.enu_to_geodetic(*shots[name][:2], 0.0)
        gps[name] = GpsFix(float(lat), float(lon), a)
    (photos / "S.JPG").write_bytes(b"")
    slon, slat, _ = frame.enu_to_geodetic(-10.0, 4.0, 0.0)
    gps["S.JPG"] = GpsFix(float(slat), float(slon), start_alt)
    rec = SimpleNamespace(
        image_names=sorted(shots),
        camera=lambda name: (None, SimpleNamespace(C_enu=shots[name])))
    read = lambda p: gps[Path(p).name]  # noqa: E731
    return frame, rec, read


class StartPointTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.photos = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def za(self, **kw):
        st = StartConfig(**{"image": "S.JPG", **kw})
        return ZAnchorConfig(mode="start_point", start=st)

    def test_known_offset_recovered(self):
        c = 7.3     # h_ODM = alt_EXIF + c
        frame, rec, read = synthetic(self.photos, c, [11.5, 12.1, 15.0, 19.9, 25.4],
                                     [0.02, -0.03, 0.0, 0.05, -0.01])
        dem = ConstDem(143.66)
        with mock.patch("geodetect.exif.read_gps", side_effect=read):
            zc = start_point(self.za(image_height_m=1.0), rec, dem, frame, self.photos)
        # grunt = EXIF 0 -> z_grunt_ODM = c; offset = NMT - c
        self.assertAlmostEqual(zc.offset, 143.66 - c, delta=0.02)
        self.assertEqual(zc.method, "start_point")
        st = zc.stats
        self.assertEqual(st["n_cameras"], 5)                       # S.JPG wykluczone
        self.assertAlmostEqual(st["median_hodm_minus_exif"], c, delta=0.02)
        self.assertAlmostEqual(st["mad_hodm_minus_exif"], 0.02, delta=0.015)
        self.assertEqual(st["start_xy_source"], "exif:S.JPG")
        # punkt gruntu startu (EXIF 0) -> H npm = NMT
        self.assertAlmostEqual(0.0 + st["median_hodm_minus_exif"] + zc.offset, 143.66, delta=1e-9)
        # kontrola barometru i wariant porownawczy "grunt = 1.4 m"
        self.assertAlmostEqual(st["start_image_alt_exif"], 1.4)
        self.assertAlmostEqual(st["baro_check_m"], 0.4)
        self.assertAlmostEqual(zc.offset - st["offset_if_ground_at_image_alt"], 1.4)

    def test_latlon_priority_and_ground_alt(self):
        frame, rec, read = synthetic(self.photos, 2.0, [10, 12, 14], [0, 0, 0])
        dem = ConstDem(100.0, slope_x=0.01)        # 1 cm / m: wplyw XY startu na NMT
        lat, lon = 53.82385, 21.43530
        with mock.patch("geodetect.exif.read_gps", side_effect=read):
            zc = start_point(self.za(latlon=[lat, lon], ground_alt_exif_m=0.5), rec, dem, frame, self.photos)
        st = zc.stats
        self.assertEqual(st["start_xy_source"], "latlon")
        x, y = frame.lonlat_to_puwg92(lon, lat)
        self.assertAlmostEqual(st["start_x_2180"], float(x), places=6)
        self.assertAlmostEqual(zc.offset, dem.sample(x, y) - (0.5 + 2.0), delta=0.01)
        self.assertNotIn("baro_check_m", st)
        self.assertGreater(st["start_xy_exif_vs_used_m"], 1.0)
        self.assertAlmostEqual(st["nmt_exif_minus_used"],
                               0.01 * (st["exif_start_x_2180"] - st["start_x_2180"]), places=6)

    def test_errors(self):
        frame, rec, read = synthetic(self.photos, 2.0, [10, 12], [0, 0])
        with mock.patch("geodetect.exif.read_gps", side_effect=read):
            with self.assertRaises(ValueError):                  # za malo kamer
                start_point(self.za(), rec, ConstDem(100), frame, self.photos)
            with self.assertRaises(ValueError):                  # brak startu
                start_point(ZAnchorConfig(mode="start_point"), rec, ConstDem(100), frame, self.photos)

    def test_dispatcher(self):
        frame, rec, read = synthetic(self.photos, 2.0, [10, 12, 14], [0, 0, 0])
        cfg = PipelineConfig(photos_dir=self.photos, z_anchor=self.za())
        with mock.patch("geodetect.exif.read_gps", side_effect=read):
            zc = calibrate_z(cfg, ConstDem(100), rec, frame)
        self.assertEqual(zc.method, "start_point")
        self.assertNotIn("per_camera", zc.describe())

    def test_config_from_yaml_dict(self):
        cfg = PipelineConfig.from_dict({"z_anchor": {"mode": "start_point", "start": {
            "image": "SING0380.JPG", "latlon": [53.82, 21.43], "ground_alt_exif_m": 0.0}}})
        self.assertIsInstance(cfg.z_anchor.start, StartConfig)
        self.assertEqual(cfg.excluded_images(), {"SING0380.JPG"})
        self.assertEqual(PipelineConfig().excluded_images(), set())


class ExclusionTest(unittest.TestCase):
    def test_run_batch_skips_start_image(self):
        with tempfile.TemporaryDirectory() as d:
            photos = Path(d)
            for n in ("S.JPG", "A.JPG", "B.JPG"):
                (photos / n).write_bytes(b"")
            cfg = PipelineConfig(photos_dir=photos)
            cfg.z_anchor.start.image = "S.JPG"
            ctx = mock.Mock()
            ctx.reconstruction.image_names = ["A.JPG", "B.JPG"]
            with mock.patch.object(batch_mod, "run_single",
                                   side_effect=lambda c, n, x: mock.Mock(image_name=n, detections=[], misses=[])), \
                    self.assertLogs("geodetect.batch", level="WARNING"):
                res = batch_mod.run_batch(cfg, ctx)
                res2 = batch_mod.run_batch(cfg, ctx, images=["S.JPG", "A.JPG"])
        self.assertEqual([r.image_name for r in res.images], ["A.JPG", "B.JPG"])
        self.assertIn("startowe", res.skipped["S.JPG"])
        self.assertEqual([r.image_name for r in res2.images], ["A.JPG"])

    def test_prepare_odm_excludes_and_removes(self):
        sys.path.insert(0, str(ROOT / "scripts"))
        from prepare_odm import prepare_odm
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "foto").mkdir()
            for n in ("S.JPG", "A.JPG", "B.JPG", "notes.txt"):
                (d / "foto" / n).write_bytes(b"x")
            (d / "odm" / "images").mkdir(parents=True)
            (d / "odm" / "images" / "S.JPG").write_bytes(b"x")       # zostalo z poprzedniego kopiowania
            cfg = PipelineConfig(photos_dir=d / "foto", odm_dir=d / "odm")
            cfg.z_anchor.start.image = "S.JPG"
            r = prepare_odm(cfg)
            self.assertEqual(sorted(r.copied), ["A.JPG", "B.JPG"])
            self.assertEqual(r.excluded, ["S.JPG"])
            self.assertEqual(r.removed, ["S.JPG"])
            self.assertEqual(sorted(p.name for p in (d / "odm" / "images").iterdir()), ["A.JPG", "B.JPG"])
            self.assertEqual(sorted(prepare_odm(cfg).unchanged), ["A.JPG", "B.JPG"])


class ManualPointsTest(unittest.TestCase):
    def test_csv_and_detector(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "pts.csv"
            p.write_text("image,u,v,visible,source,conf\nA.JPG,100.5,200,1,x,\nB.JPG,,,,,\n"
                         "C.JPG,5,6,0,,\n", encoding="utf-8")
            pts = read_manual_points(p)
            self.assertTrue(pts["A.JPG"].visible)
            self.assertFalse(pts["B.JPG"].visible)
            self.assertFalse(pts["C.JPG"].visible)
            cfg = PipelineConfig().detect
            cfg.source, cfg.manual_csv, cfg.manual_class, cfg.manual_box_px = "manual", p, "sports ball", 4
            det = make_detector(Path("nie_istnieje.pt"), cfg)
            self.assertIsInstance(det, ManualDetector)
            (a,) = det.detect(Path("x") / "A.JPG")
            self.assertEqual(a.cls_name, "sports ball")
            self.assertEqual(a.center, (100.5, 200.0))
            self.assertEqual(det.detect("B.JPG"), [])
            cfg.source = "inne"
            with self.assertRaises(ValueError):
                make_detector(Path("x.pt"), cfg)

    def test_top_k_per_class(self):
        d = [Detection(0, "a", c, (0, 0, 1, 1)) for c in (0.1, 0.5, 0.3)] + [Detection(1, "b", 0.2, (0, 0, 1, 1))]
        out = top_k_per_class(d, 1)
        self.assertEqual(sorted((x.cls_name, x.conf) for x in out), [("a", 0.5), ("b", 0.2)])
        self.assertEqual(len(top_k_per_class(d, None)), 4)


class ValidationGeometryTest(unittest.TestCase):
    def test_along_across(self):
        # kamera na poludnie od GT, trafienie 2 m dalej na polnoc i 1 m na wschod
        al, ac = along_across((101.0, 202.0), (100.0, 200.0), (100.0, 150.0))
        self.assertAlmostEqual(al, 2.0)
        self.assertAlmostEqual(ac, 1.0)

    def test_nmt_overshoot_formula(self):
        # dach 6.65 m, kat padania 45 deg -> trafienie w NMT 6.65 m dalej od kamery
        self.assertAlmostEqual(predicted_overshoot(6.65, 45.0), 6.65)
        self.assertAlmostEqual(predicted_overshoot(6.65, 0.0), 0.0)

    def test_fit_and_budget(self):
        f = fit_line([1, 2, 3, 4], [2.1, 3.9, 6.0, 8.1])
        self.assertAlmostEqual(f["slope"], 2.0, delta=0.05)
        self.assertIsNone(fit_line([1, 1, 1], [1, 2, 3]))
        self.assertAlmostEqual(combined_sigma([0.3, 0.4, None]), 0.5)
        self.assertTrue(math.isnan(combined_sigma([None])))


    def test_helmert_and_transfer(self):
        src = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 8.0], [7.0, 5.0]])
        th = math.radians(2.0)
        R = 1.001 * np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
        dst = src @ R.T + [3.8, -0.1]
        h = helmert2d(src, dst)
        self.assertAlmostEqual(h["scale"], 1.001, places=6)
        self.assertAlmostEqual(h["rot_deg"], 2.0, places=6)
        self.assertLess(h["rms"], 1e-9)
        np.testing.assert_allclose(h["apply"]([[5, 5]])[0], np.array([5, 5]) @ R.T + [3.8, -0.1])
        one = helmert2d([[1, 1]], [[2, 3]])
        np.testing.assert_allclose(one["apply"]([[0, 0]])[0], [1, 2])
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "pairs.csv"
            p.write_text("id,kind,gugik_x,gugik_y,odm_x,odm_y\n1,punkt_dachu,13,9,10,10\n"
                         "2,punkt_dachu,23,9,20,10\npilka,pilka,,,15,12\n", encoding="utf-8")
            t = gt_transfer(p, {"dx_m": 0.6, "dy_m": -1.25})
            self.assertAlmostEqual(t["x"], 18.0 - 0.6)
            self.assertAlmostEqual(t["y"], 11.0 + 1.25)
            self.assertIsNone(gt_transfer(Path(d) / "brak.csv", None))

    def test_csv_detection_source(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "best.csv"
            p.write_text("image,u,v,visible,conf,x1,y1,x2,y2\nA.JPG,15,25,1,0.46,10,20,20,30\n"
                         "B.JPG,,,0,,,,,\n", encoding="utf-8")
            cfg = PipelineConfig().detect
            cfg.source, cfg.detections_csv, cfg.manual_class = "csv", p, "sports ball"
            det = make_detector(Path("x.pt"), cfg)
            (a,) = det.detect("A.JPG")
            self.assertEqual((a.conf, a.bbox, a.source), (0.46, (10, 20, 20, 30), "csv"))
            self.assertEqual(det.detect("B.JPG"), [])


if __name__ == "__main__":
    unittest.main()
