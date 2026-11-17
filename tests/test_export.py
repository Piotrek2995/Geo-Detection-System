"""Eksport przebiegu (GPKG, KML, GeoJSON, raport) na syntetycznym wyniku + run_pipeline."""

import json
import sqlite3
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest import mock

import numpy as np

from geodetect.aggregate import aggregate
from geodetect.batch import BatchResult
from geodetect.config import AggregateConfig, PipelineConfig, SizeConfig
from geodetect.detect import Detection
from geodetect.export import export_gpkg, export_kml, export_run, objects_geojson
from geodetect.gpkg import geometry_blob, parse_geometry, read_gpkg
from geodetect.pipeline import CameraPose, GeoDetection, ImageResult
from geodetect.raycast import RayHit
from geodetect.size import FLAG_SIZE, SizeEstimate
from geodetect.zcal import ZCalibration

X0, Y0 = 660300.0, 664850.0


def _lonlat(x, y):
    from pyproj import Transformer
    return Transformer.from_crs(2180, 4326, always_xy=True).transform(x, y)


def make_batch(photos_dir=Path(".")) -> BatchResult:
    cfg = PipelineConfig(photos_dir=photos_dir)
    cfg.size.ranges_m = {"namiot": [1.5, 6.0]}
    images = []
    specs = [  # (zdjecie, klasa, dx, dy, rozmiar)
        ("A.JPG", "namiot", 0.0, 0.0, 4.0), ("B.JPG", "namiot", 0.4, -0.3, 4.4),
        ("B.JPG", "namiot", 40.0, 10.0, 25.0), ("A.JPG", "people", 8.0, 2.0, 1.7)]
    for k, name in enumerate(("A.JPG", "B.JPG")):
        cx, cy = X0 - 30 + 10 * k, Y0 - 20
        lon, lat = _lonlat(cx, cy)
        cam = CameraPose(np.zeros(3), cx, cy, 180.0, lon, lat)
        r = ImageResult(name, None, None, None, cam, ZCalibration(126.0, "dem_alignment"))
        for img, cls, dx, dy, s in specs:
            if img != name:
                continue
            x, y = X0 + dx, Y0 + dy
            lo, la = _lonlat(x, y)
            hit = RayHit(np.zeros(3), x, y, 135.5, lo, la, 50.0, "nmt_gugik_rest", 135.5)
            gd = GeoDetection(name, Detection(0, cls, 0.9, (100, 100, 180, 160)), (140.0, 130.0), hit, 40.0, False)
            gd.size = SizeEstimate(s, s, s, s, "plane")
            r.detections.append(gd)
        images.append(r)
    b = BatchResult(images, {}, z_calibration=ZCalibration(126.0, "dem_alignment"), config=cfg)
    b.objects = aggregate(b.detections, AggregateConfig(eps_m=2.5), SizeConfig(ranges_m=cfg.size.ranges_m))
    return b


class GpkgTest(unittest.TestCase):
    def test_geometry_roundtrip(self):
        self.assertEqual(parse_geometry(geometry_blob((1.0, 2.0, 3.0), 2180, "POINT")),
                         ("POINT", (1.0, 2.0, 3.0), 2180))
        t, c, s = parse_geometry(geometry_blob([(0, 0, 1), (1, 2, 3)], 2180, "LINESTRING"))
        self.assertEqual((t, [tuple(p) for p in c]), ("LINESTRING", [(0, 0, 1), (1, 2, 3)]))

    def test_export_and_read_back(self):
        b = make_batch()
        with tempfile.TemporaryDirectory() as d:
            p = export_gpkg(b, Path(d) / "r.gpkg")
            con = sqlite3.connect(p)
            self.assertEqual(con.execute("PRAGMA application_id").fetchone()[0], 0x47504B47)
            self.assertEqual(con.execute("PRAGMA integrity_check").fetchone()[0], "ok")
            srs = {r[0] for r in con.execute("SELECT srs_id FROM gpkg_spatial_ref_sys")}
            self.assertTrue({-1, 0, 2180, 4326} <= srs)
            con.close()
            L = read_gpkg(p)
        self.assertEqual({k: len(v) for k, v in L.items()},
                         {"objects": 3, "detections": 4, "cameras": 2, "rays": 4})
        objs = {(o["class"], o["n_views"]): o for o in L["objects"]}
        tent = objs[("namiot", 2)]
        self.assertEqual(tent["status"], "confirmed")
        self.assertAlmostEqual(tent["size_m"], 4.2, places=2)
        self.assertEqual(tent["geometry"][0], "POINT")
        self.assertAlmostEqual(tent["geometry"][1][0], X0 + 0.2, places=3)
        self.assertAlmostEqual(tent["geometry"][1][2], 135.5, places=3)
        self.assertEqual(tent["srs_id"], 2180)
        self.assertIn("dem_alignment", tent["georef_mode"])
        roof = objs[("namiot", 1)]
        self.assertEqual(roof["flags"], FLAG_SIZE)
        self.assertEqual(roof["status"], "single_view")
        ray = L["rays"][0]["geometry"]
        self.assertEqual(ray[0], "LINESTRING")
        self.assertAlmostEqual(ray[1][0][2], 180.0)


class KmlGeojsonTest(unittest.TestCase):
    def test_kml(self):
        b = make_batch()
        with tempfile.TemporaryDirectory() as d:
            p = export_kml(b, Path(d) / "o.kml")
            root = ET.parse(p).getroot()
        ns = {"k": "http://www.opengis.net/kml/2.2"}
        folders = root.findall(".//k:Folder", ns)
        self.assertEqual([f.find("k:name", ns).text for f in folders], ["Obiekty", "Detekcje"])
        pms = folders[0].findall("k:Placemark", ns)
        self.assertEqual(len(pms), 3)
        flagged = [pm for pm in pms if pm.find("k:styleUrl", ns).text.endswith("_flag")]
        self.assertEqual(len(flagged), 1)
        lon, lat, h = map(float, flagged[0].find(".//k:coordinates", ns).text.split(","))
        elon, elat = _lonlat(X0 + 40, Y0 + 10)
        self.assertAlmostEqual(lon, elon, places=6)
        self.assertAlmostEqual(lat, elat, places=6)

    def test_geojson_wgs84_and_attributes(self):
        b = make_batch()
        fc = objects_geojson(b)
        f = fc["features"][0]
        lon, lat, h = f["geometry"]["coordinates"]
        self.assertTrue(21 < lon < 22 and 53 < lat < 54)
        for k in ("class_label", "n_views", "rms_m", "max_m", "mean_conf", "status", "size_m", "flags",
                  "surface", "georef_mode", "qc_status", "x_2180", "y_2180", "altitude_m"):
            self.assertIn(k, f["properties"])


class ExportRunTest(unittest.TestCase):
    def test_full_export(self):
        import cv2
        with tempfile.TemporaryDirectory() as d:
            photos = Path(d) / "photos"
            photos.mkdir()
            for n in ("A.JPG", "B.JPG"):
                cv2.imwrite(str(photos / n), np.full((300, 400, 3), 120, np.uint8))
            b = make_batch(photos)
            out = Path(d) / "run"
            paths = export_run(b, out)
            for k in ("gpkg", "kml", "objects_geojson", "detections_csv", "map_html", "report_html", "report_json"):
                self.assertTrue(paths[k].exists(), k)
            rep = json.loads(paths["report_json"].read_text(encoding="utf-8"))
            self.assertEqual(rep["statistics"]["flagged"], 1)
            self.assertTrue((out / rep["objects"][0]["thumbnail"]).exists())
            html = paths["report_html"].read_text(encoding="utf-8")
            self.assertIn(FLAG_SIZE, html)
            self.assertIn("map.html", html)
            self.assertIn("data:image/jpeg;base64", paths["map_html"].read_text(encoding="utf-8"))


class RunPipelineTest(unittest.TestCase):
    def test_cancel_and_progress(self):
        from geodetect import batch as batch_mod
        from geodetect.run import CancelToken, run_pipeline
        with tempfile.TemporaryDirectory() as d:
            photos = Path(d)
            for n in ("A.JPG", "B.JPG", "C.JPG"):
                (photos / n).write_bytes(b"")
            cfg = PipelineConfig(photos_dir=photos, output_dir=photos / "out")
            ctx = mock.Mock()
            ctx.reconstruction.image_names = ["A.JPG", "B.JPG", "C.JPG"]
            ctx.qc = None
            tok = CancelToken()
            seen = []

            def fake_run_single(cfg, name, ctx):
                if name == "B.JPG":
                    tok.cancel()                  # GUI klika "Przerwij" w trakcie
                return mock.Mock(image_name=name, detections=[], misses=[])

            with mock.patch.object(batch_mod, "run_single", side_effect=fake_run_single), \
                    mock.patch.object(batch_mod, "save_dem_cache"), self.assertLogs("geodetect", "WARNING"):
                res = run_pipeline(cfg, progress_cb=seen.append, cancel=tok, all_images=True, ctx=ctx)
            self.assertTrue(res.cancelled)
            self.assertFalse(res.ok)
            self.assertEqual(res.export, {})
            self.assertEqual([p.message for p in seen if p.stage == "images"], ["A.JPG", "B.JPG"])
            self.assertTrue(all(0 <= p.fraction <= 1 for p in seen))

            with mock.patch.object(batch_mod, "run_single", side_effect=lambda c, n, x: mock.Mock(
                    image_name=n, detections=[], misses=[])), mock.patch.object(batch_mod, "save_dem_cache"):
                res = run_pipeline(cfg, progress_cb=seen.append, all_images=True, ctx=ctx, export=False)
            self.assertTrue(res.ok)
            self.assertEqual(seen[-1].stage, "done")
            self.assertEqual(seen[-1].fraction, 1.0)

    def test_error_reported_not_raised(self):
        from geodetect.run import run_pipeline
        cfg = PipelineConfig()
        with mock.patch("geodetect.run.build_context", side_effect=FileNotFoundError("brak ODM")),                 self.assertLogs("geodetect.run", "ERROR"):
            res = run_pipeline(cfg)
        self.assertIn("brak ODM", res.error)
        self.assertFalse(res.ok)


if __name__ == "__main__":
    unittest.main()
