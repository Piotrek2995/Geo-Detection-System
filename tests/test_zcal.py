import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from pyproj import Transformer

from geodetect.batch import BatchResult
from geodetect.config import PipelineConfig, ZAnchorConfig
from geodetect.detect import Detection
from geodetect.diagnostics import multiview_z_scan
from geodetect.geo import EnuFrame, GeoRef
from geodetect.odm import CameraEO, CameraIO
from geodetect.pipeline import GeoDetection
from geodetect.raycast import pixel_to_ray, raycast
from geodetect.zcal import ZCalibration, dem_alignment, fit_plane, ground_frame

REF = GeoRef(lat=53.823177673972694, lon=21.436584949186468, alt=0.0)
# proj4 zamiast "EPSG:32634": odporne na obcy proj.db (np. PROJ_LIB z PostGIS) w GDAL
UTM34_PROJ4 = "+proj=utm +zone=34 +datum=WGS84 +units=m +no_defs"
IO = CameraIO(fx=1000, fy=1000, cx=500, cy=500, dist=np.zeros(5), width=1000, height=1000)


class FlatDem:
    def __init__(self, z):
        self.z = z

    def sample(self, x, y):
        return self.z


class FakeRec:
    """Minimalna rekonstrukcja: shots = {nazwa: CameraEO}."""

    def __init__(self, eos: dict):
        self.shots = eos
        self.path = Path("reconstruction.topocentric.json")

    @property
    def image_names(self):
        return sorted(self.shots)

    def camera(self, name):
        return IO, self.shots[name]


def look_at(C: np.ndarray, P: np.ndarray) -> CameraEO:
    """Kamera w C patrzaca na P (x w prawo, y w dol, z w przod)."""
    z = (P - C) / np.linalg.norm(P - C)
    x = np.cross(z, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return CameraEO(C_enu=C.astype(float), R_wc=np.vstack([x, y, z]))


def project(eo: CameraEO, P: np.ndarray) -> tuple[float, float]:
    pc = eo.R_wc @ (P - eo.C_enu)
    return IO.fx * pc[0] / pc[2] + IO.cx, IO.fy * pc[1] / pc[2] + IO.cy


class GroundFrameTest(unittest.TestCase):
    def test_offset_from_ground_anchor(self):
        frame = EnuFrame(REF)
        anchor = CameraEO(C_enu=np.array([12.0, -7.0, 0.4]), R_wc=np.eye(3))
        rec = FakeRec({"START.JPG": anchor})
        z = ground_frame(rec, "START.JPG", FlatDem(131.2), frame, lens_height_m=0.15)
        _, _, h = frame.enu_to_geodetic(12.0, -7.0, 0.4)
        self.assertAlmostEqual(z.offset, 131.2 + 0.15 - h, places=6)
        self.assertEqual(z.method, "ground_frame")
        # po kalibracji kamera-anchor ma wysokosc NMT + obiektyw
        _, _, H = EnuFrame(REF, z.offset).enu_to_puwg92(12.0, -7.0, 0.4)
        self.assertAlmostEqual(H, 131.35, places=6)

    def test_missing_anchor_is_clear_error(self):
        rec = FakeRec({"A.JPG": CameraEO(np.zeros(3), np.eye(3))})
        with self.assertRaisesRegex(ValueError, "nie zostalo zrekonstruowane"):
            ground_frame(rec, "START.JPG", FlatDem(0), EnuFrame(REF), 0.15)
        with self.assertRaisesRegex(ValueError, "wymaga z_anchor.image"):
            ground_frame(rec, None, FlatDem(0), EnuFrame(REF), 0.15)


class DemAlignmentTest(unittest.TestCase):
    """Syntetyczny DEM ODM w UTM 34N = NMT - (offset + przechyl). Odzyskujemy oba."""
    OFFSET = 127.5
    A, B = 0.004, -0.065   # [m/m] w EPSG:2180 -> nachylenie ~3.73 deg

    def nmt(self, x, y):
        return 135.0 + 0.01 * (x - 660330.0)   # prawdziwy teren tez lekko nachylony

    def make_rasters(self, d: Path, with_tree: bool):
        import rasterio
        from rasterio.transform import from_origin
        res, w, h = 1.0, 200, 160
        x0, y0 = 528640.0, 5964100.0
        cols, rows = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
        ux, uy = x0 + cols * res, y0 - rows * res
        x2180, y2180 = Transformer.from_crs("EPSG:32634", "EPSG:2180", always_xy=True).transform(ux, uy)
        self.cx, self.cy = float(np.mean(x2180)), float(np.mean(y2180))
        tilt = self.A * (x2180 - self.cx) + self.B * (y2180 - self.cy)
        dtm = self.nmt(x2180, y2180) - self.OFFSET - tilt
        dsm = dtm.copy()
        if with_tree:
            dsm[:, : w // 2] += 8.0     # zachodnia polowa "zalesiona" -> wykluczona
        paths = {}
        for name, arr in (("dtm.tif", dtm), ("dsm.tif", dsm)):
            paths[name] = d / name
            with rasterio.open(paths[name], "w", driver="GTiff", height=h, width=w, count=1,
                               dtype="float64", crs=UTM34_PROJ4,
                               transform=from_origin(x0, y0, res, res), nodata=-9999.0) as ds:
                ds.write(arr, 1)
        return paths["dtm.tif"], paths["dsm.tif"]

    def dem(self):
        test = self

        class Nmt:
            def sample(self, x, y):
                return float(test.nmt(x, y))
        return Nmt()

    def test_recovers_offset_and_tilt(self):
        with tempfile.TemporaryDirectory() as d:
            dtm, dsm = self.make_rasters(Path(d), with_tree=True)
            za = ZAnchorConfig(mode="dem_alignment", n_samples=300, buffer_m=10.0)
            z = dem_alignment(za, dtm, dsm, self.dem())
            p = z.alignment.plane
            self.assertAlmostEqual(p.a, self.A, places=5)
            self.assertAlmostEqual(p.b, self.B, places=5)
            self.assertAlmostEqual(p.c + p.a * (self.cx - p.x0) + p.b * (self.cy - p.y0),
                                   self.OFFSET, places=3)
            self.assertAlmostEqual(p.slope_deg, np.degrees(np.arctan(np.hypot(self.A, self.B))),
                                   places=4)
            self.assertAlmostEqual(p.aspect_deg, (np.degrees(np.arctan2(self.A, self.B)) + 360) % 360,
                                   places=3)
            # tylko "otwarta" (wschodnia) polowa — DSM-DTM < 0.5 m
            ux = Transformer.from_crs("EPSG:2180", "EPSG:32634", always_xy=True).transform(
                z.alignment.x_2180, z.alignment.y_2180)[0]
            self.assertTrue(np.all(ux > 528640.0 + 100.0))
            self.assertEqual(z.stats["n_points"], 300)
            self.assertIsNone(z.tilt)

    def test_apply_tilt_returns_plane(self):
        with tempfile.TemporaryDirectory() as d:
            dtm, dsm = self.make_rasters(Path(d), with_tree=False)
            za = ZAnchorConfig(mode="dem_alignment", n_samples=200, apply_tilt=True)
            z = dem_alignment(za, dtm, dsm, self.dem())
            self.assertIsNotNone(z.tilt)
            self.assertEqual(z.method, "dem_alignment+tilt")
            # H po korekcie w dowolnym punkcie = NMT
            x, y = self.cx + 40.0, self.cy - 30.0
            h_odm = self.nmt(x, y) - self.OFFSET - (self.A * 40.0 + self.B * -30.0)
            self.assertAlmostEqual(h_odm + z.offset + z.tilt(x, y), self.nmt(x, y), places=3)

    def test_missing_dem_mentions_docker(self):
        with self.assertRaisesRegex(FileNotFoundError, "--rerun-from odm_georeferencing --dtm --dsm"):
            dem_alignment(ZAnchorConfig(), Path("nope/dtm.tif"), Path("nope/dsm.tif"), FlatDem(0))

    def test_fit_plane_ignores_outliers(self):
        rng = np.random.default_rng(1)
        x, y = rng.uniform(0, 100, 200), rng.uniform(0, 100, 200)
        d = 5.0 + 0.02 * x - 0.01 * y + rng.normal(0, 0.05, 200)
        d[:10] += 30.0
        p = fit_plane(x, y, d)
        self.assertAlmostEqual(p.a, 0.02, places=3)
        self.assertAlmostEqual(p.b, -0.01, places=3)
        self.assertEqual(p.n_used, 190)


class MultiviewScanTest(unittest.TestCase):
    def test_finds_true_offset(self):
        true_offset, ground = 50.0, 100.0
        dem = FlatDem(ground)
        P = np.array([0.0, 0.0, ground - true_offset])
        eos = {}
        for i, (az, dist) in enumerate([(0, 50), (60, 70), (130, 45), (200, 80), (290, 60)]):
            a = np.radians(az)
            C = np.array([dist * np.sin(a), dist * np.cos(a), 95.0])
            eos[f"S{i}.JPG"] = look_at(C, P)
        rec = FakeRec(eos)
        base = true_offset + 5.0     # zly offset -> rozrzut
        frame = EnuFrame(REF, base)
        cfg = PipelineConfig()
        cfg.raycast.step_m = 2.0
        dets = []
        for name, eo in eos.items():
            uv = project(eo, P)
            o, d = pixel_to_ray(uv, IO, eo)
            hit = raycast(o, d, dem, frame, cfg.raycast)
            inc = float(np.degrees(np.arccos(-d[2])))
            dets.append(GeoDetection(name, Detection(0, "namiot", 0.9, (0, 0, 1, 1)), uv, hit,
                                     inc, False))
        batch = BatchResult(images=[SimpleNamespace(image_name="x", detections=dets, misses=[])],
                            skipped={})
        ctx = SimpleNamespace(reconstruction=rec, z_calibration=ZCalibration(base, "test"),
                              frame=frame, dem=dem, config=cfg)
        scan = multiview_z_scan(batch, ctx, span_m=10.0, step_m=0.25)
        self.assertEqual(scan.n_objects, 1)
        self.assertEqual(scan.n_detections, 5)
        self.assertAlmostEqual(scan.best_offset, true_offset, delta=0.25)
        self.assertLess(scan.best_rms, 0.1)
        self.assertGreater(scan.rms[0], 1.0)


if __name__ == "__main__":
    unittest.main()
