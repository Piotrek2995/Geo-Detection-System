"""Punkt reprezentatywny: punkt nadiru, styk z gruntem na syntetycznej masce, fallbacki."""

import math
import unittest

import numpy as np

from geodetect.config import PipelineConfig, PointConfig
from geodetect.odm import CameraEO, CameraIO
from geodetect.points import (choose_point, down_direction, footprint_pixels, ground_contact,
                              nadir_point)
from geodetect.segment import ObjectMask, clean_mask
from tests.test_raycast import nadir_camera


def oblique_camera(pitch_deg=45.0, height=50.0):
    """Kamera patrzy na polnoc, pochylona o pitch_deg od poziomu (90 = nadir)."""
    io = CameraIO(fx=1000, fy=1000, cx=500, cy=500, dist=np.zeros(5), width=1000, height=1000)
    p = math.radians(pitch_deg)
    z = np.array([0.0, math.cos(p), -math.sin(p)])     # os optyczna
    x = np.array([1.0, 0.0, 0.0])                       # prawo = wschod
    y = np.cross(z, x)                                  # dol obrazu
    return io, CameraEO(C_enu=np.array([0.0, 0.0, height]), R_wc=np.vstack([x, y, z]))


def project(io, eo, X):
    pc = eo.R_wc @ (np.asarray(X, float) - eo.C_enu)
    return io.K[0, 0] * pc[0] / pc[2] + io.K[0, 2], io.K[1, 1] * pc[1] / pc[2] + io.K[1, 2]


def rect_mask(u0, v0, u1, v1):
    return ObjectMask(np.ones((v1 - v0, u1 - u0), bool), u0, v0)


class NadirPointTest(unittest.TestCase):
    def test_nadir_camera(self):
        io, eo = nadir_camera(45.0)
        self.assertEqual(nadir_point(io, eo), (500.0, 500.0))
        d, off = down_direction((800.0, 500.0), io, eo)
        np.testing.assert_allclose(d, [-1.0, 0.0], atol=1e-6)   # piony zbiegaja sie do srodka
        self.assertAlmostEqual(off, math.degrees(math.atan(0.3)), places=4)
        _, off0 = down_direction((500.0, 500.0), io, eo)
        self.assertAlmostEqual(off0, 0.0, places=6)

    def test_oblique_camera_matches_projection_of_vertical(self):
        io, eo = oblique_camera(45.0, 50.0)
        # punkt nadiru = rzut punktu dowolnie gleboko pod kamera (tu daleko pod nia)
        nu, nv = nadir_point(io, eo)
        pu, pv = project(io, eo, [0.0, 0.0, -1e6])
        self.assertAlmostEqual(nu, pu, delta=1e-3)
        self.assertAlmostEqual(nv, pv, delta=1e-2)
        # obraz odcinka pionowego (slup na gruncie, 30 m przed kamera): kierunek gora -> dol
        top, foot = project(io, eo, [5.0, 30.0, 1.0]), project(io, eo, [5.0, 30.0, 0.0])
        seg = np.subtract(foot, top)
        d, off = down_direction(top, io, eo)
        np.testing.assert_allclose(d, seg / np.linalg.norm(seg), atol=1e-3)
        self.assertGreater(off, 30.0)


class GroundContactTest(unittest.TestCase):
    def test_contact_is_edge_towards_nadir(self):
        io, eo = nadir_camera(45.0)
        m = rect_mask(800, 480, 900, 520)                 # obiekt na prawo od nadiru
        u, v = ground_contact(m, io, eo, min_offnadir_deg=5.0, band=0.05)
        self.assertAlmostEqual(u, 801.0, delta=1.5)       # lewa krawedz (od strony nadiru)
        self.assertAlmostEqual(v, 500.0, delta=1.0)

    def test_oblique_tent_contact_at_bottom_edge(self):
        io, eo = oblique_camera(45.0, 50.0)
        m = rect_mask(450, 300, 550, 380)                 # pionowo nad srodkiem kadru
        u, v = ground_contact(m, io, eo, 5.0, 0.05)
        self.assertAlmostEqual(u, 500.0, delta=1.0)
        self.assertGreater(v, 378.0)                      # dolna krawedz (w strone nadiru)

    def test_near_nadir_falls_back_to_centroid(self):
        io, eo = nadir_camera(45.0)
        cfg = PointConfig()
        m = rect_mask(480, 480, 520, 520)
        ch = choose_point("mask_ground_contact", (480, 480, 520, 520), io, eo, cfg, m)
        self.assertEqual((ch.used, ch.fallback), ("mask_centroid", "near_nadir"))
        self.assertEqual(ch.uv, (500.0, 500.0))


class FallbackTest(unittest.TestCase):
    def setUp(self):
        self.io, self.eo = oblique_camera()
        self.cfg = PointConfig()
        self.bbox = (100.0, 100.0, 200.0, 180.0)

    def test_bbox_methods(self):
        ch = choose_point("bbox_bottom_center", self.bbox, self.io, self.eo, self.cfg)
        self.assertEqual((ch.used, ch.uv), ("bbox_bottom_center", (150.0, 180.0)))
        ch = choose_point("bbox_center", self.bbox, self.io, self.eo, self.cfg)
        self.assertEqual(ch.uv, (150.0, 140.0))

    def test_no_mask(self):
        ch = choose_point("mask_centroid", self.bbox, self.io, self.eo, self.cfg, None)
        self.assertEqual((ch.used, ch.fallback, ch.uv), ("bbox_center", "no_mask", (150.0, 140.0)))

    def test_rejected_masks(self):
        tiny = rect_mask(140, 130, 145, 135)
        m, q = clean_mask(tiny, self.bbox, self.cfg)
        self.assertEqual(q.reason, "too_small")
        ch = choose_point("mask_footprint_median", self.bbox, self.io, self.eo, self.cfg, m, q)
        self.assertEqual((ch.used, ch.fallback), ("bbox_center", "mask_rejected:too_small"))
        self.assertIsNone(ch.footprint_uvs)
        full = rect_mask(99, 99, 201, 181)               # prostokat = caly bbox
        _, q = clean_mask(full, self.bbox, self.cfg)
        self.assertTrue(q.touches_all_edges)
        self.assertEqual(q.reason, "fills_bbox")
        # poprawna maska tez dotyka krawedzi ciasnego bboxa, ale nie wypelnia go -> ok
        ell = np.zeros((80, 100), bool)
        yy, xx = np.mgrid[0:80, 0:100]
        ell[((xx - 49.5) / 50) ** 2 + ((yy - 39.5) / 40) ** 2 <= 1] = True
        _, q = clean_mask(ObjectMask(ell, 100, 100), self.bbox, self.cfg)
        self.assertTrue(q.ok)
        self.assertTrue(q.touches_all_edges)

    def test_largest_component_and_fragmented(self):
        a = np.zeros((80, 100), bool)
        a[10:70, 10:60] = True                           # duza skladowa
        a[5:10, 90:95] = True                            # szum
        m, q = clean_mask(ObjectMask(a, 100, 100), self.bbox, self.cfg)
        self.assertEqual(q.n_components, 2)
        self.assertEqual(m.area, 60 * 50)
        self.assertTrue(q.ok)
        b = np.zeros((80, 100), bool)
        b[10:40, 10:40] = True
        b[45:75, 60:90] = True                           # dwie rowne czesci
        _, q = clean_mask(ObjectMask(b, 100, 100), self.bbox, self.cfg)
        self.assertEqual(q.reason, "fragmented")

    def test_footprint_sampling(self):
        m = rect_mask(100, 100, 200, 180)
        p1 = footprint_pixels(m, 50, seed=0)
        np.testing.assert_array_equal(p1, footprint_pixels(m, 50, seed=0))
        self.assertEqual(p1.shape, (50, 2))
        ch = choose_point("mask_footprint_median", self.bbox, self.io, self.eo,
                          PointConfig(footprint_samples=50), m)
        self.assertEqual(ch.used, "mask_footprint_median")
        self.assertEqual(len(ch.footprint_uvs), 50)


class PointConfigTest(unittest.TestCase):
    def test_per_class_and_validation(self):
        cfg = PipelineConfig.from_dict({"point": {"method": "bbox_center",
                                                  "per_class": {"people": "bbox_bottom_center"}}})
        self.assertEqual(cfg.point.method_for("people"), "bbox_bottom_center")
        self.assertEqual(cfg.point.method_for("namiot"), "bbox_center")
        self.assertFalse(cfg.point.needs_masks())
        cfg.point.per_class["namiot"] = "nonsense"
        with self.assertRaises(ValueError):
            cfg.point.validate()


if __name__ == "__main__":
    unittest.main()
