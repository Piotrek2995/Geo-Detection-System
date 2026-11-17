"""Wyrownanie 7p (geodetect/align.py): syntetyczna scena z zadana transformacja."""

import math
import unittest

import numpy as np

from geodetect.align import (AlignConfigView, Similarity, estimate_similarity, polar_rotation,
                             rot_matrix)
from geodetect.geo import EnuFrame, GeoRef


def scene(x, y, building=True):
    """Teren nachylony 2 % + budynek 12x8 m z dachem dwuspadowym (okap 4 m, kalenica 7 m)
    i drugi, mniejszy budynek z dachem plaskim (rozroznia obrot wokol pionu)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    z = 100.0 + (0.02 * (x - 50) + 0.01 * (y - 50) if building else 0.0 * x)
    if building:
        inb = (np.abs(x - 50) <= 6) & (np.abs(y - 50) <= 4)
        roof = 4.0 + 3.0 * (1 - np.abs(y - 50) / 4)
        z = np.where(inb, z + roof, z)
        inb2 = (x >= 60) & (x <= 64) & (y >= 40) & (y <= 46)
        z = np.where(inb2, z + 3.0, z)
    return z


def raster_scene(building=True, res=0.5):
    """Scena jako raster probkowany biliniowo (jak NMPT): gradient niezerowy na krawedziach."""
    from geodetect.surface import Grid, sample_grid
    g = np.arange(0, 110, res) + res / 2
    X, Y = np.meshgrid(g, g[::-1])
    grid = Grid(scene(X, Y, building), 0.0, 110.0, res, res, "EPSG:2180")
    return lambda x, y: sample_grid(grid, x, y, "bilinear")


def sample_points(building=True, step=0.5, seed=0):
    rng = np.random.default_rng(seed)
    g = np.arange(30, 75, step) + 0.13
    X, Y = np.meshgrid(g, g)
    X, Y = X.ravel(), Y.ravel()
    # DSM po wygladzeniu do rozdzielczosci NMPT: te same krawedzie co raster referencyjny
    Z = raster_scene(building)(X, Y) + rng.normal(0, 0.03, X.size)
    # "zmiany": samochod 4x2 m (1.5 m) nieobecny w NMPT
    car = (np.abs(X - 40) <= 2) & (np.abs(Y - 62) <= 1)
    Z[car] += 1.5
    return np.column_stack([X, Y, Z])


class SimilarityTest(unittest.TestCase):
    def test_apply_inverse_and_tilt(self):
        sim = Similarity(np.array([1, -2, 3, 0.02, -0.01, 0.3, math.log(0.9)]), np.array([5., 6., 7.]))
        P = np.random.default_rng(1).normal(size=(10, 3)) * 20
        np.testing.assert_allclose(sim.inverse(sim.apply(P)), P, atol=1e-9)
        self.assertAlmostEqual(sim.s, 0.9)
        rz = Similarity(np.array([0, 0, 0, 0, 0, 0.5, 0]), np.zeros(3))
        self.assertAlmostEqual(rz.tilt_deg, 0.0, places=9)        # obrot wokol pionu nie pochyla
        R = rot_matrix([0.1, 0.2, -0.3])
        np.testing.assert_allclose(polar_rotation(2.5 * R), R, atol=1e-12)


class EstimateTest(unittest.TestCase):
    def _truth(self):
        c0 = np.array([50.0, 50.0, 100.0])
        p = np.array([1.5, -2.0, 140.0, math.radians(3.0), math.radians(-2.0), math.radians(12.0),
                      math.log(1 / 0.85)])
        return Similarity(p, c0)

    def _run(self, building, priors, cfg=None):
        truth = self._truth()
        Pt = sample_points(building)                     # punkty w ukladzie docelowym
        P = truth.inverse(Pt)                            # "ODM": zdeformowany
        cams_t = np.array([[45, 45, 125.0], [55, 52, 128.0], [50, 58, 122.0], [42, 55, 130.0]])
        cams = truth.inverse(cams_t)
        kw = {}
        if priors:
            rng = np.random.default_rng(3)
            kw = dict(cams=cams, cam_gps_xy=cams_t[:, :2] + rng.normal(0, 0.5, (4, 2)),
                      cam_baro_h=cams_t[:, 2] + rng.normal(0, 0.1, 4))
        cfg = cfg or AlignConfigView(search_xy_m=4, search_step_m=1, search_rot_deg=15,
                                     search_rot_step_deg=5, search_scales=(0.9, 1.0, 1.1, 1.2))
        init = Similarity.identity(np.array([P[:, 0].mean(), P[:, 1].mean(), np.median(P[:, 2])]),
                                   tz=138.0)
        res = estimate_similarity(P, raster_scene(building), init, cfg, surface_res_m=0.5, **kw)
        return truth, res, P

    def test_recovers_7_parameters(self):
        truth, res, P = self._run(building=True, priors=True)
        # porownanie w punktach (c0 rozne -> porownujemy dzialanie transformacji)
        err = np.abs(res.sim.apply(P) - truth.apply(P)).max()
        self.assertLess(err, 0.10)
        self.assertAlmostEqual(res.sim.s, truth.s, delta=0.002)
        self.assertAlmostEqual(res.sim.tilt_deg, truth.tilt_deg, delta=0.05)
        self.assertLess(res.residuals_after["nmad"], 0.06)
        self.assertLess(res.n_inliers, res.n_points)        # samochod odrzucony
        self.assertLess(res.residuals_before["nmad"], 10)
        self.assertTrue(all(np.isfinite(v) for v in res.std.values()))
        self.assertEqual(res.weak, [])                      # budynki wyznaczaja wszystko

    def test_vertical_only_surface_without_priors(self):
        _, res, _ = self._run(building=True, priors=False)
        self.assertLess(res.residuals_after["nmad"], 0.06)

    def test_flat_terrain_horizontal_unobservable(self):
        cfg = AlignConfigView(search_xy_m=0, search_rot_deg=0)
        truth, res, P = self._run(building=False, priors=True, cfg=cfg)
        # z poziomej plaszczyzny nie da sie wyznaczyc przesuniecia poziomego ani obrotu wokol
        # pionu -> wiaze je prior GPS; raport uwarunkowania to pokazuje (skala jest tu wyznaczona
        # slabo, tylko przez zmiennosc wysokosci pochylonego modelu)
        for k in ("tx", "ty", "rz"):
            self.assertIn(k, res.weak)
        self.assertGreater(res.cond_surface, 1e8)
        # pochylenie i wysokosc nadal wyznacza powierzchnia
        for k in ("tz", "rx", "ry"):
            self.assertNotIn(k, res.weak)
        self.assertAlmostEqual(res.sim.tilt_deg, truth.tilt_deg, delta=0.1)


class AlignedReconstructionTest(unittest.TestCase):
    def test_camera_transform_consistent_with_points(self):
        from types import SimpleNamespace

        from geodetect.align import AlignedReconstruction, LocalFrame
        from geodetect.odm import CameraEO, CameraIO
        from geodetect.raycast import pixel_to_ray
        fr = EnuFrame(GeoRef(53.8238, 21.4352, 0.0))
        lf = LocalFrame(fr)
        c0 = lf.to_local(np.array([0.0, 0.0, 0.0]))[0]
        sim = Similarity(np.array([2.0, -1.0, 143.0, 0.05, -0.03, 0.33, math.log(1.15)]), c0)
        R0 = rot_matrix([math.pi, 0, 0])                  # kamera patrzy w dol
        io = CameraIO(1000, 1000, 500, 400, np.zeros(5), 1000, 800)
        eo = CameraEO(C_enu=np.array([5.0, -3.0, 20.0]), R_wc=R0)
        base = SimpleNamespace(ref=fr.ref, path="x", cameras={}, shots={}, image_names=["A"],
                               camera=lambda n: (io, eo))
        ar = AlignedReconstruction(base, fr, sim)
        _, eo2 = ar.camera("A")
        # punkt na promieniu przed transformacja -> po transformacji lezy na nowym promieniu
        for uv in ((500, 400), (100, 700), (900, 50)):
            o, d = pixel_to_ray(uv, io, eo)
            X = o + 15.0 * d
            X2 = ar.map_enu(X)[0]
            o2, d2 = pixel_to_ray(uv, io, eo2)
            v = X2 - o2
            cosang = float(v @ d2 / np.linalg.norm(v))
            self.assertGreater(cosang, 1 - 1e-8)
            self.assertAlmostEqual(np.linalg.norm(v), 15.0 * 1.15, delta=0.01)


if __name__ == "__main__":
    unittest.main()
