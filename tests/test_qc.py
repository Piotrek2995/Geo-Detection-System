"""Kontrola jakosci rekonstrukcji (geodetect/qc.py) na danych syntetycznych."""

import unittest

import numpy as np

from geodetect.config import QcConfig
from geodetect.qc import (BAD, INFO, NA, OK, REC_7P, REC_BASE, REC_CALIB, WARN, check_baseline, check_focal,
                          check_surface, check_surface_modes, check_vertical_scale, expected_focal_exif,
                          fit_plane, summarize, surface_stats)


class QcTest(unittest.TestCase):
    cfg = QcConfig()

    def test_vertical_scale(self):
        rng = np.random.default_rng(0)
        alt = np.linspace(10, 25, 19)
        ok = check_vertical_scale(1.0 * alt + 3 + rng.normal(0, 0.2, 19), alt, self.cfg)
        self.assertEqual(ok.status, OK)
        self.assertAlmostEqual(ok.value, 1.0, delta=0.05)
        bad = check_vertical_scale(0.68 * alt + 4.7 + rng.normal(0, 0.2, 19), alt, self.cfg)  # lot "pilka"
        self.assertEqual(bad.status, BAD)
        warn = check_vertical_scale(1.15 * alt + rng.normal(0, 0.05, 19), alt, self.cfg)
        self.assertEqual(warn.status, WARN)
        # krotki zakres wysokosci -> nieokreslona
        self.assertEqual(check_vertical_scale(alt[:5] * 0 + 40 + rng.normal(0, 1, 5), 40 + rng.normal(0, 0.5, 5),
                                              self.cfg).status, NA)

    def test_vertical_scale_uncertain_slope_not_flagged(self):
        # duzy szum przy zakresie 6 m: odchylenie k mniejsze niz 2 sigma nachylenia -> OK
        rng = np.random.default_rng(1)
        alt = np.linspace(43, 49, 26)
        c = check_vertical_scale(1.14 * alt - 6.7 + rng.normal(0, 1.0, 26), alt, self.cfg)
        self.assertEqual(c.status, OK, c.text)

    def test_focal(self):
        self.assertAlmostEqual(expected_focal_exif(13, 2560, 1440), 0.3447, places=4)
        # vs EXIF: najwyzej ostrzezenie (EXIF nie uwzglednia kadrowania)
        self.assertEqual(check_focal(0.528, self.cfg, 13, 2560, 1440).status, INFO)
        self.assertEqual(check_focal(0.36, self.cfg, 13, 2560, 1440).status, OK)
        ref = QcConfig(camera_focal=0.528)
        self.assertEqual(check_focal(0.374, ref).status, BAD)
        self.assertEqual(check_focal(0.52, ref).status, OK)
        self.assertEqual(check_focal(0.5, QcConfig()).status, NA)

    def test_surface(self):
        rng = np.random.default_rng(2)
        good = check_surface(5.0 + rng.normal(0, 0.2, 500), self.cfg)       # stala roznica nie szkodzi
        self.assertEqual(good.status, OK)
        x = np.linspace(-30, 30, 500)
        tilted = check_surface(np.tan(np.radians(5)) * x + rng.normal(0, 0.2, 500), self.cfg)
        self.assertEqual(tilted.status, BAD)                               # pochylenie 5 deg na 60 m
        self.assertEqual(check_surface([1.0] * 5, self.cfg).status, NA)

    def test_plane_fit_robust(self):
        rng = np.random.default_rng(3)
        xy = rng.uniform(-30, 30, (2000, 2))
        r = 0.05 * xy[:, 0] - 0.02 * xy[:, 1] + 1.0 + rng.normal(0, 0.1, 2000)
        r[:500] += rng.uniform(3, 15, 500)              # 25 % odstajacych (drzewa, zmiany)
        coef, use = fit_plane(xy, r)
        c0 = 1.0 + 0.05 * xy[:, 0].mean() - 0.02 * xy[:, 1].mean()   # wyraz wolny w centroidzie
        np.testing.assert_allclose(coef, [0.05, -0.02, c0], atol=0.01)
        self.assertLess(use[:500].mean(), 0.05)

    def test_surface_modes_before_after(self):
        rng = np.random.default_rng(4)
        xy = rng.uniform(-30, 30, (1500, 2))
        noise = rng.normal(0, 0.3, 1500)
        tilted = np.tan(np.radians(3.5)) * xy[:, 1] + noise          # gps_only: przechyl 3.5 deg
        aligned = noise                                               # po 7p
        b, a = surface_stats(xy, tilted), surface_stats(xy, aligned)
        self.assertAlmostEqual(b["tilt_deg"], 3.5, delta=0.1)
        self.assertLess(b["nmad_plane"], 0.4)                         # ksztalt zgodny, tylko przechyl
        self.assertGreater(b["nmad_raw"], 1.0)
        surf, tilt = check_surface_modes(b, b, self.cfg, mode="gps_only + dem_alignment")
        self.assertEqual((surf.status, tilt.status), (OK, WARN))    # dobry lot bez 7p: ostrzezenie
        surf, tilt = check_surface_modes(b, a, self.cfg, mode="similarity_nmpt")
        self.assertEqual((surf.status, tilt.status), (OK, OK))
        self.assertRegex(tilt.text, r"3\.\d\d° → 0\.\d\d°")
        bad = surface_stats(xy, np.tan(np.radians(9)) * xy[:, 0] + noise)
        self.assertEqual(check_surface_modes(bad, bad, self.cfg)[1].status, BAD)
        self.assertEqual(check_surface_modes(None, {"n": 3}, self.cfg)[0].status, NA)

    def test_no_7p_recommendation_when_aligned(self):
        rng = np.random.default_rng(5)
        xy = rng.uniform(-30, 30, (500, 2))
        st = surface_stats(xy, np.tan(np.radians(4)) * xy[:, 0] + rng.normal(0, 0.2, 500))
        checks = check_surface_modes(st, st, self.cfg)
        self.assertTrue(summarize(checks).recommendations[0].startswith(REC_7P))
        self.assertEqual(summarize(checks, aligned=True).recommendations, [])

    def test_baseline(self):
        g = np.array([[0, 0], [80, 0], [80, 60], [0, 60]], float)
        self.assertEqual(check_baseline(g, 1.0, self.cfg).status, OK)
        self.assertEqual(check_baseline(g / 8, 3.0, self.cfg).status, BAD)  # ~10 m: lot "pilka"

    def test_summary_recommendations(self):
        alt = np.linspace(10, 25, 19)
        geom = check_vertical_scale(0.68 * alt + 4.7, alt, self.cfg)
        surf = check_surface(np.linspace(-6, 6, 200), self.cfg)
        base = check_baseline(np.array([[0, 0], [9, 0]], float), 3.0, self.cfg)
        rep = summarize([geom, surf, base])
        self.assertEqual(rep.status, BAD)
        self.assertEqual(rep.recommendations[0], REC_CALIB)
        self.assertTrue(rep.recommendations[1].startswith(REC_7P))
        self.assertIn(REC_BASE, rep.recommendations)
        ok = summarize([check_vertical_scale(alt + 1, alt, self.cfg)])
        self.assertEqual((ok.status, ok.recommendations), (OK, []))
        self.assertIn("Kontrola jakości", rep.lines()[0])


if __name__ == "__main__":
    unittest.main()
