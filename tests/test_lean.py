"""Przesuniecie dachu zalezne od wysokosci (geodetect/lean.py, validation.gt_transfer)."""

import csv
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from geodetect.lean import LeanModel, edge_band, edge_height, lean_model
from geodetect.validation import gt_transfer

V = (0.10, -0.22)          # przesuniecie na ortofotomapie [m] na 1 m wysokosci


def gable_height(x, y):
    """Dach dwuspadowy nad prostokatem x 0..12, y 0..8: okap 3 m, kalenica (x = 6) 7 m."""
    if 0 <= x <= 12 and 0 <= y <= 8:
        return 3.0 + 4.0 * (1 - abs(x - 6) / 6)
    return 0.0


class LeanModelTest(unittest.TestCase):
    def test_correct_inverts_lean(self):
        lm = LeanModel(*V, 5.0, 0.5, -1.1)
        for xt, yt in ((6.0, 4.0), (1.0, 4.0), (11.0, 2.0), (-5.0, -5.0)):
            h = gable_height(xt, yt)
            xo, yo = xt + V[0] * h, yt + V[1] * h          # punkt widziany na ortofotomapie
            x, y, hh = lm.correct(xo, yo, gable_height)
            self.assertAlmostEqual(x, xt, delta=1e-3)
            self.assertAlmostEqual(y, yt, delta=1e-3)
            self.assertAlmostEqual(hh, h, delta=1e-3)

    def test_edge_height_and_model(self):
        res = 0.05
        n_x, n_y = int(16 / res), int(12 / res)          # siatka x -2..14, y -2..10
        x0, y1 = -2.0, 10.0
        fp = np.zeros((n_y, n_x), bool)
        rows, cols = np.mgrid[0:n_y, 0:n_x]
        xs, ys = x0 + (cols + 0.5) * res, y1 - (rows + 0.5) * res
        fp[(xs >= 0) & (xs <= 12) & (ys >= 0) & (ys <= 8)] = True
        band = edge_band(fp, res)
        self.assertTrue(band.any() and (band <= fp).all())
        h, n = edge_height(fp, x0, y1, res, np.vectorize(gable_height))
        # pas 0.5-1.5 m od krawedzi: okapy (~3.5 m) i szczyty (pelny profil) -> 3..7 m
        self.assertGreater(n, 1000)
        self.assertTrue(3.0 < h < 7.0, h)
        lm = lean_model(0.6, -1.25, h)
        self.assertAlmostEqual(lm.vx * h, 0.6)
        with self.assertLogs("geodetect.lean", "WARNING"):
            self.assertIsNone(lean_model(0.6, -1.25, 0.1))     # brak wysokosci -> brak modelu


class GtTransferTest(unittest.TestCase):
    def test_height_dependent_transfer(self):
        # ODM -> prawda: obrot 19 deg, skala 0.85, przesuniecie; GUGiK = prawda + V * H
        ang, s, t = math.radians(19), 0.85, np.array([100.0, 200.0])
        R = np.array([[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]])
        to_true = lambda p: s * R @ np.asarray(p) + t                     # noqa: E731
        from_true = lambda q: np.linalg.solve(s * R, np.asarray(q) - t)   # noqa: E731

        def H(x, y):
            return gable_height(x - 100.0, y - 200.0)
        true_pts = [(101.0, 201.0), (111.0, 201.5), (106.0, 207.0), (102.0, 206.5)]   # okapy i kalenica
        ball_true = (104.0, 203.0)                                                       # na polaci, H ~5.7
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "pairs.csv"
            with open(p, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["id", "kind", "odm_x", "odm_y", "gugik_x", "gugik_y"])
                w.writeheader()
                for i, q in enumerate(true_pts):
                    o = from_true(q)
                    h = H(*q)
                    w.writerow({"id": i, "kind": "punkt_dachu", "odm_x": o[0], "odm_y": o[1],
                                "gugik_x": q[0] + V[0] * h, "gugik_y": q[1] + V[1] * h})
                o = from_true(ball_true)
                w.writerow({"id": "b", "kind": "pilka", "odm_x": o[0], "odm_y": o[1], "gugik_x": "", "gugik_y": ""})
            lean = {"dx_m": V[0] * 5.0, "dy_m": V[1] * 5.0, "vx_per_m": V[0], "vy_per_m": V[1], "h_edges_m": 5.0}
            new = gt_transfer(p, lean, H)
            old = gt_transfer(p, {"dx_m": lean["dx_m"], "dy_m": lean["dy_m"]})
        self.assertEqual(new["lean_mode"], "height")
        self.assertLess(math.hypot(new["x"] - ball_true[0], new["y"] - ball_true[1]), 0.01)
        self.assertLess(new["helmert_rms"], 1e-3)       # zbieznosc iteracji 1 mm
        self.assertAlmostEqual(new["ball_height"], H(*ball_true), places=3)
        self.assertEqual(old["lean_mode"], "uniform")
        # staly wektor (dla H = 5 m) przy punktach na 3-7 m: blad pozycji pilki i reszty Helmerta
        self.assertGreater(old["helmert_rms"], 0.05)


if __name__ == "__main__":
    unittest.main()
