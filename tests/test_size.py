"""Rozmiar fizyczny z bboxa (geodetect/size.py) na syntetycznej kamerze."""

import math
import unittest

import numpy as np

from geodetect.aggregate import AggregatedObject, apply_size
from geodetect.config import SizeConfig
from geodetect.odm import CameraEO, CameraIO
from geodetect.size import FLAG_SIZE, detection_size, size_flags


def camera(height: float, tilt_deg: float = 0.0):
    """Kamera nad (0, 0) na wysokosci height, pochylona o tilt_deg od nadiru w strone polnocy."""
    io = CameraIO(fx=1000, fy=1000, cx=1000, cy=750, dist=np.zeros(5), width=2000, height=1500)
    nadir = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64)   # world -> camera
    t = -math.radians(tilt_deg)
    # obrot kamery wokol jej osi x: os optyczna odchyla sie od pionu w strone N
    Rx = np.array([[1, 0, 0], [0, math.cos(t), -math.sin(t)], [0, math.sin(t), math.cos(t)]])
    return io, CameraEO(C_enu=np.array([0.0, 0.0, height]), R_wc=Rx @ nadir)


def project(io, eo, P):
    pc = eo.R_wc @ (np.asarray(P, float) - eo.C_enu)
    return io.fx * pc[0] / pc[2] + io.cx, io.fy * pc[1] / pc[2] + io.cy


def bbox_of_square(io, eo, center, side):
    cx, cy = center
    pts = [project(io, eo, (cx + dx, cy + dy, 0.0)) for dx in (-side / 2, side / 2) for dy in (-side / 2, side / 2)]
    us, vs = zip(*pts)
    return min(us), min(vs), max(us), max(vs)


def size_of(io, eo, center, side):
    hit = np.array([center[0], center[1], 0.0])
    rng = float(np.linalg.norm(hit - eo.C_enu))
    d = (hit - eo.C_enu) / rng
    inc = math.degrees(math.acos(-d[2]))
    return detection_size(bbox_of_square(io, eo, center, side), io, eo, hit, rng, inc), inc


class SizeTest(unittest.TestCase):
    def test_nadir(self):
        io, eo = camera(50.0)
        s, inc = size_of(io, eo, (0.0, 0.0), 4.0)
        self.assertEqual(s.method, "plane")
        self.assertAlmostEqual(s.size_m, 4.0, delta=0.02)
        self.assertAlmostEqual(s.simple_m, 4.0, delta=0.02)

    def test_oblique_incidence_corrected(self):
        # kamera pochylona 50 deg, obiekt 3 m na osi optycznej: w kierunku N skrocony o cos(50)
        io, eo = camera(40.0, 50.0)
        y = 40.0 * math.tan(math.radians(50))
        s, inc = size_of(io, eo, (0.0, y), 3.0)
        self.assertAlmostEqual(inc, 50.0, delta=0.5)
        self.assertAlmostEqual(s.width_m, 3.0, delta=0.1)       # w poprzek promienia (E-W)
        self.assertAlmostEqual(s.length_m, 3.0, delta=0.15)     # wzdluz (N-S) po korekcie kata
        self.assertAlmostEqual(s.size_m, 3.0, delta=0.15)

    def test_oblique_elongated_along_ray(self):
        # prostokat 1 m (E-W) x 3 m (N-S, wzdluz promienia), kat padania 50 deg:
        # bbox_px * R / f daje ~3 * cos(50) = 1.9 m, rzut na plaszczyzne ~3 m
        io, eo = camera(40.0, 50.0)
        y = 40.0 * math.tan(math.radians(50))
        pts = [project(io, eo, (dx, y + dy, 0.0)) for dx in (-0.5, 0.5) for dy in (-1.5, 1.5)]
        us, vs = zip(*pts)
        hit = np.array([0.0, y, 0.0])
        rng = float(np.linalg.norm(hit - eo.C_enu))
        s = detection_size((min(us), min(vs), max(us), max(vs)), io, eo, hit, rng, 50.0)
        self.assertAlmostEqual(s.size_m, 3.0, delta=0.15)
        self.assertAlmostEqual(s.simple_m, 3.0 * math.cos(math.radians(50)), delta=0.25)

    def test_grazing_falls_back_to_simple(self):
        io, eo = camera(10.0, 80.0)
        y = 10.0 * math.tan(math.radians(80))
        s, _ = size_of(io, eo, (0.0, y), 2.0)
        self.assertEqual(s.method, "simple")

    def test_flags_and_object_median(self):
        ranges = {"namiot": [1.5, 6.0]}
        self.assertEqual(size_flags("namiot", 25.0, ranges), [FLAG_SIZE])
        self.assertEqual(size_flags("namiot", 4.0, ranges), [])
        self.assertEqual(size_flags("people", 25.0, ranges), [])          # klasa bez zakresu
        self.assertEqual(size_flags("namiot", float("nan"), ranges), [])

        class D:   # minimalna detekcja
            def __init__(self, s, border=False):
                from geodetect.size import SizeEstimate
                self.size = SizeEstimate(s, s, s, s, "plane")
                self.touches_border = border
        o = AggregatedObject(0, "namiot", 0, 0, 0, 0, 0, 3, [], 0, 0, 0.9, 0, 0)
        apply_size(o, [D(24.0), D(26.0), D(2.0, border=True)], SizeConfig(ranges_m=ranges))
        self.assertAlmostEqual(o.size_m, 25.0)       # ucieta krawedzia pominieta
        self.assertEqual(o.size_n, 2)
        self.assertEqual(o.flags, [FLAG_SIZE])


if __name__ == "__main__":
    unittest.main()
