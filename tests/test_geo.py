import math
import unittest

import numpy as np
from pyproj import Proj

from geodetect.geo import EnuFrame, GeoRef

# reference_lla.json z lotu testowego
REF = GeoRef(lat=53.823177673972694, lon=21.436584949186468, alt=0.0)


class EnuToPuwg92Test(unittest.TestCase):
    def setUp(self):
        self.frame = EnuFrame(REF)
        self.ref_x, self.ref_y = self.frame.lonlat_to_puwg92(REF.lon, REF.lat)

    def test_origin_maps_to_reference_lla(self):
        lon, lat, h = self.frame.enu_to_geodetic(0.0, 0.0, 0.0)
        self.assertAlmostEqual(lon, REF.lon, places=10)
        self.assertAlmostEqual(lat, REF.lat, places=10)
        self.assertAlmostEqual(h, REF.alt, places=6)
        x, y, H = self.frame.enu_to_puwg92(0.0, 0.0, 0.0)
        self.assertAlmostEqual(x, self.ref_x, places=4)
        self.assertAlmostEqual(y, self.ref_y, places=4)

    def test_300m_east_differs_from_naive_by_convergence(self):
        x, y, _ = self.frame.enu_to_puwg92(300.0, 0.0, 0.0)
        dy = y - self.ref_y              # naiwnie: 0
        dx = x - (self.ref_x + 300.0)    # naiwnie: 0
        # zbieznosc PUWG 1992 w tym miejscu ~1.97 deg -> ~10 m w osi Y
        gamma = Proj("EPSG:2180").get_factors(REF.lon, REF.lat).meridian_convergence
        self.assertGreater(gamma, 1.9)
        self.assertLess(gamma, 2.05)
        self.assertGreater(abs(dy), 5.0)
        self.assertLess(abs(dy), 15.0)
        self.assertAlmostEqual(dy, 300.0 * math.sin(math.radians(gamma)), delta=0.1)
        # w osi X tylko skala odwzorowania i cos(gamma): rzad decymetrow
        self.assertLess(abs(dx), 0.5)

    def test_roundtrip_geodetic_enu(self):
        e, n, u = 120.0, -80.0, 45.0
        lon, lat, h = self.frame.enu_to_geodetic(e, n, u)
        e2, n2, u2 = self.frame.geodetic_to_enu(lon, lat, h)
        np.testing.assert_allclose([e2, n2, u2], [e, n, u], atol=1e-6)

    def test_z_offset_added_to_height(self):
        f = EnuFrame(REF, z_offset=136.8)
        _, _, H = f.enu_to_puwg92(0.0, 0.0, 45.0)
        self.assertAlmostEqual(H, 45.0 + 136.8, places=6)

    def test_vectorized(self):
        x, y, H = self.frame.enu_to_puwg92(np.array([0.0, 300.0]), np.zeros(2), np.zeros(2))
        self.assertEqual(np.shape(x), (2,))


if __name__ == "__main__":
    unittest.main()
