import unittest

import numpy as np

from geodetect.config import RaycastConfig
from geodetect.geo import EnuFrame, GeoRef
from geodetect.odm import CameraEO, CameraIO
from geodetect.raycast import pixel_to_ray, raycast

REF = GeoRef(lat=53.823177673972694, lon=21.436584949186468, alt=0.0)


class FlatDem:
    def __init__(self, z):
        self.z = z

    def sample(self, x, y):
        return self.z


def nadir_camera(height):
    io = CameraIO(fx=1000, fy=1000, cx=500, cy=500, dist=np.zeros(5), width=1000, height=1000)
    # kamera patrzy w dol: z_cam = -U, x_cam = E, y_cam = -N (gora obrazu = polnoc)
    R_wc = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64)
    return io, CameraEO(C_enu=np.array([0.0, 0.0, height]), R_wc=R_wc)


class RaycastTest(unittest.TestCase):
    def test_nadir_ray_hits_flat_terrain_below_camera(self):
        z_offset = 100.0
        frame = EnuFrame(REF, z_offset=z_offset)
        io, eo = nadir_camera(45.0)
        o, d = pixel_to_ray((500, 500), io, eo)
        np.testing.assert_allclose(d, [0, 0, -1], atol=1e-12)
        hit = raycast(o, d, FlatDem(110.0), frame, RaycastConfig())
        self.assertIsNotNone(hit)
        self.assertAlmostEqual(hit.h, 110.0, delta=0.01)
        self.assertAlmostEqual(hit.range_m, 45.0 - (110.0 - z_offset), delta=0.01)
        ref_x, ref_y = frame.lonlat_to_puwg92(REF.lon, REF.lat)
        self.assertAlmostEqual(hit.x_2180, ref_x, delta=0.01)
        self.assertAlmostEqual(hit.y_2180, ref_y, delta=0.01)

    def test_oblique_ray_east(self):
        frame = EnuFrame(REF, z_offset=0.0)
        io, eo = nadir_camera(50.0)
        # piksel na prawo od srodka -> promien na wschod, 45 deg
        o, d = pixel_to_ray((1500, 500), io, eo)
        hit = raycast(o, d, FlatDem(0.0), frame, RaycastConfig())
        self.assertAlmostEqual(hit.enu[0], 50.0, delta=0.05)
        self.assertAlmostEqual(hit.enu[1], 0.0, delta=0.05)

    def test_upward_ray_returns_none(self):
        frame = EnuFrame(REF)
        hit = raycast(np.zeros(3), np.array([0, 0, 1.0]), FlatDem(0.0), frame, RaycastConfig())
        self.assertIsNone(hit)


if __name__ == "__main__":
    unittest.main()
