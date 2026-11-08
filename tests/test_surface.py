"""Powierzchnie ray castingu: pierwsze przeciecie, mozaika arkuszy, nodata, uklad odm_local."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from pyproj import Transformer

from geodetect.config import PipelineConfig, RaycastConfig
from geodetect.dem import GugikDem
from geodetect.geo import EnuFrame, GeoRef
from geodetect.raycast import pixel_to_ray, raycast
from geodetect.surface import (VERTICAL_NPM, VERTICAL_ODM, Grid, RasterSurface, build_surface,
                               load_mosaic, sample_grid)
from tests.test_raycast import REF, nadir_camera

FRAME0 = EnuFrame(REF, z_offset=0.0)
REF_X, REF_Y = (float(v) for v in FRAME0.lonlat_to_puwg92(REF.lon, REF.lat))


def grid_around(z_fn, half=80, res=1.0, crs="EPSG:2180", cx=REF_X, cy=REF_Y) -> Grid:
    """Siatka (2*half)^2 m wokol (cx, cy); z_fn(dx, dy) na srodkach pikseli."""
    x0, y0 = cx - half, cy + half          # krawedzie pikseli w cx + k*res
    n = int(2 * half / res)
    xc = x0 + (np.arange(n) + 0.5) * res
    yc = y0 - (np.arange(n) + 0.5) * res
    X, Y = np.meshgrid(xc, yc)
    return Grid(z_fn(X - cx, Y - cy).astype(float), x0, y0, res, res, crs)


def east_ray(height=50.0):
    """Promien z kamery na wysokosci `height` nad originem, 45 deg w dol na wschod."""
    io, eo = nadir_camera(height)
    return pixel_to_ray((1500, 500), io, eo)


class FirstIntersectionTest(unittest.TestCase):
    def wall(self, dx, dy):
        z = np.full(dx.shape, 10.0)
        z[(dx >= 20.0) & (dx < 21.0)] = 35.0        # sciana 1 piksel grubosci, 25 m nad gruntem
        return z

    def test_hits_thin_wall_not_ground_behind(self):
        surf = RasterSurface(grid_around(self.wall), "nmpt_local", VERTICAL_NPM, "nearest")
        o, d = east_ray()
        hit = raycast(o, d, surf, FRAME0, RaycastConfig())
        self.assertIsNotNone(hit)
        # promien na e=20 ma h=30 < 35 -> trafienie w lico sciany, nie w grunt na e=40
        self.assertAlmostEqual(hit.enu[0], 20.0, delta=0.15)
        self.assertAlmostEqual(hit.h, 30.0, delta=0.2)
        self.assertEqual(hit.surface, "nmpt_local")
        self.assertAlmostEqual(hit.surface_h, 35.0)

    def test_step_follows_raster_resolution(self):
        cfg = RaycastConfig(step_m=5.0)
        self.assertEqual(cfg.step_for(1.0), 1.0)
        self.assertEqual(cfg.step_for(None), 5.0)
        # krok 5 m bez ograniczenia rozdzielczoscia przeskakuje sciane 1 m -> grunt za nia
        coarse = RaycastConfig(step_m=5.0, raster_step_factor=10.0)
        surf = RasterSurface(grid_around(self.wall), "nmpt_local", VERTICAL_NPM, "nearest")
        hit = raycast(*east_ray(), surf, FRAME0, coarse)
        self.assertGreater(hit.enu[0], 30.0)

    def test_flat_ground_without_wall(self):
        surf = RasterSurface(grid_around(lambda dx, dy: np.full(dx.shape, 10.0)),
                             "nmt_local", VERTICAL_NPM)
        hit = raycast(*east_ray(), surf, FRAME0, RaycastConfig())
        self.assertAlmostEqual(hit.enu[0], 40.0, delta=0.05)
        self.assertAlmostEqual(hit.h, 10.0, delta=0.05)


class NodataTest(unittest.TestCase):
    def test_sample_grid_nan_and_bilinear_fallback(self):
        z = np.array([[1.0, 2.0], [3.0, np.nan]])
        g = Grid(z, 0.0, 2.0, 1.0, 1.0, "EPSG:2180")
        # srodek siatki: sasiad nodata -> najblizszy piksel zamiast NaN
        self.assertEqual(sample_grid(g, [0.9], [1.1])[0], 1.0)
        self.assertTrue(np.isnan(sample_grid(g, [1.5], [0.5])[0]))     # sam piksel nodata
        self.assertTrue(np.isnan(sample_grid(g, [5.0], [5.0])[0]))     # poza rastrem
        full = Grid(np.array([[1.0, 2.0], [3.0, 4.0]]), 0.0, 2.0, 1.0, 1.0, "EPSG:2180")
        np.testing.assert_allclose(sample_grid(full, [1.0], [1.0])[0], 2.5)  # srednia 4 pikseli

    def test_ray_skips_nodata_before_ground(self):
        def fn(dx, dy):
            z = np.full(dx.shape, 10.0)
            z[(dx > 5.0) & (dx < 30.0)] = np.nan     # dziura w danych pod promieniem
            return z
        surf = RasterSurface(grid_around(fn), "nmt_local", VERTICAL_NPM)
        hit = raycast(*east_ray(), surf, FRAME0, RaycastConfig())
        self.assertAlmostEqual(hit.enu[0], 40.0, delta=0.05)

    def test_all_nodata_gives_no_hit(self):
        surf = RasterSurface(grid_around(lambda dx, dy: np.full(dx.shape, np.nan)),
                             "nmt_local", VERTICAL_NPM)
        self.assertIsNone(raycast(*east_ray(), surf, FRAME0, RaycastConfig(t_max_m=200)))


def write_asc(path: Path, z: np.ndarray, xll: float, yll: float, cell: float, nodata=-9999.0):
    z = np.where(np.isnan(z), nodata, z)
    header = (f"ncols {z.shape[1]}\nnrows {z.shape[0]}\nxllcorner {xll}\nyllcorner {yll}\n"
              f"cellsize {cell}\nNODATA_value {nodata}\n")
    path.write_text(header + "\n".join(" ".join(f"{v:g}" for v in row) for row in z) + "\n")


class MosaicTest(unittest.TestCase):
    def test_two_asc_sheets_side_by_side(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            west = np.full((4, 3), 100.0)
            west[0, 2] = np.nan
            write_asc(d / "a_west.asc", west, 1000.0, 2000.0, 1.0)
            write_asc(d / "b_east.asc", np.full((4, 3), 200.0), 1003.0, 2000.0, 1.0)
            g = load_mosaic(sorted(d.glob("*.asc")))
            self.assertEqual(g.z.shape, (4, 6))
            self.assertEqual(g.bounds, (1000.0, 2000.0, 1006.0, 2004.0))
            self.assertEqual(sample_grid(g, [1000.5], [2000.5], "nearest")[0], 100.0)
            self.assertEqual(sample_grid(g, [1005.5], [2003.5], "nearest")[0], 200.0)
            self.assertTrue(np.isnan(sample_grid(g, [1002.5], [2003.5], "nearest")[0]))
            # przyciecie do obszaru
            gc = load_mosaic(sorted(d.glob("*.asc")), bounds=(1002.2, 2000.0, 1004.1, 2004.0))
            self.assertEqual(gc.bounds, (1002.0, 2000.0, 1005.0, 2004.0))

    def test_xyz_and_asc_mix_and_overlap(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            write_asc(d / "a.asc", np.full((2, 2), 50.0), 0.0, 0.0, 1.0)
            # XYZ: srodki pikseli, nachodzi 1 kolumna na a.asc (pierwszy arkusz wygrywa)
            rows = [f"{x + 0.5} {y + 0.5} 70" for x in (1, 2, 3) for y in (0, 1)]
            (d / "b.xyz").write_text("\n".join(rows) + "\n")
            g = load_mosaic([d / "a.asc", d / "b.xyz"])
            self.assertEqual(g.bounds, (0.0, 0.0, 4.0, 2.0))
            np.testing.assert_array_equal(g.z, [[50, 50, 70, 70], [50, 50, 70, 70]])

    def test_resolution_mismatch_raises(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            write_asc(d / "a.asc", np.ones((2, 2)), 0.0, 0.0, 1.0)
            write_asc(d / "b.asc", np.ones((2, 2)), 2.0, 0.0, 0.5)
            with self.assertRaises(ValueError):
                load_mosaic([d / "a.asc", d / "b.asc"])

    def test_build_surface_nmt_local_from_folder(self):
        with tempfile.TemporaryDirectory() as d:
            nmt = Path(d) / "nmt"
            nmt.mkdir()
            x0, y0 = np.floor(REF_X) - 50, np.floor(REF_Y) - 50
            write_asc(nmt / "w.asc", np.full((100, 50), 12.0), x0, y0, 1.0)
            write_asc(nmt / "e.asc", np.full((100, 50), 12.0), x0 + 50, y0, 1.0)
            cfg = PipelineConfig()
            cfg.surface.source, cfg.surface.nmt_dir = "nmt_local", nmt
            surf = build_surface(cfg, (REF_X - 10, REF_Y - 10, REF_X + 10, REF_Y + 10))
            self.assertEqual((surf.name, surf.vertical, surf.resolution_m), ("nmt_local", "npm", 1.0))
            io, eo = nadir_camera(40.0)
            hit = raycast(*pixel_to_ray((500, 500), io, eo), surf, EnuFrame(REF, 0.0),
                          RaycastConfig())
            self.assertAlmostEqual(hit.h, 12.0, delta=0.01)


class OdmLocalSurfaceTest(unittest.TestCase):
    def test_odm_local_compared_without_z_offset(self):
        """DSM ODM ma wysokosci ODM: porownanie bez z_offset, ale hit.h nadal w npm."""
        to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32634", always_xy=True)
        ux, uy = to_utm.transform(REF.lon, REF.lat)
        grid = grid_around(lambda dx, dy: np.full(dx.shape, 2.0), crs="EPSG:32634", cx=ux, cy=uy)
        surf = RasterSurface(grid, "odm_dsm", VERTICAL_ODM)
        frame = EnuFrame(REF, z_offset=126.0)
        io, eo = nadir_camera(45.0)
        hit = raycast(*pixel_to_ray((500, 500), io, eo), surf, frame, RaycastConfig())
        self.assertIsNotNone(hit)
        self.assertAlmostEqual(hit.range_m, 43.0, delta=0.02)      # 45 - 2, nie 45 - 128
        self.assertAlmostEqual(hit.h, 128.0, delta=0.02)          # npm = 2 + z_offset
        self.assertAlmostEqual(hit.surface_h, 2.0, delta=1e-6)    # wartosc DSM w ukl. ODM
        self.assertAlmostEqual(hit.x_2180, REF_X, delta=0.02)
        hit.ground_h = 125.5
        self.assertAlmostEqual(hit.above_ground_m, 2.5, delta=0.02)


class GugikPointListTest(unittest.TestCase):
    def test_sample_many_uses_point_list_northing_first(self):
        session = mock.Mock()
        session.get.return_value = mock.Mock(
            ok=True, status_code=200, text="664811 660330 139.6,664812 660331 0")
        dem = GugikDem("u", session=session)
        z = dem.sample_many(np.array([660330.2, 660330.8]), np.array([664811.1, 664811.9]))
        self.assertEqual(z[0], 139.6)
        self.assertTrue(np.isnan(z[1]))                           # "0" = brak danych
        params = session.get.call_args.kwargs["params"]
        self.assertEqual(params["request"], "GetHByPointList")
        self.assertEqual(params["list"], "664811 660330,664812 660331")
        dem.sample_many(np.array([660330.0]), np.array([664811.0]))  # z cache
        self.assertEqual(session.get.call_count, 1)


if __name__ == "__main__":
    unittest.main()
