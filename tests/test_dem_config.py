import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from geodetect.config import PipelineConfig
from geodetect.dem import GugikDem


class GugikAxisOrderTest(unittest.TestCase):
    def test_x_param_is_northing(self):
        """GetHByXY: x = northing, y = easting (sprawdzone na Sniezce: 1603 m)."""
        session = mock.Mock()
        session.get.return_value = mock.Mock(ok=True, text="139.6")
        dem = GugikDem("https://example/nmt/", session=session)
        self.assertEqual(dem.sample(660330.0, 664811.0), 139.6)
        params = session.get.call_args.kwargs["params"]
        self.assertEqual(params["x"], 664811.0)
        self.assertEqual(params["y"], 660330.0)
        dem.sample(660330.2, 664811.3)  # ten sam metr -> cache
        self.assertEqual(session.get.call_count, 1)

    def test_zero_means_no_data(self):
        session = mock.Mock()
        session.get.return_value = mock.Mock(ok=True, text="0")
        self.assertIsNone(GugikDem("u", session=session).sample(1.0, 2.0))


class ConfigTest(unittest.TestCase):
    def test_defaults(self):
        cfg = PipelineConfig()
        self.assertEqual(cfg.raycast.step_m, 5.0)
        self.assertEqual(cfg.dem.source, "gugik")

    def test_from_json_resolves_paths_and_nested(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "c.json"
            p.write_text(json.dumps({"images": "A.JPG", "odm_dir": "odm",
                                     "raycast": {"step_m": 2.0},
                                     "dem": {"source": "raster", "raster_path": "nmt.tif"}}))
            cfg = PipelineConfig.from_file(p)
        self.assertEqual(cfg.images, ["A.JPG"])
        self.assertEqual(cfg.odm_dir, Path(d).resolve() / "odm")
        self.assertEqual(cfg.dem.raster_path, Path(d).resolve() / "nmt.tif")
        self.assertEqual(cfg.raycast.step_m, 2.0)
        self.assertEqual(cfg.raycast.bisect_iters, 25)

    def test_example_yaml_loads(self):
        cfg = PipelineConfig.from_file(Path(__file__).parents[1] / "config.yaml")
        self.assertEqual(cfg.images, ["SING0304.JPG", "SING0316.JPG"])

    def test_unknown_key_rejected(self):
        with self.assertRaises(ValueError):
            PipelineConfig.from_dict({"raycast": {"stepp": 1}})


if __name__ == "__main__":
    unittest.main()
