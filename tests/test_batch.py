import tempfile
import unittest
from pathlib import Path
from unittest import mock

from geodetect import batch as batch_mod
from geodetect.config import PipelineConfig


class RunBatchSkipTest(unittest.TestCase):
    def test_skips_images_missing_in_reconstruction_or_on_disk(self):
        with tempfile.TemporaryDirectory() as d:
            photos = Path(d)
            for name in ("A.JPG", "B.JPG", "EXTRA.JPG"):  # EXTRA nie ma w ODM
                (photos / name).write_bytes(b"")
            cfg = PipelineConfig(photos_dir=photos)
            ctx = mock.Mock()
            ctx.reconstruction.image_names = ["A.JPG", "B.JPG", "GONE.JPG"]  # GONE nie ma na dysku

            def fake_run_single(cfg, name, ctx):
                return mock.Mock(image_name=name, detections=[], misses=[])

            calls = []
            with mock.patch.object(batch_mod, "run_single", side_effect=fake_run_single), \
                    self.assertLogs("geodetect.batch", level="WARNING"):
                res = batch_mod.run_batch(cfg, ctx, progress_cb=lambda i, n, nm: calls.append((i, n, nm)))

        self.assertEqual([r.image_name for r in res.images], ["A.JPG", "B.JPG"])
        self.assertEqual(set(res.skipped), {"EXTRA.JPG", "GONE.JPG"})
        self.assertEqual(calls, [(1, 2, "A.JPG"), (2, 2, "B.JPG")])
        self.assertEqual(res.objects, [])
        self.assertIn("pominiete: 2", batch_mod.summary_text(res))

    def test_explicit_list_skips_unknown(self):
        ctx = mock.Mock()
        ctx.reconstruction.image_names = ["A.JPG"]
        cfg = PipelineConfig(photos_dir=Path("."))
        with mock.patch.object(batch_mod, "run_single"), \
                self.assertLogs("geodetect.batch", level="WARNING"):
            res = batch_mod.run_batch(cfg, ctx, images=["X.JPG"])
        self.assertEqual(res.skipped, {"X.JPG": "brak w rekonstrukcji ODM"})


if __name__ == "__main__":
    unittest.main()
