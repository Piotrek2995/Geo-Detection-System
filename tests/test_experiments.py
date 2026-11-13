"""Eksperymenty: parser YAML (osie / lista / bledy), nadpisania configu, cache detekcji
miedzy wariantami, metryki na danych syntetycznych (P/R/F1, podzbiory, prog, przesuniecia,
flaga pokrycia NMPT)."""

import math
import tempfile
import textwrap
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from geodetect.config import PipelineConfig
from geodetect.detect import Detection
from geodetect.diagnostics import ZScan
from geodetect.experiments import (CachingDetector, DetectionStore, apply_override, cover_flags,
                                   detection_key, detection_shifts, expand_variants, gt_evaluate,
                                   load_experiment, match_object, prf, shift, suggest_threshold,
                                   variant_config)
from geodetect.gt_labels import GtBox, GtImage


def write_exp(tmp: Path, body: str) -> Path:
    (tmp / "config.yaml").write_text("photos_dir: zdjecia\nmodel_path: models/a.pt\n", encoding="utf-8")
    p = tmp / "exp.yaml"
    p.write_text(textwrap.dedent(body), encoding="utf-8")
    return p


class ParserTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_axes_cartesian_product(self):
        v = expand_variants({"surface.source": ["nmt_local", "odm_dsm"],
                             "detect.tiling.enabled": [False, True]}, None)
        self.assertEqual([x.name for x in v], ["nmt_local | off", "nmt_local | on",
                                               "odm_dsm | off", "odm_dsm | on"])
        self.assertEqual(v[3].overrides, {"surface.source": "odm_dsm", "detect.tiling.enabled": True})

    def test_model_path_name_is_stem(self):
        v = expand_variants({"model": ["models/stary.pt", "models/nowy.pt"]}, None)
        self.assertEqual([x.name for x in v], ["stary", "nowy"])

    def test_explicit_list_and_derive(self):
        p = write_exp(self.tmp, """
            name: t
            base_config: config.yaml
            overrides: {aggregate.eps_m: 10}
            variants:
              - name: a
                set: {z_anchor.mode: agl_assumption}
              - name: b
                derive: {z_offset: multiview_check, from: a}
            reference: b
            metrics: {min_views_scan: 4}
        """)
        e = load_experiment(p)
        self.assertEqual([v.name for v in e.variants], ["a", "b"])
        self.assertEqual(e.reference_variant, "b")
        self.assertEqual(e.metrics.min_views_scan, 4)
        self.assertEqual(e.base_config, (self.tmp / "config.yaml").resolve())

    def test_errors(self):
        cases = {
            "unknown_key": "name: t\nbase_config: config.yaml\nfoo: 1\naxes: {x: [1]}\n",
            "no_variants": "name: t\nbase_config: config.yaml\n",
            "bad_reference": "name: t\nbase_config: config.yaml\naxes: {surface.source: [a]}\nreference: zzz\n",
            "bad_metric": "name: t\nbase_config: config.yaml\naxes: {surface.source: [a]}\nmetrics: {nope: 1}\n",
            "bad_derive": ("name: t\nbase_config: config.yaml\nvariants:\n"
                           "  - {name: a, derive: {z_offset: multiview_check, from: nie_ma}}\n"),
            "duplicate": ("name: t\nbase_config: config.yaml\nvariants:\n"
                          "  - {name: a}\n  - {name: a}\n"),
        }
        for label, body in cases.items():
            with self.subTest(label):
                with self.assertRaises(ValueError):
                    load_experiment(write_exp(self.tmp, body))

    def test_apply_override(self):
        cfg = PipelineConfig()
        apply_override(cfg, "detect.tiling.enabled", True, self.tmp)
        apply_override(cfg, "detect.model", "models/x.pt", self.tmp)        # alias
        apply_override(cfg, "point", {"method": "mask_centroid"}, self.tmp)  # slownik w sekcje
        self.assertTrue(cfg.detect.tiling.enabled)
        self.assertEqual(cfg.model_path, (self.tmp / "models/x.pt").resolve())
        self.assertEqual(cfg.point.method, "mask_centroid")
        with self.assertRaises(ValueError):
            apply_override(cfg, "detect.nie_ma", 1, self.tmp)
        with self.assertRaises(ValueError):
            apply_override(cfg, "nie_ma.pole", 1, self.tmp)

    def test_variant_config_does_not_touch_base(self):
        p = write_exp(self.tmp, """
            name: t
            base_config: config.yaml
            overrides: {aggregate.eps_m: 10}
            axes: {surface.source: [nmt_local, odm_dsm]}
        """)
        e = load_experiment(p)
        base = PipelineConfig.from_file(e.base_config)
        c = variant_config(base, e, e.variants[1])
        self.assertEqual((c.surface.source, c.aggregate.eps_m), ("odm_dsm", 10))
        self.assertEqual(base.surface.source, PipelineConfig().surface.source)
        self.assertNotEqual(base.aggregate.eps_m, 10)


class FakeDetector:
    def __init__(self, cfg):
        self.names = {0: "namiot", 1: "people"}
        self.calls = 0

    def detect(self, path):
        self.calls += 1
        return [Detection(0, "namiot", 0.9, (0, 0, 10, 10))]


class CacheTest(unittest.TestCase):
    def test_geometry_variants_share_detections(self):
        made = []

        def factory(cfg):
            made.append(FakeDetector(cfg))
            return made[-1]
        store = DetectionStore(factory)
        a = PipelineConfig()
        b = PipelineConfig()
        b.surface.source, b.z_anchor.mode, b.point.method = "odm_dsm", "fixed", "mask_centroid"
        c = PipelineConfig()
        c.detect.tiling.enabled = True
        self.assertEqual(detection_key(a), detection_key(b))
        self.assertNotEqual(detection_key(a), detection_key(c))
        for cfg in (a, b, a, b):
            for img in ("x.jpg", "y.jpg"):
                CachingDetector(store, cfg).detect(Path("zdjecia") / img)
        self.assertEqual(len(made), 1)          # model ladowany raz
        self.assertEqual(made[0].calls, 2)      # kazde zdjecie raz
        CachingDetector(store, c).detect(Path("x.jpg"))
        self.assertEqual(len(made), 2)          # inna sekcja detect -> nowy detektor
        self.assertEqual(store.calls, 3)
        self.assertEqual(CachingDetector(store, a).names[1], "people")

    def test_model_change_invalidates(self):
        a, b = PipelineConfig(), PipelineConfig()
        b.model_path = Path("models/inny.pt")
        self.assertNotEqual(detection_key(a), detection_key(b))


def det(cls, conf, bbox):
    return Detection(0 if cls == "namiot" else 1, cls, conf, bbox)


class MetricsTest(unittest.TestCase):
    def setUp(self):
        self.gt = {
            "a.jpg": GtImage("a.jpg", [GtBox("namiot", (0, 0, 100, 100)),
                                       GtBox("people", (200, 200, 220, 260))], "train", Path("x"), 0.0),
            "b.jpg": GtImage("b.jpg", [GtBox("people", (50, 50, 70, 110))], "valid", Path("y"), 0.0),
        }
        self.dets = {
            "a.jpg": [det("namiot", 0.9, (2, 2, 100, 98)),          # TP
                      det("namiot", 0.55, (500, 500, 600, 600)),    # FP (krzak)
                      det("people", 0.31, (201, 199, 221, 258))],   # TP z niskim conf
            "b.jpg": [det("people", 0.91, (400, 400, 420, 460))],   # FP (brak etykiety?) ; FN
            "c.jpg": [det("namiot", 0.8, (0, 0, 10, 10))],          # bez GT -> pomijane
        }

    def test_prf(self):
        self.assertEqual(prf(3, 1, 0), (0.75, 1.0, 2 * 0.75 / 1.75))
        p, r, f1 = prf(0, 0, 0)
        self.assertTrue(math.isnan(p) and math.isnan(r) and math.isnan(f1))
        self.assertEqual(prf(0, 2, 1)[2], 0.0)

    def test_gt_evaluate_thresholds_and_subsets(self):
        res = gt_evaluate(self.dets, self.gt, ["namiot", "people"], 0.25, 0.5,
                          ["a.jpg", "b.jpg", "c.jpg"])
        self.assertEqual((res["namiot"]["tp"], res["namiot"]["fp"], res["namiot"]["fn"]), (1, 1, 0))
        self.assertEqual((res["people"]["tp"], res["people"]["fp"], res["people"]["fn"]), (1, 1, 1))
        self.assertEqual(res["people"]["fp_list"][0][0], "b.jpg")
        hi = gt_evaluate(self.dets, self.gt, ["namiot", "people"], 0.5, 0.5, ["a.jpg", "b.jpg"])
        self.assertEqual((hi["people"]["tp"], hi["people"]["fn"]), (0, 2))   # 0.31 odcieta
        honest = gt_evaluate(self.dets, self.gt, ["namiot", "people"], 0.25, 0.5, ["b.jpg"])
        self.assertEqual(honest["namiot"]["tp"] + honest["namiot"]["fp"] + honest["namiot"]["fn"], 0)
        self.assertEqual((honest["people"]["fp"], honest["people"]["fn"]), (1, 1))

    def test_suggest_threshold(self):
        # TP 0.31, 0.9; FP 0.53 -> prog 0.31 lepszy niz 0.6 (recall), FP 0.53 tolerowany
        t, f1 = suggest_threshold([0.31, 0.9], [0.53], 2, 0.25)
        self.assertAlmostEqual(t, 0.25)
        self.assertAlmostEqual(f1, 0.8)
        # FP ponizej wszystkich TP -> prog tuz nad FP
        t, f1 = suggest_threshold([0.8, 0.9], [0.3, 0.5], 2, 0.25)
        self.assertAlmostEqual(t, 0.8)
        self.assertAlmostEqual(f1, 1.0)

    def test_shift_and_match(self):
        o = SimpleNamespace(cls_name="namiot", x_2180=10.0, y_2180=0.0, h=101.0)
        ref = SimpleNamespace(cls_name="namiot", x_2180=7.0, y_2180=4.0, h=100.0)
        s = shift(o, ref)
        self.assertAlmostEqual(s["dist"], 5.0)
        self.assertAlmostEqual(s["dh"], 1.0)
        self.assertTrue(math.isnan(shift(o, None)["dist"]))
        far = SimpleNamespace(cls_name="namiot", x_2180=100.0, y_2180=0.0, h=0.0)
        other = SimpleNamespace(cls_name="people", x_2180=10.0, y_2180=0.0, h=0.0)
        self.assertIs(match_object(o, [far, other, ref], 6.0), ref)
        self.assertIsNone(match_object(o, [far, other], 6.0))

    def test_detection_shifts(self):
        def g(img, cls, bbox, x, y):
            return SimpleNamespace(image_name=img, detection=det(cls, 0.9, bbox),
                                   hit=SimpleNamespace(x_2180=x, y_2180=y))
        ref = SimpleNamespace(detections=[g("a", "people", (0, 0, 1, 1), 0, 0),
                                          g("b", "people", (0, 0, 1, 1), 0, 0)])
        var = SimpleNamespace(detections=[g("a", "people", (0, 0, 1, 1), 3, 4),
                                          g("b", "people", (0, 0, 1, 1), 0, 1),
                                          g("c", "people", (0, 0, 1, 1), 9, 9)])   # brak w ref
        d = detection_shifts(var, ref, ["people"])["people"]
        self.assertEqual(d["n"], 2)
        self.assertAlmostEqual(d["median"], 3.0)
        self.assertAlmostEqual(d["max"], 5.0)

    def test_cover_flags(self):
        nmt = np.full(6, 100.0)
        nmpt = np.array([100.0, 100.5, 112.0, 112.0, 100.0, 100.0])   # drzewo 12 m
        high = np.array([150.0, 140.0, 130.0, 120.0, 110.0, 100.0])   # nad koronami
        f = cover_flags(high, nmt, nmpt, 2.0)
        self.assertTrue(f["over_cover"])
        self.assertFalse(f["through_cover"])
        self.assertAlmostEqual(f["obstacle_m"], 12.0)
        self.assertAlmostEqual(f["min_clearance_m"], 8.0)
        low = np.array([115.0, 113.0, 111.0, 109.0, 105.0, 100.0])    # przez korone
        f = cover_flags(low, nmt, nmpt, 2.0)
        self.assertTrue(f["through_cover"])
        self.assertAlmostEqual(f["max_cover_through_m"], 12.0)
        f = cover_flags(high, nmt, np.full(6, 101.0), 2.0)            # otwarty teren
        self.assertFalse(f["over_cover"])
        self.assertTrue(math.isnan(f["min_clearance_m"]))

    def test_zscan_without_objects(self):
        s = ZScan(np.array([1.0, 2.0]), np.array([np.nan, np.nan]), np.zeros(2, int), 1.0, 0, 0)
        self.assertFalse(s.valid)
        self.assertTrue(math.isnan(s.best_offset) and math.isnan(s.refined_offset))


if __name__ == "__main__":
    unittest.main()
