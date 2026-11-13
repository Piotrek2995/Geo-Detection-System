"""Detekcja: klasy po nazwie, walidacja, progi per klasa, kafle i scalanie, list_models,
maski z modelu segmentacyjnego, status obiektow."""

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from geodetect.aggregate import aggregate
from geodetect.config import AggregateConfig, DetectConfig, PipelineConfig
from geodetect.detect import Detection, Detector, list_models, model_info
from geodetect.pipeline import choose_points, touches_border
from geodetect.segment import mask_from_polygons
from geodetect.tiling import Tile, merge_detections, tile_grid, touches_interior_edge
from tests.test_aggregate import gd
from tests.test_points import oblique_camera

NAMES = {0: "namiot", 1: "people"}


# ───────────── atrapa modelu ultralytics ─────────────

class _T:
    """Minimalny odpowiednik tensora (item / tolist / indeks)."""

    def __init__(self, v):
        self.v = v

    def item(self):
        return self.v

    def tolist(self):
        return list(self.v)

    def __getitem__(self, i):
        return _T(self.v)


class FakeBox:
    def __init__(self, cls, conf, xyxy):
        self.cls, self.conf, self.xyxy = _T(cls), _T(conf), _T(xyxy)


def fake_result(boxes, polygons=None):
    masks = None if polygons is None else SimpleNamespace(xy=polygons)
    return SimpleNamespace(boxes=[FakeBox(*b) for b in boxes], masks=masks)


class FakeYOLO:
    task = "detect"
    names = NAMES

    def __init__(self, path):
        self.overrides = {"imgsz": 1024}
        self.ckpt = {"train_args": {"imgsz": 1024}}
        self.calls = []
        self.results = []

    def predict(self, sources, **kw):
        self.calls.append((len(sources), kw))
        out, self.results = self.results[:len(sources)], self.results[len(sources):]
        return out


def make_detector(cfg=None, task="detect"):
    with mock.patch("ultralytics.YOLO", FakeYOLO):
        d = Detector(Path("fake.pt"), cfg or DetectConfig())
    d.model.task = d.task = task
    return d


class ClassConfigTest(unittest.TestCase):
    def test_classes_by_name(self):
        d = make_detector(DetectConfig(classes=["people"]))
        self.assertEqual(d.class_ids, [1])
        self.assertIsNone(make_detector(DetectConfig(classes="all")).class_ids)
        self.assertIsNone(make_detector(DetectConfig(classes=None)).class_ids)

    def test_unknown_class_lists_available(self):
        for cfg in (DetectConfig(classes=["namiot", "czlowiek"]),
                    DetectConfig(conf_per_class={"auto": 0.5}),
                    DetectConfig(display_names={"Namiot": "x"}),
                    DetectConfig(colors={"rower": "#000000"})):
            with self.assertRaises(ValueError) as cm:
                make_detector(cfg)
            self.assertIn("dostepne klasy: ['namiot', 'people']", str(cm.exception))

    def test_imgsz_from_training_and_override(self):
        self.assertEqual(make_detector().imgsz, 1024)
        self.assertEqual(make_detector(DetectConfig(imgsz=640)).imgsz, 640)

    def test_conf_per_class_and_names_from_model(self):
        d = make_detector(DetectConfig(conf=0.25, conf_per_class={"people": 0.5}))
        d.model.results = [fake_result([(0, 0.3, (0, 0, 10, 10)), (1, 0.4, (20, 20, 30, 30)),
                                        (1, 0.6, (40, 40, 50, 50))])]
        dets = d.detect(Path("x.jpg"))
        self.assertEqual([(x.cls_name, x.conf) for x in dets], [("namiot", 0.3), ("people", 0.6)])
        kw = d.model.calls[0][1]
        self.assertEqual(kw["conf"], 0.25)          # model z najnizszym progiem, potem filtr
        self.assertEqual((kw["imgsz"], kw["iou"], kw["device"]), (1024, 0.7, None))

    def test_display_and_device(self):
        c = DetectConfig(display_names={"people": "człowiek"}, device="cpu")
        self.assertEqual((c.display("people"), c.display("namiot")), ("człowiek", "namiot"))
        self.assertEqual(c.torch_device(), "cpu")
        self.assertIsNone(DetectConfig(device="auto").torch_device())

    def test_yaml_tiling_nested(self):
        cfg = PipelineConfig.from_dict({"detect": {"classes": "all", "tiling": {
            "enabled": True, "classes": ["people"], "overlap": 0.25}}})
        self.assertTrue(cfg.detect.tiling.enabled)
        self.assertEqual(cfg.detect.tiling.overlap, 0.25)
        self.assertIsNone(cfg.detect.class_list())


# ───────────── kafle ─────────────

def det(cls, conf, bbox, trunc=False, tile=None, source="tile"):
    return Detection(0 if cls == "namiot" else 1, cls, conf, tuple(map(float, bbox)),
                     source=source, tile_truncated=trunc, tile_index=tile)


class TileGridTest(unittest.TestCase):
    def test_grid_covers_image_with_overlap(self):
        tiles = tile_grid(2560, 1440, 1024, 0.2)
        self.assertEqual({t.x0 for t in tiles}, {0, 819, 1536})
        self.assertEqual({t.y0 for t in tiles}, {0, 416})
        self.assertTrue(all(t.x1 - t.x0 == 1024 and t.y1 - t.y0 == 1024 for t in tiles))
        self.assertEqual(max(t.x1 for t in tiles), 2560)
        self.assertEqual(tile_grid(800, 600, 1024, 0.2), [Tile(0, 0, 800, 600)])

    def test_interior_edge_vs_image_border(self):
        t = Tile(0, 0, 1024, 1024)
        W, H = 2560, 1440
        # prawa krawedz kafla jest wewnetrzna -> uciety; lewa/gorna to krawedzie zdjecia
        self.assertTrue(touches_interior_edge((1000, 500, 1024, 540), t, W, H, 3))
        self.assertFalse(touches_interior_edge((0, 0, 40, 40), t, W, H, 3))
        # touches_border (zadanie 2) dotyczy zdjecia, nie kafla
        self.assertFalse(touches_border((1000, 500, 1024, 540), W, H, 2.0))
        self.assertTrue(touches_border((0, 0, 40, 40), W, H, 2.0))


class MergeTest(unittest.TestCase):
    def test_truncated_dropped_when_whole_exists(self):
        whole = det("namiot", 0.9, (990, 500, 1080, 560), tile=1)
        cut = det("namiot", 0.95, (990, 500, 1024, 560), trunc=True, tile=0)
        out = merge_detections([cut, whole])
        self.assertEqual([d.bbox for d in out], [whole.bbox])

    def test_complementary_pieces_are_joined(self):
        # obiekt 700..1200 wiekszy niz zakladka 819..1024 -> uciety w obu kaflach
        a = det("namiot", 0.8, (700, 400, 1024, 520), trunc=True, tile=0)
        b = det("namiot", 0.7, (819, 402, 1200, 518), trunc=True, tile=1)
        out = merge_detections([a, b])
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].bbox, (700.0, 400.0, 1200.0, 520.0))
        self.assertEqual((out[0].source, out[0].tile_truncated, out[0].conf), ("merged", False, 0.8))

    def test_duplicate_slivers_not_merged_and_rejected_by_witness_tile(self):
        tiles = tile_grid(2560, 1440, 1024, 0.2)
        # ten sam pasek na prawej krawedzi kafli (0,0) i (0,416); kafel 819 widzi go w calosci
        s1 = det("people", 0.39, (1017, 638, 1024, 678), trunc=True, tile=0)
        s2 = det("people", 0.42, (1017, 637, 1024, 678), trunc=True, tile=3)
        self.assertEqual(merge_detections([s1, s2], tiles=tiles), [])
        # bez informacji o kafelkach: deduplikacja, nie laczenie -> jeden uciety box
        out = merge_detections([s1, s2])
        self.assertEqual(len(out), 1)
        self.assertTrue(out[0].tile_truncated)
        self.assertEqual(out[0].conf, 0.42)

    def test_witness_tile_with_detection_keeps_piece(self):
        tiles = tile_grid(2560, 1440, 1024, 0.2)
        piece = det("people", 0.6, (1010, 600, 1024, 640), trunc=True, tile=0)
        other = det("people", 0.5, (1015, 598, 1022, 642), tile=1)     # kafel 1 cos widzi
        out = merge_detections([piece, other], tiles=tiles)
        self.assertEqual(len(out), 1)            # zawarty w sobie nawzajem -> jeden box

    def test_nms_between_tile_and_full_image(self):
        a = det("people", 0.9, (100, 100, 140, 180), tile=0)
        b = det("people", 0.6, (102, 101, 141, 178), source="full")
        c = det("namiot", 0.8, (100, 100, 140, 180), tile=0)            # inna klasa zostaje
        out = merge_detections([a, b, c])
        self.assertEqual(sorted((d.cls_name, d.conf) for d in out), [("namiot", 0.8), ("people", 0.9)])


class TiledDetectorTest(unittest.TestCase):
    def test_end_to_end_tiled_with_cut_object(self):
        import cv2
        W, H = 2560, 1440
        cfg = DetectConfig()
        cfg.tiling.enabled, cfg.tiling.include_full = True, False
        d = make_detector(cfg)
        tiles = tile_grid(W, H, 1024, 0.2)
        # scena: namiot na granicy (przecina krawedz kafla 0 i poczatek kafla 1), osoba w srodku
        scene = [(0, 0.9, (700, 300, 1200, 420)), (1, 0.8, (1300, 700, 1340, 780))]
        results = []
        for t in tiles:
            boxes = []
            for cls, conf, (x1, y1, x2, y2) in scene:
                cx1, cy1, cx2, cy2 = max(x1, t.x0), max(y1, t.y0), min(x2, t.x1), min(y2, t.y1)
                if cx2 - cx1 > 2 and cy2 - cy1 > 2:
                    boxes.append((cls, conf, (cx1 - t.x0, cy1 - t.y0, cx2 - t.x0, cy2 - t.y0)))
            results.append(fake_result(boxes))
        d.model.results = results
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "img.jpg"
            cv2.imwrite(str(p), np.zeros((H, W, 3), np.uint8))
            out = d.detect(p)
        got = sorted((x.cls_name, x.bbox, x.tile_truncated) for x in out)
        self.assertEqual(got, [("namiot", (700.0, 300.0, 1200.0, 420.0), False),
                               ("people", (1300.0, 700.0, 1340.0, 780.0), False)])
        self.assertEqual(d.model.calls[0][0], len(tiles))     # kafle w jednym wywolaniu


# ───────────── modele ─────────────

class _FakeNet:
    def __init__(self, names, task):
        self.names, self.task = names, task


class ListModelsTest(unittest.TestCase):
    def test_list_models_metadata(self):
        import torch
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            torch.save({"model": _FakeNet(NAMES, "detect"), "date": "2026-03-30",
                        "train_args": {"imgsz": 1024, "task": "detect", "data": "d.yaml"}},
                       tmp / "b_det.pt")
            torch.save({"model": _FakeNet(["kot"], "segment"),
                        "train_args": {"imgsz": [640, 640], "task": "segment"}}, tmp / "a_seg.pt")
            torch.save({"model": {"w": 1}}, tmp / "sam.pt")       # nie-YOLO -> pomijany
            (tmp / "notes.txt").write_text("x")
            ms = list_models(tmp)
            self.assertEqual([m.name for m in ms], ["a_seg.pt", "b_det.pt"])
            seg, dt = ms
            self.assertEqual((seg.task, seg.names, seg.imgsz), ("segment", {0: "kot"}, 640))
            self.assertEqual((dt.task, dt.names, dt.imgsz, dt.date), ("detect", NAMES, 1024,
                                                                       "2026-03-30"))
            self.assertGreater(dt.size_mb, 0)
            self.assertIn("klasy: namiot, people", dt.label)
            self.assertIsNone(model_info(tmp / "sam.pt"))
        self.assertEqual(list_models(Path("nie_ma_takiego")), [])


# ───────────── maski z modelu segmentacyjnego ─────────────

class SegModelMaskTest(unittest.TestCase):
    def test_polygons_from_segment_result(self):
        d = make_detector(task="segment")
        poly = np.array([[10, 10], [50, 10], [50, 40], [10, 40]], float)
        d.model.results = [fake_result([(0, 0.9, (10, 10, 50, 40))], [poly])]
        out = d.detect(Path("x.jpg"))
        np.testing.assert_array_equal(out[0].polygons[0], poly)

    def test_mask_from_polygons_and_pipeline_skips_sam(self):
        # romb wpisany w bbox (prostokat wypelniajacy bbox zostalby odrzucony: fills_bbox)
        poly = np.array([[150, 100], [200, 140], [150, 180], [100, 140]], float)
        m = mask_from_polygons([poly], (100, 100, 200, 180), (1000, 1000))
        u, v = m.pixels().mean(axis=0)
        self.assertAlmostEqual(u, 150.5, delta=1.0)
        self.assertAlmostEqual(v, 140.5, delta=1.0)
        self.assertAlmostEqual(m.area, 100 * 80 / 2, delta=150)
        cfg = PipelineConfig()
        cfg.point.method = "mask_centroid"
        ctx = SimpleNamespace(config=cfg, segmenter=None)
        io, eo = oblique_camera()
        dd = Detection(0, "namiot", 0.9, (100, 100, 200, 180), polygons=[poly])
        ch = choose_points(ctx, Path("x.jpg"), [dd], io, eo)[0]
        self.assertEqual((ch.used, ch.fallback), ("mask_centroid", None))
        self.assertIsNone(ctx.segmenter)                  # SAM nie zostal utworzony
        self.assertAlmostEqual(ch.uv[0], 150.5, delta=1.0)


class ObjectStatusTest(unittest.TestCase):
    def test_confirmed_vs_single_view(self):
        x0, y0 = 660330.0, 664811.0
        dets = [gd("A.JPG", x0, y0), gd("B.JPG", x0 + 0.3, y0), gd("C.JPG", x0 + 50, y0)]
        objs = aggregate(dets, AggregateConfig(eps_m=2.5, min_views=2))
        status = sorted((o.n_views, o.status) for o in objs)
        self.assertEqual(status, [(1, "single_view"), (2, "confirmed")])
        objs = aggregate(dets, AggregateConfig(eps_m=2.5, min_views=3))
        self.assertTrue(all(o.status == "single_view" for o in objs))


if __name__ == "__main__":
    unittest.main()
