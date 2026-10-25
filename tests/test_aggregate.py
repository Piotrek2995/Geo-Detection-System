import unittest

import numpy as np

from geodetect.aggregate import aggregate, dbscan
from geodetect.config import AggregateConfig, FilterConfig
from geodetect.detect import Detection
from geodetect.pipeline import GeoDetection, rejection_reason, touches_border
from geodetect.raycast import RayHit

X0, Y0 = 660330.0, 664811.0


def gd(image, x, y, cls="namiot", conf=0.9, inc=20.0, rng=60.0, rejected=None):
    hit = RayHit(np.zeros(3), x, y, 135.5, 21.43, 53.82, rng)
    return GeoDetection(image, Detection(0, cls, conf, (100, 100, 200, 200)), (150, 150),
                        hit, inc, False, rejected_reason=rejected)


class DbscanTest(unittest.TestCase):
    def test_two_groups_and_noise(self):
        pts = np.array([[0, 0], [1, 0], [0, 1], [10, 10], [10.5, 10], [50, 50]], float)
        labels = dbscan(pts, eps=2.0, min_samples=2)
        self.assertEqual(len(set(labels[:3])), 1)
        self.assertEqual(len(set(labels[3:5])), 1)
        self.assertNotEqual(labels[0], labels[3])
        self.assertEqual(labels[5], -1)

    def test_min_samples_1_keeps_singletons(self):
        labels = dbscan(np.array([[0, 0], [100, 0]], float), eps=2.5, min_samples=1)
        self.assertEqual(sorted(labels), [0, 1])


class AggregateTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.dets = []
        # obiekt A (namiot) z 5 zdjec, obiekt B (namiot) 20 m dalej z 3 zdjec, szum ~0.5 m
        for i in range(5):
            dx, dy = rng.normal(0, 0.5, 2)
            self.dets.append(gd(f"A{i}.JPG", X0 + dx, Y0 + dy))
        for i in range(3):
            dx, dy = rng.normal(0, 0.5, 2)
            self.dets.append(gd(f"B{i}.JPG", X0 + 20 + dx, Y0 + dy))
        # osoba w miejscu obiektu A — inna klasa, nie moze sie scalic z namiotem
        self.dets.append(gd("A0.JPG", X0, Y0, cls="people"))

    def test_two_objects_per_class_separately(self):
        objs = aggregate(self.dets, AggregateConfig(eps_m=2.5))
        tents = sorted((o for o in objs if o.cls_name == "namiot"), key=lambda o: o.x_2180)
        self.assertEqual(len(tents), 2)
        self.assertEqual([o.n_views for o in tents], [5, 3])
        self.assertEqual(len([o for o in objs if o.cls_name == "people"]), 1)
        # srodek = dokladnie srednia probek (wagi rowne)
        self.assertAlmostEqual(tents[0].x_2180, np.mean([d.hit.x_2180 for d in self.dets[:5]]))
        self.assertAlmostEqual(tents[1].y_2180, np.mean([d.hit.y_2180 for d in self.dets[5:8]]))
        self.assertAlmostEqual(tents[1].x_2180, X0 + 20, delta=1.5)
        self.assertLess(tents[0].rms_m, 1.0)
        self.assertGreaterEqual(tents[0].max_m, tents[0].rms_m)
        self.assertTrue(all(d.object_id is not None for d in self.dets))
        self.assertEqual(len({d.object_id for d in self.dets}), 3)

    def test_rms_and_max_exact(self):
        dets = [gd("a.JPG", X0 - 1, Y0), gd("b.JPG", X0 + 1, Y0)]
        (o,) = aggregate(dets, AggregateConfig())
        self.assertAlmostEqual(o.x_2180, X0)
        self.assertAlmostEqual(o.rms_m, 1.0)
        self.assertAlmostEqual(o.max_m, 1.0)
        self.assertEqual(o.images, ["a.JPG", "b.JPG"])

    def test_weighted_mean(self):
        dets = [gd("a.JPG", X0, Y0, conf=0.9, inc=0.0), gd("b.JPG", X0 + 2, Y0, conf=0.3, inc=60.0)]
        eq = aggregate(dets, AggregateConfig(weights="equal"))[0]
        cf = aggregate(dets, AggregateConfig(weights="conf"))[0]
        ci = aggregate(dets, AggregateConfig(weights="cos_incidence"))[0]
        self.assertAlmostEqual(eq.x_2180, X0 + 1.0)
        self.assertAlmostEqual(cf.x_2180, X0 + 2 * 0.3 / 1.2)
        self.assertAlmostEqual(ci.x_2180, X0 + 2 * 0.5 / 1.5)
        with self.assertRaises(ValueError):
            aggregate(dets, AggregateConfig(weights="xxx"))

    def test_duplicate_image_in_cluster_flagged(self):
        dets = [gd("S1.JPG", X0, Y0, cls="people"), gd("S1.JPG", X0 + 1.2, Y0, cls="people"),
                gd("S2.JPG", X0 + 0.6, Y0, cls="people")]
        with self.assertLogs("geodetect.aggregate", level="WARNING"):
            (o,) = aggregate(dets, AggregateConfig(eps_m=2.5))
        self.assertEqual(o.duplicate_images, ["S1.JPG"])
        self.assertEqual(o.n_views, 3)
        self.assertEqual(o.images, ["S1.JPG", "S2.JPG"])

    def test_rejected_not_aggregated(self):
        dets = [gd("a.JPG", X0, Y0), gd("b.JPG", X0, Y0, rejected="range_m>100")]
        objs = aggregate(dets, AggregateConfig())
        self.assertEqual(objs[0].n_views, 1)
        self.assertIsNone(dets[1].object_id)


class FilterTest(unittest.TestCase):
    def test_touches_border(self):
        self.assertTrue(touches_border((0, 10, 50, 50), 2560, 1440, 2))
        self.assertTrue(touches_border((10, 10, 2559, 50), 2560, 1440, 2))
        self.assertFalse(touches_border((10, 10, 50, 50), 2560, 1440, 2))

    def test_rejection_reasons(self):
        d = gd("a.JPG", X0, Y0, inc=50.0, rng=120.0)
        self.assertIsNone(rejection_reason(d, FilterConfig()))
        self.assertEqual(rejection_reason(d, FilterConfig(max_range_m=100)), "range_m>100")
        self.assertEqual(rejection_reason(d, FilterConfig(max_incidence_deg=45)), "incidence_deg>45")
        d.touches_border = True
        self.assertEqual(rejection_reason(d, FilterConfig(exclude_border=True)), "touches_border")


if __name__ == "__main__":
    unittest.main()
