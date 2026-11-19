"""Testy API biblioteki uzywanego przez GUI: odmrun, coverage, photos, runs, qc (opisy, auto),
config.save, detect (prompt). Bez sieci, Dockera i GPU (atrapy runner/popen/get)."""

import io
import json
import logging
import math
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from geodetect import coverage, odmrun, photos, runs
from geodetect.config import DetectConfig, PipelineConfig
from geodetect.detect import is_open_vocab_model, prompt_class_names
from geodetect.qc import (BAD, INFO, OK, REC_7P, REC_AUTO, REC_BASE, WARN, QcCheck, QcReport, auto_georef_choice,
                          compare_georef,
                          explain, explain_report, summarize)
from tests.test_surface import write_asc


def _cfg(tmp: Path, **kw) -> PipelineConfig:
    c = PipelineConfig.from_dict({}, base_dir=tmp)
    c.photos_dir, c.odm_dir = tmp / "zdj", tmp / "odm"
    for k, v in kw.items():
        setattr(c, k, v)
    return c


class OdmRunTest(unittest.TestCase):
    def test_command(self):
        cmd = odmrun.odm_command(Path("odm"))
        self.assertEqual(cmd[:3], ["docker", "run", "--rm"])
        self.assertTrue(cmd[4].endswith(":/datasets/code"))
        self.assertIn("--dsm", cmd)
        self.assertNotIn("--use-fixed-camera-params", cmd)
        fixed = odmrun.odm_command(Path("odm"), fixed_cameras=True, extra=("--fast-orthophoto",))
        self.assertIn("/datasets/code/cameras.json", fixed)
        self.assertEqual(fixed[-1], "--fast-orthophoto")

    def test_prepare_excludes_start_image(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            c = _cfg(d)
            c.photos_dir.mkdir()
            for n in ("A.JPG", "B.JPG", "START.JPG", "notes.txt"):
                (c.photos_dir / n).write_bytes(b"x" * 10)
            c.z_anchor.start.image = "START.JPG"
            (c.odm_dir / "images").mkdir(parents=True)
            (c.odm_dir / "images" / "START.JPG").write_bytes(b"old")      # z poprzedniego przygotowania
            r = odmrun.prepare_odm(c)
            self.assertEqual(r.copied, ["A.JPG", "B.JPG"])
            self.assertEqual(r.excluded, ["START.JPG"])
            self.assertEqual(r.removed, ["START.JPG"])
            self.assertEqual(r.n_images, 2)
            self.assertFalse((c.odm_dir / "images" / "START.JPG").exists())
            self.assertEqual(odmrun.prepare_odm(c).unchanged, ["A.JPG", "B.JPG"])

    def test_cameras_from_reconstruction_and_copy(self):
        with tempfile.TemporaryDirectory() as d:
            src, dst = Path(d) / "src", Path(d) / "dst"
            (src / "opensfm").mkdir(parents=True)
            cams = {"v2 cam 2560 1440 brown 0.36": {"projection_type": "brown", "focal_x": 0.52}}
            (src / "opensfm" / "reconstruction.json").write_text(json.dumps([{"cameras": cams, "shots": {}}]))
            self.assertEqual(odmrun.camera_calibration(src), cams)
            p = odmrun.write_cameras_json(src, dst)
            self.assertEqual(json.loads(p.read_text()), cams)
            self.assertEqual(odmrun.camera_calibration(dst), cams)       # cameras.json ma pierwszenstwo
            with self.assertRaises(FileNotFoundError):
                odmrun.camera_calibration(Path(d) / "pusty")

    def test_docker_status(self):
        def runner_ok(cmd, **kw):
            if cmd[1] == "info":
                return SimpleNamespace(returncode=0, stdout="27.1.1\n", stderr="")
            return SimpleNamespace(returncode=1, stdout="", stderr="no such image")
        s = odmrun.docker_status(runner=runner_ok)
        self.assertTrue(s.ok)
        self.assertEqual(s.version, "27.1.1")
        self.assertFalse(s.odm_image)
        self.assertIn("pobrany", s.message)

        def runner_down(cmd, **kw):
            return SimpleNamespace(returncode=1, stdout="", stderr="error during connect: pipe not found")
        s = odmrun.docker_status(runner=runner_down)
        self.assertFalse(s.ok)
        self.assertIn("nie jest uruchomiony", s.message)

        def runner_timeout(cmd, **kw):
            raise subprocess.TimeoutExpired(cmd, 1)
        self.assertFalse(odmrun.docker_status(runner=runner_timeout).ok)

    def test_run_odm_streams_and_cancels(self):
        class Proc:
            def __init__(self, lines):
                self.stdout = io.StringIO("".join(f"{x}\n" for x in lines))
                self.killed = False
                self.rc = 0

            def kill(self):
                self.killed = True

            def wait(self):
                return self.rc

            def poll(self):
                return self.rc

        procs = []

        def popen(cmd, **kw):
            procs.append(Proc(["[INFO] Running opensfm stage", "[INFO] Running odm_dem stage", "done"]))
            return procs[-1]
        got = []
        r = odmrun.run_odm(["x"], line_cb=got.append, popen=popen)
        self.assertTrue(r.ok)
        self.assertEqual(len(got), 3)
        with self.assertLogs("geodetect.odmrun", "WARNING"):
            r = odmrun.run_odm(["x"], cancel=lambda: True, popen=popen)
        self.assertTrue(r.cancelled)
        self.assertIsNone(r.returncode)
        self.assertTrue(procs[-1].killed)
        self.assertEqual(r.log_tail, ["[INFO] Running opensfm stage"])

    def test_outputs_ok(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            self.assertFalse(any(odmrun.odm_outputs_ok(d).values()))
            (d / "opensfm").mkdir()
            (d / "opensfm" / "reconstruction.topocentric.json").write_text("[]")
            (d / "opensfm" / "reference_lla.json").write_text("{}")
            ok = odmrun.odm_outputs_ok(d)
            self.assertTrue(ok["rekonstrukcja"] and ok["reference_lla"])
            self.assertFalse(ok["DSM"])


class CoverageTest(unittest.TestCase):
    def test_check_folder(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            write_asc(d / "a.asc", np.full((10, 10), 100.0), 1000.0, 2000.0, 1.0)
            ok = coverage.check_folder("NMPT", d, (1002.0, 2002.0, 1008.0, 2008.0))
            self.assertTrue(ok.ok, ok.message)
            self.assertEqual(ok.files, ["a.asc"])
            self.assertIn("pełne pokrycie", ok.message)
            out = coverage.check_folder("NMPT", d, (1005.0, 2002.0, 1015.0, 2008.0))
            self.assertFalse(out.ok)
            self.assertAlmostEqual(out.missing_m["E"], 5.0)
            self.assertIn("E 5 m", out.message)
            z = np.full((10, 10), 100.0)
            z[4:6, 4:6] = -9999.0
            write_asc(d / "a.asc", z, 1000.0, 2000.0, 1.0)
            holes = coverage.check_folder("NMPT", d, (1002.0, 2002.0, 1008.0, 2008.0))
            self.assertFalse(holes.ok)
            self.assertGreater(holes.nodata_frac, 0)
            self.assertFalse(coverage.check_folder("NMT", d / "brak", (0, 0, 1, 1)).ok)
            self.assertFalse(coverage.check_folder("NMT", None, (0, 0, 1, 1)).ok)

    def test_rest_status(self):
        calls = []

        def get_ok(url, params, timeout):
            calls.append(params)
            return SimpleNamespace(text="135.42")
        s = coverage.gugik_rest_status("http://x", x=660000.0, y=664000.0, get=get_ok)
        self.assertTrue(s.ok)
        # GetHByXY: x = wspolrzedna "Y" PUWG (northing), y = "X"
        self.assertEqual(calls[0]["x"], "664000.00")
        self.assertFalse(coverage.gugik_rest_status("http://x", get=lambda *a, **k: SimpleNamespace(text="blad")).ok)

        def get_err(*a, **k):
            raise ConnectionError("offline")
        s = coverage.gugik_rest_status("http://x", get=get_err)
        self.assertFalse(s.ok)
        self.assertIn("ConnectionError", s.message)


class PhotosTest(unittest.TestCase):
    def test_scan_without_exif(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "b.jpg").write_bytes(b"not a jpeg")
            (d / "a.JPG").write_bytes(b"not a jpeg")
            (d / "x.txt").write_text("-")
            self.assertEqual([p.name for p in photos.list_photos(d)], ["a.JPG", "b.jpg"])
            logging.disable(logging.CRITICAL)        # biblioteka EXIF loguje "File format not recognized"
            try:
                ph = photos.scan_photos(d)
            finally:
                logging.disable(logging.NOTSET)
            self.assertEqual(len(ph), 2)
            self.assertFalse(any(p.has_gps for p in ph))
            self.assertTrue(all(p.error for p in ph))
        self.assertEqual(photos.list_photos(Path("nie_ma_takiego_folderu")), [])

    def test_suggest_start(self):
        P = photos.PhotoInfo
        ph = [P("1.JPG", Path("1"), 53, 21, 45.0), P("2.JPG", Path("2"), 53, 21, 1.1),
              P("3.JPG", Path("3"), 53, 21, 0.9)]
        self.assertEqual(photos.suggest_start_image(ph), "2.JPG")
        self.assertIsNone(photos.suggest_start_image(ph[:1]))


def write_run(d: Path, name: str, objs: list[dict], params: dict, qc: dict | None = None) -> Path:
    r = d / "runs" / name
    r.mkdir(parents=True)
    feats = [{"type": "Feature", "geometry": {"type": "Point", "coordinates": [o["lon"], o["lat"]]},
              "properties": {k: v for k, v in o.items() if k not in ("lon", "lat")}} for o in objs]
    (r / "objects.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": feats}))
    dets = [{"type": "Feature", "geometry": {"type": "Point", "coordinates": [o["lon"], o["lat"]]},
             "properties": {"image": f"{i}.JPG", "class": o["class"], "object_id": o["object_id"], "conf": c,
                            "touches_border": tb, "bbox": [0, 0, 10, 10]}}
            for o in objs for i, (c, tb) in enumerate([(0.5, False), (0.9, True), (0.7, False)])]
    (r / "detections.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": dets}))
    rep = {"created": name, "parameters": params, "qc": qc,
           "statistics": {"images_processed": 3, "objects": len(objs), "confirmed": len(objs), "flagged": 0},
           "files": {"objects_geojson": "objects.geojson"}}
    (r / "run_report.json").write_text(json.dumps(rep))
    return r


class RunsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)
        o = lambda i, cls, x, y: {"object_id": i, "class": cls, "x_2180": x, "y_2180": y, "altitude_m": 100.0,  # noqa
                                  "lon": 21.0 + x * 1e-5, "lat": 53.0 + y * 1e-5, "status": "confirmed",
                                  "n_views": 3}
        self.a = write_run(self.d, "2026-01-01_100000", [o(0, "namiot", 0, 0), o(1, "people", 50, 0)],
                           {"model": "models/A.pt", "conf": 0.25, "aggregate": {"eps_m": 2.5}},
                           {"status": "OK", "recommendations": [], "checks": []})
        self.b = write_run(self.d, "2026-01-02_100000", [o(0, "namiot", 3, 4), o(1, "people", 200, 0)],
                           {"model": "models/B.pt", "conf": 0.25, "aggregate": {"eps_m": 3.0}})
        (self.d / "runs" / "pusty").mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def test_list_and_load(self):
        rs = runs.list_runs(self.d)
        self.assertEqual([r.name for r in rs], ["2026-01-02_100000", "2026-01-01_100000"])
        self.assertEqual(rs[1].qc_status, "OK")
        self.assertEqual(rs[0].model, "B.pt")
        self.assertEqual(runs.list_runs(self.d / "brak"), [])
        run = runs.load_run(self.a)
        self.assertEqual(len(run.objects), 2)
        self.assertAlmostEqual(run.object(1)["lon"], 21.0005)
        v = run.views(0)
        self.assertEqual([x["conf"] for x in v], [0.7, 0.5, 0.9])       # ucieta krawedzia na koncu
        self.assertIsNone(run.object(99))

    def test_compare(self):
        m, diff = runs.compare_runs(runs.load_run(self.a), runs.load_run(self.b))
        pair = [x for x in m if x.a_id is not None and x.b_id is not None]
        self.assertEqual(len(pair), 1)
        self.assertAlmostEqual(pair[0].shift_m, 5.0)
        self.assertEqual(pair[0].dh_m, 0.0)
        self.assertEqual(sum(1 for x in m if x.b_id is None), 1)
        self.assertEqual(sum(1 for x in m if x.a_id is None), 1)
        self.assertEqual(diff["aggregate.eps_m"], (2.5, 3.0))
        self.assertNotIn("conf", diff)

    def test_run_map(self):
        m = runs.run_map(runs.load_run(self.a), selected=0)
        html = m.get_root().render()
        self.assertIn("Legenda", html)
        self.assertIn("Esri World Imagery", html)


class QcGuiTest(unittest.TestCase):
    def _rep(self, *checks, aligned=False):
        return summarize(list(checks), aligned)

    def test_focal_info_does_not_degrade_status(self):
        rep = self._rep(QcCheck("focal", INFO, 53.0, "", {"source": "EXIF f35 = 13 mm"}),
                        QcCheck("tilt", OK, 0.5, ""))
        self.assertEqual(rep.status, OK)
        v = explain_report(rep)
        self.assertEqual(v["checks"][-1][1], INFO)               # informacja na koncu
        self.assertIn("normalne", v["checks"][-1][2])

    def test_explain_without_code_names(self):
        rep = self._rep(QcCheck("tilt", WARN, 3.9, "", {}), QcCheck("baseline", BAD, 6.0, "", {"span_m": 40}),
                        QcCheck("surface", OK, 0.3, ""), QcCheck("vertical_scale", OK, 1.02, ""))
        v = explain_report(rep)
        self.assertEqual(v["top"], 2)
        self.assertEqual([c[1] for c in v["checks"][:2]], [BAD, WARN])
        self.assertIn(REC_7P, rep.recommendations)
        self.assertIn(REC_BASE, rep.recommendations)
        texts = [t for t, _ in v["actions"]] + [c[2] for c in v["checks"]]
        for t in texts:
            for code in ("similarity_nmpt", "georef.mode", "z_anchor", "—"):
                self.assertNotIn(code, t)
        self.assertIn(("Na stronie Georeferencja wybierz wyrównanie do NMPT.", "georef_nmpt"), v["actions"])
        self.assertIn("3.9°", explain(rep.check("tilt")))

    def test_auto_choice_stage1(self):
        self.assertEqual(auto_georef_choice(None)[0], "gps_only")
        ok = self._rep(QcCheck("tilt", OK, 0.4, ""), QcCheck("surface", OK, 0.3, ""))
        self.assertIsNone(auto_georef_choice(ok)[0])                    # -> porownanie wariantow
        mild = self._rep(QcCheck("tilt", INFO, 3.9, ""), QcCheck("vertical_scale", OK, 0.97, ""))
        self.assertIsNone(auto_georef_choice(mild)[0])
        squashed = self._rep(QcCheck("vertical_scale", BAD, 0.68, ""), QcCheck("tilt", INFO, 3.0, ""))
        mode, why = auto_georef_choice(squashed)
        self.assertEqual(mode, "similarity_nmpt")
        self.assertIn("0.68", why)
        tilted = self._rep(QcCheck("tilt", WARN, 14.0, ""))
        mode, why = auto_georef_choice(tilted)
        self.assertEqual(mode, "similarity_nmpt")
        self.assertIn("14.0°", why)

    def test_mild_tilt_recommends_auto_not_7p(self):
        rep = self._rep(QcCheck("tilt", INFO, 3.9, ""), QcCheck("surface", OK, 0.3, ""))
        self.assertEqual(rep.status, OK)
        self.assertEqual(rep.recommendations, [REC_AUTO])
        self.assertEqual(explain_report(rep)["actions"][0][1], "georef_auto")
        self.assertIn("lekko pochylony", explain(rep.check("tilt")))

    def test_from_dict_legacy_tilt_warning(self):
        old = {"status": WARN, "recommendations": [REC_7P], "checks": [
            {"name": "tilt", "status": WARN, "value": 3.9, "text": "", "details": {"mode": "gps_only"}},
            {"name": "baseline", "status": WARN, "value": 3.2, "text": "", "details": {}}]}
        rep = QcReport.from_dict(old)
        self.assertEqual(rep.check("tilt").status, INFO)
        self.assertEqual(rep.status, WARN)                              # baza nadal ostrzezeniem
        self.assertEqual(rep.recommendations, [REC_AUTO])
        self.assertEqual(QcReport.from_dict(rep.to_dict()).recommendations, [REC_AUTO])


    def test_from_dict_old_focal_warning_becomes_info(self):
        old = {"status": WARN, "recommendations": [], "checks": [
            {"name": "focal", "status": WARN, "value": 53.0, "text": "", "details": {"source": "EXIF f35 = 13 mm"}},
            {"name": "tilt", "status": OK, "value": 0.5, "text": "", "details": {}}]}
        rep = QcReport.from_dict(old)
        self.assertEqual(rep.check("focal").status, INFO)
        self.assertEqual(rep.status, OK)

    def test_from_dict_roundtrip(self):
        rep = self._rep(QcCheck("tilt", WARN, 3.9, "t", {"a": 1}))
        back = QcReport.from_dict(rep.to_dict())
        self.assertEqual(back.status, rep.status)
        self.assertEqual(back.check("tilt").details, {"a": 1})
        self.assertEqual(back.recommendations, rep.recommendations)


class GeorefCompareTest(unittest.TestCase):
    """Etap 2 reguly auto: rozrzut obiektow i reszty DSM - NMPT w obu wariantach."""
    cfg = PipelineConfig().georef

    def test_7p_increases_spread_keeps_standard(self):
        c = compare_georef(0.38, 1.0, 2, 1.20, 0.45, self.cfg)
        self.assertEqual(c.mode, "gps_only")
        self.assertEqual(c.reason, "Wybrano standard: wyrównanie do NMPT zwiększyłoby rozrzut obiektów z 0.38 do 1.00 m.")

    def test_7p_reduces_spread(self):
        c = compare_georef(2.0, 0.6, 3, 1.5, 0.5, self.cfg)
        self.assertEqual(c.mode, "similarity_nmpt")
        self.assertIn("z 2.00 do 0.60 m", c.reason)

    def test_small_gain_keeps_standard(self):
        self.assertEqual(compare_georef(1.0, 0.85, 3, 1.0, 0.5, self.cfg).mode, "gps_only")   # 15% < 20%

    def test_worse_residuals_keep_standard(self):
        c = compare_georef(1.0, 0.5, 3, 0.40, 0.60, self.cfg)
        self.assertEqual(c.mode, "gps_only")
        self.assertIn("pogorszyłoby zgodność z NMPT", c.reason)

    def test_no_objects_decided_by_residuals(self):
        nan = float("nan")
        self.assertEqual(compare_georef(nan, nan, 0, 1.6, 0.5, self.cfg).mode, "similarity_nmpt")
        self.assertEqual(compare_georef(nan, nan, 0, 0.5, 0.48, self.cfg).mode, "gps_only")
        self.assertEqual(compare_georef(nan, nan, 0, nan, nan, self.cfg).mode, "gps_only")

    def test_multiview_spread(self):
        from geodetect.batch import multiview_spread
        o = lambda n, rms, ok=True: SimpleNamespace(confirmed=ok, n_views=n, rms_m=rms)   # noqa: E731
        s, n = multiview_spread([o(26, 0.4), o(2, 5.0), o(4, 1.0, ok=False), o(3, 0.8)], 3)
        self.assertEqual(n, 2)
        self.assertAlmostEqual(s, math.sqrt((26 * 0.16 + 3 * 0.64) / 29))
        self.assertEqual(multiview_spread([o(2, 1.0)], 3)[1], 0)

    def test_choose_variant_uses_qc_residuals(self):
        from geodetect.batch import choose_georef_variant
        o = lambda rms: [SimpleNamespace(confirmed=True, n_views=10, rms_m=rms)]          # noqa: E731
        qc = lambda nmad: QcReport([QcCheck("surface", OK, 0.3, "", {"after": {"nmad_raw": nmad}})], OK, [])  # noqa: E731
        cfg = PipelineConfig()
        c = choose_georef_variant(cfg, o(0.38), o(1.0), qc(1.2), qc(0.45))
        self.assertEqual((c.mode, c.n_objects), ("gps_only", 1))
        c = choose_georef_variant(cfg, o(3.0), o(0.9), qc(2.0), qc(0.5))
        self.assertEqual(c.mode, "similarity_nmpt")
        self.assertAlmostEqual(c.nmad_7p, 0.5)


class ConfigDetectGuiTest(unittest.TestCase):
    def test_save_roundtrip_relative_paths(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            c = PipelineConfig.from_dict({"photos_dir": "zdj", "model_path": "models/m.pt",
                                          "detect": {"prompt_text": ["ball"], "visual_prompt_image": "zdj/a.jpg"}},
                                         base_dir=d)
            p = c.save(d / "sub" / ".." / "preset.yaml")
            text = p.read_text(encoding="utf-8")
            self.assertIn("photos_dir: zdj", text)
            self.assertIn("visual_prompt_image: zdj/a.jpg", text)
            back = PipelineConfig.from_file(p)
            self.assertEqual(back.photos_dir, (d / "zdj").resolve())
            self.assertEqual(back.detect.prompt_text, ["ball"])

    def test_prompt_class_names(self):
        self.assertIsNone(prompt_class_names(DetectConfig()))
        self.assertEqual(prompt_class_names(DetectConfig(prompt_text=["ball", "tent"])), ["ball", "tent"])
        vp = DetectConfig(prompt_text=["x"], visual_prompt_image=Path("a.jpg"),
                          visual_prompt_boxes=[[0, 0, 1, 1], [2, 2, 3, 3]], visual_prompt_classes=["piłka"])
        self.assertEqual(prompt_class_names(vp), ["piłka"])           # wizualny ma pierwszenstwo
        self.assertTrue(vp.uses_visual_prompt())

    def test_open_vocab_detection(self):
        YOLOESegModel = type("YOLOESegModel", (), {})
        WorldModel = type("WorldModel", (), {})
        DetectionModel = type("DetectionModel", (), {})
        self.assertTrue(is_open_vocab_model(YOLOESegModel()))
        self.assertTrue(is_open_vocab_model(WorldModel()))
        self.assertFalse(is_open_vocab_model(DetectionModel()))


class GeorefAutoTest(unittest.TestCase):
    """pipeline.georeference z georef.mode "auto": QC wariantu gps_only decyduje o wyrownaniu do NMPT."""

    def run_auto(self, qc_gps, align_exc=None):
        from unittest import mock

        from geodetect import pipeline
        from geodetect.zcal import ZCalibration
        cfg = PipelineConfig.from_dict({"georef": {"mode": "auto"}})
        rec = SimpleNamespace(ref=None)
        aligned = SimpleNamespace(reconstruction="REC_7P", result=SimpleNamespace(sim=SimpleNamespace(apply=None)),
                                  stats=lambda: {})
        qc_calls = []

        def fake_qc(config, rec_gps, base_frame, z_cal, alignment, dem):
            qc_calls.append(alignment)
            return qc_gps if alignment is None else "QC_PO_7P"
        align = mock.Mock(side_effect=align_exc) if align_exc else mock.Mock(return_value=aligned)
        with mock.patch.object(pipeline, "EnuFrame"),                 mock.patch.object(pipeline, "calibrate_z", return_value=ZCalibration(126.0, "dem_alignment")),                 mock.patch.object(pipeline, "_run_qc", side_effect=fake_qc),                 mock.patch("geodetect.align.align_reconstruction", align):
            g = pipeline.georeference(cfg, dem=object(), reconstruction=rec)
        return cfg, g, qc_calls

    def test_auto_compares_when_no_deformation(self):
        mild = summarize([QcCheck("tilt", INFO, 3.9, ""), QcCheck("surface", OK, 0.3, "")])
        self.assertEqual(mild.recommendations, [REC_AUTO])
        cfg, g, calls = self.run_auto(mild)
        self.assertEqual(g.mode, "gps_only")                  # wariant glowny: standard
        self.assertIsNotNone(g.alternative)                   # 7p do porownania w run_batch
        self.assertEqual(g.alternative.mode, "similarity_nmpt")
        self.assertEqual(g.alternative.reconstruction, "REC_7P")
        self.assertEqual(g.alternative.qc, "QC_PO_7P")
        self.assertEqual(g.alternative.config.georef.mode, "similarity_nmpt")
        self.assertEqual(len(calls), 2)
        self.assertEqual(g.qc.recommendations, [])            # bez "wybierz tryb auto" w trybie auto
        self.assertEqual(mild.recommendations, [REC_AUTO])    # raport z cache QC nienaruszony
        self.assertEqual(cfg.georef.mode, "auto")             # config uzytkownika bez zmian
        self.assertIn("porównanie", g.decision)

    def test_auto_aligns_deformed_model(self):
        squashed = summarize([QcCheck("vertical_scale", BAD, 0.68, ""), QcCheck("tilt", WARN, 14.0, "")])
        cfg, g, calls = self.run_auto(squashed)
        self.assertEqual(g.mode, "similarity_nmpt")
        self.assertIsNone(g.alternative)
        self.assertEqual(g.reconstruction, "REC_7P")
        self.assertEqual(g.z_calibration.offset, 0.0)
        self.assertEqual(g.qc, "QC_PO_7P")                    # raport po wyrownaniu
        self.assertEqual(g.config.georef.mode, "similarity_nmpt")
        self.assertIn("0.68", g.decision)

    def test_auto_falls_back_without_nmpt(self):
        mild = summarize([QcCheck("tilt", INFO, 3.0, "")])
        with self.assertLogs("geodetect.pipeline", "WARNING"):
            cfg, g, calls = self.run_auto(mild, FileNotFoundError("brak arkuszy NMPT"))
        self.assertEqual(g.mode, "gps_only")
        self.assertIsNone(g.alternative)
        self.assertIn("nie powiodło", g.decision)

if __name__ == "__main__":
    unittest.main()
