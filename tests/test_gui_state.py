"""Testy logiki GUI bez streamlit (app/state.py): walidacja, mapowanie ustawien, presety,
preferencje, wybor powierzchni, zadania w tle (z atrapami przebiegu i ODM)."""

import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from app import state as S
from geodetect.config import PipelineConfig
from geodetect.coverage import FolderCoverage, RestStatus
from geodetect.photos import PhotoInfo
from geodetect.run import Progress, RunResult


def project(tmp: Path) -> PipelineConfig:
    c = PipelineConfig.from_dict({}, base_dir=tmp)
    c.photos_dir, c.odm_dir, c.output_dir = tmp / "zdj", tmp / "odm", tmp / "out"
    c.photos_dir.mkdir()
    model = tmp / "m.pt"
    model.write_bytes(b"0")
    c.model_path = model
    return c


def model_info(names=("namiot", "people"), open_vocab=False, task="detect"):
    return SimpleNamespace(names=dict(enumerate(names)), task=task, open_vocab=open_vocab, name="m.pt")


ALL_OK = {"rekonstrukcja": True, "reference_lla": True, "DSM": True, "DTM": True, "ortofotomapa": True}


def wait(job, timeout=10.0):
    t0 = time.time()
    while job.running and time.time() - t0 < timeout:
        time.sleep(0.02)
    job.join(1)


class ProjectTest(unittest.TestCase):
    def test_validate_project(self):
        with tempfile.TemporaryDirectory() as d:
            c = project(Path(d))
            ph = [PhotoInfo("A.JPG", Path("A"), 53, 21, 45), PhotoInfo("B.JPG", Path("B"), None, None, None, "x")]
            iss = S.validate_project(c, ph, ALL_OK)
            self.assertFalse(S.has_errors(iss))
            self.assertIn("bez pozycji GPS", iss[0].message)
            S.set_start(c, "C.JPG")
            c.z_anchor.mode = "start_point"
            self.assertIn("nie istnieje", S.validate_project(c, ph, ALL_OK)[-1].message)
            S.set_start(c, None, None)
            msgs = [i.message for i in S.validate_project(c, ph, {**ALL_OK, "rekonstrukcja": False})]
            self.assertTrue(any("punktu startu" in m for m in msgs))
            self.assertTrue(any("Brak rekonstrukcji" in m for m in msgs))
            S.set_start(c, None, (21.0, 53.0))                    # zamienione lat/lon
            self.assertTrue(any("poza Polską" in i.message for i in S.validate_project(c, ph, ALL_OK)))
            c.photos_dir = Path(d) / "brak"
            self.assertTrue(S.has_errors(S.validate_project(c)))

    def test_odm_status_and_sidebar(self):
        self.assertEqual(S.odm_status(ALL_OK), (True, "Rekonstrukcja gotowa."))
        ready, text = S.odm_status({**ALL_OK, "DTM": False})
        self.assertTrue(ready)
        self.assertIn("brakuje: model terenu (DTM)", text)
        self.assertFalse(S.odm_status({**ALL_OK, "reference_lla": False})[0])
        c = PipelineConfig.from_dict({"model_path": "models/MODEL5.pt"})
        self.assertEqual(S.project_status(c, 26, ALL_OK), "Zdjęcia: 26 · Rekonstrukcja: gotowa · Model: MODEL5")


class DetectionTest(unittest.TestCase):
    def test_roundtrip_and_unknown_classes_ignored(self):
        with tempfile.TemporaryDirectory() as d:
            c = project(Path(d))
            c.size.ranges_m = {"sports ball": [0.1, 0.5], "namiot": [1.5, 6]}
            s = S.detection_settings(c)
            s.classes = ["people"]
            s.display_names = {"people": "człowiek", "namiot": "namiot"}
            s.conf_per_class = {"people": 0.3}
            s.point_per_class = {"people": "bbox_bottom_center", "namiot": "bbox_center"}
            with self.assertLogs("app.state", "INFO") as lg:
                S.apply_detection(c, s, ["namiot", "people"])
            self.assertTrue(any("sports ball" in m for m in lg.output))
            self.assertEqual(c.detect.classes, ["people"])
            self.assertEqual(c.detect.display_names, {"people": "człowiek"})      # identyczne nazwy pomijane
            self.assertEqual(c.point.per_class, {"people": "bbox_bottom_center"})  # rowne domyslnej pomijane
            self.assertEqual(set(c.size.ranges_m), {"namiot"})
            self.assertEqual(S.detection_settings(c).class_conf("people"), 0.3)
            self.assertEqual(S.validate_detection(c, model_info()), [])

    def test_validate_detection_messages(self):
        with tempfile.TemporaryDirectory() as d:
            c = project(Path(d))
            c.detect.classes = ["kot"]
            self.assertIn("Model nie zna klas: kot", S.validate_detection(c, model_info())[0].message)
            c.detect.classes = []
            self.assertIn("Nie wybrano", S.validate_detection(c, model_info())[0].message)
            c.detect.classes = "all"
            c.size.ranges_m = {"namiot": [5, 1]}
            self.assertIn("minimum", S.validate_detection(c, model_info())[0].message)
            c.size.ranges_m = {}
            ov = model_info((), open_vocab=True)
            self.assertIn("czego szukać", S.validate_detection(c, ov)[0].message)
            c.detect.prompt_text = ["ball"]
            self.assertEqual(S.validate_detection(c, ov), [])
            c.model_path = Path(d) / "brak.pt"
            self.assertIn("Nie ma pliku modelu", S.validate_detection(c, ov)[0].message)

    def test_model_summary_and_prompt(self):
        c = PipelineConfig.from_dict({"detect": {"display_names": {"people": "człowiek"}}})
        self.assertEqual(S.model_summary(model_info(), c), "Detekcja, 2 klasy: człowiek, namiot")
        self.assertEqual(S.model_summary(model_info(("a",) * 1), c), "Detekcja, 1 klasa: a")
        self.assertIn("5 klas", S.model_summary(model_info(tuple("abcde")), c))
        self.assertIn("opisu", S.model_summary(model_info((), True, "segment"), c))
        self.assertEqual(S.parse_prompt(" ball, tent;\nperson ,"), ["ball", "tent", "person"])
        self.assertIsNone(S.parse_prompt(" , "))


class GeorefTest(unittest.TestCase):
    def cov(self, nmt=True, nmpt=True):
        return {"nmt_local": FolderCoverage("NMT", Path("n"), nmt, message="ok" if nmt else "brak arkuszy"),
                "nmpt_local": FolderCoverage("NMPT", Path("p"), nmpt, message="ok" if nmpt else "brak arkuszy")}

    def test_resolve_surface(self):
        up, down = RestStatus(True, "ok"), RestStatus(False, "offline")
        self.assertEqual(S.resolve_surface(S.surface_availability(up, self.cov(), ALL_OK))[0], "nmpt_local")
        self.assertEqual(S.resolve_surface(S.surface_availability(up, self.cov(nmpt=False), ALL_OK))[0],
                         "nmt_gugik_rest")
        self.assertEqual(S.resolve_surface(S.surface_availability(down, self.cov(nmpt=False), ALL_OK))[0],
                         "nmt_local")
        self.assertEqual(S.resolve_surface(S.surface_availability(down, None, ALL_OK))[0], "odm_dsm")

    def test_validate_georef(self):
        c = PipelineConfig.from_dict({})
        c.surface.source, c.georef.mode, c.z_anchor.mode = "nmpt_local", "similarity_nmpt", "dem_alignment"
        msgs = [i for i in S.validate_georef(c, self.cov(nmpt=False), RestStatus(True, ""), {**ALL_OK, "DSM": False})]
        self.assertTrue(S.has_errors(msgs))
        self.assertTrue(any("nie pokrywają" in i.message for i in msgs))
        self.assertTrue(any("Wyrównanie do NMPT" in i.message for i in msgs))
        self.assertTrue(any("DSM" in i.message for i in msgs))
        c.georef.mode, c.surface.source = "auto", "nmt_gugik_rest"
        msgs = S.validate_georef(c, self.cov(nmpt=False), RestStatus(True, ""), ALL_OK)
        self.assertEqual([i.level for i in msgs], [S.INFO])
        msgs = S.validate_georef(c, self.cov(), RestStatus(False, ""), ALL_OK)
        self.assertIn("nie odpowiada", msgs[0].message)
        for i in msgs:
            self.assertNotIn("z_anchor", i.message)

    def test_action_and_summary(self):
        c = S.gui_defaults(PipelineConfig.from_dict({}))
        self.assertEqual(c.georef.mode, "auto")
        self.assertEqual(dict(S.settings_summary(c, 12))["Georeferencja"], "Automatycznie (zalecane)")
        S.apply_action(c, "georef_nmpt")
        self.assertEqual(c.georef.mode, "similarity_nmpt")
        with self.assertRaises(ValueError):
            S.apply_action(c, "nieznana")
        self.assertIn("Łączenie widoków", dict(S.settings_details(c)))
        r = SimpleNamespace(name="2026-11-17_113813", n_objects=5, model="MODEL5.pt")
        self.assertEqual(S.run_label(r), "17.11.2026 11:38 · 5 obiektów · MODEL5")
        self.assertEqual(S.param_label("aggregate.eps_m"), "odległość łączenia [m]")
        self.assertEqual(S.param_label("conf_per_class.people"), "próg pewności: people")


class PresetPrefsTest(unittest.TestCase):
    def test_preset_changes_only_detection(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            c = project(d)
            c.detect.classes = ["person"]
            c.detect.display_names = {"person": "człowiek"}
            p = S.save_preset(c, 'Ludzie: (model "COCO")', d)
            self.assertEqual(p.stem, "Ludzie (model COCO)")
            self.assertEqual(S.list_presets(d), ["Ludzie (model COCO)"])
            (d / "inny").mkdir()
            other = project(d / "inny")
            other.z_anchor.mode = "fixed"
            loaded = S.load_preset("Ludzie (model COCO)", other, d)
            self.assertEqual(loaded.detect.class_list(), ["person"])
            self.assertEqual(loaded.photos_dir, other.photos_dir)        # projekt bez zmian
            self.assertEqual(loaded.z_anchor.mode, "fixed")
            with self.assertRaises(FileNotFoundError):
                S.load_preset("brak", other, d)
            (d / "zly.yaml").write_text("photos_dir: x\n")
            with self.assertRaises(ValueError):
                S.load_preset("zly", other, d)
            with self.assertRaises(ValueError):
                S.preset_slug(' / ')

    def test_repo_presets_load(self):
        names = S.list_presets()
        self.assertIn("Ludzie (model COCO)", names)
        base = S.base_config()
        for n in names:
            c = S.load_preset(n, base)
            self.assertEqual(c.photos_dir, base.photos_dir)
            self.assertFalse(Path(str(S.preset_dict(c)["model_path"])).is_absolute())

    def test_prefs_and_last_project(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            self.assertEqual(S.load_prefs(d), {"surface_auto": True, "research_mode": False})
            S.save_prefs({"surface_auto": False, "research_mode": True}, d)
            p = S.load_prefs(d)
            self.assertFalse(p["surface_auto"])
            self.assertTrue(S.research_mode(p))
            with mock.patch.dict("os.environ", {S.RESEARCH_ENV: "1"}):
                self.assertTrue(S.research_mode({}))
            with mock.patch.dict("os.environ", {S.RESEARCH_ENV: ""}):
                self.assertFalse(S.research_mode({}))
            self.assertIsNone(S.last_project(d))
            c = project(d)
            S.remember_project(c, d)
            self.assertEqual(S.last_project(d).photos_dir, c.photos_dir)
            (d / S.LAST_PROJECT).write_text("detect: [zly")
            with self.assertLogs("app.state", "WARNING"):
                self.assertIsNone(S.last_project(d))


class JobTest(unittest.TestCase):
    def test_job_progress_logs_error(self):
        def work(job):
            import logging
            logging.getLogger("geodetect.test").info("krok 1")
            job.progress = 0.5
            return 42
        j = S.Job("t", work).start()
        wait(j)
        self.assertEqual((j.status, j.result), ("done", 42))
        self.assertTrue(any("krok 1" in x for x in j.logs))

        def boom(job):
            raise ValueError("zly plik")
        j = S.Job("t", boom).start()
        wait(j)
        self.assertEqual(j.status, "error")
        self.assertIn("zly plik", j.error)

    def test_run_job_with_fake_pipeline(self):
        calls = {}

        def fake_run(cfg, progress_cb, cancel, all_images, images, out_dir):
            calls["cfg"] = cfg
            for i, n in enumerate(["A.JPG", "B.JPG"], 1):
                progress_cb(Progress("images", i, 2, n, 0.1 + 0.4 * i))
                if cancel():
                    return RunResult(cancelled=True)
            return RunResult(objects=[1, 2, 3])
        c = PipelineConfig.from_dict({})
        j = S.start_run_job(c, run=fake_run)
        wait(j)
        self.assertEqual(j.status, "done")
        self.assertEqual(j.items, ["A.JPG", "B.JPG"])
        self.assertEqual(j.message, "Gotowe: 3 obiektów")
        self.assertIsNot(calls["cfg"], c)                         # przebieg na kopii configu

        def slow_run(cfg, progress_cb, cancel, **kw):
            while not cancel():
                time.sleep(0.01)
            return RunResult(cancelled=True)
        j = S.start_run_job(c, run=slow_run)
        j.cancel()
        wait(j)
        self.assertEqual(j.status, "cancelled")

        j = S.start_run_job(c, run=lambda *a, **k: RunResult(error="RuntimeError: brak NMT"))
        wait(j)
        self.assertEqual(j.status, "error")
        self.assertIn("brak NMT", j.error)

    def test_odm_job_with_fake_runner(self):
        with tempfile.TemporaryDirectory() as d:
            c = project(Path(d))
            (c.photos_dir / "A.JPG").write_bytes(b"x")
            got = {}

            def runner(cmd, line_cb, cancel):
                got["cmd"] = cmd
                line_cb("[INFO] Running openmvs stage")
                return SimpleNamespace(returncode=0, cancelled=False)
            j = S.start_odm_job(c, runner=runner)
            wait(j)
            self.assertEqual(j.status, "done", j.error)
            self.assertEqual(j.progress, 1.0)
            self.assertTrue((c.odm_dir / "images" / "A.JPG").exists())
            self.assertEqual(got["cmd"][0], "docker")
            j = S.start_odm_job(c, runner=lambda cmd, line_cb, cancel: SimpleNamespace(returncode=1, cancelled=False))
            wait(j)
            self.assertEqual(j.status, "error")
            self.assertIn("kod 1", j.error)

    def test_experiment_job(self):
        class FakeRunner:
            def __init__(self, exp, out, images, store):
                self.exp = exp

            def run(self):
                return Path("output/experiments") / self.exp.name
        j = S.start_experiment_job(S.list_experiments()[0], runner_cls=FakeRunner)
        wait(j)
        self.assertEqual(j.status, "done", j.error)


if __name__ == "__main__":
    unittest.main()
