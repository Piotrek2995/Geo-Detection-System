"""Smoke testy stron GUI (streamlit.testing AppTest) na syntetycznym projekcie: 3 zdjecia,
atrapa modelu i detektora, bez sieci (GUGiK, Docker) i bez zapisu stanu GUI uzytkownika."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from geodetect.config import PipelineConfig
from geodetect.coverage import RestStatus
from geodetect.detect import ModelInfo
from geodetect.odmrun import DockerStatus
from tests.test_gui_api import write_run

try:
    from streamlit.testing.v1 import AppTest
except ImportError:                       # streamlit nie zainstalowany (srodowisko tylko CLI)
    AppTest = None

APP = Path(__file__).resolve().parents[1] / "app"


@unittest.skipIf(AppTest is None, "brak streamlit")
class GuiPagesSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import logging

        import cv2
        from streamlit import logger as st_logger
        st_logger.set_log_level(logging.ERROR)         # "missing ScriptRunContext" w trybie testowym
        cls.tmp = tempfile.TemporaryDirectory()
        d = Path(cls.tmp.name)
        (d / "zdj").mkdir()
        for i in range(3):
            cv2.imwrite(str(d / "zdj" / f"IMG_{i}.JPG"), np.full((90, 160, 3), 80 + 40 * i, np.uint8))
        (d / "m.pt").write_bytes(b"0")
        cls.cfg = PipelineConfig.from_dict({"photos_dir": "zdj", "odm_dir": "odm", "output_dir": "out",
                                            "model_path": "m.pt",
                                            "detect": {"display_names": {"people": "człowiek"}},
                                            "georef": {"mode": "auto"}}, base_dir=d)
        o = {"object_id": 0, "class": "people", "x_2180": 0.0, "y_2180": 0.0, "altitude_m": 100.0, "lon": 21.0,
             "lat": 53.0, "status": "confirmed", "n_views": 3, "mean_conf": 0.9, "flags": []}
        qc = {"status": "OSTRZEŻENIE", "recommendations": ["rozważ georef.mode: similarity_nmpt (wyrównanie 7p do NMPT)"],
              "checks": [{"name": "tilt", "status": "OSTRZEŻENIE", "value": 3.9, "text": "", "details": {}},
                         {"name": "focal", "status": "informacja", "value": 53.0, "text": "",
                          "details": {"source": "EXIF"}}]}
        write_run(d / "out", "2026-01-01_100000", [o], {"model": "m.pt", "display_names": {"people": "człowiek"},
                                                        "photos_dir": str(d / "zdj")}, qc)
        write_run(d / "out", "2026-01-02_100000", [dict(o, x_2180=1.0)], {"model": "m.pt"})
        info = ModelInfo(d / "m.pt", "m.pt", "detect", {0: "namiot", 1: "people"}, 1024, 1.0)
        det = SimpleNamespace(detect=lambda p: [SimpleNamespace(bbox=(10, 10, 40, 40), conf=0.91, cls_name="people")])
        cls.patches = [
            mock.patch("app.ui.models", lambda folder: [info]),
            mock.patch("app.ui.detector_for", lambda c: det),
            mock.patch("app.ui.rest_status", lambda url: RestStatus(True, "ok", 0.1)),
            mock.patch("app.ui.docker_status", lambda: DockerStatus(False, None, False, "brak")),
            mock.patch("app.ui.coverage_for", lambda c: None),
            mock.patch("app.state.remember_project", lambda cfg, folder=None: None),
            mock.patch("app.state.save_prefs", lambda prefs, folder=None: None),
            mock.patch("geodetect.detect.model_info", lambda p: info),
        ]
        for p in cls.patches:
            p.start()

    @classmethod
    def tearDownClass(cls):
        for p in cls.patches:
            p.stop()
        cls.tmp.cleanup()

    def page(self, name: str, **state) -> "AppTest":
        at = AppTest.from_file(str(APP / name), default_timeout=60)
        at.session_state["cfg"] = PipelineConfig.from_dict(self.cfg.to_dict())
        at.session_state["prefs"] = {"surface_auto": True, "research_mode": False}
        for k, v in state.items():
            at.session_state[k] = v
        at.run()
        self.assertFalse(at.exception, [e.value for e in at.exception])
        return at

    def test_main_navigation(self):
        at = self.page("main.py")
        self.assertIn("Zdjęcia: 3", " ".join(c.value for c in at.sidebar.caption))
        self.assertEqual(at.title[0].value, "Projekt")

    def test_projekt(self):
        at = self.page("views/projekt.py")
        self.assertTrue(any("Brak rekonstrukcji" in e.value for e in list(at.error) + list(at.warning)))

    def test_detekcja_preview(self):
        at = self.page("views/detekcja.py")
        self.assertIn("Detekcja, 2 klasy: człowiek, namiot", [c.value for c in at.caption])
        self.assertEqual(at.multiselect[0].value, ["namiot", "people"])
        btn = next(b for b in at.button if b.label == "Pokaż detekcje")
        btn.click().run()
        self.assertFalse(at.exception)
        self.assertEqual(at.session_state["preview"][0], "IMG_0.JPG")

    def test_georeferencja(self):
        at = self.page("views/georeferencja.py")
        self.assertEqual(at.radio[0].value, "auto")
        self.assertEqual(at.session_state["cfg"].georef.mode, "auto")

    def test_uruchom_blocked_without_reconstruction(self):
        at = self.page("views/uruchom.py")
        self.assertTrue(any("Projekt:" in e.value for e in at.error))
        run_btn = next(b for b in at.button if b.label == "Uruchom przetwarzanie")
        self.assertTrue(run_btn.disabled)

    def test_wyniki(self):
        at = self.page("views/wyniki.py", sel_obj=0,
                       result_dir=str(Path(self.tmp.name) / "out" / "runs" / "2026-01-01_100000"))
        texts = " ".join(m.value for m in at.markdown)
        self.assertIn("Obiekt 0: człowiek", [s.value for s in at.subheader])
        self.assertIn("Na stronie Georeferencja wybierz wyrównanie do NMPT.", texts)
        self.assertNotIn("similarity_nmpt", texts)


if __name__ == "__main__":
    unittest.main()
